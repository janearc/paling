// Package consume is paling's INBOUND Kafka boundary, the mirror of the emit
// package. The fleet (delightd / fleet-svc) produces OrchestrationCommand
// messages onto a topic to drive a bento through the pipeline; this consumer
// decodes the Confluent Schema-Registry protobuf frame and relays each command
// to the bare-metal daemon's HTTP control plane. As with emission, no Python
// ever touches Kafka -- the sidecar owns the wire format on both directions.
//
// AVAILABILITY: a down broker must never break the daemon. New() pings the
// broker and returns an error the caller logs-and-proceeds on; Run() retries
// the consume loop with backoff so a broker that comes up later is picked up
// without restarting the sidecar. Relay failures to the daemon are logged and
// the record is still committed -- the daemon is the system of record for work,
// and the fleet re-issues idempotent commands (command_id) rather than relying
// on Kafka redelivery to a process that may be mid-restart.
package consume

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/twmb/franz-go/pkg/kgo"
	"google.golang.org/protobuf/proto"

	palingeventsv1 "paling-sidecar/gen/go/paling/events/v1"
)

// Consumer pulls OrchestrationCommands off Kafka and relays them to the daemon.
// A nil *Consumer is a valid no-op so the sidecar can hold one unconditionally
// when Kafka is unreachable -- inbound orchestration is best-effort, exactly
// like outbound emission.
type Consumer struct {
	client    *kgo.Client
	daemonURL string
	http      *http.Client
}

// New connects the consumer to the orchestration topic in its own consumer
// group. An error means inbound orchestration is unavailable; the caller logs
// it and proceeds with a nil Consumer rather than failing the sidecar.
func New(ctx context.Context, brokers []string, topic, group, daemonURL string) (*Consumer, error) {
	if len(brokers) == 0 {
		return nil, fmt.Errorf("no kafka brokers configured")
	}
	cl, err := kgo.NewClient(
		kgo.SeedBrokers(brokers...),
		kgo.ConsumerGroup(group),
		kgo.ConsumeTopics(topic),
		// start at the latest offset: orchestration is live control traffic, not
		// a backlog to replay. a restarted sidecar acts on new commands only.
		kgo.ConsumeResetOffset(kgo.NewOffset().AtEnd()),
	)
	if err != nil {
		return nil, err
	}
	if err := cl.Ping(ctx); err != nil {
		cl.Close()
		return nil, fmt.Errorf("kafka unreachable: %w", err)
	}
	return &Consumer{
		client:    cl,
		daemonURL: daemonURL,
		http:      &http.Client{Timeout: 5 * time.Second},
	}, nil
}

// Close releases the consumer.
func (c *Consumer) Close() {
	if c != nil && c.client != nil {
		c.client.Close()
	}
}

// Run drives the poll loop until ctx is cancelled. A nil Consumer returns
// immediately. Poll errors are retried with exponential backoff + jitter so a
// transient broker outage degrades into retry rather than a crash.
func (c *Consumer) Run(ctx context.Context) {
	if c == nil {
		return
	}
	b := backoff.NewExponentialBackOff()
	b.InitialInterval = 100 * time.Millisecond
	b.MaxInterval = 30 * time.Second
	b.MaxElapsedTime = 0 // never give up; the sidecar lives as long as the fleet
	b.RandomizationFactor = 0.1

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		fetches := c.client.PollFetches(ctx)
		if errs := fetches.Errors(); len(errs) > 0 {
			// ctx cancellation surfaces here as an error too; bail cleanly.
			if ctx.Err() != nil {
				return
			}
			for _, e := range errs {
				log.Printf("orchestration poll error (topic %s): %v", e.Topic, e.Err)
			}
			d := b.NextBackOff()
			select {
			case <-ctx.Done():
				return
			case <-time.After(d):
			}
			continue
		}
		b.Reset()
		fetches.EachRecord(func(rec *kgo.Record) {
			c.handle(ctx, rec)
		})
	}
}

// handle decodes one record and relays it to the daemon. Decode failures are
// logged and dropped (a malformed command is not retryable); relay failures are
// logged after a bounded retry. Either way the loop keeps moving -- one bad or
// undeliverable command must not wedge the stream.
func (c *Consumer) handle(ctx context.Context, rec *kgo.Record) {
	cmd, err := decode(rec.Value)
	if err != nil {
		log.Printf("orchestration decode failed (dropping): %v", err)
		return
	}
	if err := c.relay(ctx, cmd); err != nil {
		log.Printf("orchestration relay to daemon failed (command_id=%s): %v", cmd.GetCommandId(), err)
	}
}

// relay POSTs the decoded command to the daemon's /orchestrate endpoint, with a
// bounded exponential backoff so a daemon mid-restart is given a few chances
// before the command is abandoned to the fleet's idempotent re-issue.
func (c *Consumer) relay(ctx context.Context, cmd *palingeventsv1.OrchestrationCommand) error {
	payload := map[string]any{
		"command_id":      cmd.GetCommandId(),
		"trace_id":        cmd.GetTraceId(),
		"bento_id":        cmd.GetBentoId(),
		"action":          cmd.GetAction().String(),
		"parameters_json": cmd.GetParametersJson(),
		"issued_by":       cmd.GetIssuedBy(),
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	op := func() error {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.daemonURL, bytes.NewReader(body))
		if err != nil {
			return backoff.Permanent(err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := c.http.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 500 {
			return fmt.Errorf("daemon returned %d", resp.StatusCode)
		}
		if resp.StatusCode >= 400 {
			// a 4xx is the daemon rejecting the command; retrying won't help.
			return backoff.Permanent(fmt.Errorf("daemon rejected command: %d", resp.StatusCode))
		}
		return nil
	}

	b := backoff.NewExponentialBackOff()
	b.InitialInterval = 100 * time.Millisecond
	b.MaxInterval = 5 * time.Second
	b.RandomizationFactor = 0.1
	return backoff.Retry(op, backoff.WithContext(backoff.WithMaxRetries(b, 5), ctx))
}

// decode strips the Confluent Schema-Registry protobuf wire framing and
// unmarshals an OrchestrationCommand. Frame layout (mirror of emit.encode):
//
//	byte 0    : magic 0x00
//	bytes 1-4 : schema id, big-endian (ignored on read -- the type is fixed)
//	N bytes   : message-index (single 0x00 for a first/only message, else a
//	            zig-zag varint count followed by that many zig-zag varint indices)
//	rest      : serialized protobuf payload
//
// OrchestrationCommand is the only message in its file, so the producer writes
// the 0x00 single-byte index; the general varint path is handled regardless so
// a future multi-message file does not silently corrupt decoding.
func decode(frame []byte) (*palingeventsv1.OrchestrationCommand, error) {
	if len(frame) < 5 || frame[0] != 0x00 {
		return nil, fmt.Errorf("not a confluent SR frame (len=%d)", len(frame))
	}
	rest := frame[5:] // skip magic + 4-byte schema id

	// message-index: a leading single 0x00 is the first-message optimization.
	if len(rest) == 0 {
		return nil, fmt.Errorf("frame truncated before message-index")
	}
	if rest[0] == 0x00 {
		rest = rest[1:]
	} else {
		count, n := binary.Varint(rest)
		if n <= 0 {
			return nil, fmt.Errorf("bad message-index count varint")
		}
		rest = rest[n:]
		for i := int64(0); i < count; i++ {
			_, n := binary.Varint(rest)
			if n <= 0 {
				return nil, fmt.Errorf("bad message-index varint")
			}
			rest = rest[n:]
		}
	}

	var cmd palingeventsv1.OrchestrationCommand
	if err := proto.Unmarshal(rest, &cmd); err != nil {
		return nil, fmt.Errorf("protobuf unmarshal: %w", err)
	}
	return &cmd, nil
}
