// Package emit is paling's Kafka emission, owned entirely by the Go sidecar so
// that no Python touches Kafka, the Schema Registry, or protobuf (the design
// doc's tier-0 stance). Events are encoded in the Confluent Schema-Registry
// protobuf wire format so standard Confluent consumers (and obs-svc) decode them.
//
// TRADEOFFS vs confluent-kafka (librdkafka): we use franz-go (pure Go, no cgo)
// and hand-roll both the wire framing and schema registration. A framing bug is
// SILENT -- produce succeeds and only a consumer fails to deserialize. The
// message-index is the sharp edge here: it is NOT always the single 0x00 byte
// delightd's BackupEvent used. That is only the optimization for the FIRST
// message in a file. BanchanLifecycleEvent is the second message in
// banchan_event.proto, so its index is computed from the descriptor and zig-zag
// encoded per the Confluent format. Getting this wrong is exactly the class of
// bug the official serde would hide from us.
package emit

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"
	"google.golang.org/protobuf/proto"
)

// Publisher produces protobuf events in the Confluent SR wire format. A nil
// *Publisher is a valid no-op, so the sidecar can hold one unconditionally and
// let a disabled or unreachable Kafka be silent -- emission must never block the
// work it describes.
type Publisher struct {
	client *kgo.Client
	http   *http.Client
	srURL  string

	mu        sync.Mutex
	schemaIDs map[string]int32 // subject -> registry id, cached after first use
}

// New connects the producer. An error means emission is unavailable; the caller
// should log it and proceed with a nil Publisher rather than failing.
func New(ctx context.Context, brokers []string, schemaRegistryURL string) (*Publisher, error) {
	if len(brokers) == 0 {
		return nil, fmt.Errorf("no kafka brokers configured")
	}
	cl, err := kgo.NewClient(
		kgo.SeedBrokers(brokers...),
		kgo.RequiredAcks(kgo.AllISRAcks()),
		kgo.ProducerLinger(5*time.Millisecond),
	)
	if err != nil {
		return nil, err
	}
	if err := cl.Ping(ctx); err != nil {
		cl.Close()
		return nil, fmt.Errorf("kafka unreachable: %w", err)
	}
	return &Publisher{
		client:    cl,
		http:      &http.Client{Timeout: 5 * time.Second},
		srURL:     schemaRegistryURL,
		schemaIDs: map[string]int32{},
	}, nil
}

// Close releases the producer.
func (p *Publisher) Close() {
	if p != nil && p.client != nil {
		p.client.Close()
	}
}

// Publish registers the schema (lazily, once per subject) and produces msg to
// topic. A nil Publisher is a no-op; errors are returned for the caller to log.
func (p *Publisher) Publish(ctx context.Context, topic, subject, schemaText, key string, msg proto.Message) error {
	if p == nil {
		return nil
	}
	id, err := p.schemaID(ctx, subject, schemaText)
	if err != nil {
		return fmt.Errorf("schema registration: %w", err)
	}
	frame, err := encode(id, msg)
	if err != nil {
		return err
	}
	rec := &kgo.Record{Topic: topic, Key: []byte(key), Value: frame}
	return p.client.ProduceSync(ctx, rec).FirstErr()
}

// encode builds the Confluent protobuf wire format:
//
//	byte 0    : magic 0x00
//	bytes 1-4 : schema id, big-endian
//	N bytes   : message-index (see messageIndex)
//	rest      : serialized protobuf payload
func encode(schemaID int32, msg proto.Message) ([]byte, error) {
	payload, err := proto.Marshal(msg)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, 8+len(payload))
	out = append(out, 0x00)
	var id [4]byte
	binary.BigEndian.PutUint32(id[:], uint32(schemaID))
	out = append(out, id[:]...)
	out = append(out, messageIndex(msg)...)
	out = append(out, payload...)
	return out, nil
}

// messageIndex returns the Confluent message-index bytes for a top-level message.
// The index is the message's position among the file's messages. The array [0]
// (first message) is written as a single 0x00 byte; any other index is written
// as a zig-zag varint count (1) followed by the zig-zag varint index.
func messageIndex(msg proto.Message) []byte {
	idx := msg.ProtoReflect().Descriptor().Index()
	if idx == 0 {
		return []byte{0x00}
	}
	b := binary.AppendVarint(nil, 1)            // count = 1, zig-zag varint
	b = binary.AppendVarint(b, int64(idx))      // the index, zig-zag varint
	return b
}

// schemaID returns the registry id for subject, registering schemaText on the
// first call for that subject and caching the result for subsequent emits.
func (p *Publisher) schemaID(ctx context.Context, subject, schemaText string) (int32, error) {
	p.mu.Lock()
	if id, ok := p.schemaIDs[subject]; ok {
		p.mu.Unlock()
		return id, nil
	}
	p.mu.Unlock()

	id, err := p.registerSchema(ctx, subject, schemaText)
	if err != nil {
		return 0, err
	}
	p.mu.Lock()
	p.schemaIDs[subject] = id
	p.mu.Unlock()
	return id, nil
}

// registerSchema POSTs schemaText under subject (RecordNameStrategy) and returns
// the registry-assigned id. Re-registering an identical schema is idempotent.
func (p *Publisher) registerSchema(ctx context.Context, subject, schemaText string) (int32, error) {
	body, err := json.Marshal(map[string]string{"schemaType": "PROTOBUF", "schema": schemaText})
	if err != nil {
		return 0, err
	}
	url := fmt.Sprintf("%s/subjects/%s/versions", strings.TrimRight(p.srURL, "/"), subject)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(body)))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/vnd.schemaregistry.v1+json")

	resp, err := p.http.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return 0, fmt.Errorf("registry returned %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var out struct {
		ID int32 `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, err
	}
	return out.ID, nil
}
