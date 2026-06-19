package consume

import (
	"context"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"
	"google.golang.org/protobuf/proto"

	palingeventsv1 "paling-sidecar/gen/go/paling/events/v1"
)

// frameFor builds a Confluent SR protobuf frame for a first/only message, the
// exact bytes emit.encode produces for an index-0 message. OrchestrationCommand
// is the only message in its file, so this is the on-wire shape the producer
// uses; decode must invert it.
func frameFor(t *testing.T, schemaID int32, msg proto.Message) []byte {
	t.Helper()
	payload, err := proto.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	out := []byte{0x00}
	var id [4]byte
	binary.BigEndian.PutUint32(id[:], uint32(schemaID))
	out = append(out, id[:]...)
	out = append(out, 0x00) // single-byte message-index (first message)
	out = append(out, payload...)
	return out
}

func TestDecode_RoundTrip(t *testing.T) {
	cmd := &palingeventsv1.OrchestrationCommand{
		CommandId: "c1",
		TraceId:   "t1",
		BentoId:   "b1",
		Action:    palingeventsv1.OrchestrationAction_ORCHESTRATION_ACTION_TRAIN,
		IssuedBy:  "fleet-svc",
	}
	got, err := decode(frameFor(t, 11, cmd))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.GetCommandId() != "c1" || got.GetBentoId() != "b1" {
		t.Errorf("round-trip mismatch: %+v", got)
	}
	if got.GetAction() != palingeventsv1.OrchestrationAction_ORCHESTRATION_ACTION_TRAIN {
		t.Errorf("action = %v", got.GetAction())
	}
}

func TestDecode_MultiIndexVarint(t *testing.T) {
	// exercise the general varint message-index path (count=1, index=0) to prove
	// a future multi-message file would still decode. payload is an empty cmd.
	payload, _ := proto.Marshal(&palingeventsv1.OrchestrationCommand{CommandId: "x"})
	frame := []byte{0x00, 0x00, 0x00, 0x00, 0x07}
	frame = binary.AppendVarint(frame, 1) // count
	frame = binary.AppendVarint(frame, 0) // index 0
	frame = append(frame, payload...)
	got, err := decode(frame)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.GetCommandId() != "x" {
		t.Errorf("got %q", got.GetCommandId())
	}
}

func TestDecode_RejectsShortFrame(t *testing.T) {
	if _, err := decode([]byte{0x00, 0x01}); err == nil {
		t.Error("expected error on truncated frame")
	}
}

func TestDecode_RejectsBadMagic(t *testing.T) {
	if _, err := decode([]byte{0x01, 0x00, 0x00, 0x00, 0x07, 0x00}); err == nil {
		t.Error("expected error on non-zero magic")
	}
}

func TestDecode_TruncatedBeforeIndex(t *testing.T) {
	// magic + 4-byte id and nothing else: no room for a message-index.
	if _, err := decode([]byte{0x00, 0x00, 0x00, 0x00, 0x07}); err == nil {
		t.Error("expected error when frame ends before message-index")
	}
}

func TestDecode_BadCountVarint(t *testing.T) {
	// a non-zero first index byte signals the varint path; an incomplete varint
	// for the count must be rejected, not panic.
	frame := []byte{0x00, 0x00, 0x00, 0x00, 0x07, 0xFF}
	if _, err := decode(frame); err == nil {
		t.Error("expected error on malformed count varint")
	}
}

func TestDecode_UnmarshalFailure(t *testing.T) {
	// a valid frame header followed by bytes that are not a valid protobuf must
	// surface as a decode error.
	frame := []byte{0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0xFF, 0xFF, 0xFF}
	if _, err := decode(frame); err == nil {
		t.Error("expected protobuf unmarshal error")
	}
}

func TestRelay_PostsDecodedCommand(t *testing.T) {
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()

	c := &Consumer{daemonURL: srv.URL, http: srv.Client()}
	cmd := &palingeventsv1.OrchestrationCommand{CommandId: "c1", BentoId: "b1"}
	if err := c.relay(context.Background(), cmd); err != nil {
		t.Fatalf("relay: %v", err)
	}
	if atomic.LoadInt32(&hits) != 1 {
		t.Errorf("daemon hit count = %d, want 1", hits)
	}
}

func TestRelay_PermanentOn4xx(t *testing.T) {
	// a 4xx is the daemon rejecting the command; relay must not retry forever.
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	c := &Consumer{daemonURL: srv.URL, http: srv.Client()}
	err := c.relay(context.Background(), &palingeventsv1.OrchestrationCommand{CommandId: "c1"})
	if err == nil {
		t.Fatal("expected error on 4xx")
	}
	if atomic.LoadInt32(&hits) != 1 {
		t.Errorf("4xx must not be retried: hit count = %d, want 1", hits)
	}
}

func TestRelay_NewRequestErrorIsPermanent(t *testing.T) {
	// an unconstructable request (bad URL) must fail without infinite retry.
	c := &Consumer{daemonURL: "://bad", http: http.DefaultClient}
	if err := c.relay(context.Background(), &palingeventsv1.OrchestrationCommand{CommandId: "c1"}); err == nil {
		t.Fatal("expected error for malformed daemon URL")
	}
}

func TestHandle_DecodeFailureDropsRecord(t *testing.T) {
	// a malformed frame must be dropped (logged), never relayed, and must not
	// wedge the loop -- so no daemon call should happen.
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
	}))
	defer srv.Close()
	c := &Consumer{daemonURL: srv.URL, http: srv.Client()}
	c.handle(context.Background(), &kgo.Record{Value: []byte{0xFF}}) // bad magic
	if atomic.LoadInt32(&hits) != 0 {
		t.Errorf("malformed frame must not be relayed: hits = %d", hits)
	}
}

func TestHandle_ValidFrameRelays(t *testing.T) {
	// the happy path through handle: decode a real frame and relay it.
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()
	c := &Consumer{daemonURL: srv.URL, http: srv.Client()}
	cmd := &palingeventsv1.OrchestrationCommand{CommandId: "c1", BentoId: "b1"}
	c.handle(context.Background(), &kgo.Record{Value: frameFor(t, 11, cmd)})
	if atomic.LoadInt32(&hits) != 1 {
		t.Errorf("valid frame should relay once: hits = %d", hits)
	}
}

func TestNew_RejectsEmptyBrokers(t *testing.T) {
	// no brokers configured is an immediate, non-panicking error so the caller
	// proceeds with a nil consumer (inbound disabled, daemon unaffected).
	if _, err := New(context.Background(), nil, "t", "g", "http://x"); err == nil {
		t.Fatal("expected error for empty brokers")
	}
}

func TestNew_UnreachableBrokerErrors(t *testing.T) {
	// a broker that does not answer must surface as an error within the ping,
	// not hang the sidecar startup.
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancelled: New's Ping returns promptly rather than dialing forever
	if _, err := New(ctx, []string{"127.0.0.1:1"}, "t", "g", "http://x"); err == nil {
		t.Fatal("expected error for unreachable broker")
	}
}

func TestNilConsumer_RunReturns(t *testing.T) {
	// a nil consumer is a valid no-op when kafka is unreachable.
	var c *Consumer
	c.Run(context.Background()) // must not panic
	c.Close()                   // must not panic
}

func TestRun_PollErrorBacksOffThenStops(t *testing.T) {
	// drive the poll-error branch: a consumer subscribed to a topic on a dead
	// broker produces fetch errors, which Run logs and backs off on. A short
	// deadline then cancels the loop, proving the backoff path is non-fatal and
	// honours cancellation mid-backoff.
	cl, err := kgo.NewClient(
		kgo.SeedBrokers("127.0.0.1:1"),
		kgo.ConsumeTopics("paling.orchestration"),
		kgo.ConsumerGroup("test-grp"),
	)
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	c := &Consumer{client: cl, daemonURL: "http://x", http: http.DefaultClient}
	defer c.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	done := make(chan struct{})
	go func() { c.Run(ctx); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not return after context deadline")
	}
}

func TestRun_StopsOnCancelledContext(t *testing.T) {
	// Run must exit promptly when its context is cancelled (clean shutdown),
	// exercising the poll-loop's cancellation path without a live broker.
	cl, err := kgo.NewClient(kgo.SeedBrokers("127.0.0.1:1"))
	if err != nil {
		t.Fatalf("client: %v", err)
	}
	c := &Consumer{client: cl, daemonURL: "http://x", http: http.DefaultClient}
	defer c.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	done := make(chan struct{})
	go func() { c.Run(ctx); close(done) }()
	<-done // returns or the test times out -> failure
}
