package emit

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	palingeventsv1 "paling-sidecar/gen/go/paling/events/v1"
)

// TestNew_RejectsEmptyBrokers pins the fail-closed contract: no brokers means
// emission is unavailable and the caller proceeds with a nil Publisher.
func TestNew_RejectsEmptyBrokers(t *testing.T) {
	if _, err := New(context.Background(), nil, "http://x"); err == nil {
		t.Fatal("expected error for empty brokers")
	}
}

// TestNew_UnreachableBroker proves a dead broker surfaces as an error (via the
// startup Ping) rather than hanging the sidecar.
func TestNew_UnreachableBroker(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancelled so Ping returns promptly instead of dialing forever
	if _, err := New(ctx, []string{"127.0.0.1:1"}, "http://x"); err == nil {
		t.Fatal("expected error for unreachable broker")
	}
}

// TestNilPublisher_IsNoOp confirms a nil *Publisher accepts Publish/Close as
// no-ops, the property that lets the sidecar hold one unconditionally.
func TestNilPublisher_IsNoOp(t *testing.T) {
	var p *Publisher
	if err := p.Publish(context.Background(), "t", "s", "schema", "k", &palingeventsv1.BanchanLifecycleEvent{}); err != nil {
		t.Errorf("nil Publish should be a no-op, got %v", err)
	}
	p.Close() // must not panic
}

// TestPublish_SchemaErrorPropagates drives Publish far enough to register a
// schema and fail there (registry returns 500), proving a registration failure
// is returned for the caller to log rather than producing a malformed record.
func TestPublish_SchemaErrorPropagates(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	p := &Publisher{http: srv.Client(), srURL: srv.URL, schemaIDs: map[string]int32{}}
	err := p.Publish(context.Background(), "topic", "subj", "schema", "key", &palingeventsv1.BanchanLifecycleEvent{})
	if err == nil {
		t.Fatal("expected schema-registration error to propagate")
	}
}

// TestSchemaID_RegistersOnceAndCaches exercises registerSchema against a fake
// Schema Registry and asserts the per-subject cache: the registry is hit once,
// the second call is served from the cache.
func TestSchemaID_RegistersOnceAndCaches(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/vnd.schemaregistry.v1+json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id": 42}`))
	}))
	defer srv.Close()

	p := &Publisher{
		http:      srv.Client(),
		srURL:     srv.URL,
		schemaIDs: map[string]int32{},
	}
	id, err := p.schemaID(context.Background(), "paling.events.v1.Banchan", "schema-text")
	if err != nil {
		t.Fatalf("schemaID: %v", err)
	}
	if id != 42 {
		t.Errorf("id = %d, want 42", id)
	}
	if _, err := p.schemaID(context.Background(), "paling.events.v1.Banchan", "schema-text"); err != nil {
		t.Fatalf("cached schemaID: %v", err)
	}
	if hits != 1 {
		t.Errorf("registry hit %d times, want 1 (second call must be cached)", hits)
	}
}

// TestRegisterSchema_Non200IsError verifies a registry rejection surfaces as an
// error the caller logs rather than a silent bad schema id.
func TestRegisterSchema_Non200IsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusConflict)
		_, _ = w.Write([]byte("incompatible"))
	}))
	defer srv.Close()

	p := &Publisher{http: srv.Client(), srURL: srv.URL, schemaIDs: map[string]int32{}}
	if _, err := p.registerSchema(context.Background(), "subj", "schema"); err == nil {
		t.Fatal("expected error on non-200 registry response")
	}
}
