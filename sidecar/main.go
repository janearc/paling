// ==============================================================================
// Paling Service Mesh Sidecar (Go)
// ==============================================================================
// This service operates as the cluster-facing interface for the Paling ecosystem.
// While the primary Paling daemon MUST run on bare-metal to access Apple Silicon
// (Metal GPU) for MLX operations, the rest of the hyperscaler fleet relies on
// containerized service discovery (Traefik) and observability (Prometheus).
//
// Core Responsibilities:
//  1. Service Mesh Bridging: Exposes the bare-metal Paling node to the cluster
//     via Traefik docker labels (discovery is declarative, not a runtime POST;
//     see the service-discovery note above pollPaling).
//  2. Liveness Polling: Continuously polls the bare-metal daemon across the
//     host boundary (`host.docker.internal:8090`) to ensure MLX hasn't crashed.
//  3. Telemetry Export: Aggregates error rates, time-of-day histograms, and
//     success counters, re-exposing them to the cluster's Prometheus scraper
//     on port 9090.
//  4. Fault Tolerance: Implements exponential backoff and jitter on all network
//     boundaries to prevent thundering herds during cluster-wide reboots.
//
// ==============================================================================
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"strings"

	"github.com/cenkalti/backoff/v4"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/protobuf/types/known/timestamppb"

	"paling-sidecar/emit"
	observabilityv1 "paling-sidecar/gen/go/observability/v1"
	palingeventsv1 "paling-sidecar/gen/go/paling/events/v1"
	palingproto "paling-sidecar/proto"
)

const (
	topicObservability = "observability.events"
	topicPaling        = "paling.events"
	subjectHeartbeat   = "observability.v1.ServiceHealthHeartbeat"
	subjectBanchan     = "paling.events.v1.BanchanLifecycleEvent"
)

var startTime = time.Now()

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// startHeartbeat emits an observability.v1.ServiceHealthHeartbeat on a ticker.
// Best-effort: a publish failure is logged, never fatal.
func startHeartbeat(ctx context.Context, pub *emit.Publisher) {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			hb := &observabilityv1.ServiceHealthHeartbeat{
				ServiceName:        "paling",
				CurrentState:       observabilityv1.HealthState_HEALTH_STATE_GREEN,
				UptimeSeconds:      uint32(time.Since(startTime).Seconds()),
				InternalLoadMetric: 0,
				Timestamp:          timestamppb.Now(),
				IdempotencyKey:     fmt.Sprintf("paling-hb-%d", time.Now().UnixNano()),
			}
			if err := pub.Publish(ctx, topicObservability, subjectHeartbeat, palingproto.ObservabilitySchema, "paling", hb); err != nil {
				log.Printf("heartbeat emit failed: %v", err)
			}
		}
	}
}

// emitIntake is the HTTP endpoint paling's bare-metal daemon POSTs domain events
// to. The sidecar owns the protobuf/Schema-Registry encoding, so Python never
// touches Kafka. A nil publisher accepts and drops (best-effort).
func emitIntake(ctx context.Context, pub *emit.Publisher) http.HandlerFunc {
	type req struct {
		EventID     string `json:"event_id"`
		TraceID     string `json:"trace_id"`
		BentoID     string `json:"bento_id"`
		BanchanName string `json:"banchan_name"`
		State       string `json:"state"`
		ErrorMsg    string `json:"error_message"`
	}
	return func(w http.ResponseWriter, r *http.Request) {
		var in req
		if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		ev := &palingeventsv1.BanchanLifecycleEvent{
			EventId:      in.EventID,
			TraceId:      in.TraceID,
			OccurredAt:   timestamppb.Now(),
			BentoId:      in.BentoID,
			BanchanName:  in.BanchanName,
			State:        banchanState(in.State),
			ErrorMessage: in.ErrorMsg,
		}
		if err := pub.Publish(ctx, topicPaling, subjectBanchan, palingproto.BanchanSchema, in.BentoID, ev); err != nil {
			log.Printf("banchan emit failed: %v", err)
			http.Error(w, "emit failed", http.StatusBadGateway)
			return
		}
		w.WriteHeader(http.StatusAccepted)
	}
}

func banchanState(s string) palingeventsv1.BanchanState {
	switch strings.ToUpper(s) {
	case "QUEUED", "NOT_STARTED":
		return palingeventsv1.BanchanState_BANCHAN_STATE_QUEUED
	case "IN_PROGRESS":
		return palingeventsv1.BanchanState_BANCHAN_STATE_IN_PROGRESS
	case "PARTIAL":
		return palingeventsv1.BanchanState_BANCHAN_STATE_PARTIAL
	case "NEEDS_MASSAGE":
		return palingeventsv1.BanchanState_BANCHAN_STATE_NEEDS_MASSAGE
	case "DONE", "COMPLETED":
		return palingeventsv1.BanchanState_BANCHAN_STATE_COMPLETED
	case "FAILED":
		return palingeventsv1.BanchanState_BANCHAN_STATE_FAILED
	default:
		return palingeventsv1.BanchanState_BANCHAN_STATE_UNSPECIFIED
	}
}

var (
	errorHistogram = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "paling_error_time_of_day",
		Help:    "Histogram of errors based on the hour of the day (0-23).",
		Buckets: prometheus.LinearBuckets(0, 1, 24),
	})
	errorCounter = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "paling_poll_errors_total",
		Help: "Total number of polling errors",
	})
	successCounter = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "paling_poll_success_total",
		Help: "Total number of successful polls",
	})
)

func init() {
	prometheus.MustRegister(errorHistogram)
	prometheus.MustRegister(errorCounter)
	prometheus.MustRegister(successCounter)
	rand.Seed(time.Now().UnixNano())
}

func doWithRetries(operation func() error) error {
	b := backoff.NewExponentialBackOff()
	b.InitialInterval = 100 * time.Millisecond
	b.MaxInterval = 30 * time.Second
	b.RandomizationFactor = 0.1 // 10% jitter

	notify := func(err error, d time.Duration) {
		log.Printf("Operation failed: %v. Retrying in %v...", err, d)
	}

	// Wrap with max retries to preserve the original 8-attempt behavior
	return backoff.RetryNotify(operation, backoff.WithMaxRetries(b, 8), notify)
}

// NOTE on service discovery (issue #9): the sidecar does NOT POST a runtime
// registration to delightd. delightd has no such endpoint -- it discovers
// services two ways, neither of which is an HTTP call from the service:
//
//  1. Traefik routing: the docker provider reads the traefik.* labels on this
//     container (see docker-compose.yml). The bare-metal daemon, which is off
//     the docker network, is routed via Traefik's file provider; the daemon
//     installs its own route into ~/var/traefik/dynamic/paling.yml at startup
//     (it owns the host filesystem).
//  2. Agent skills: delightd's skill aggregator scans ~/work/<project>/mcp.json.
//     paling ships mcp.json at its repo root, so delightd surfaces the daemon's
//     operations as agent tools with no registration call.
//
// The previous registerWithDelightd() POSTed to a nonexistent route on the wrong
// port and was dead code; it has been removed in favour of the real mechanisms.

func pollPaling() {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		err := doWithRetries(func() error {
			resp, err := http.Get("http://host.docker.internal:8090/health")
			if err != nil {
				return err
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return fmt.Errorf("bad status: %d", resp.StatusCode)
			}
			return nil
		})

		if err != nil {
			log.Printf("Failed to poll paling: %v", err)
			errorCounter.Inc()
			hour := float64(time.Now().Hour())
			errorHistogram.Observe(hour)
		} else {
			successCounter.Inc()
		}
	}
}

func main() {
	log.Println("Starting Paling Go Sidecar...")

	// Service discovery is declarative, not a runtime POST: Traefik reads this
	// container's docker labels, the daemon installs its own Traefik file-route,
	// and delightd's skill aggregator scans paling's mcp.json. See the note on
	// the removed registerWithDelightd above.
	log.Println("Discovery: traefik docker labels + daemon file-route + mcp.json (no runtime registration)")

	// Polling loop
	go pollPaling()

	// Kafka emission (best-effort): the sidecar is paling's only producer, so
	// no Python touches Kafka/Schema-Registry/protobuf. A failure here disables
	// emission but never stops the sidecar.
	emitCtx, emitCancel := context.WithCancel(context.Background())
	defer emitCancel()
	var publisher *emit.Publisher
	if pub, err := emit.New(emitCtx, strings.Split(getenv("KAFKA_BROKERS", "kafka:9092"), ","), getenv("SCHEMA_REGISTRY_URL", "http://schema-registry:8081")); err != nil {
		log.Printf("Kafka emission disabled: %v", err)
	} else {
		publisher = pub
		defer publisher.Close()
		log.Println("Kafka emission ready")
		go startHeartbeat(emitCtx, publisher)
	}
	http.HandleFunc("/emit", emitIntake(emitCtx, publisher))

	// Expose metrics and health
	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok","service":"paling-sidecar"}`))
	})

	server := &http.Server{Addr: ":9090"}

	go func() {
		log.Println("Listening on :9090")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down sidecar...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server Shutdown Failed:%+v", err)
	}
	log.Println("Sidecar exited properly")
}
