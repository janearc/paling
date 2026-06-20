// ==============================================================================
// Paling Service Mesh Sidecar (Go)
// ==============================================================================
// This is the container-side companion to the Paling daemon. The daemon itself
// MUST run on bare-metal to reach Apple Silicon (Metal GPU) for MLX operations,
// so it cannot live on the docker network; this sidecar runs in a container and
// bridges that bare-metal process into the rest of the fleet's container-based
// service discovery (Traefik) and metrics scraping (Prometheus).
//
// The value this sidecar earns is in the bad cases, not the steady state. When
// the whole cluster reboots at once -- a host power-cycle, a docker substrate
// restart, recovery from an outage that took the broker and registry down with
// it -- every network boundary it owns retries with exponential backoff and
// jitter and every downstream dependency is treated as optionally-absent. The
// sidecar comes back up on its own and re-attaches to whatever is available,
// rather than wedging or requiring a hand-restart. Day-to-day polling and
// emission are the easy path; converging back to a working state after a
// cluster-wide failure is the job it exists to do.
//
// Concretely it does four things:
//  1. Service bridging: exposes the bare-metal Paling node to the fleet via
//     Traefik docker labels (discovery is declarative, not a runtime POST;
//     see the service-discovery note above pollPaling).
//  2. Liveness polling: continuously polls the bare-metal daemon across the
//     host boundary (`host.docker.internal:8090`) to detect an MLX crash.
//  3. Telemetry export: aggregates error rates, time-of-day histograms, and
//     success counters, re-exposing them to the Prometheus scraper on :9090.
//  4. Retry on every boundary: exponential backoff with jitter on all network
//     calls, so a simultaneous restart of many services does not produce a
//     synchronized retry storm against a recovering dependency.
//
// ==============================================================================
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
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

	"paling-sidecar/consume"
	"paling-sidecar/emit"
	observabilityv1 "paling-sidecar/gen/go/observability/v1"
	palingeventsv1 "paling-sidecar/gen/go/paling/events/v1"
	palingproto "paling-sidecar/proto"
)

const (
	topicObservability = "observability.events"
	topicPaling        = "paling.events"
	topicOrchestration = "paling.orchestration"
	orchestrationGroup = "paling-sidecar"
	subjectHeartbeat   = "observability.v1.ServiceHealthHeartbeat"
	subjectBanchan     = "paling.events.v1.BanchanLifecycleEvent"
	// defaultDaemonBase is the bare-metal `paling serve` daemon's base URL as seen
	// from inside the cluster. host.k3d.internal is the name k3d injects into
	// CoreDNS for the host gateway; it resolves to the host at runtime, so no host
	// IP is baked in. Override with PALING_DAEMON_BASE (a different name or a LAN IP).
	defaultDaemonBase = "http://host.k3d.internal:8090"
)

var startTime = time.Now()

// log is the single logger this file uses. It emits structured JSON to stderr so
// the lines are ingestible by the fleet's log pipeline rather than free-form text
// on stdout. Everything below logs through this -- there is no bare fmt.Print or
// stdlib log.Println anywhere in the sidecar, so the output format is uniform.
var log = slog.New(slog.NewJSONHandler(os.Stderr, nil))

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
				log.Error("heartbeat emit failed", "err", err)
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
			log.Error("banchan emit failed", "err", err)
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
		log.Warn("operation failed, retrying", "err", err, "retry_in", d)
	}

	// Wrap with max retries to preserve the original 8-attempt behavior
	return backoff.RetryNotify(operation, backoff.WithMaxRetries(b, 8), notify)
}

// service discovery (landed via issue #9): the sidecar does NOT POST a runtime
// registration to delightd. delightd has no such endpoint -- it discovers
// services two ways, neither of which is an HTTP call from the service:
//
//  1. Traefik routing: the docker provider reads the traefik.* labels on this
//     container (see docker-compose.yml). The bare-metal daemon, which is off
//     the docker network, is routed via Traefik's file provider; the daemon
//     installs its own route into ${PALING_VAR}/var/traefik/dynamic/paling.yml
//     at startup (it owns the host filesystem).
//  2. Agent skills: delightd's skill aggregator scans ~/work/<project>/mcp.json.
//     paling ships mcp.json at its repo root, so delightd surfaces the daemon's
//     operations as agent tools with no registration call.
//
// The previous registerWithDelightd() POSTed to a nonexistent route on the wrong
// port and was dead code; it has been removed in favour of the real mechanisms.

func pollPaling(base string) {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		err := doWithRetries(func() error {
			resp, err := http.Get(base + "/health")
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
			log.Error("failed to poll paling", "err", err)
			errorCounter.Inc()
			hour := float64(time.Now().Hour())
			errorHistogram.Observe(hour)
		} else {
			successCounter.Inc()
		}
	}
}

func main() {
	log.Info("starting paling go sidecar")

	// Service discovery is declarative, not a runtime POST: Traefik reads this
	// container's docker labels, the daemon installs its own Traefik file-route,
	// and delightd's skill aggregator scans paling's mcp.json. See the note on
	// the removed registerWithDelightd above.
	log.Info("discovery is declarative", "mechanisms", "traefik docker labels + daemon file-route + mcp.json", "runtime_registration", false)

	// Polling loop against the bare-metal daemon over the host boundary. The base
	// URL resolves the host gateway by name (host.k3d.internal), so no host IP is
	// baked in; PALING_DAEMON_BASE overrides it.
	daemonBase := getenv("PALING_DAEMON_BASE", defaultDaemonBase)
	go pollPaling(daemonBase)

	// Kafka emission (best-effort): the sidecar is paling's only producer, so
	// no Python touches Kafka/Schema-Registry/protobuf. A failure here disables
	// emission but never stops the sidecar.
	emitCtx, emitCancel := context.WithCancel(context.Background())
	defer emitCancel()
	var publisher *emit.Publisher
	if pub, err := emit.New(emitCtx, strings.Split(getenv("KAFKA_BROKERS", "kafka:9092"), ","), getenv("SCHEMA_REGISTRY_URL", "http://schema-registry:8081")); err != nil {
		log.Warn("kafka emission disabled", "err", err)
	} else {
		publisher = pub
		defer publisher.Close()
		log.Info("kafka emission ready")
		go startHeartbeat(emitCtx, publisher)
	}
	http.HandleFunc("/emit", emitIntake(emitCtx, publisher))

	// Inbound orchestration (best-effort): consume OrchestrationCommands off
	// Kafka and relay them to the bare-metal daemon. A down broker disables
	// inbound control but never stops the sidecar -- symmetric with emission.
	brokers := strings.Split(getenv("KAFKA_BROKERS", "kafka:9092"), ",")
	daemonOrchestrateURL := getenv("PALING_DAEMON_ORCHESTRATE_URL", daemonBase+"/orchestrate")
	if cons, err := consume.New(emitCtx, brokers, topicOrchestration, orchestrationGroup, daemonOrchestrateURL); err != nil {
		log.Warn("kafka orchestration consumer disabled", "err", err)
	} else {
		defer cons.Close()
		log.Info("kafka orchestration consumer ready")
		go cons.Run(emitCtx)
	}

	// Expose metrics and health
	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok","service":"paling-sidecar"}`))
	})

	server := &http.Server{Addr: ":9090"}

	go func() {
		log.Info("listening", "addr", ":9090")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("shutting down sidecar")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Error("server shutdown failed", "err", err)
		os.Exit(1)
	}
	log.Info("sidecar exited properly")
}
