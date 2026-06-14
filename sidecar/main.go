// ==============================================================================
// Paling Service Mesh Sidecar (Go)
// ==============================================================================
// This service operates as the cluster-facing interface for the Paling ecosystem.
// While the primary Paling daemon MUST run on bare-metal to access Apple Silicon
// (Metal GPU) for MLX operations, the rest of the hyperscaler fleet relies on 
// containerized service discovery (Traefik) and observability (Prometheus).
// 
// Core Responsibilities:
// 1. Service Mesh Bridging: Registers the bare-metal Paling node with the 
//    central `delightd` control plane and exposes standard ports to Traefik.
// 2. Liveness Polling: Continuously polls the bare-metal daemon across the 
//    host boundary (`host.docker.internal:8090`) to ensure MLX hasn't crashed.
// 3. Telemetry Export: Aggregates error rates, time-of-day histograms, and 
//    success counters, re-exposing them to the cluster's Prometheus scraper 
//    on port 9090.
// 4. Fault Tolerance: Implements exponential backoff and jitter on all network 
//    boundaries to prevent thundering herds during cluster-wide reboots.
// ==============================================================================
package main

import (
	"bytes"
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

	"github.com/cenkalti/backoff/v4"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

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

func registerWithDelightd() error {
	// Construct the service registration payload for the control plane.
	// This declares our presence to delightd so Traefik can dynamically route 
	// cluster traffic to our exposed sidecar port.
	payload := map[string]interface{}{
		"service": "paling",
		"port":    9090, // We expose sidecar on 9090
	}
	data, _ := json.Marshal(payload)
	req, err := http.NewRequest("POST", "http://localhost:8080/projects/paling/register", bytes.NewBuffer(data))
	if err != nil {
		return err
	}
	
	// Execute the registration HTTP request. We assume delightd is reachable 
	// locally or via standard mesh routing rules.
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode >= 400 {
		return fmt.Errorf("failed to register, status code: %d", resp.StatusCode)
	}
	return nil
}

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

	// Register
	go func() {
		if err := doWithRetries(registerWithDelightd); err != nil {
			log.Printf("Warning: delightd registration failed: %v", err)
		} else {
			log.Println("Registered with delightd.")
		}
	}()

	// Polling loop
	go pollPaling()

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
