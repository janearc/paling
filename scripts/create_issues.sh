#!/bin/bash
set -e

echo "Creating GitHub issues for the Paling recovery plan..."

gh issue create --title "uv: blacklist NVIDIA/CUDA on macOS/arm64; regenerate lock" --body "uv respects platform markers natively. Exclude NVIDIA/CUDA wheels on macOS/arm64. Regenerate uv.lock and ensure ~/.cache/uv is guarded."
gh issue create --title "daemon: paling serve subcommand + internal HTTP API" --body "Add paling serve daemon. Implement bento/banchan state machines. Expose /health, /metrics, /bento/:id/status."
gh issue create --title "infra: LaunchDaemon plist for self-healing startup" --body "Create com.jane.paling.plist with KeepAlive=true, RunAtLoad=true. Invokes uv run paling serve inside project venv."
gh issue create --title "kafka: define and register Protobuf banchan event schema" --body "Define paling/proto/banchan_event.proto. Must include optional tool_event reserved field. Needs Schema Registry (Buf or Redpanda) before 1.0 stable release."
gh issue create --title "kafka: producer + consumer in paling daemon" --body "Producer for banchan state transitions. Consumer for inbound orchestration events. Add exponential backoff on all retries."
gh issue create --title "banchan: partial ingestion / resume for pre-computed banchan" --body "Detect pre-computed banchan (e.g. existing taxonometry files) and transition state directly to PARTIAL, resuming without restarting."
gh issue create --title "sidecar: Go sidecar with full Prometheus instrumentation" --body "Go service on golang:1.23. Polls paling /metrics and /health. Includes full prometheus/client_golang instrumentation with SRE-grade temporality metrics."
gh issue create --title "infra: Traefik labels + delightd registration" --body "Update docker-compose.yml to route sidecar via Traefik. Register service with delightd automatically."
gh issue create --title "design: edge-exploration algorithm (Kalman vs UMAP etc)" --body "Vector-space edge-exploration algorithm is underspecified. Do not implement Kalman until algorithm is agreed. Compare with UMAP, t-SNE, etc."
gh issue create --title "future: acceptance criteria / external tool executor" --body "Acceptance-criteria pipeline placeholder. BanchanExecutor stub. Tool may be in any language. Communication via Kafka events + stdout capture. Milestone: paling-acceptance-criteria."

echo "All issues created successfully!"
