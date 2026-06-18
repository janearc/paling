# 04 - The Service Mesh (Daemonization)

Paling performs highly compute-intensive tasks (profiling, taxonometry generation, QLoRA training) that take significant time. It cannot remain a simple blocking CLI tool. It must graduate into the operational mesh alongside `delightd` and `obs-svc`.

> **Note (2026-06):** `transparent` has been sunset. Telemetry no longer scrapes a per-host report; services emit Protobuf events over Kafka, and `obs-svc-agg` aggregates them. Paling's telemetry target is therefore Kafka → `obs-svc`, not `transparent`.

## Hybrid Architecture

Following the success of the `comfyui-svc` deployment:
*   Paling's core (MLX, Python) runs natively on macOS (or inside an optimized container if possible, but MLX requires metal access so native Python `uv` environment is mandated).
*   A Go sidecar runs alongside Paling and is its **Kafka emitter**: it publishes health/uptime/progress as `observability.v1` heartbeats and `paling.events.v1` domain events (Confluent Schema Registry protobuf), consumed by `obs-svc`. Keeping emission in the Go sidecar reuses the fleet's franz-go producer convention and keeps Python out of the Kafka/Schema-Registry path.
*   Paling exposes an API (via MCP or HTTP REST) to allow external orchestration.

## Asynchronous Bento Operations

The user must be able to ask `delightd`: *"hey how long until paling is done with the wonder-documents-123 bento?"*
This requires Paling to maintain a state machine of active jobs and expose their progress.

## Knockout List

- [ ] Refactor `paling.py` to support running as a long-lived daemon instead of just a single-execution script.
- [ ] Build out the Go sidecar (similar to ComfyUI) as Paling's Kafka emitter: dual-emit `observability.v1` heartbeats + `paling.events.v1` domain events to Kafka, consumed by `obs-svc`.
- [ ] Implement a REST/MCP endpoint in the Paling daemon that returns the status of a specific Bento (e.g., `GET /bento/wonder-documents-123/status`).
- [ ] Add percentage-based progress tracking to the MLX training loop and Taxonometry generation, surfacing this data to the sidecar.
- [ ] Create a `docker-compose.yml` (or LaunchDaemon plist) to codify the hybrid deployment of the Paling daemon + Go sidecar.
