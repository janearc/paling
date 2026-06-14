# 04 - The Service Mesh (Daemonization)

Paling performs highly compute-intensive tasks (profiling, taxonometry generation, QLoRA training) that take significant time. It cannot remain a simple blocking CLI tool. It must graduate into the operational mesh alongside `delightd` and `transparent`.

## Hybrid Architecture

Following the success of the `comfyui-svc` deployment:
*   Paling's core (MLX, Python) runs natively on macOS (or inside an optimized container if possible, but MLX requires metal access so native Python `uv` environment is mandated).
*   A Go sidecar runs alongside Paling to provide telemetry, exposing health, uptime, and progress metrics to `transparent`.
*   Paling exposes an API (via MCP or HTTP REST) to allow external orchestration.

## Asynchronous Bento Operations

The user must be able to ask `delightd`: *"hey how long until paling is done with the wonder-sigils-123 bento?"*
This requires Paling to maintain a state machine of active jobs and expose their progress.

## Knockout List

- [ ] Refactor `paling.py` to support running as a long-lived daemon instead of just a single-execution script.
- [ ] Build a Go sidecar for Paling (similar to ComfyUI) that listens on a port and serves `/metrics` for `transparent`.
- [ ] Implement a REST/MCP endpoint in the Paling daemon that returns the status of a specific Bento (e.g., `GET /bento/wonder-sigils-123/status`).
- [ ] Add percentage-based progress tracking to the MLX training loop and Taxonometry generation, surfacing this data to the sidecar.
- [ ] Create a `docker-compose.yml` (or LaunchDaemon plist) to codify the hybrid deployment of the Paling daemon + Go sidecar.
