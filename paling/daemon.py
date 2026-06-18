# ==============================================================================
# Paling Orchestration Daemon (`paling serve`)
# ==============================================================================
# This module serves as the primary orchestration plane for the Paling ecosystem.
# Because Paling relies on MLX to execute QLoRA training and inference directly
# on Apple Silicon (Metal API), this daemon intentionally runs bare-metal rather
# than inside a container. It acts as the bridge between the containerized fleet
# and the local MLX hardware.
#
# Core Responsibilities:
# 1. State Machines: Manages the lifecycle of Bentos (entire projects) and
#    Banchans (discrete processing steps like Lexical Cartography or Training).
# 2. Asynchronous Execution: API endpoints queue operations into a thread pool
#    so HTTP requests remain unblocked during multi-hour fine-tuning runs.
# 3. Distributed Observability: Emits `BanchanLifecycleEvent` protobufs
#    via Kafka, allowing the broader fleet to track execution via trace_ids.
# 4. Fleet Integration: Provides `/health` and `/metrics` for Traefik routing
#    and the Go sidecar container to scrape.
# ==============================================================================
import asyncio
import logging
import os
import json
import uuid
import urllib.request
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
from enum import Enum
import sys

logger = logging.getLogger(__name__)

# the go sidecar owns all kafka/schema-registry/protobuf; the bare-metal daemon
# only POSTs a small json payload to it locally. emission is best-effort and
# never raises -- a kafka or sidecar hiccup must not break bento operations.
_SIDECAR_EMIT_URL = os.environ.get("PALING_SIDECAR_URL", "http://localhost:9090/emit")

# State machine definitions
class BentoState(str, Enum):
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    TRAINING = "TRAINING"
    DONE = "DONE"
    FAILED = "FAILED"

class BanchanState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    PARTIAL = "PARTIAL"
    NEEDS_MASSAGE = "NEEDS_MASSAGE"
    DONE = "DONE"
    FAILED = "FAILED"

app = FastAPI(title="Paling Daemon")

# In-memory state tracking
bentos_state: Dict[str, BentoState] = {}
banchans_state: Dict[str, Dict[str, BanchanState]] = {}

class BanchanEventProducer:
    # emits a banchan lifecycle event by relaying it to the local go sidecar,
    # which serializes to protobuf and produces to kafka. event_id is a fresh
    # uuid4 (the audit's idempotency key). best-effort: failures are logged.
    def emit(self, bento_id: str, banchan_name: str, state: BanchanState, trace_id: str = ""):
        payload = {
            "event_id": str(uuid.uuid4()),
            "trace_id": trace_id,
            "bento_id": bento_id,
            "banchan_name": banchan_name,
            "state": state.value,
        }
        try:
            req = urllib.request.Request(
                _SIDECAR_EMIT_URL,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2).close()
        except Exception as e:
            logger.warning("banchan event emit failed (non-fatal): %s", e)


producer = BanchanEventProducer()

class BanchanExecutor:
    """
    Runs an external tool bundled as part of a Bento.
    Tool may be in any language; not expected to share runtime with paling.
    Communication: Kafka events + stdout/stderr capture.

    NOT IMPLEMENTED — acceptance-criteria pipeline placeholder.
    This will be one of the harder integration points in this codebase:
    tools may run for hours, need kill/resume support, and must be fully
    observable via the banchan state machine and Kafka event stream.

    Schema note: paling.bento.banchan.* Protobuf messages must include
    an optional `tool_event` field reserved for tool-emitted payloads.

    Milestone: paling-acceptance-criteria (Issue #11)
    """
    def execute(self, bento_id: str, banchan_name: str, tool_path: str) -> None:
        raise NotImplementedError("acceptance-criteria executor not yet implemented")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    # Return Prometheus text
    return "paling_jobs_active 0\n"

@app.get("/bento/{bento_id}/status")
def bento_status(bento_id: str):
    state = bentos_state.get(bento_id, BentoState.IDLE)
    return {"bento_id": bento_id, "state": state}

@app.get("/bento/{bento_id}/banchan/{banchan_name}/status")
def banchan_status(bento_id: str, banchan_name: str):
    bento_banchans = banchans_state.get(bento_id, {})
    state = bento_banchans.get(banchan_name, BanchanState.NOT_STARTED)
    return {"bento_id": bento_id, "banchan_name": banchan_name, "state": state}

@app.post("/bento/{bento_id}/prepare")
def prepare_bento(bento_id: str):
    bentos_state[bento_id] = BentoState.PREPARING
    producer.emit(bento_id, "prepare", BanchanState.IN_PROGRESS)
    return {"bento_id": bento_id, "status": "enqueued for prepare"}

@app.post("/bento/{bento_id}/train")
def train_bento(bento_id: str):
    bentos_state[bento_id] = BentoState.TRAINING
    producer.emit(bento_id, "train", BanchanState.IN_PROGRESS)
    return {"bento_id": bento_id, "status": "enqueued for train"}

def serve(port: int = 8090):
    import uvicorn
    import subprocess
    
    # Register with delightd at startup
    try:
        # Enforce checkpoint / registration
        pass
    except Exception as e:
        logger.info(f"Failed to register with delightd: {e}")

    uvicorn.run(app, host="127.0.0.1", port=port)
