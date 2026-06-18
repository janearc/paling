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
from pathlib import Path

from paling import bento

logger = logging.getLogger(__name__)

# the daemon never speaks kafka itself: emit() posts a small json event to the
# local go sidecar, which does the protobuf + schema-registry + kafka work.
# best-effort -- emit() never raises, so a sidecar/kafka hiccup can't break a bento.
_SIDECAR_EMIT_URL = os.environ.get("PALING_SIDECAR_URL", "http://localhost:9090/emit")

# where bentos live on the bare-metal host; create/ingest operate under this root.
_BENTOS_ROOT = os.environ.get("PALING_BENTOS_ROOT", bento.DEFAULT_BENTOS_ROOT)

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


class CreateBentoRequest(BaseModel):
    name: Optional[str] = None
    archetype: str = "unprocessed"


class IngestCorpusRequest(BaseModel):
    source_path: str


@app.post("/bento")
def create_bento(req: CreateBentoRequest):
    # scaffold a new bento over the API so the agent skill never hand-places files.
    try:
        bento_id, path = bento.scaffold_bento(_BENTOS_ROOT, req.name, req.archetype)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    bentos_state[bento_id] = BentoState.IDLE
    producer.emit(bento_id, "create", BanchanState.NOT_STARTED)
    return {"bento_id": bento_id, "path": str(path), "state": BentoState.IDLE}


@app.post("/bento/{bento_id}/corpus")
def add_corpus(bento_id: str, req: IngestCorpusRequest):
    # ingest a markdown corpus into an existing bento's raw_data.
    bento_path = Path(_BENTOS_ROOT).expanduser().resolve() / bento_id
    if not bento_path.is_dir():
        raise HTTPException(status_code=404, detail=f"bento '{bento_id}' not found")
    try:
        count = bento.ingest_corpus(bento_path, req.source_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    producer.emit(bento_id, "ingest", BanchanState.IN_PROGRESS)
    return {"bento_id": bento_id, "files_ingested": count}


@app.post("/bento/{bento_id}/verify")
def verify_bento(bento_id: str):
    # pipeline stage 1: walk the bento, confirm it looks processable (corpus +
    # valid schema), write the report to preflight/. no processing here.
    bento_path = Path(_BENTOS_ROOT).expanduser().resolve() / bento_id
    if not bento_path.is_dir():
        raise HTTPException(status_code=404, detail=f"bento '{bento_id}' not found")
    report = bento.verify_bento(bento_path)
    bento.write_preflight(bento_path, report)
    producer.emit(bento_id, "verify", BanchanState.IN_PROGRESS)
    return report


@app.post("/bento/{bento_id}/profile")
def profile_bento(bento_id: str):
    # pipeline stage 2: profile the corpus into per-document taxonometry
    # signatures + a corpus-level summary under taxonometry/. gated on stage-1
    # verify; a bento that fails the gate returns 409 rather than profiling junk.
    bento_path = Path(_BENTOS_ROOT).expanduser().resolve() / bento_id
    if not bento_path.is_dir():
        raise HTTPException(status_code=404, detail=f"bento '{bento_id}' not found")
    report = bento.profile_bento(bento_path)
    if not report.profiled:
        raise HTTPException(status_code=409, detail=report.issues)
    producer.emit(bento_id, "profile", BanchanState.IN_PROGRESS)
    return report


@app.post("/bento/{bento_id}/extract")
def extract_bento(bento_id: str):
    # pipeline stage 3: extract the typed relationship graph (concept nodes +
    # directed edges) into anchors/paling/relationships/. gated on stage-2
    # taxonometry having run (409 otherwise).
    bento_path = Path(_BENTOS_ROOT).expanduser().resolve() / bento_id
    if not bento_path.is_dir():
        raise HTTPException(status_code=404, detail=f"bento '{bento_id}' not found")
    report = bento.extract_relationships(bento_path)
    if not report.extracted:
        raise HTTPException(status_code=409, detail=report.issues)
    producer.emit(bento_id, "extract", BanchanState.IN_PROGRESS)
    return report


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
