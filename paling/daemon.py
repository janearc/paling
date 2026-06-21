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
import logging
import os
import json
import uuid
import urllib.request
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from collections import OrderedDict
from enum import Enum
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


class OrchestrationCommand(BaseModel):
    # inbound orchestration relayed by the go sidecar from kafka. the fleet
    # (delightd/fleet-svc) produces these to drive a bento through the pipeline
    # without hand-curling the daemon. command_id is the idempotency key.
    command_id: str
    bento_id: str
    action: str
    trace_id: str = ""
    parameters_json: str = ""
    issued_by: str = ""


# maps the wire OrchestrationAction enum names (and their bare verbs) to the
# bento state the action drives. unknown actions are rejected with 400 so the
# sidecar marks the command permanently failed rather than retrying forever.
_ORCHESTRATION_ACTIONS = {
    "ORCHESTRATION_ACTION_PREPARE": ("prepare", BentoState.PREPARING),
    "ORCHESTRATION_ACTION_TRAIN": ("train", BentoState.TRAINING),
    "PREPARE": ("prepare", BentoState.PREPARING),
    "TRAIN": ("train", BentoState.TRAINING),
}

# command_ids already accepted, so a redelivered command is a no-op. bounded in
# size: a redelivery window of recent ids is all idempotency needs here.
_seen_commands: "OrderedDict[str, str]" = OrderedDict()
_SEEN_COMMANDS_MAX = 4096


def _remember_command(command_id: str) -> bool:
    # returns True if this command_id is new (and records it), False if already
    # seen. keeps the daemon idempotent under at-least-once kafka delivery.
    if command_id in _seen_commands:
        return False
    _seen_commands[command_id] = ""
    while len(_seen_commands) > _SEEN_COMMANDS_MAX:
        _seen_commands.popitem(last=False)
    return True


@app.post("/orchestrate")
def orchestrate(cmd: OrchestrationCommand):
    # control-plane entry the sidecar relays inbound kafka commands to. validates
    # the action, dedupes on command_id, drives the bento state machine, and
    # emits a lifecycle event carrying the originating trace_id.
    mapping = _ORCHESTRATION_ACTIONS.get(cmd.action.upper())
    if mapping is None:
        raise HTTPException(status_code=400, detail=f"unknown action '{cmd.action}'")

    if not _remember_command(cmd.command_id):
        return {
            "command_id": cmd.command_id,
            "bento_id": cmd.bento_id,
            "status": "duplicate",
        }

    banchan_name, target_state = mapping
    bentos_state[cmd.bento_id] = target_state
    producer.emit(cmd.bento_id, banchan_name, BanchanState.IN_PROGRESS, trace_id=cmd.trace_id)
    logger.info(
        "orchestration accepted: command_id=%s bento_id=%s action=%s issued_by=%s",
        cmd.command_id,
        cmd.bento_id,
        banchan_name,
        cmd.issued_by,
    )
    return {
        "command_id": cmd.command_id,
        "bento_id": cmd.bento_id,
        "action": banchan_name,
        "status": "accepted",
    }


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


@app.post("/bento/{bento_id}/questions")
def questions_bento(bento_id: str):
    # pipeline stage 4: generate questions per context by iterating the
    # gap_generation model to convergence, persisting them under
    # anchors/paling/questions/. gated on stage-2 taxonometry (409 otherwise).
    bento_path = Path(_BENTOS_ROOT).expanduser().resolve() / bento_id
    if not bento_path.is_dir():
        raise HTTPException(status_code=404, detail=f"bento '{bento_id}' not found")
    report = bento.generate_questions(bento_path)
    if not report.generated:
        raise HTTPException(status_code=409, detail=report.issues)
    producer.emit(bento_id, "questions", BanchanState.IN_PROGRESS)
    return report


@app.post("/bento/{bento_id}/answers")
def answers_bento(bento_id: str):
    # pipeline stage 5: answer each stage-4 question by iterating the model to
    # convergence, writing the review shape under anchors/paling/review/. gated on
    # stage-4 questions existing (409 otherwise).
    bento_path = Path(_BENTOS_ROOT).expanduser().resolve() / bento_id
    if not bento_path.is_dir():
        raise HTTPException(status_code=404, detail=f"bento '{bento_id}' not found")
    report = bento.generate_answers(bento_path)
    if not report.generated:
        raise HTTPException(status_code=409, detail=report.issues)
    producer.emit(bento_id, "answers", BanchanState.IN_PROGRESS)
    return report


@app.post("/bento/{bento_id}/curate")
def curate_bento(bento_id: str):
    # pipeline stage 6: grade stage-5 answers with the summarization model and
    # write the curated review under anchors/paling/curated/. gated on stage-5
    # review existing (409 otherwise).
    bento_path = Path(_BENTOS_ROOT).expanduser().resolve() / bento_id
    if not bento_path.is_dir():
        raise HTTPException(status_code=404, detail=f"bento '{bento_id}' not found")
    report = bento.curate_review(bento_path)
    if not report.curated:
        raise HTTPException(status_code=409, detail=report.issues)
    producer.emit(bento_id, "curate", BanchanState.IN_PROGRESS)
    return report


@app.post("/bento/{bento_id}/dataset")
def dataset_bento(bento_id: str):
    # pipeline stage 7: project the curated review (stage 6) into
    # output/train.jsonl + output/valid.jsonl for the trainer. gated on stage-6
    # curated output existing (409 otherwise).
    bento_path = Path(_BENTOS_ROOT).expanduser().resolve() / bento_id
    if not bento_path.is_dir():
        raise HTTPException(status_code=404, detail=f"bento '{bento_id}' not found")
    report = bento.build_training_data(bento_path)
    if not report.built:
        raise HTTPException(status_code=409, detail=report.issues)
    producer.emit(bento_id, "dataset", BanchanState.IN_PROGRESS)
    return report


def serve(port: int = 8090):
    import uvicorn
    from paling import discovery

    # fleet discovery (issue #9): the daemon is bare-metal and off the docker
    # network, so traefik cannot discover it via docker labels (those route the
    # sidecar). install the daemon's own traefik file-provider route so it becomes
    # mesh-routable at paling-daemon.local. best-effort -- a failure here never
    # blocks serving. delightd separately surfaces paling via repo-root mcp.json.
    result = discovery.install_daemon_route()
    if not result.installed:
        logger.info("traefik route not installed (non-fatal): %s", result.message)

    uvicorn.run(app, host="127.0.0.1", port=port)
