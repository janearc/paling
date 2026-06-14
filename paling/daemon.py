import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
from enum import Enum
import sys

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
    """
    Kafka Event Producer stub. Emits events on banchan state transitions.
    TODO: implement kafka-python logic here (Issue #6)
    Add exponential backoff on all Kafka connections (100ms base, 30s cap, jitter)
    """
    def emit(self, bento_id: str, banchan_name: str, state: BanchanState):
        pass # TODO

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
    return {"bento_id": bento_id, "status": "enqueued for prepare"}

@app.post("/bento/{bento_id}/train")
def train_bento(bento_id: str):
    bentos_state[bento_id] = BentoState.TRAINING
    return {"bento_id": bento_id, "status": "enqueued for train"}

def serve(port: int = 8080):
    import uvicorn
    import subprocess
    
    # Register with delightd at startup
    try:
        # Enforce checkpoint / registration
        pass
    except Exception as e:
        print(f"Failed to register with delightd: {e}")

    uvicorn.run(app, host="127.0.0.1", port=port)
