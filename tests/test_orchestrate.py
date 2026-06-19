# tests for the inbound orchestration control-plane endpoint (issue #6). the
# go sidecar consumes OrchestrationCommands off kafka and relays them here; these
# tests pin the contract the sidecar relies on: action validation, command_id
# idempotency, state-machine drive, and trace_id propagation onto the emit.
from fastapi.testclient import TestClient

from paling import daemon


def _client(tmp_path, monkeypatch):
    # temp bentos root + capture emitted events instead of hitting a live sidecar.
    monkeypatch.setattr(daemon, "_BENTOS_ROOT", str(tmp_path))
    emitted = []
    monkeypatch.setattr(
        daemon.producer,
        "emit",
        lambda *a, **k: emitted.append((a, k)),
    )
    # reset the dedupe window so tests are independent of execution order.
    daemon._seen_commands.clear()
    return TestClient(daemon.app), emitted


def test_orchestrate_train_accepted(tmp_path, monkeypatch):
    c, emitted = _client(tmp_path, monkeypatch)
    r = c.post(
        "/orchestrate",
        json={
            "command_id": "cmd-1",
            "bento_id": "b1",
            "action": "ORCHESTRATION_ACTION_TRAIN",
            "trace_id": "trace-xyz",
            "issued_by": "fleet-svc",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "accepted"
    assert body["action"] == "train"
    assert daemon.bentos_state["b1"] == daemon.BentoState.TRAINING
    # the originating trace_id must ride along on the emitted lifecycle event.
    assert emitted and emitted[0][1].get("trace_id") == "trace-xyz"


def test_orchestrate_bare_verb_accepted(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post(
        "/orchestrate",
        json={"command_id": "cmd-2", "bento_id": "b2", "action": "prepare"},
    )
    assert r.status_code == 200
    assert daemon.bentos_state["b2"] == daemon.BentoState.PREPARING


def test_orchestrate_unknown_action_rejected(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post(
        "/orchestrate",
        json={"command_id": "cmd-3", "bento_id": "b3", "action": "explode"},
    )
    # a 4xx tells the sidecar this command is permanently bad (no infinite retry).
    assert r.status_code == 400


def test_orchestrate_idempotent_on_duplicate(tmp_path, monkeypatch):
    c, emitted = _client(tmp_path, monkeypatch)
    payload = {
        "command_id": "dup-1",
        "bento_id": "b4",
        "action": "ORCHESTRATION_ACTION_TRAIN",
    }
    first = c.post("/orchestrate", json=payload)
    second = c.post("/orchestrate", json=payload)
    assert first.json()["status"] == "accepted"
    assert second.json()["status"] == "duplicate"
    # the duplicate must not re-emit / re-enqueue work.
    assert len(emitted) == 1


def test_orchestrate_missing_fields_422(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/orchestrate", json={"bento_id": "b5"})
    assert r.status_code == 422


def test_seen_commands_window_is_bounded(tmp_path, monkeypatch):
    # the idempotency window must stay bounded so a long-lived daemon does not
    # leak memory under a high command volume.
    _client(tmp_path, monkeypatch)
    for i in range(daemon._SEEN_COMMANDS_MAX + 50):
        daemon._remember_command(f"c{i}")
    assert len(daemon._seen_commands) <= daemon._SEEN_COMMANDS_MAX
