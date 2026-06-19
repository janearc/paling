# configurable model client.
#
# paling never hard-codes where a model lives. it asks delightd -- the fleet's
# LLM discovery authority (GET /discovery/llms) -- to resolve a logical model
# name (e.g. "mistral") to a healthy provider endpoint, then calls it. this
# couples paling to delightd on purpose: per the fleet's availability mandate,
# consumers rely on delightd and fail closed if it's down rather than each one
# carrying its own fallback. this is the single spot where paling trades
# portability for fleet integration -- acceptable because the fleet is the only
# deployment.

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# delightd's control port defaults to 8080; override host/port for a remote one.
DEFAULT_DELIGHTD_URL = os.environ.get("PALING_DELIGHTD_URL", "http://localhost:8080")


class ModelUnavailable(RuntimeError):
    # raised when no healthy backend can be resolved for a requested model.
    pass


def resolve_model(model_name, delightd_url=None, timeout=3):
    # resolve a logical model name to a backend via delightd's /discovery/llms:
    # {"provider", "url", "model"} for the first healthy provider that serves a
    # model whose name contains `model_name` (ollama reports e.g. "mistral:latest"
    # / "mistral-small:24b", so we match by substring). returns None if delightd
    # is unreachable or nothing healthy serves it -- fail-closed, no local fallback.
    base = (delightd_url or DEFAULT_DELIGHTD_URL).rstrip("/")
    try:
        with urllib.request.urlopen(base + "/discovery/llms", timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception as e:
        logger.warning("delightd LLM discovery unreachable (%s): %s", base, e)
        return None

    want = model_name.lower()
    for src in data.get("sources", []):
        if not src.get("healthy"):
            continue
        for m in src.get("models", []):
            if want in m.lower():
                return {"provider": src.get("provider"), "url": src.get("url"), "model": m}
    logger.warning("no healthy delightd provider serves model %r", model_name)
    return None


def generate(model_name, prompt, delightd_url=None, timeout=120, **opts):
    # resolve `model_name` via delightd and run a single (non-streaming) ollama
    # completion, returning the generated text. raises ModelUnavailable if no
    # backend can be resolved -- the caller decides whether that's fatal (a
    # model-optional stage can catch it and degrade).
    backend = resolve_model(model_name, delightd_url=delightd_url)
    if backend is None:
        raise ModelUnavailable(f"no backend serves model {model_name!r} (is delightd up?)")

    payload = {"model": backend["model"], "prompt": prompt, "stream": False}
    payload.update(opts)
    req = urllib.request.Request(
        backend["url"].rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data.get("response", "")
