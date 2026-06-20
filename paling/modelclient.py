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

# delightd's control port is 8088 (see delightd docker-compose.yml and
# delightd_architecture.md); override host/port for a remote daemon.
DEFAULT_DELIGHTD_URL = os.environ.get("PALING_DELIGHTD_URL", "http://localhost:8088")

# delightd reports backend URLs as seen from inside the mesh (e.g. ollama at
# host.docker.internal under compose, or host.k3d.internal under k3d), but paling
# runs bare-metal, where those cluster/mesh names do not resolve. rewrite the
# mesh-only hostnames to a bare-metal-reachable host (loopback, since the model
# backends run on the same host); override with PALING_BACKEND_HOST.
_MESH_HOSTS = ("host.docker.internal", "host.k3d.internal")


def _reachable_url(url):
    # map a mesh-internal backend URL to one reachable from the bare-metal daemon.
    # non-mesh URLs pass through unchanged.
    if not url:
        return url
    host = os.environ.get("PALING_BACKEND_HOST", "127.0.0.1")
    for m in _MESH_HOSTS:
        url = url.replace("//" + m + ":", "//" + host + ":").replace("//" + m + "/", "//" + host + "/")
    return url


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
                return {"provider": src.get("provider"),
                        "url": _reachable_url(src.get("url")), "model": m}
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


# --- seq2seq (encoder-decoder) provider --------------------------------------
# decoder-only chat models (mistral, llama) are served by ollama and resolved
# through delightd's discovery above. encoder-decoder (seq2seq) models -- the
# current gap_generation worker is flan-t5, swappable -- are not served by ollama,
# so until model-svc stands up a discoverable seq2seq endpoint we run them
# IN-PROCESS via transformers on Metal (the bare-metal carve-out that is the
# reason paling serve is not containerized). this is the one provider that does
# not yet go through delightd discovery; it is the local transport behind the same
# client surface, and swaps to the model-svc gateway later without callers changing.

# logical model name -> hugging face id. the bento schema's routing.gap_generation
# names the logical model ("flan-t5-large"); this maps it to weights already in
# the read-only HF cache. add seq2seq models here as the pipeline grows.
_SEQ2SEQ_MODELS = {
    "flan-t5-large": "google/flan-t5-large",
}


class _Seq2SeqBackend:
    # lazily loads a seq2seq model + tokenizer once and reuses them across calls.
    # loading pulls torch/transformers (heavy), so it is deferred until a seq2seq
    # generation is actually requested -- the rest of the daemon never imports torch.
    def __init__(self, hf_id):
        self.hf_id = hf_id
        self._tok = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        # local imports: torch/transformers are only needed on this path.
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("loading seq2seq model %s in-process", self.hf_id)
        self._tok = AutoTokenizer.from_pretrained(self.hf_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_id)
        # prefer Metal (MPS) when present -- the reason this runs bare-metal.
        if torch.backends.mps.is_available():
            model = model.to("mps")
        model.eval()
        self._model = model

    def generate(self, prompt, max_new_tokens=384, temperature=0.8, top_p=0.95,
                 top_k=75, do_sample=True):
        # single (non-batched) generation. flan's encoder caps at 512 input
        # tokens, so we truncate rather than error on an over-long context.
        self._ensure_loaded()
        import torch

        inputs = self._tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=1,
            )
        return self._tok.decode(out[0], skip_special_tokens=True)


_seq2seq_backends = {}


def get_seq2seq(model_name):
    # resolve a logical seq2seq model name to a cached in-process backend. raises
    # ModelUnavailable for an unknown name (fail-closed, like the decoder path) so
    # a stage never silently generates with the wrong model.
    hf_id = _SEQ2SEQ_MODELS.get(model_name)
    if hf_id is None:
        raise ModelUnavailable(f"no seq2seq backend for model {model_name!r}")
    backend = _seq2seq_backends.get(model_name)
    if backend is None:
        backend = _Seq2SeqBackend(hf_id)
        _seq2seq_backends[model_name] = backend
    return backend


def generate_seq2seq(model_name, prompt, **opts):
    # generate a single completion from an in-process seq2seq model. mirrors
    # generate() for the decoder path so a caller picks the provider by model
    # architecture without caring about transport.
    return get_seq2seq(model_name).generate(prompt, **opts)
