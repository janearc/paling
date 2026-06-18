import json
import urllib.error

from paling import modelclient


class _FakeResp:
    # minimal stand-in for the urlopen context-manager response.
    def __init__(self, body):
        self._b = json.dumps(body).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _discovery(sources):
    return {"status": "ok", "sources": sources}


def test_resolve_model_found(monkeypatch):
    src = {"provider": "ollama", "url": "http://h:11434",
           "models": ["llama3:8b", "mistral-small:24b"], "healthy": True}

    def fake_urlopen(req, timeout=None):
        return _FakeResp(_discovery([src]))

    monkeypatch.setattr(modelclient.urllib.request, "urlopen", fake_urlopen)
    backend = modelclient.resolve_model("mistral")
    assert backend["url"] == "http://h:11434"
    assert "mistral" in backend["model"].lower()


def test_resolve_skips_unhealthy_and_unserved(monkeypatch):
    sources = [
        {"provider": "down", "url": "http://x", "models": ["mistral:latest"], "healthy": False},
        {"provider": "ollama", "url": "http://h", "models": ["llama3"], "healthy": True},
    ]

    def fake_urlopen(req, timeout=None):
        return _FakeResp(_discovery(sources))

    monkeypatch.setattr(modelclient.urllib.request, "urlopen", fake_urlopen)
    assert modelclient.resolve_model("mistral") is None


def test_resolve_delightd_down_fails_closed(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(modelclient.urllib.request, "urlopen", fake_urlopen)
    assert modelclient.resolve_model("mistral") is None


def test_generate_resolves_then_calls_provider(monkeypatch):
    src = {"provider": "ollama", "url": "http://h:11434",
           "models": ["mistral:latest"], "healthy": True}

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if "/discovery/llms" in url:
            return _FakeResp(_discovery([src]))
        assert url == "http://h:11434/api/generate"
        return _FakeResp({"response": "EXTRACTED"})

    monkeypatch.setattr(modelclient.urllib.request, "urlopen", fake_urlopen)
    assert modelclient.generate("mistral", "extract from: ...") == "EXTRACTED"


def test_generate_raises_when_unavailable(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeResp(_discovery([]))

    monkeypatch.setattr(modelclient.urllib.request, "urlopen", fake_urlopen)
    try:
        modelclient.generate("mistral", "hi")
        assert False, "expected ModelUnavailable"
    except modelclient.ModelUnavailable:
        pass
