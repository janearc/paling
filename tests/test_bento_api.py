from fastapi.testclient import TestClient

from paling import daemon


def _client(tmp_path, monkeypatch):
    # point the daemon at a temp bentos root and stub emission (no live sidecar).
    monkeypatch.setattr(daemon, "_BENTOS_ROOT", str(tmp_path))
    monkeypatch.setattr(daemon.producer, "emit", lambda *a, **k: None)
    return TestClient(daemon.app)


def test_create_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/bento", json={"name": "mybento"})
    assert r.status_code == 200
    body = r.json()
    assert body["bento_id"] == "mybento"
    assert (tmp_path / "mybento" / "raw_data").is_dir()
    assert (tmp_path / "mybento" / "schema" / "schema.json").is_file()


def test_create_bento_conflict(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "dup"})
    r = c.post("/bento", json={"name": "dup"})
    assert r.status_code == 409


def test_add_corpus(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "b1"})

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# doc a")
    (corpus / "b.md").write_text("# doc b")
    (corpus / "ignore.txt").write_text("not markdown")

    r = c.post("/bento/b1/corpus", json={"source_path": str(corpus)})
    assert r.status_code == 200
    assert r.json()["files_ingested"] == 2
    assert (tmp_path / "b1" / "raw_data" / "a.md").is_file()
    assert not (tmp_path / "b1" / "raw_data" / "ignore.txt").exists()


def test_add_corpus_missing_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/bento/ghost/corpus", json={"source_path": str(tmp_path)})
    assert r.status_code == 404


def test_verify_valid_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "v1"})
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# doc a")
    c.post("/bento/v1/corpus", json={"source_path": str(corpus)})

    r = c.post("/bento/v1/verify")
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["corpus_files"] == 1
    assert body["archetype"] == "unprocessed"
    assert (tmp_path / "v1" / "preflight" / "preflight.json").is_file()


def test_verify_empty_bento_fails_gate(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "empty"})
    r = c.post("/bento/empty/verify")
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert any("no .md corpus" in i for i in body["issues"])


def test_verify_missing_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/bento/ghost/verify")
    assert r.status_code == 404
