import json

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


def test_profile_valid_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "p1"})
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # one document with distinctive vocabulary, one deliberately flat/generic.
    (corpus / "ethic.md").write_text(
        "# Stewardship\n\nThe steward bears asymmetric obligation toward the "
        "vulnerable. Beneficence is a structural duty, not optional largesse."
    )
    (corpus / "flat.md").write_text(
        "# Notes\n\nThis is a thing. It is here. We did it. It works and is good."
    )
    c.post("/bento/p1/corpus", json={"source_path": str(corpus)})

    r = c.post("/bento/p1/profile")
    assert r.status_code == 200
    body = r.json()
    assert body["profiled"] is True
    assert body["documents"] == 2
    # the flat document carries no rare terms -> surfaced as thin.
    assert "flat" in body["thin_documents"]
    assert "ethic" not in body["thin_documents"]

    tax = tmp_path / "p1" / "taxonometry"
    assert (tax / "corpus.json").is_file()
    assert (tax / "signatures" / "ethic-taxonometry.json").is_file()
    assert (tax / "signatures" / "flat-taxonometry.json").is_file()


def test_profile_ungated_bento_conflicts(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    # empty bento: no corpus, so the stage-1 verify gate fails.
    c.post("/bento", json={"name": "empty"})
    r = c.post("/bento/empty/profile")
    assert r.status_code == 409


def test_profile_missing_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/bento/ghost/profile")
    assert r.status_code == 404


def _extract_corpus(tmp_path):
    # three cross-referencing concept docs: an ethic that names the other two.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# Ethic of Care\n\nCare depends on Stewardship and on the Repair Process."
    )
    (corpus / "b.md").write_text("# Stewardship\n\nStewardship tends the garden.")
    (corpus / "c.md").write_text("# Repair Process\n\nThe repair process restores things.")
    return corpus


def test_extract_valid_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "x1"})
    c.post("/bento/x1/corpus", json={"source_path": str(_extract_corpus(tmp_path))})
    c.post("/bento/x1/profile")  # stage-2 gate must pass first

    r = c.post("/bento/x1/extract")
    assert r.status_code == 200
    body = r.json()
    assert body["extracted"] is True
    assert body["nodes"] >= 3
    # the ethic doc references the other two concepts -> at least two edges.
    assert body["edges"] >= 2
    # cue-lexicon typing from the titles.
    assert "ethic" in body["by_kind"]
    assert "process" in body["by_kind"]
    assert body["coverage"]["corpus_files"] == 3

    rels = tmp_path / "x1" / "anchors" / "paling" / "relationships"
    assert (rels / "nodes.jsonl").is_file()
    assert (rels / "edges.jsonl").is_file()
    assert (rels / "graph.json").is_file()


def test_extract_requires_profile(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "x2"})
    c.post("/bento/x2/corpus", json={"source_path": str(_extract_corpus(tmp_path))})
    # no profile run -> stage-2 gate fails.
    r = c.post("/bento/x2/extract")
    assert r.status_code == 409


def test_extract_missing_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/bento/ghost/extract")
    assert r.status_code == 404


def test_questions_valid_bento(tmp_path, monkeypatch):
    from paling import modelclient
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "q1"})
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # distinctive vocabulary so the doc isn't pruned as thin by stage 2.
    (corpus / "care.md").write_text(
        "# Ethic of Care\n\nCare bears asymmetric obligation toward the vulnerable; "
        "beneficence is a structural duty, not optional largesse."
    )
    c.post("/bento/q1/corpus", json={"source_path": str(corpus)})
    c.post("/bento/q1/profile")  # stage-2 gate must pass first

    # stub the seq2seq model so the route runs without loading flan.
    monkeypatch.setattr(modelclient, "generate_seq2seq", lambda *a, **k: "Q> what is care?")
    r = c.post("/bento/q1/questions")
    assert r.status_code == 200
    body = r.json()
    assert body["generated"] is True
    assert body["questions_total"] >= 1
    assert (tmp_path / "q1" / "anchors" / "paling" / "questions").is_dir()


def test_questions_requires_profile(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "q2"})
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "care.md").write_text("# Care\n\na body about care and duty")
    c.post("/bento/q2/corpus", json={"source_path": str(corpus)})
    # no profile run -> stage-2 gate fails.
    r = c.post("/bento/q2/questions")
    assert r.status_code == 409


def test_questions_missing_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/bento/ghost/questions")
    assert r.status_code == 404


def _seed_questions(tmp_path, name):
    # a bento with corpus + schema (verify passes) and a stage-4 questions file.
    (tmp_path / name / "raw_data" / "d.md").write_text("# D\n\nbody about disingenerosity")
    qdir = tmp_path / name / "anchors" / "paling" / "questions"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "d.json").write_text(json.dumps({
        "context_id": "d", "source_doc": "d.md", "context": "ctx",
        "questions": ["what is disingenerosity?"],
    }))


def test_answers_valid_bento(tmp_path, monkeypatch):
    from paling import modelclient
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "a1"})
    _seed_questions(tmp_path, "a1")
    monkeypatch.setattr(modelclient, "generate_seq2seq", lambda *a, **k: "the withholding of good faith")
    r = c.post("/bento/a1/answers")
    assert r.status_code == 200
    body = r.json()
    assert body["generated"] is True
    assert body["questions_answered"] == 1
    assert (tmp_path / "a1" / "anchors" / "paling" / "review" / "d.json").is_file()


def test_answers_requires_questions(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "a2"})
    (tmp_path / "a2" / "raw_data" / "d.md").write_text("# D\n\nbody")
    # no stage-4 questions -> gate fails.
    r = c.post("/bento/a2/answers")
    assert r.status_code == 409


def test_answers_missing_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/bento/ghost/answers")
    assert r.status_code == 404


def _seed_review(tmp_path, name):
    # a bento with corpus + schema (verify passes) and a stage-5 review file.
    (tmp_path / name / "raw_data" / "d.md").write_text("# D\n\nbody")
    rdir = tmp_path / name / "anchors" / "paling" / "review"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "d.json").write_text(json.dumps({
        "context_id": "d", "source_doc": "d.md", "context": "ctx",
        "questions": [{"question": "what is disingenerosity?", "answers": ["a"], "approved": False}],
    }))


def test_curate_valid_bento(tmp_path, monkeypatch):
    from paling import modelclient
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "c1"})
    _seed_review(tmp_path, "c1")
    monkeypatch.setattr(modelclient, "generate", lambda *a, **k: "RATING: 5\nSYNTHESIS: x")
    r = c.post("/bento/c1/curate")
    assert r.status_code == 200
    body = r.json()
    assert body["curated"] is True
    assert body["approved"] == 1
    assert (tmp_path / "c1" / "anchors" / "paling" / "curated" / "d.json").is_file()


def test_curate_requires_review(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "c2"})
    (tmp_path / "c2" / "raw_data" / "d.md").write_text("# D\n\nbody")
    # no stage-5 review -> gate fails.
    r = c.post("/bento/c2/curate")
    assert r.status_code == 409


def test_curate_missing_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/bento/ghost/curate")
    assert r.status_code == 404


def _seed_curated(tmp_path, name):
    # a bento with corpus (verify passes) and a stage-6 curated review file.
    (tmp_path / name / "raw_data" / "d.md").write_text("# D\n\nbody")
    cdir = tmp_path / name / "anchors" / "paling" / "curated"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "d.json").write_text(json.dumps({
        "context_id": "d", "source_doc": "d.md", "context": "ctx",
        "questions": [{"question": "what is disingenerosity?", "answers": ["a"],
                       "rating": 5, "synthesis_answer": "synth", "approved": True}],
    }))


def test_dataset_valid_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "d1"})
    _seed_curated(tmp_path, "d1")
    r = c.post("/bento/d1/dataset")
    assert r.status_code == 200
    body = r.json()
    assert body["built"] is True
    assert body["pairs"] == 1
    assert (tmp_path / "d1" / "output" / "train.jsonl").is_file()


def test_dataset_requires_curated(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    c.post("/bento", json={"name": "d2"})
    (tmp_path / "d2" / "raw_data" / "d.md").write_text("# D\n\nbody")
    # no stage-6 curated output -> gate fails.
    r = c.post("/bento/d2/dataset")
    assert r.status_code == 409


def test_dataset_missing_bento(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/bento/ghost/dataset")
    assert r.status_code == 404
