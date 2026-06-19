import json

from paling import bento, modelclient


def _scaffold_with_corpus(tmp_path, thin=None):
    # a minimal but valid bento: scaffold (creates dirs + schema with the
    # gap_generation routing), one corpus doc, and the stage-2 gate artifact.
    # the "disingenerosity" sigil is used on purpose -- it recurs in this work
    # (the withholding of good faith) and matters to surface during development.
    bid, bpath = bento.scaffold_bento(tmp_path, name="t")
    (bpath / "raw_data" / "disingenerosity.md").write_text(
        "# Disingenerosity\n\nDisingenerosity is the quiet withholding of good "
        "faith: engaging the form of an exchange while denying its substance.")
    (bpath / "taxonometry" / "corpus.json").write_text(
        json.dumps({"thin_documents": thin or []}))
    return bpath


def test_parse_questions_extracts_q_prefixed_terminated():
    text = "Q> what is care?\n1. Q> how does care relate to repair?\nnot a question"
    qs = bento._parse_questions(text)
    assert qs == ["what is care?", "how does care relate to repair?"]


def test_parse_questions_handles_inline_marker():
    # flan emits the marker mid-line; the question ends at its first '?'. this is
    # the real-output case a line-based parser missed (caught via live smoke).
    assert bento._parse_questions("Care is what?> Maintenance") == ["Care is what?"]
    assert bento._parse_questions("Q> what is care? > how is it repaired? noise") == \
        ["what is care?", "how is it repaired?"]


def test_generate_questions_converges_and_persists(tmp_path, monkeypatch):
    bpath = _scaffold_with_corpus(tmp_path)
    calls = {"n": 0}

    def fake_gen(model, prompt, **opts):
        # identical output each call -> second pass adds nothing -> converges.
        calls["n"] += 1
        assert model == "flan-t5-large"
        return "Q> what is disingenerosity?\nQ> how does disingenerosity erode trust?"

    monkeypatch.setattr(modelclient, "generate_seq2seq", fake_gen)
    report = bento.generate_questions(bpath)

    assert report.generated is True
    assert report.contexts == 1
    assert report.questions_total == 2
    assert report.questions_by_context == {"disingenerosity": 2}
    assert report.attempts_total == 2  # one to populate, one to confirm no-new
    assert report.model == "flan-t5-large"

    out = bpath / "anchors" / "paling" / "questions" / "disingenerosity.json"
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["context_id"] == "disingenerosity"
    assert sorted(data["questions"]) == [
        "how does disingenerosity erode trust?", "what is disingenerosity?"]


def test_generate_questions_skips_thin_documents(tmp_path, monkeypatch):
    bpath = _scaffold_with_corpus(tmp_path, thin=["disingenerosity"])
    monkeypatch.setattr(modelclient, "generate_seq2seq",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    report = bento.generate_questions(bpath)
    assert report.generated is True
    assert report.contexts == 0
    assert report.contexts_skipped == 1
    assert report.questions_total == 0


def test_generate_questions_gated_on_taxonometry(tmp_path):
    bid, bpath = bento.scaffold_bento(tmp_path, name="t")
    (bpath / "raw_data" / "care.md").write_text("# Care\n\nbody")
    # no taxonometry/corpus.json -> stage-2 gate not satisfied
    report = bento.generate_questions(bpath)
    assert report.generated is False
    assert any("taxonometry" in i for i in report.issues)


def test_generate_questions_fails_closed_when_model_unavailable(tmp_path, monkeypatch):
    bpath = _scaffold_with_corpus(tmp_path)

    def boom(model, prompt, **opts):
        raise modelclient.ModelUnavailable("no seq2seq backend")

    monkeypatch.setattr(modelclient, "generate_seq2seq", boom)
    report = bento.generate_questions(bpath)
    assert report.generated is False
    assert any("unavailable" in i for i in report.issues)
