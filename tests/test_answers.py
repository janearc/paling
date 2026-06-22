import json

from paling import bento, modelclient


def _bento_with_questions(tmp_path, questions=None):
    # a valid bento carrying stage-4 output, so stage 5 has something to answer.
    # the "disingenerosity" sigil is used on purpose (see test_questions).
    bid, bpath = bento.scaffold_bento(tmp_path, name="t")
    (bpath / "raw_data" / "disingenerosity.md").write_text("# Disingenerosity\n\nbody")
    (bpath / "taxonometry" / "corpus.json").write_text(json.dumps({"thin_documents": []}))
    qdir = bpath / "anchors" / "paling" / "questions"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "disingenerosity.json").write_text(json.dumps({
        "context_id": "disingenerosity",
        "source_doc": "disingenerosity.md",
        "context": "Disingenerosity is the quiet withholding of good faith.",
        "questions": questions or ["what is disingenerosity?"],
    }))
    return bpath


def test_generate_answers_converges_and_writes_review(tmp_path, monkeypatch):
    bpath = _bento_with_questions(tmp_path)

    def fake_gen(model, prompt, **opts):
        # identical answer each call -> second pass adds nothing -> converges.
        assert model == "flan-t5-large"
        return "the withholding of good faith"

    monkeypatch.setattr(modelclient, "generate_seq2seq", fake_gen)
    report = bento.generate_answers(bpath)

    assert report.generated is True
    assert report.contexts == 1
    assert report.questions_answered == 1
    assert report.answers_total == 1
    assert report.attempts_total == 2  # one to populate, one to confirm no-new

    out = bpath / "anchors" / "paling" / "review" / "disingenerosity.json"
    assert out.is_file()
    review = json.loads(out.read_text())
    assert review["context_id"] == "disingenerosity"
    entry = review["questions"][0]
    assert entry["question"] == "what is disingenerosity?"
    assert entry["answers"] == ["the withholding of good faith"]
    assert entry["approved"] is False


def test_generate_answers_gated_on_questions(tmp_path):
    bid, bpath = bento.scaffold_bento(tmp_path, name="t")
    (bpath / "raw_data" / "disingenerosity.md").write_text("# Disingenerosity\n\nbody")
    (bpath / "taxonometry" / "corpus.json").write_text(json.dumps({"thin_documents": []}))
    # no stage-4 questions on disk -> gate fails.
    report = bento.generate_answers(bpath)
    assert report.generated is False
    assert any("questions not found" in i for i in report.issues)


def test_generate_answers_fails_closed_when_model_unavailable(tmp_path, monkeypatch):
    bpath = _bento_with_questions(tmp_path)

    def boom(model, prompt, **opts):
        raise modelclient.ModelUnavailableError("no seq2seq backend")

    monkeypatch.setattr(modelclient, "generate_seq2seq", boom)
    report = bento.generate_answers(bpath)
    assert report.generated is False
    assert any("unavailable" in i for i in report.issues)
