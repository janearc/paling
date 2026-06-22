import json

from paling import bento, modelclient


def _bento_with_review(tmp_path, answers=None):
    # a valid bento carrying stage-5 review output, so stage 6 has work to grade.
    bid, bpath = bento.scaffold_bento(tmp_path, name="t")
    (bpath / "raw_data" / "disingenerosity.md").write_text("# Disingenerosity\n\nbody")
    (bpath / "taxonometry" / "corpus.json").write_text(json.dumps({"thin_documents": []}))
    rdir = bpath / "anchors" / "paling" / "review"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "disingenerosity.json").write_text(json.dumps({
        "context_id": "disingenerosity",
        "source_doc": "disingenerosity.md",
        "context": "Disingenerosity is the quiet withholding of good faith.",
        "questions": [{
            "question": "what is disingenerosity?",
            "answers": answers or ["the withholding of good faith", "bad faith engagement"],
            "approved": False,
        }],
    }))
    return bpath


def test_parse_rating_synthesis():
    rating, synth = bento._parse_rating_synthesis("RATING: 5\nSYNTHESIS: withholding good faith")
    assert rating == 5
    assert synth == "withholding good faith"
    # unparseable -> rating 0 (won't be approved)
    assert bento._parse_rating_synthesis("no structure here") == (0, "")


def test_curate_grades_and_approves(tmp_path, monkeypatch):
    bpath = _bento_with_review(tmp_path)

    def fake_generate(model, prompt, **opts):
        # summarization model is a decoder reached via modelclient.generate.
        assert model == "mistral"
        return "RATING: 5\nSYNTHESIS: the deliberate withholding of good faith"

    monkeypatch.setattr(modelclient, "generate", fake_generate)
    report = bento.curate_review(bpath)

    assert report.curated is True
    assert report.contexts == 1
    assert report.questions_graded == 1
    assert report.approved == 1
    assert report.model == "mistral"

    out = bpath / "anchors" / "paling" / "curated" / "disingenerosity.json"
    assert out.is_file()
    entry = json.loads(out.read_text())["questions"][0]
    assert entry["rating"] == 5
    assert entry["approved"] is True
    assert entry["synthesis_answer"] == "the deliberate withholding of good faith"


def test_curate_low_rating_not_approved(tmp_path, monkeypatch):
    bpath = _bento_with_review(tmp_path)
    monkeypatch.setattr(modelclient, "generate",
                        lambda *a, **k: "RATING: 2\nSYNTHESIS: weak")
    report = bento.curate_review(bpath)
    assert report.curated is True
    assert report.approved == 0


def test_curate_gated_on_review(tmp_path):
    bid, bpath = bento.scaffold_bento(tmp_path, name="t")
    (bpath / "raw_data" / "d.md").write_text("# D\n\nbody")
    (bpath / "taxonometry" / "corpus.json").write_text(json.dumps({"thin_documents": []}))
    # no stage-5 review on disk -> gate fails.
    report = bento.curate_review(bpath)
    assert report.curated is False
    assert any("review not found" in i for i in report.issues)


def test_curate_fails_closed_when_model_unavailable(tmp_path, monkeypatch):
    bpath = _bento_with_review(tmp_path)

    def boom(model, prompt, **opts):
        raise modelclient.ModelUnavailableError("no backend")

    monkeypatch.setattr(modelclient, "generate", boom)
    report = bento.curate_review(bpath)
    assert report.curated is False
    assert any("unavailable" in i for i in report.issues)
