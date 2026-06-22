import json

from paling import bento


def _bento_with_curated(tmp_path, questions):
    # a valid bento carrying stage-6 curated output, so stage 7 has pairs to project.
    bid, bpath = bento.scaffold_bento(tmp_path, name="t")
    (bpath / "raw_data" / "disingenerosity.md").write_text("# Disingenerosity\n\nbody")
    (bpath / "taxonometry" / "corpus.json").write_text(json.dumps({"thin_documents": []}))
    cdir = bpath / "anchors" / "paling" / "curated"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "disingenerosity.json").write_text(json.dumps({
        "context_id": "disingenerosity",
        "source_doc": "disingenerosity.md",
        "context": "Disingenerosity is the quiet withholding of good faith.",
        "questions": questions,
    }))
    return bpath


def test_build_training_data_keeps_only_approved(tmp_path):
    bpath = _bento_with_curated(tmp_path, [
        {"question": "what is disingenerosity?", "answers": ["x"],
         "rating": 5, "synthesis_answer": "the withholding of good faith", "approved": True},
        {"question": "is this approved?", "answers": ["y"],
         "rating": 2, "synthesis_answer": "weak", "approved": False},
    ])
    report = bento.build_training_data(bpath)

    assert report.built is True
    assert report.pairs == 1
    assert report.skipped_unapproved == 1
    assert report.train + report.valid == 1

    train = (bpath / "output" / "train.jsonl")
    valid = (bpath / "output" / "valid.jsonl")
    assert train.is_file() and valid.is_file()
    lines = [ln for ln in (train.read_text() + valid.read_text()).splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][1]["content"] == "what is disingenerosity?"
    assert rec["messages"][2]["content"] == "the withholding of good faith"


def test_build_training_data_falls_back_to_first_answer(tmp_path):
    bpath = _bento_with_curated(tmp_path, [
        {"question": "q?", "answers": ["first candidate", "second"],
         "rating": 4, "synthesis_answer": "", "approved": True},
    ])
    report = bento.build_training_data(bpath)
    assert report.built is True
    line = [ln for ln in (bpath / "output" / "train.jsonl").read_text().splitlines() if ln.strip()]
    line += [ln for ln in (bpath / "output" / "valid.jsonl").read_text().splitlines() if ln.strip()]
    rec = json.loads(line[0])
    assert rec["messages"][2]["content"] == "first candidate"


def test_build_training_data_gated_on_curated(tmp_path):
    bid, bpath = bento.scaffold_bento(tmp_path, name="t")
    (bpath / "raw_data" / "d.md").write_text("# D\n\nbody")
    (bpath / "taxonometry" / "corpus.json").write_text(json.dumps({"thin_documents": []}))
    # no stage-6 curated output -> gate fails.
    report = bento.build_training_data(bpath)
    assert report.built is False
    assert any("curated review not found" in i for i in report.issues)


def test_build_training_data_no_approved_pairs(tmp_path):
    bpath = _bento_with_curated(tmp_path, [
        {"question": "q?", "answers": ["a"], "rating": 1,
         "synthesis_answer": "a", "approved": False},
    ])
    report = bento.build_training_data(bpath)
    assert report.built is False
    assert report.skipped_unapproved == 1
    assert any("no approved training pairs" in i for i in report.issues)
