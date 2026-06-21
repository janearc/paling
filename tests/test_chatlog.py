"""Tests for the second-stage chatlog ingest (extractor JSON -> chat data).

The fixtures here are SMALL SYNTHETIC stand-ins for the upstream extractor's
output. The real quell logs are private and are never committed; these only
exercise the parsing rules.
"""

import json
from pathlib import Path

import pytest

from paling import chatlog
from paling.dataset import build_chatlog_datasets


def _entry(extracted, host=None, timestamp=None):
    return {
        "raw": "<<garbage>>",
        "extracted": extracted,
        "authors": [4],          # decoy counter; must be ignored
        "timestamp": timestamp,
        "host": host,
    }


def _synthetic_messages():
    """A synthetic extractor messages-dict covering every rule.

    Insertion order is authoritative; timestamps are deliberately
    non-monotonic to prove they are not used for ordering.
    """
    return {
        # telemetry crumbs (too short) -> dropped
        "00000000-0000-0000-0000-000000000001": _entry("status", timestamp=900.0),
        # the custom-instructions marker -> dropped
        "00000000-0000-0000-0000-000000000002": _entry(
            "Original custom instructions no longer available", timestamp=800.0
        ),
        # first substantial host-present entry -> SYSTEM
        "00000000-0000-0000-0000-000000000003": _entry(
            "You are Aethelquell, a creature of stone and patience.",
            host="93f46c0698549fc6-AMS", timestamp=700.0,
        ),
        # assistant reply streamed as two consecutive host==None rows -> merged
        "00000000-0000-0000-0000-000000000004": _entry(
            "She turns to face you, unhurried.", timestamp=600.0
        ),
        "00000000-0000-0000-0000-000000000005": _entry(
            "The space between you narrows by a breath.", timestamp=500.0
        ),
        # painter (host present) turn -> USER
        "00000000-0000-0000-0000-000000000006": _entry(
            "hi, do you know who I am?", host="93f46ca199fe9fc6-AMS", timestamp=400.0
        ),
        # short crumb between turns -> dropped
        "00000000-0000-0000-0000-000000000007": _entry("ok", timestamp=350.0),
        # assistant reply -> ASSISTANT
        "00000000-0000-0000-0000-000000000008": _entry(
            "I know the shape of what you bring with you.", timestamp=300.0
        ),
        # second painter turn -> USER
        "00000000-0000-0000-0000-000000000009": _entry(
            "I'm here to meet you. I paint.", host="93f46e70484f1cba-AMS", timestamp=200.0
        ),
        # final assistant -> ASSISTANT
        "00000000-0000-0000-0000-00000000000a": _entry(
            "Then paint, and we will see what holds.", timestamp=100.0
        ),
    }


def test_parse_roles_order_and_merge():
    turns = chatlog.parse_chatlog_messages(_synthetic_messages())
    roles = [t["role"] for t in turns]
    # system first, then alternating user/assistant -- NOT sorted by timestamp.
    assert roles == [
        "system", "assistant", "user", "assistant", "user", "assistant"
    ]

    # The two streamed assistant rows were merged into one turn.
    first_assistant = turns[1]["content"]
    assert "She turns to face you" in first_assistant
    assert "space between you narrows" in first_assistant

    # System prompt is the first substantial host-present entry.
    assert turns[0]["content"].startswith("You are Aethelquell")

    # The marker and short crumbs are gone.
    joined = " ".join(t["content"] for t in turns)
    assert "Original custom instructions" not in joined
    assert "status" not in turns[0]["content"]


def test_character_records():
    turns = chatlog.parse_chatlog_messages(_synthetic_messages())
    recs = chatlog.character_records(turns)
    assert len(recs) == 1
    msgs = recs[0]["messages"]
    assert [m["role"] for m in msgs] == [
        "system", "assistant", "user", "assistant", "user", "assistant"
    ]


def test_painter_pairs_are_reactive():
    turns = chatlog.parse_chatlog_messages(_synthetic_messages())
    pairs = chatlog.painter_records(turns)
    # Two painter (user) turns, each paired with the assistant turn it answers.
    assert len(pairs) == 2
    for p in pairs:
        roles = [m["role"] for m in p["messages"]]
        assert roles == ["user", "assistant"]

    # First painter answers the merged opening assistant turn.
    assert "space between you narrows" in pairs[0]["messages"][0]["content"]
    assert pairs[0]["messages"][1]["content"] == "hi, do you know who I am?"
    # Second painter answers the next assistant turn.
    assert "shape of what you bring" in pairs[1]["messages"][0]["content"]


def test_leading_painter_turn_skipped():
    # A painter turn with no preceding assistant context is not emitted.
    msgs = {
        "a": _entry("You are a system prompt of sufficient length here.",
                    host="aaaaaaaaaaaaaaaa-AMS"),
        "b": _entry("opening painter line with nothing to react to yet",
                    host="bbbbbbbbbbbbbbbb-AMS"),
        "c": _entry("the assistant finally responds with something."),
    }
    turns = chatlog.parse_chatlog_messages(msgs)
    pairs = chatlog.painter_records(turns)
    assert pairs == []


def test_normalize_punctuation_handles_mojibake():
    # Smart quotes and an em-dash collapse to ascii.
    assert chatlog.normalize_punctuation("“hi”—there") == '"hi"--there'
    # Literal backslash-n becomes a real newline.
    assert chatlog.normalize_punctuation("a\\nb") == "a\nb"


def test_build_chatlog_datasets_end_to_end(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "synthetic-001.json").write_text(
        json.dumps({"messages": _synthetic_messages()})
    )

    results = build_chatlog_datasets(
        input_dir=str(in_dir), output_dir=str(out_dir), val_split=0.0
    )
    assert set(results) == {"character", "painter"}

    # Each dataset lands in its OWN subdir with standard train/valid names, so
    # `paling train --data out/character` (or out/painter) works directly.
    char_lines = []
    for name in ("train.jsonl", "valid.jsonl"):
        f = out_dir / "character" / name
        if f.is_file():
            char_lines += [l for l in f.read_text().splitlines() if l.strip()]
    assert len(char_lines) == 1
    rec = json.loads(char_lines[0])
    assert rec["messages"][0]["role"] == "system"

    painter_lines = []
    for name in ("train.jsonl", "valid.jsonl"):
        f = out_dir / "painter" / name
        if f.is_file():
            painter_lines += [l for l in f.read_text().splitlines() if l.strip()]
    assert len(painter_lines) == 2
    for line in painter_lines:
        prec = json.loads(line)
        assert [m["role"] for m in prec["messages"]] == ["user", "assistant"]

    # A manifest records exactly what was produced from which inputs.
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest["inputs"]) == 1
    assert manifest["inputs"][0]["character_records"] == 1
    assert manifest["inputs"][0]["painter_pairs"] == 2
    assert manifest["failed"] == []


def test_short_host_present_turn_survives():
    # A short host-PRESENT row ("cut the shit" == 12 chars) is a real painter
    # jab, not a telemetry crumb, and must be KEPT as a user turn -- only
    # host-LESS short rows are dropped as crumbs.
    msgs = {
        "a": _entry("You are a system prompt of sufficient length here.",
                    host="aaaaaaaaaaaaaaaa-AMS"),
        "b": _entry("The assistant says something substantial enough."),
        # short host-less crumb -> dropped
        "c": _entry("status"),
        # short host-PRESENT painter jab -> kept as user
        "d": _entry("cut the shit", host="dddddddddddddddd-AMS"),
    }
    turns = chatlog.parse_chatlog_messages(msgs)
    roles = [t["role"] for t in turns]
    assert roles == ["system", "assistant", "user"]
    assert turns[-1]["content"] == "cut the shit"
    # the host-less crumb did not survive
    assert all("status" != t["content"] for t in turns)


def test_malformed_input_raises_by_default_and_skips_when_opted_in(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "good.json").write_text(
        json.dumps({"messages": _synthetic_messages()})
    )
    (in_dir / "bad.json").write_text("{ this is not valid json")

    out_dir_fail = tmp_path / "out_fail"
    # Default: a malformed file makes the whole build RAISE (no silent loss).
    with pytest.raises(Exception) as excinfo:
        build_chatlog_datasets(
            input_dir=str(in_dir), output_dir=str(out_dir_fail), val_split=0.0
        )
    assert "bad.json" in str(excinfo.value)

    # skip_bad=True: the bad file is skipped-with-warning, the good one builds.
    out_dir_skip = tmp_path / "out_skip"
    results = build_chatlog_datasets(
        input_dir=str(in_dir),
        output_dir=str(out_dir_skip),
        val_split=0.0,
        skip_bad=True,
    )
    assert set(results) == {"character", "painter"}
    manifest = json.loads((out_dir_skip / "manifest.json").read_text())
    assert len(manifest["failed"]) == 1
    assert "bad.json" in manifest["failed"][0]["file"]
    assert len(manifest["inputs"]) == 1
    assert "good.json" in manifest["inputs"][0]["file"]


def test_multiple_input_files_all_processed(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "one.json").write_text(
        json.dumps({"messages": _synthetic_messages()})
    )
    (in_dir / "two.json").write_text(
        json.dumps({"messages": _synthetic_messages()})
    )

    results = build_chatlog_datasets(
        input_dir=str(in_dir), output_dir=str(out_dir), val_split=0.0
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest["inputs"]) == 2
    files = {Path(i["file"]).name for i in manifest["inputs"]}
    assert files == {"one.json", "two.json"}
    # Both files contributed: character records doubled vs a single-file run.
    char_train, char_valid = results["character"]
    assert char_train + char_valid == 2
