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
    # Two painter (user) turns; each record carries the FULL running history in
    # the painter's inverted frame and ends with the painter turn as assistant.
    assert len(pairs) == 2

    for p in pairs:
        msgs = p["messages"]
        # Every record ends with the painter turn as the assistant target.
        assert msgs[-1]["role"] == "assistant"
        # The flipped-frame history precedes it: system stays system, the
        # character's turns are users, prior painter turns are assistants. No
        # raw "assistant"-roled character turn leaks through.
        assert msgs[0]["role"] == "system"
        for m in msgs[:-1]:
            assert m["role"] in ("system", "user", "assistant")

    # First painter record: system + the merged opening character turn (as
    # user) -> painter turn (as assistant).
    first = pairs[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user", "assistant"]
    assert first[0]["content"].startswith("You are Aethelquell")
    assert "space between you narrows" in first[1]["content"]
    assert first[-1]["content"] == "hi, do you know who I am?"

    # Second painter record carries the FULL escalation history: system, both
    # character turns as users, the FIRST painter turn folded in as assistant,
    # then the second painter turn as the assistant target.
    second = pairs[1]["messages"]
    assert [m["role"] for m in second] == [
        "system", "user", "assistant", "user", "assistant"
    ]
    # The earlier painter turn is now prior context (an assistant turn).
    assert second[2]["content"] == "hi, do you know who I am?"
    # The character turn it answered is present as user context.
    assert "shape of what you bring" in second[3]["content"]
    assert second[-1]["content"] == "I'm here to meet you. I paint."


def test_painter_opener_emitted_with_system_context():
    # A leading painter turn (the opener) is NO LONGER skipped when a system
    # prompt is present: it emits as [system] + [opener-as-assistant].
    msgs = {
        "a": _entry("You are a system prompt of sufficient length here.",
                    host="aaaaaaaaaaaaaaaa-AMS"),
        "b": _entry("opening painter line with nothing to react to yet",
                    host="bbbbbbbbbbbbbbbb-AMS"),
        "c": _entry("the assistant finally responds with something."),
    }
    turns = chatlog.parse_chatlog_messages(msgs)
    pairs = chatlog.painter_records(turns)
    assert len(pairs) == 1
    opener = pairs[0]["messages"]
    assert [m["role"] for m in opener] == ["system", "assistant"]
    assert opener[0]["content"].startswith("You are a system prompt")
    assert opener[-1]["content"] == "opening painter line with nothing to react to yet"


def test_painter_skipped_only_when_history_empty():
    # With NO system turn, a leading painter turn has genuinely empty history
    # and is skipped; the painter turn after the first character turn is kept.
    turns = [
        {"role": "user", "content": "leading painter jab, nothing before it"},
        {"role": "assistant", "content": "the character finally says something."},
        {"role": "user", "content": "now the painter reacts to that"},
    ]
    pairs = chatlog.painter_records(turns)
    assert len(pairs) == 1
    msgs = pairs[0]["messages"]
    # Only the character turn (as user) precedes the painter target; the
    # skipped leading painter turn never entered the history.
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert "character finally says" in msgs[0]["content"]
    assert msgs[-1]["content"] == "now the painter reacts to that"


def test_normalize_punctuation_handles_mojibake():
    # Smart quotes and an em-dash collapse to ascii.
    assert chatlog.normalize_punctuation("“hi”—there") == '"hi"--there'
    # Literal backslash-n becomes a real newline.
    assert chatlog.normalize_punctuation("a\\nb") == "a\nb"


def test_normalize_punctuation_ellipsis_and_nbsp():
    # Horizontal ellipsis collapses to three dots.
    assert chatlog.normalize_punctuation("wait…") == "wait..."
    # Non-breaking space becomes a regular space.
    assert chatlog.normalize_punctuation("a b") == "a b"
    # En-dash collapses to a single hyphen.
    assert chatlog.normalize_punctuation("1–2") == "1-2"


def test_normalize_punctuation_utf8_latin1_mojibake():
    # The common UTF-8-read-as-Latin-1 mojibake forms for the punctuation that
    # actually appears in the quell logs. Build each from its real codepoint so
    # the test cannot drift from the source bytes.
    def moji(ch):
        return ch.encode("utf-8").decode("latin-1")

    assert chatlog.normalize_punctuation(moji("—")) == "--"   # em-dash
    assert chatlog.normalize_punctuation(moji("–")) == "-"    # en-dash
    assert chatlog.normalize_punctuation(moji("…")) == "..."  # ellipsis
    assert chatlog.normalize_punctuation(moji("’")) == "'"    # right single quote
    assert chatlog.normalize_punctuation(moji("“")) == '"'    # left double quote
    assert chatlog.normalize_punctuation(moji("”")) == '"'    # right double quote


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
        roles = [m["role"] for m in prec["messages"]]
        # Painter records now carry running history: lead with system, end with
        # the painter turn as the assistant target.
        assert roles[0] == "system"
        assert roles[-1] == "assistant"

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
