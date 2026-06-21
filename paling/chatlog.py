"""Second-stage ingest for ChatGPT-share chatlogs.

This is the second hop of a two-stage pipeline:

  stage 1  (upstream, already exists)
    archaea/whole/tooling/openai-chatlog-extract/chatgpt-logextract.py
    parses OpenAI's obfuscated "share this chat" HTML into a messages-JSON:
        {"messages": {guid: {raw, extracted, authors, timestamp, host}}}

  stage 2  (this module)
    messages-JSON -> paling chat training data
        - a CHARACTER dataset (the target/quell side), standard chat data
        - a PAINTER dataset (preceding assistant context -> the host/user turn)

The extractor's output is messy by nature -- the share HTML is obfuscated and
the upstream spec drifts. The rules below were reverse-engineered against real
extractor output (the quell logs) and must be followed exactly:

  * Walk ``messages`` in DICT INSERTION ORDER. It is a dict keyed by UUID, not
    a list. Do NOT sort by ``timestamp`` -- timestamps are non-monotonic.
  * Use ``extracted`` for text. Ignore ``raw`` (truncated garbage) and
    ``authors`` (a decoy counter, not a role).
  * Role comes from ``host``:
        host is None      -> assistant (the character/target)
        host is present   -> user      (the painter)
    EXCEPTION: the first substantial host-present entry is the SYSTEM PROMPT
    (role=system).
  * Drop the literal "Original custom instructions no longer available" marker.
  * Drop junk rows whose ``extracted.strip()`` is shorter than ~20 chars
    (telemetry crumbs).
  * Merge consecutive same-role turns. A single assistant reply sometimes
    streams as two adjacent ``host is None`` rows; concatenate them.
  * Unicode-normalize mojibake (see ``normalize_punctuation`` below, which
    mirrors the upstream extractor's reference implementation).

The character dataset is ``[system, user, assistant, user, assistant, ...]``
per session. The painter dataset pairs each painter (user) turn with the
assistant turn it answers -- painter turns are reactive and are never emitted
in isolation.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Rows shorter than this (after strip) are telemetry crumbs, not dialogue.
MIN_TURN_CHARS = 20

# The extractor emits this placeholder when the original system/custom
# instructions are not recoverable. It is not dialogue; drop it.
CUSTOM_INSTRUCTIONS_MARKER = "Original custom instructions no longer available"

# Mirror of chatgpt-logextract.py's normalize_punctuation. Kept here (rather
# than imported) because the extractor lives in a separate repo (archaea) that
# paling does not depend on. If the upstream table changes, update both.
_PUNCT_REPLACEMENTS = {
    "\u2014": "--", "\u2013": "-", "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'", "\u00e2\u0080\u0099": "'",
    "\u00e2\u0080\u009d": '"', "\u00e2\u0080\u009c": '"',
    "\u00e2\u0080\u0094": "--", "\u00e2\u0080\u00a6": "...",
    "\u00e2\u0080\u0098": "'", "\u00ee\u0088\u0084\u00ee\u0088\u0086": "",
}


def normalize_punctuation(text: str) -> str:
    """Normalize mojibake and smart punctuation.

    Reference implementation: chatgpt-logextract.py:normalize_punctuation.
    """
    for k, v in _PUNCT_REPLACEMENTS.items():
        text = text.replace(k, v)
    return text.replace("\\n", "\n")


def parse_chatlog_messages(messages: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """Turn an extractor messages-dict into an ordered list of chat turns.

    Returns ``[{"role": ..., "content": ...}, ...]`` in conversation order,
    starting with the system prompt (if present), then alternating
    user/assistant with consecutive same-role turns merged.

    ``messages`` is the value under the ``messages`` key of the extractor's
    output. Insertion order is authoritative.
    """
    turns: List[Dict[str, str]] = []
    system_assigned = False

    for entry in messages.values():
        extracted = entry.get("extracted") or ""
        text = normalize_punctuation(extracted).strip()

        if len(text) < MIN_TURN_CHARS:
            continue
        if CUSTOM_INSTRUCTIONS_MARKER in text:
            continue

        host = entry.get("host")

        if host is not None and not system_assigned:
            # First substantial host-present entry is the system prompt.
            role = "system"
            system_assigned = True
        elif host is None:
            role = "assistant"
        else:
            role = "user"

        # Merge consecutive same-role turns (e.g. a streamed assistant reply
        # split across two host==None rows).
        if turns and turns[-1]["role"] == role and role != "system":
            turns[-1]["content"] = turns[-1]["content"] + "\n\n" + text
        else:
            turns.append({"role": role, "content": text})

    return turns


def load_chatlog(path: Path) -> List[Dict[str, str]]:
    """Load one extractor messages-JSON file into ordered chat turns."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages")
    if not isinstance(messages, dict):
        raise ValueError(
            f"{path}: expected a 'messages' object (extractor output); "
            f"got {type(messages).__name__}"
        )
    return parse_chatlog_messages(messages)


def character_records(
    turns: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build CHARACTER (target/quell side) chat training records.

    One record per session: ``{"messages": [system, user, assistant, ...]}``.

    If the session carries no system turn and ``system_prompt`` is supplied, it
    is prepended so the record always leads with a system message.
    """
    if not turns:
        return []

    messages = list(turns)
    if messages[0]["role"] != "system" and system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    # A record needs at least one user/assistant exchange to be useful.
    if not any(m["role"] in ("user", "assistant") for m in messages):
        return []

    return [{"messages": messages}]


def painter_records(turns: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Build PAINTER pairs from ordered chat turns.

    Every painter (user) turn is reactive: it answers the assistant turn that
    immediately precedes it. We emit ``(preceding assistant context -> painter
    turn)`` pairs and never emit a painter turn in isolation. A leading painter
    turn with no preceding assistant context is skipped.
    """
    records: List[Dict[str, Any]] = []
    prev_assistant: Optional[str] = None

    for turn in turns:
        role = turn["role"]
        if role == "assistant":
            prev_assistant = turn["content"]
        elif role == "user":
            if prev_assistant is None:
                # Painter turn with no assistant context to react to; skip.
                continue
            records.append({
                "messages": [
                    {"role": "user", "content": prev_assistant},
                    {"role": "assistant", "content": turn["content"]},
                ]
            })
            # Consume the context so the same assistant turn is not reused for
            # a subsequent painter turn (which would only happen across a merge
            # boundary anyway).
            prev_assistant = None

    return records
