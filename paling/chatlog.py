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
    (telemetry crumbs) -- but ONLY when ``host`` is None. Telemetry crumbs are
    always host-less; a short host-PRESENT row is a real painter jab ("cut the
    shit") and is the highest-signal painter data, so it is always kept.
  * Merge consecutive same-role turns. A single assistant reply sometimes
    streams as two adjacent ``host is None`` rows; concatenate them.
  * Unicode-normalize mojibake (see ``normalize_punctuation`` below).

Normalization contract (separation of duties): paling ALWAYS normalizes text
its own way, regardless of what the upstream extractor or a bento-builder did.
Those layers may normalize too -- that is fine -- but paling re-normalizes
unconditionally. paling owns the canonical form of its own training data; it
does not trust an upstream's normalization to match paling's table. This is a
deliberate decision (Max), not redundant work: the upstream table drifts, and
paling's table is maintained independently to be correct for paling's corpus.

The character dataset is ``[system, user, assistant, user, assistant, ...]``
per session. The painter dataset carries the FULL running conversation in the
painter's frame (roles inverted): the character's words become ``user`` context
and the painter's words become the ``assistant`` target, so the painter can
learn multi-turn escalation rather than a single isolated reaction.
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

# paling's OWN punctuation/mojibake normalization table. This started as a copy
# of chatgpt-logextract.py's normalize_punctuation but is now maintained
# independently: paling owns the canonical form of its training data and
# re-normalizes regardless of upstream (see module docstring). Order in the
# dict matters only insofar as multi-byte mojibake keys must be replaced before
# their single-character counterparts -- here they are disjoint, so order is
# irrelevant.
_PUNCT_REPLACEMENTS = {
    # Smart punctuation -> ascii.
    "\u2014": "--",      # em-dash
    "\u2013": "-",       # en-dash
    "\u201c": '"',       # left double quote
    "\u201d": '"',       # right double quote
    "\u2018": "'",       # left single quote
    "\u2019": "'",       # right single quote
    "\u2026": "...",     # horizontal ellipsis
    "\u00a0": " ",       # non-breaking space
    # UTF-8-as-Latin-1 mojibake (a leading "\u00e2\u0080" = the bytes for the
    # U+20xx punctuation block misread one byte at a time).
    "\u00e2\u0080\u0099": "'",     # right single quote
    "\u00e2\u0080\u0098": "'",     # left single quote
    "\u00e2\u0080\u009c": '"',     # left double quote
    "\u00e2\u0080\u009d": '"',     # right double quote
    "\u00e2\u0080\u0094": "--",    # em-dash
    "\u00e2\u0080\u0093": "-",     # en-dash
    "\u00e2\u0080\u00a6": "...",   # ellipsis
    # Private-use glyph pair the extractor leaves behind.
    "\u00ee\u0088\u0084\u00ee\u0088\u0086": "",
}


def normalize_punctuation(text: str) -> str:
    """Normalize mojibake and smart punctuation to paling's canonical ascii form.

    paling owns this table and applies it unconditionally; see the module
    docstring for the separation-of-duties contract.
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
        host = entry.get("host")

        # Drop truly-empty text regardless of host.
        if not text:
            continue
        # Apply the telemetry-crumb length floor ONLY to host-less rows.
        # Painter turns (host present) are always kept even when short: a
        # 12-char "cut the shit" is the highest-signal painter data, not a
        # telemetry crumb. Telemetry crumbs are always host-less.
        if host is None and len(text) < MIN_TURN_CHARS:
            continue
        if CUSTOM_INSTRUCTIONS_MARKER in text:
            continue

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
    """Build PAINTER records carrying the full running conversation.

    The painter's skill is multi-turn escalation ("you've dodged me, push
    harder"), so each painter turn is emitted with the ENTIRE prior conversation
    as context -- not just the immediately-preceding assistant turn. The frame
    is inverted into the painter's point of view:

      * a ``system`` turn stays ``system``;
      * a character (``assistant``) turn becomes a ``user`` turn -- the
        character's words are the painter's context;
      * a painter (``user``) turn becomes the ``assistant`` target.

    For each painter turn we emit ``{"messages": history + [painter as
    assistant]}``, then fold that painter turn into the running history (as an
    ``assistant`` turn) so later examples see it. A painter turn is skipped ONLY
    when the history is genuinely empty (nothing to react to). When a system
    prompt is present the history is never empty, so a leading painter turn (the
    opener) IS emitted as ``[system] + [opener-as-assistant]`` -- it is no longer
    dropped.

    No windowing: sessions are short and we carry full history. Windowing is a
    deferred follow-up.
    """
    records: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []

    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if role == "system":
            history.append({"role": "system", "content": content})
        elif role == "assistant":
            # The character's words are the painter's context.
            history.append({"role": "user", "content": content})
        elif role == "user":
            if not history:
                # Degenerate: a painter turn with no prior context at all.
                continue
            records.append({
                "messages": list(history) + [
                    {"role": "assistant", "content": content},
                ]
            })
            # Fold this painter turn into the running history so later examples
            # carry it as prior conversation.
            history.append({"role": "assistant", "content": content})

    return records
