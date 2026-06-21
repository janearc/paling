"""Second-stage ingest: ChatGPT-share chatlogs -> paling chat training data.

Stage 1 (upstream, in archaea) scrapes a "share this chat" page into a
messages-JSON blob. This module is stage 2: it turns that blob into two
datasets -- a CHARACTER dataset (the assistant/target side) and a PAINTER
dataset (the human/prompter side). The per-message parsing rules are
reverse-engineered from real logs and explained inline at each step below.
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

# This table fixes TWO kinds of broken text and nothing else:
#   1. smart punctuation (curly quotes, em/en dashes, ellipsis) -> plain ascii
#   2. "mojibake" -- UTF-8 bytes that got misread as Latin-1, so a single
#      character like an em-dash comes through as gibberish; we map those byte
#      soups back to the ascii character they were meant to be.
#
# POLICY (deliberate, Max): we DO NOT blanket-strip non-ascii. It is tempting to
# just throw away everything that isn't plain ascii, but models -- ChatGPT
# especially -- talk in emoji, and those emoji carry real signal. Stripping them
# would lose meaning. So we only repair the known-broken forms above and leave
# every other non-ascii character (emoji included) untouched.
#
# paling owns this table and applies it to its own data unconditionally, even if
# an upstream tool already normalized -- paling does not trust upstream to match
# paling's canonical form (see "Normalization is paling's job" in the doc).
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
    """Repair smart punctuation and mojibake to ascii; leave emoji alone."""
    # Apply each repair from the table above, in order.
    for k, v in _PUNCT_REPLACEMENTS.items():
        text = text.replace(k, v)
    # The scrape sometimes leaves a literal backslash-n instead of a real
    # newline; turn it back into an actual line break.
    return text.replace("\\n", "\n")


def parse_chatlog_messages(messages: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """Turn the scraped messages-dict into an ordered [{"role", "content"}] list."""
    # IMPORTANT: walk the dict in INSERTION ORDER, which is the real
    # conversation order. The "timestamp" field looks tempting but is
    # non-monotonic garbage -- do not sort by it.
    turns: List[Dict[str, str]] = []
    # We only learn who said something from "host". The very first host-said
    # thing turns out to be the system prompt; this flag tracks whether we've
    # already claimed it.
    system_assigned = False

    for entry in messages.values():
        # Use the cleaned "extracted" text. "raw" is truncated junk and
        # "authors" is a decoy counter (not a real role) -- both ignored.
        extracted = entry.get("extracted") or ""
        text = normalize_punctuation(extracted).strip()
        host = entry.get("host")

        # Empty after stripping -> nothing to keep.
        if not text:
            continue
        # Tiny rows are usually telemetry crumbs ("status"), not dialogue, so we
        # drop them -- BUT only when host is empty. A short row WITH a host is a
        # real human jab like "cut the shit": that is the highest-signal painter
        # data, so host-present rows are kept no matter how short.
        if host is None and len(text) < MIN_TURN_CHARS:
            continue
        # A placeholder the scraper leaves when it can't recover the real system
        # prompt; it isn't dialogue, drop it.
        if CUSTOM_INSTRUCTIONS_MARKER in text:
            continue

        # Assign the role. host present + none claimed yet = the system prompt;
        # host present afterwards = the human (painter/"user"); no host = the
        # model (character/"assistant").
        if host is not None and not system_assigned:
            role = "system"
            system_assigned = True
        elif host is None:
            role = "assistant"
        else:
            role = "user"

        # A single reply sometimes arrives split across two rows. If this row is
        # the same role as the previous turn, glue it onto that turn instead of
        # starting a new one (but never merge into the system prompt).
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
    """Build the CHARACTER side: one chat record per session, [system, user, ...]."""
    if not turns:
        return []

    messages = list(turns)
    # Always lead with a system message. If the log had none and the caller
    # gave us a fallback prompt, stick it on the front.
    if messages[0]["role"] != "system" and system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    # A lone system message with no actual exchange is useless training data.
    if not any(m["role"] in ("user", "assistant") for m in messages):
        return []

    return [{"messages": messages}]


def painter_records(turns: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Build the PAINTER side: each human turn as a target, with full prior history."""
    # We want the painter (the human) to learn multi-turn escalation -- "you
    # dodged me, push harder" -- so every painter turn is trained WITH the whole
    # conversation so far, not just the last reply. To do that we flip the frame
    # into the painter's point of view: the model's words become the "user"
    # context and the painter's words become the "assistant" target we predict.
    # (No windowing -- sessions are short so we carry the full history. Windowing
    # is a deferred follow-up.)
    records: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []

    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if role == "system":
            # System prompt stays as-is at the front of the history.
            history.append({"role": "system", "content": content})
        elif role == "assistant":
            # The character spoke; in the painter's frame that is context, so
            # it goes into history as a "user" turn.
            history.append({"role": "user", "content": content})
        elif role == "user":
            # A painter turn. Skip it ONLY if there is literally nothing before
            # it to react to (e.g. a log that opens on the human with no system
            # prompt). If a system prompt is present, history isn't empty, so
            # even the opening painter line is emitted.
            if not history:
                continue
            # Emit one training record: everything so far + this painter turn as
            # the assistant target.
            records.append({
                "messages": list(history) + [
                    {"role": "assistant", "content": content},
                ]
            })
            # Then fold this painter turn into the running history so the NEXT
            # painter turn sees it as prior conversation.
            history.append({"role": "assistant", "content": content})

    return records
