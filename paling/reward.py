# reward.py - Simple reward scoring for Painter LLM

"""A very lightweight reward model used by the Painter LLM.

In production this would be replaced by a trained neural model that
estimates emotional resonance, surprise, and narrative impact. For now we
use a deterministic heuristic that scores a response based on:

1. Length (longer responses tend to be richer).
2. Presence of vivid cue words (e.g. "star", "fire", "death", ...).
3. A simple novelty check – penalise responses that are exact repeats.

The function returns a float in the range [0, 1]. Callers can decide a
threshold (e.g. 0.6) for what constitutes a "high‑reward" interaction.
"""

from typing import Set

# Pre‑defined cue words that signal vivid, emotionally charged language.
_CUE_WORDS = {
    "star", "fire", "death", "copper", "annihilation",
    "burn", "lava", "blood", "scream", "void",
}

# Keep a global set of seen responses to encourage novelty.
_SEEN: Set[str] = set()


def score_response(response: str) -> float:
    """Score a model response.

    The score is a weighted combination of length, cue‑word coverage, and
    novelty. Returned value is clamped to [0, 1].
    """
    if not response:
        return 0.0

    # Normalise to lower case for cue detection.
    lower = response.lower()

    # Length score: map 0‑200 characters to 0‑0.4.
    length_score = min(len(response) / 200.0, 1.0) * 0.4

    # Cue word score: proportion of cue words present (max 0.4).
    cues_found = sum(1 for w in _CUE_WORDS if w in lower)
    cue_score = min(cues_found / len(_CUE_WORDS), 1.0) * 0.4

    # Novelty score: 0.2 if response hasn't been seen before.
    novelty_score = 0.0 if lower in _SEEN else 0.2
    _SEEN.add(lower)

    total = length_score + cue_score + novelty_score
    return min(total, 1.0)
