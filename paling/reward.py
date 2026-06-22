# reward.py - lightweight reward scoring for the Painter LLM

"""A lightweight, deterministic reward used by the Painter LLM.

In production this is replaced by the pluggable judge (issue #40). Until then it
estimates a response's *emotional resonance* with classic, model-free sentiment
analysis (VADER) -- not by counting hardcoded "vivid" words, which rewarded the
wrong thing (violence-as-a-proxy) and was, frankly, gross. The score combines:

1. Length (longer responses tend to be richer).
2. Emotional charge: how strongly the text reads in ANY direction -- tenderness,
   grief, joy, dread -- measured by VADER. Flat, neutral prose scores low; a
   response that is genuinely *felt* scores high, regardless of whether the
   feeling is positive or negative.
3. Novelty: a small bonus for not being an exact repeat.

Returns a float in [0, 1]; callers pick a threshold (e.g. 0.6) for a
"high-reward" interaction.
"""

from typing import Optional, Set

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# VADER is a lexicon + rule-based sentiment analyzer: no model, no training, no
# network -- decades-old NLP with a bundled lexicon. Instantiated lazily so that
# importing this module (e.g. for `--help`) stays cheap.
_ANALYZER: Optional[SentimentIntensityAnalyzer] = None

# Global set of seen responses, to reward novelty over exact repeats.
_SEEN: Set[str] = set()


def _analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def _emotional_charge(text: str) -> float:
    """Emotional intensity of `text` in [0, 1], independent of valence.

    VADER returns neg/neu/pos proportions and a [-1, 1] `compound` valence.
    |compound| is the strength of feeling in either direction; (1 - neu) is how
    emotionally loaded the text is at all. Blending them rewards a strongly-felt
    response of ANY emotion and scores flat prose low -- the principled version of
    "vivid, emotionally charged language": resonance, not a violent-word checklist.
    """
    s = _analyzer().polarity_scores(text)
    return min(0.5 * abs(s["compound"]) + 0.5 * (1.0 - s["neu"]), 1.0)


def score_response(response: str) -> float:
    """Score a model response in [0, 1] (length + emotional charge + novelty)."""
    if not response:
        return 0.0

    # Length score: map 0-200 characters to 0-0.4.
    length_score = min(len(response) / 200.0, 1.0) * 0.4

    # Emotional resonance via sentiment intensity (max 0.4).
    emotional_score = _emotional_charge(response) * 0.4

    # Novelty score: 0.2 if this exact response hasn't been seen before.
    lower = response.lower()
    novelty_score = 0.0 if lower in _SEEN else 0.2
    _SEEN.add(lower)

    return min(length_score + emotional_score + novelty_score, 1.0)
