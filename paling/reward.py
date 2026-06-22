"""Reward scoring for the Painter LLM.

Scores a candidate response in [0, 1] on whether it earns a place in the
character's voice. Three signals: emotional resonance (the main one -- a character
who only says flat, neutral things is lifeless), length (longer tends to be
richer), and novelty (an exact repeat earns nothing). Callers threshold the score
(e.g. 0.6) to keep the high-reward responses.

Emotional resonance is measured with sentiment analysis, not a keyword list, so a
response that is strongly felt -- tender, grieving, joyful, afraid -- scores high
whatever its subject, and nothing is rewarded simply for being lurid.
"""

from typing import Optional, Set

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# vader is a lexicon + rule-based sentiment scorer: no model, no network. Created
# lazily so importing this module stays cheap.
_ANALYZER: Optional[SentimentIntensityAnalyzer] = None

# responses scored so far, so an exact repeat earns no novelty bonus.
_SEEN: Set[str] = set()


def _analyzer() -> SentimentIntensityAnalyzer:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def _emotional_charge(text: str) -> float:
    """Strength of feeling in `text`, in [0, 1], regardless of its direction."""
    # |compound| is how strong the sentiment is either way; (1 - neu) is how much
    # of the text carries feeling at all. Blended, a strongly-felt line scores high
    # and flat prose scores low.
    s = _analyzer().polarity_scores(text)
    return min(0.5 * abs(s["compound"]) + 0.5 * (1.0 - s["neu"]), 1.0)


def score_response(response: str) -> float:
    """Score a response in [0, 1] from its emotional resonance, length, and novelty."""
    if not response:
        return 0.0

    # length: 0-200 characters maps to 0-0.4.
    length_score = min(len(response) / 200.0, 1.0) * 0.4

    # emotional resonance via sentiment intensity, up to 0.4.
    emotional_score = _emotional_charge(response) * 0.4

    # novelty: 0.2 the first time a response is seen, 0 on any exact repeat.
    lower = response.lower()
    novelty_score = 0.0 if lower in _SEEN else 0.2
    _SEEN.add(lower)

    return min(length_score + emotional_score + novelty_score, 1.0)
