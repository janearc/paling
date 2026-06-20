"""Language contract for a bento.

paling does not curate a corpus -- it *validates* one. A bento declares what it
should look like (which language, or that it is deliberately multilingual); this
module decides, fast and without running a model, whether the files on disk match
that declaration. When they don't -- an English bento carrying a Turkish glossary
nobody declared -- paling refuses it loudly and kicks it back to a human, who
either fixes the declaration or curates the corpus. The system can't trust a
client to have manicured their data; it can refuse to silently train on a mess.

The detector is a deterministic script/stopword heuristic, not an LLM: the whole
point is a cheap static check that runs at acceptance time, before a single
generation pass is spent. The primitive (`detect_language`) is self-contained and
meant to migrate into the shared corpus-assessment library; the *enforcement*
(`enforce_language_contract`) is paling's, because paling owns the bento contract.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

# Stopwords that are common and discriminating within each language. Short, hand
# picked for precision over recall -- a handful of confident hits beats a long
# noisy list. English is the fallback, so it does not need to win on stopwords.
_STOPWORDS: Dict[str, frozenset] = {
    "en": frozenset("the of and to a in is that it as for with on are be this".split()),
    "tr": frozenset("ve bir bu için değil ile da de daha çok gibi olarak şey".split()),
    "es": frozenset("el la los las de que y en un una con por para se no".split()),
    "fr": frozenset("le la les des une dans que et est pour pas avec sur ne".split()),
    "nl": frozenset("de het een en van niet dat zijn op met voor te als maar".split()),
}

# Characters that are strong, near-unique fingerprints for a language. The Turkish
# dotless-i and soft-g in particular almost never appear in the others.
_SCRIPT_HINTS: Dict[str, str] = {
    "tr": "ışğİĞŞ",
    "es": "ñ¿¡",
    "fr": "çàâêëîïôœ",
    "nl": "ĳ",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# below this many real words a file is too short to judge; report "unknown".
_MIN_WORDS = 12
# a language must clear this share of the stopword signal to win outright.
_DECISION_MARGIN = 1.5


def detect_language(text: str) -> str:
    """Best-effort deterministic language code for a block of text.

    Returns an ISO-ish code from the supported set, or "unknown" when the text is
    too short or no language stands out. Never raises; never runs a model.
    """
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) < _MIN_WORDS:
        return "unknown"

    scores: Counter = Counter()
    wordset = set(words)
    for lang, stops in _STOPWORDS.items():
        scores[lang] += sum(1 for w in words if w in stops)

    # script fingerprints are decisive -- weight them heavily.
    for lang, chars in _SCRIPT_HINTS.items():
        hits = sum(text.count(c) for c in chars)
        if hits:
            scores[lang] += 3 * hits

    if not scores or max(scores.values()) == 0:
        # latin words, no stopword or script signal: assume the english fallback.
        return "en" if wordset else "unknown"

    ranked = scores.most_common()
    top_lang, top = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    if runner == 0 or top >= runner * _DECISION_MARGIN:
        return top_lang
    # a real contest between two languages in one file -- don't guess.
    return "mixed"


class FileLanguage(BaseModel):
    model_config = {"extra": "forbid"}
    path: str
    language: str


class LanguageAssessment(BaseModel):
    model_config = {"extra": "forbid"}
    # languages that clear the significance bar (not one stray foreign word).
    languages: List[str] = []
    # every scanned file and its detected language.
    files: List[FileLanguage] = []
    # count of files per detected language.
    counts: Dict[str, int] = {}


# a language is "present" (not noise) once it covers this share of the corpus or
# this many files -- whichever is easier to clear on a small corpus.
_SIGNIFICANT_SHARE = 0.05
_SIGNIFICANT_FILES = 3


class UnexpectedLanguageError(Exception):
    """A bento's files do not match its declared language set -- the WHOAPODNA.

    Raised at acceptance time so a malformed corpus is refused before any
    generation runs. The message tells the human exactly what to do: declare the
    language, or ask for a multilingual bento on purpose.
    """


def assess_languages(bento_path) -> LanguageAssessment:
    """Scan a bento's raw_data and report the languages actually present."""
    raw = Path(bento_path).expanduser().resolve() / "raw_data"
    files: List[FileLanguage] = []
    counts: Counter = Counter()
    for f in sorted(raw.rglob("*.md")):
        try:
            lang = detect_language(f.read_text(errors="replace"))
        except OSError:
            continue
        files.append(FileLanguage(path=str(f.relative_to(raw)), language=lang))
        counts[lang] += 1

    total = len(files) or 1
    significant = sorted(
        lang for lang, n in counts.items()
        if lang not in ("unknown",)
        and (n >= _SIGNIFICANT_FILES or n / total >= _SIGNIFICANT_SHARE)
    )
    return LanguageAssessment(languages=significant, files=files, counts=dict(counts))


def enforce_language_contract(
    bento_path, declared: Optional[List[str]] = None
) -> LanguageAssessment:
    """Validate a bento against its declared language set; raise on a mismatch.

    `declared` is what the bento says it should be (e.g. ["en"], or ["en","tr"]
    for a deliberate multilingual corpus). When omitted, paling infers: a single
    significant language is accepted as monolingual; more than one with no
    declaration is the error -- the human must opt into multilingual on purpose.
    """
    assessment = assess_languages(bento_path)
    present = assessment.languages

    if declared:
        allowed = {d.lower() for d in declared}
        stray = sorted(set(present) - allowed)
        if stray:
            offenders = [
                fl.path for fl in assessment.files if fl.language in stray
            ][:10]
            raise UnexpectedLanguageError(
                f"bento declares languages {sorted(allowed)} but also contains "
                f"{stray}. Offending files include {offenders}. Either fix the "
                f"declaration to {sorted(allowed | set(stray))} for a multilingual "
                f"bento, or remove the out-of-language files."
            )
        return assessment

    # no declaration: infer. monolingual is fine; multiple is a WHOAPODNA.
    if len(present) > 1:
        raise UnexpectedLanguageError(
            f"bento mixes languages {present} but declares none. paling won't "
            f"guess which one you meant -- declare a single language, or request a "
            f"multilingual bento explicitly (declared languages={present})."
        )
    return assessment
