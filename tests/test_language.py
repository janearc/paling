"""Language-contract tests.

The detector is a heuristic, so the tests assert on clearly-in-one-language
samples, not edge cases. The regression case is the real failure: the wonder
bento carried an undeclared Turkish glossary, and paling should refuse it.
"""

import pytest

from paling.language import (
    detect_language, assess_languages, enforce_language_contract,
    UnexpectedLanguageError,
)

_EN = ("The coherence of the system is the moment when being and function are in "
       "alignment. This is what makes it feel alive and present to a person here.")
_TR = ("Tutarlılık, varlık, işlev ve davranışın uyum içinde olduğu andır. Bu bir "
       "şeyin canlı hissi vermesini sağlayan şeydir, değil mi, daha çok gibi.")
_FR = ("Le système de la cohérence est le moment où une chose dans la fonction et "
       "la présence sont alignées pour les personnes, et ce n'est pas une erreur.")
_NL = ("Het systeem van de coherentie is het moment dat een ding en de functie "
       "niet aanwezig zijn voor de mensen maar als een, dat is het niet altijd.")


def test_detects_each_language():
    assert detect_language(_EN) == "en"
    assert detect_language(_TR) == "tr"
    assert detect_language(_FR) == "fr"
    assert detect_language(_NL) == "nl"


def test_short_text_is_unknown():
    assert detect_language("coherence") == "unknown"
    assert detect_language("# Title\n\nok then") == "unknown"


def _bento(tmp_path, files):
    raw = tmp_path / "raw_data"
    for rel, text in files.items():
        p = raw / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


def test_assess_reports_present_languages(tmp_path):
    b = _bento(tmp_path, {
        f"core/{n}.md": _EN for n in range(5)
    } | {
        f"turkish/{n}.md": _TR for n in range(4)
    })
    a = assess_languages(b)
    assert set(a.languages) == {"en", "tr"}
    assert a.counts["en"] == 5 and a.counts["tr"] == 4


def test_monolingual_bento_passes_without_declaration(tmp_path):
    b = _bento(tmp_path, {f"core/{n}.md": _EN for n in range(5)})
    a = enforce_language_contract(b)  # no raise
    assert a.languages == ["en"]


def test_undeclared_mix_is_whoapodna(tmp_path):
    # the real wonder failure: English corpus + an undeclared Turkish glossary.
    b = _bento(tmp_path, {
        f"core/{n}.md": _EN for n in range(8)
    } | {
        f"skillsets/turkish/glossary/{n}.md": _TR for n in range(5)
    })
    with pytest.raises(UnexpectedLanguageError, match="mixes languages"):
        enforce_language_contract(b)


def test_declared_language_with_stray_is_whoapodna(tmp_path):
    b = _bento(tmp_path, {
        f"core/{n}.md": _EN for n in range(8)
    } | {
        f"turkish/{n}.md": _TR for n in range(4)
    })
    with pytest.raises(UnexpectedLanguageError, match=r"declares languages \['en'\]"):
        enforce_language_contract(b, declared=["en"])


def test_declared_multilingual_passes(tmp_path):
    # opting into multilingual on purpose is allowed.
    b = _bento(tmp_path, {
        f"core/{n}.md": _EN for n in range(8)
    } | {
        f"turkish/{n}.md": _TR for n in range(4)
    })
    a = enforce_language_contract(b, declared=["en", "tr"])
    assert set(a.languages) == {"en", "tr"}


def test_clean_english_bento_passes_declared_en(tmp_path):
    # the wonder-en shape: all English, declared en -> clean.
    b = _bento(tmp_path, {f"core/{n}.md": _EN for n in range(10)})
    a = enforce_language_contract(b, declared=["en"])
    assert a.languages == ["en"]
