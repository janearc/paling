from wonderlib.profiling import (
    TaxonometryProfile,
    TaxonometryCorpus,
    RarityAnalyzer,
    profile_document,
    data_to_taxonometry_corpus,
)
from wonderlib.benchmark import Benchmark
from wonderlib.git_stats import GitStats, GitCommitEntry, get_git_stats
from wonderlib.markdown_xml import markdown_to_xml

__version__ = "0.1.0"

# Re-export hub: these names are the public wonderlib API, imported here so
# callers can `from wonderlib import X`. __all__ makes the re-export explicit
# (and satisfies ruff F401).
__all__ = [
    "TaxonometryProfile",
    "TaxonometryCorpus",
    "RarityAnalyzer",
    "profile_document",
    "data_to_taxonometry_corpus",
    "Benchmark",
    "GitStats",
    "GitCommitEntry",
    "get_git_stats",
    "markdown_to_xml",
    "__version__",
]
