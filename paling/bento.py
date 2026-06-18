# bento scaffolding and corpus ingestion, shared by the CLI and the daemon API.
# a bento is the atomic unit of work: a directory tree holding raw corpus,
# schema, adapters, and outputs. keeping this here (not inline in the CLI) lets
# the API create and feed bentos over HTTP -- the interface the agent skill
# drives, instead of anyone hand-placing files.

import json
import shutil
import uuid
from pathlib import Path

# the canonical bento sub-structure (mirrors `paling create bento`).
_BENTO_DIRS = (
    "raw_data",
    "schema",
    "adapters",
    "preflight",
    "taxonometry",
    "anchors/owner",
    "anchors/paling",
    "acceptance",
    "output",
)

DEFAULT_BENTOS_ROOT = "/opt/paling/var/bentos"


def scaffold_bento(base_dir, name=None, archetype="unprocessed"):
    # create a new bento tree and its schema.json under base_dir; returns
    # (bento_id, path). a missing name gets a uuid4, matching `create bento --new`.
    bento_id = name or str(uuid.uuid4())
    base_path = Path(base_dir).expanduser().resolve() / bento_id
    if base_path.exists():
        raise FileExistsError(f"bento '{bento_id}' already exists at {base_path}")

    for d in _BENTO_DIRS:
        (base_path / d).mkdir(parents=True, exist_ok=True)

    schema = {
        "archetype": archetype,
        "routing": {"gap_generation": "flan-t5-large", "summarization": "mistral"},
    }
    (base_path / "schema" / "schema.json").write_text(json.dumps(schema, indent=2))
    return bento_id, base_path


def ingest_corpus(bento_path, source_path):
    # copy the markdown corpus from source_path into the bento's raw_data and
    # return the number of files ingested. source_path may be a single .md file
    # or a directory walked recursively for .md files.
    raw = Path(bento_path) / "raw_data"
    if not raw.is_dir():
        raise FileNotFoundError(f"bento has no raw_data dir: {bento_path}")

    src = Path(source_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"corpus source does not exist: {src}")

    files = [src] if src.is_file() else sorted(p for p in src.rglob("*.md") if p.is_file())
    for f in files:
        shutil.copy2(f, raw / f.name)
    return len(files)
