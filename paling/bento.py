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

DEFAULT_BENTOS_ROOT = str(Path.home() / "var" / "paling" / "bentos")


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


def verify_bento(bento_path):
    # walk a bento and report whether it looks processable -- without doing any
    # processing. this is the pipeline's preflight gate: a bento needs a corpus
    # and a valid schema before anything downstream (extract/generate/train) runs.
    path = Path(bento_path).expanduser().resolve()
    if not path.is_dir():
        return {"bento_id": path.name, "valid": False, "issues": [f"bento dir not found: {path}"]}

    issues = []
    missing = [d for d in _BENTO_DIRS if not (path / d).is_dir()]
    if missing:
        issues.append(f"missing dirs: {', '.join(missing)}")

    raw = path / "raw_data"
    md_files = [p for p in raw.rglob("*.md") if p.is_file()] if raw.is_dir() else []
    if not md_files:
        issues.append("raw_data has no .md corpus")

    archetype = routing = None
    schema_path = path / "schema" / "schema.json"
    if schema_path.is_file():
        try:
            schema = json.loads(schema_path.read_text())
            archetype = schema.get("archetype")
            routing = schema.get("routing")
            if not archetype:
                issues.append("schema.json missing 'archetype'")
            if not routing:
                issues.append("schema.json missing 'routing'")
        except json.JSONDecodeError as e:
            issues.append(f"schema.json is invalid json: {e}")
    else:
        issues.append("missing schema/schema.json")

    return {
        "bento_id": path.name,
        "valid": not issues,
        "corpus_files": len(md_files),
        "corpus_bytes": sum(p.stat().st_size for p in md_files),
        "archetype": archetype,
        "routing": routing,
        "issues": issues,
    }


def write_preflight(bento_path, report):
    # persist a verify report to the bento's preflight/ dir so the gate result is
    # on disk (stage 1 of the pipeline writes here before stage 2 runs).
    out = Path(bento_path).expanduser().resolve() / "preflight" / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return out
