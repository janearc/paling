# bento scaffolding and corpus ingestion, shared by the CLI and the daemon API.
# a bento is the atomic unit of work: a directory tree holding raw corpus,
# schema, adapters, and outputs. keeping this here (not inline in the CLI) lets
# the API create and feed bentos over HTTP -- the interface the agent skill
# drives, instead of anyone hand-placing files.

import json
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# --- stage report structs -----------------------------------------------------
# the pipeline stages return typed reports (not bare dicts) so callers -- the
# daemon endpoints, the CLI, downstream stages -- get a stable, self-describing
# shape. FastAPI serializes these to the same JSON the skill wrapper prints.

class VerifyReport(BaseModel):
    # result of the stage-1 verify gate (see verify_bento).
    bento_id: str
    valid: bool
    issues: List[str] = []
    corpus_files: int = 0
    corpus_bytes: int = 0
    archetype: Optional[str] = None
    routing: Optional[Dict[str, Any]] = None


class TaxonometrySummary(BaseModel):
    # corpus-level rollup of the stage-2 taxonometry.
    #
    # "taxonometry" is a coined term (Max's): the lexical/statistical *signature*
    # of a document's vocabulary -- zipf-frequency rarity buckets + POS-based
    # rarity -- not a taxonomy. See wonderlib/profiling.py for how each
    # per-document signature is computed.
    documents: int = 0
    rare_terms_total: int = 0
    rare_terms_unique: int = 0
    zipf_avg: float = 0.0
    zipf_cluster: List[int] = [0, 0, 0]
    rarity_pos_avg: float = 0.0
    thin_documents: List[str] = []


class ProfileReport(BaseModel):
    # result of stage-2 profiling. the summary fields are flattened in so the
    # skill wrapper / API shape stays flat and stable.
    bento_id: str
    profiled: bool
    issues: List[str] = []
    documents: int = 0
    rare_terms_total: int = 0
    rare_terms_unique: int = 0
    zipf_avg: float = 0.0
    zipf_cluster: List[int] = [0, 0, 0]
    rarity_pos_avg: float = 0.0
    thin_documents: List[str] = []


class GraphMetrics(BaseModel):
    # cohesion metrics over the relationship graph. edges are stored directed
    # (direction carries meaning for character: A grounds B != B grounds A), but
    # these structural metrics are computed on the undirected projection -- "how
    # tightly knit is this region of meaning" doesn't care about arrow direction.
    # (gizzard's bug was mixing the two: density on directed edges, then doubled.)
    density: float = 0.0
    avg_degree: float = 0.0
    avg_clustering: float = 0.0
    directed: bool = True
    cohesion_basis: str = "undirected"


class Coverage(BaseModel):
    # which corpus documents produced a node. the pipeline used to drop inputs
    # silently (the missing-signature holes); coverage makes a hole a reported
    # signal, not something you notice later.
    corpus_files: int = 0
    documents_with_node: int = 0
    missing: List[str] = []


class ExtractReport(BaseModel):
    # result of stage-3 relationship extraction.
    bento_id: str
    extracted: bool
    issues: List[str] = []
    nodes: int = 0
    edges: int = 0
    by_kind: Dict[str, int] = {}
    by_edge_type: Dict[str, int] = {}
    metrics: Optional[GraphMetrics] = None
    low_confidence: int = 0
    coverage: Optional[Coverage] = None


def _safe_write_json(out_path, data):
    # filesystem writes fail sometimes (perms, full disk, races). persistence of
    # a stage report is a side effect, not the result the caller needs, so we log
    # and carry on rather than letting a write error take down the request.
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2))
        return True
    except OSError as e:
        logger.warning("failed to write %s (non-fatal): %s", out_path, e)
        return False


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


def verify_bento(bento_path) -> VerifyReport:
    # walk a bento and report whether it looks processable -- without doing any
    # processing. this is the pipeline's preflight gate: a bento needs a corpus
    # and a valid schema before anything downstream (extract/generate/train) runs.
    path = Path(bento_path).expanduser().resolve()
    if not path.is_dir():
        return VerifyReport(bento_id=path.name, valid=False,
                            issues=[f"bento dir not found: {path}"])

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

    return VerifyReport(
        bento_id=path.name,
        valid=not issues,
        corpus_files=len(md_files),
        corpus_bytes=sum(p.stat().st_size for p in md_files),
        archetype=archetype,
        routing=routing,
        issues=issues,
    )


def write_preflight(bento_path, report: VerifyReport):
    # persist a verify report to the bento's preflight/ dir so the gate result is
    # on disk (stage 1 of the pipeline writes here before stage 2 runs). the write
    # is best-effort -- see _safe_write_json.
    out = Path(bento_path).expanduser().resolve() / "preflight" / "preflight.json"
    _safe_write_json(out, report.model_dump())
    return out


def _summarize_taxonometry(corpus) -> TaxonometrySummary:
    # roll per-document signatures up into the corpus-level, internal-to-paling
    # metrics. the load-bearing one is thin_documents: documents with no rare
    # terms carry no distinctive character vocabulary -- they're the flat, generic
    # inputs that won't yield character signal downstream (the "thin and flat"
    # smell), so the pipeline surfaces them here before generation wastes effort.
    sigs = corpus.signatures
    n = len(sigs)
    cluster = [0, 0, 0]  # summed zipf buckets: high-rarity, medium, low
    all_terms = []
    for s in sigs:
        for i in range(3):
            cluster[i] += s.zipf_cluster[i]
        all_terms.extend(s.rare_terms)
    thin = sorted(Path(f).name.replace("-taxonometry.json", "")
                  for f in corpus.no_rare_term_filenames())
    return TaxonometrySummary(
        documents=n,
        rare_terms_total=len(all_terms),
        rare_terms_unique=len({t.lower() for t in all_terms}),
        zipf_avg=round(sum(s.zipf_avg for s in sigs) / n, 4) if n else 0.0,
        zipf_cluster=cluster,
        rarity_pos_avg=round(sum(s.rarity_pos for s in sigs) / n, 4) if n else 0.0,
        thin_documents=thin,
    )


def profile_bento(bento_path) -> ProfileReport:
    # pipeline stage 2: profile the corpus into per-document taxonometry
    # signatures (lexical rarity metrics) plus a corpus-level summary, written to
    # taxonometry/. gated on the stage-1 verify gate -- a bento that doesn't look
    # processable isn't worth profiling. runs model-free (lexical zipf/POS
    # heuristics); no MLX load, so it's fast and deterministic.
    path = Path(bento_path).expanduser().resolve()
    report = verify_bento(path)
    if not report.valid:
        return ProfileReport(
            bento_id=path.name,
            profiled=False,
            issues=["verify gate failed; fix the bento and re-verify"] + report.issues,
        )

    # lazy import: profiling pulls in spacy/torch/wordfreq, which the rest of the
    # daemon (create/ingest/verify) has no reason to load.
    from paling.profile_runner import profile_single_file
    from wonderlib.profiling import DataToTaxonometryCorpus

    raw = path / "raw_data"
    tax_dir = path / "taxonometry"
    # this run's signatures go in a dedicated subdir we own, mirroring raw_data's
    # tree. two reasons: (1) files that share a stem across raw_data subdirs would
    # otherwise clobber each other (flat {stem}-taxonometry.json names), and (2)
    # the corpus rollup globs recursively, so aggregating from a clean dir we just
    # wrote keeps any pre-existing hand-curated taxonometry (e.g. sigil/) out of
    # the metrics. we rebuild it from scratch each run.
    sig_dir = tax_dir / "signatures"
    if sig_dir.exists():
        shutil.rmtree(sig_dir)
    sig_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(p for p in raw.rglob("*.md") if p.is_file())
    for f in md_files:
        # include_git=False: raw_data is a copy outside any git tree, so per-file
        # git stats would only produce noise.
        out_dir = sig_dir / f.relative_to(raw).parent
        profile_single_file(f, out_dir, model_path=None, include_git=False)

    corpus = DataToTaxonometryCorpus(str(sig_dir))
    summary = _summarize_taxonometry(corpus)
    _safe_write_json(tax_dir / "corpus.json", summary.model_dump())

    return ProfileReport(bento_id=path.name, profiled=True, **summary.model_dump())


# --- stage 3: relationship extraction -----------------------------------------
# the structure of character is an ontology of typed concepts -- ethic / concept
# / process / axiom / system -- and the relationships between them. stage 3 maps
# the corpus into that graph. this is the deterministic, model-free Layer 1
# (cue-lexicon node typing + in-prose mention edges); a Layer 2 that routes the
# corpus through a configurable model client (mistral) for richer typed
# extraction is the next increment -- see docs/pipeline/stage3-*.md.
#
# NB: this deliberately does NOT forklift gizzard. gizzard's extractor that
# actually ran used document-granular path-string nodes (useless for us) and
# built its graph from formatted strings, so its metrics were silently empty.
# here nodes are concept-level (merged across docs by label).

_CANONICAL_KINDS = ("ethic", "concept", "process", "axiom", "system")


def _doc_title(text, stem):
    # a doc's primary concept comes from its first H1; fall back to the filename.
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return stem.replace("-", " ").replace("_", " ").strip()


def _normalize_id(label):
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _classify_kind(rel_parts, stem, title):
    # cue lexicon: the ontology bucket is encoded in the corpus path
    # (core/ethic/..., skillsets/stewardship/axiom/...), so the deepest matching
    # path component wins; otherwise look in the filename + title. non-canonical
    # buckets (primitive, people, seed, ...) fall back to the general 'concept'.
    for part in reversed(rel_parts):
        if part.lower() in _CANONICAL_KINDS:
            return part.lower()
    hay = f"{stem} {title}".lower()
    for kind in _CANONICAL_KINDS:
        if re.search(rf"\b{kind}", hay):
            return kind
    return "concept"


def extract_relationships(bento_path) -> ExtractReport:
    # pipeline stage 3: extract a typed concept graph from the corpus. gated on
    # stage-2 taxonometry (we reuse its rare_terms as sub-document concepts), and
    # transitively on stage-1 verify. deterministic Layer 1; writes the graph to
    # anchors/paling/relationships/ (machine-derived anchors), rebuilt each run.
    path = Path(bento_path).expanduser().resolve()
    verify = verify_bento(path)
    if not verify.valid:
        return ExtractReport(
            bento_id=path.name, extracted=False,
            issues=["verify gate failed; fix the bento and re-verify"] + verify.issues)
    if not (path / "taxonometry" / "corpus.json").is_file():
        return ExtractReport(
            bento_id=path.name, extracted=False,
            issues=["stage-2 taxonometry has not run; run `paling profile` first"])

    raw = path / "raw_data"
    sig_dir = path / "taxonometry" / "signatures"

    # rare terms (sub-document concepts) keyed by their source doc, from stage 2.
    rare_by_path = {}
    for sig in sig_dir.rglob("*-taxonometry.json"):
        try:
            data = json.loads(sig.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        fn = data.get("filename")
        if fn:
            rare_by_path[str(Path(fn).resolve())] = data.get("rare_terms", [])

    # pass 1: each doc contributes a primary concept node (its subject), merged
    # across docs by normalized label so a concept named in many docs is ONE node.
    nodes = {}
    docs = []
    missing = []
    md_files = sorted(p for p in raw.rglob("*.md") if p.is_file())
    for f in md_files:
        rel = str(f.relative_to(raw))
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.warning("could not read %s (non-fatal): %s", rel, e)
            missing.append(rel)
            continue
        title = _doc_title(text, f.stem)
        pid = _normalize_id(title)
        if not pid:
            missing.append(rel)
            continue
        kind = _classify_kind(f.relative_to(raw).parent.parts, f.stem, title)
        rare = rare_by_path.get(str(f.resolve()), [])
        node = nodes.setdefault(pid, {"id": pid, "kind": kind, "label": title,
                                      "source_docs": [], "rare_terms": [], "confidence": 1.0})
        node["kind"] = kind          # a primary subject carries its canonical kind
        node["confidence"] = 1.0
        if rel not in node["source_docs"]:
            node["source_docs"].append(rel)
        for t in rare:
            if t not in node["rare_terms"]:
                node["rare_terms"].append(t)
        docs.append((pid, text, rel, rare))

    # pass 2: promote distinctive rare terms to their own (lower-confidence)
    # concept nodes, unless they already exist as a primary concept.
    for pid, text, rel, rare in docs:
        for term in rare:
            tid = _normalize_id(term)
            if not tid or tid in nodes:
                continue
            n = nodes.setdefault(tid, {"id": tid, "kind": "concept", "label": term,
                                       "source_docs": [], "rare_terms": [], "confidence": 0.5})
            if rel not in n["source_docs"]:
                n["source_docs"].append(rel)

    # pass 3: directed reference edges -- if concept Q's label appears in a doc
    # whose primary concept is P, then P references Q. phrase + word-boundary
    # match (the anchored regex gizzard's :251/:275 botched), labels >= 4 chars.
    matchers = [(nid, n, re.compile(rf"\b{re.escape(n['label'])}\b", re.I))
                for nid, n in nodes.items() if len(n["label"]) >= 4]
    edges = {}
    for pid, text, rel, rare in docs:
        for nid, n, rx in matchers:
            if nid == pid:
                continue
            if rx.search(text):
                key = (pid, nid)
                conf = 1.0 if n["confidence"] >= 1.0 else 0.6
                if key in edges:
                    edges[key]["mentions"] += 1
                    edges[key]["confidence"] = max(edges[key]["confidence"], conf)
                else:
                    edges[key] = {"source": pid, "target": nid, "type": "references",
                                  "mentions": 1, "confidence": conf, "extractor": "lexical"}

    metrics = _graph_metrics(list(nodes), edges)
    by_kind = {}
    for n in nodes.values():
        by_kind[n["kind"]] = by_kind.get(n["kind"], 0) + 1
    by_edge_type = {}
    for e in edges.values():
        by_edge_type[e["type"]] = by_edge_type.get(e["type"], 0) + 1
    low_conf = sum(1 for n in nodes.values() if n["confidence"] < 1.0) \
        + sum(1 for e in edges.values() if e["confidence"] < 1.0)
    coverage = Coverage(corpus_files=len(md_files),
                        documents_with_node=len(md_files) - len(missing),
                        missing=missing)

    out_dir = path / "anchors" / "paling" / "relationships"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "nodes.jsonl", nodes.values())
    _write_jsonl(out_dir / "edges.jsonl", edges.values())
    graph = {"bento_id": path.name, "nodes": len(nodes), "edges": len(edges),
             "by_kind": by_kind, "by_edge_type": by_edge_type,
             "metrics": metrics.model_dump(), "low_confidence": low_conf,
             "coverage": coverage.model_dump()}
    _safe_write_json(out_dir / "graph.json", graph)

    return ExtractReport(bento_id=path.name, extracted=True, nodes=len(nodes),
                         edges=len(edges), by_kind=by_kind, by_edge_type=by_edge_type,
                         metrics=metrics, low_confidence=low_conf, coverage=coverage)


def _write_jsonl(out_path, rows):
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    except OSError as e:
        logger.warning("failed to write %s (non-fatal): %s", out_path, e)


def _graph_metrics(node_ids, edges) -> GraphMetrics:
    # cohesion on the UNDIRECTED projection of the directed edge set.
    n = len(node_ids)
    adj = {nid: set() for nid in node_ids}
    undirected = set()
    for (s, t) in edges:
        if s in adj and t in adj and s != t:
            adj[s].add(t)
            adj[t].add(s)
            undirected.add(frozenset((s, t)))
    u = len(undirected)
    density = round(2 * u / (n * (n - 1)), 6) if n > 1 else 0.0
    avg_degree = round(2 * u / n, 4) if n else 0.0
    clustering = []
    for nid in node_ids:
        nb = list(adj[nid])
        k = len(nb)
        if k < 2:
            clustering.append(0.0)
            continue
        links = sum(1 for i in range(k) for j in range(i + 1, k) if nb[j] in adj[nb[i]])
        clustering.append(2 * links / (k * (k - 1)))
    avg_clustering = round(sum(clustering) / n, 6) if n else 0.0
    return GraphMetrics(density=density, avg_degree=avg_degree, avg_clustering=avg_clustering)


# --- stage 4: question generation ---------------------------------------------
# the depth the 2024 pipeline had came from an iterate-to-convergence loop, not
# from hand-writing: the gap_generation model is asked for instruct-style
# questions over each context until it stops producing new ones. this is the
# faithful port of wonder-local's md_to_questions (question half). a later
# increment brings the stage-3 relationship graph into the prompt for graph-aware
# question generation -- the "bring it forward to 2026" step.
#
# the model is whatever the bento's schema routes to gap_generation -- a seq2seq
# model today (flan-t5-large), swappable for something stronger when hardware
# allows. paling does not depend on any specific model; the routed one is the
# current worker.

# the prompt was arrived at by trial and error against the current gap_generation
# model; kept verbatim so behaviour ports faithfully before we evolve it. a
# different model may want a different prompt.
_QUESTION_PROMPT = (
    "Identify three distinct concepts discussed in the following paragraph. For "
    "each concept, generate one instruct-style question that would help a model "
    "understand its meaning and how it relates to the other two concepts. Prefix "
    "each question with 'Q>' on a new line.\n\nParagraph: {context}"
)

# stop a context once a generation adds no new unique questions, or after this
# many attempts (the model can loop without converging on dense text).
_MAX_QUESTION_ATTEMPTS = 10


# typed record of one context's converged questions, persisted for stage 5 (answers).
class ContextQuestions(BaseModel):
    context_id: str
    source_doc: str
    context: str
    questions: List[str] = []


# typed result of stage-4 question generation (the daemon and skill serialize this).
class QuestionsReport(BaseModel):
    bento_id: str
    generated: bool
    issues: List[str] = []
    contexts: int = 0
    contexts_skipped: int = 0
    questions_total: int = 0
    questions_by_context: Dict[str, int] = {}
    attempts_total: int = 0
    model: Optional[str] = None


# extract questions from a model completion by splitting on the 'Q>' marker.
def _parse_questions(text):
    # the gap_generation model emits the marker inline as often as on its own line
    # (e.g. "Care is what?> Maintenance"), so split on the marker, then keep each
    # segment up to its first '?' -- dropping trailing noise and any leading
    # enumeration ("1. "). faithful to wonder-local's split-on-marker parser.
    out = []
    for chunk in re.split(r"[Qq]?>", text):
        chunk = chunk.strip()
        if "?" not in chunk:
            continue
        q = chunk[: chunk.index("?") + 1]
        q = re.sub(r"^\s*\d+[.)]?\s*", "", q).strip()
        if len(q) > 1 and q.endswith("?"):
            out.append(q)
    return out


# stage 4: iterate the gap_generation model to convergence per context, persisting
# the questions for stage-5 answering.
def generate_questions(bento_path) -> QuestionsReport:
    # gated on stage-2 taxonometry (and transitively stage-1 verify). thin
    # documents (no rare-term character signal) are skipped rather than fed to
    # generation.
    from paling import modelclient

    path = Path(bento_path).expanduser().resolve()
    verify = verify_bento(path)
    if not verify.valid:
        return QuestionsReport(
            bento_id=path.name, generated=False,
            issues=["verify gate failed; fix the bento and re-verify"] + verify.issues)
    corpus_summary = path / "taxonometry" / "corpus.json"
    if not corpus_summary.is_file():
        return QuestionsReport(
            bento_id=path.name, generated=False,
            issues=["stage-2 taxonometry has not run; run `paling profile` first"])

    model = (verify.routing or {}).get("gap_generation", "flan-t5-large")
    try:
        thin = set(json.loads(corpus_summary.read_text()).get("thin_documents", []))
    except (OSError, json.JSONDecodeError):
        thin = set()

    out_dir = path / "anchors" / "paling" / "questions"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = path / "raw_data"
    md_files = sorted(p for p in raw.rglob("*.md") if p.is_file())
    contexts = skipped = total_q = attempts_total = 0
    by_context = {}
    issues = []

    for f in md_files:
        cid = _normalize_id(f.stem) or f.stem
        if f.stem in thin or f.name in thin:
            skipped += 1
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError as e:
            logger.warning("could not read %s (non-fatal): %s", f.name, e)
            skipped += 1
            continue
        if not text:
            skipped += 1
            continue

        prompt = _QUESTION_PROMPT.format(context=text)
        unique = set()
        attempts = 0
        # iterate to convergence: regenerate until a pass yields no new question.
        while attempts < _MAX_QUESTION_ATTEMPTS:
            try:
                completion = modelclient.generate_seq2seq(
                    model, prompt, temperature=0.8, top_p=0.95, top_k=75)
            except modelclient.ModelUnavailable as e:
                # fail closed: a missing generation model is fatal to the stage,
                # not a context to silently drop.
                issues.append(f"model {model!r} unavailable: {e}")
                return QuestionsReport(bento_id=path.name, generated=False,
                                       issues=issues, model=model)
            attempts += 1
            prev = len(unique)
            unique.update(_parse_questions(completion))
            if len(unique) == prev:
                break

        attempts_total += attempts
        questions = sorted(unique)
        contexts += 1
        total_q += len(questions)
        by_context[cid] = len(questions)
        cq = ContextQuestions(context_id=cid, source_doc=str(f.relative_to(raw)),
                              context=text, questions=questions)
        _safe_write_json(out_dir / f"{cid}.json", cq.model_dump())

    return QuestionsReport(
        bento_id=path.name, generated=True, contexts=contexts,
        contexts_skipped=skipped, questions_total=total_q,
        questions_by_context=by_context, attempts_total=attempts_total,
        model=model, issues=issues)


# --- stage 5: answer generation -----------------------------------------------
# the second half of the convergence engine: for each question stage 4 produced,
# iterate the gap_generation model to convergence on answers (grounded in the
# context), collecting the multi-answer set that stage 6 (pre-human curation) and
# a human then grade. faithful port of wonder-local's md_to_questions answer half.
# output is the review shape -- question + candidate answers + approved=False.

# the answer prompt grounds the model in the context; kept close to the 2024 tool.
_ANSWER_PROMPT = "{question} Answer based only on this context:\n\n{context}"

# stop a question once an answer pass yields nothing new, or after this many
# attempts (answers converge faster than questions; the 2024 tool used 5).
_MAX_ANSWER_ATTEMPTS = 5


# one question with its converged candidate answers, awaiting curation.
class QuestionEntry(BaseModel):
    question: str
    answers: List[str] = []
    approved: bool = False


# the review record for one context: each question with its candidate answers.
class ContextReview(BaseModel):
    context_id: str
    source_doc: str
    context: str
    questions: List[QuestionEntry] = []


# typed result of stage-5 answer generation.
class AnswersReport(BaseModel):
    bento_id: str
    generated: bool
    issues: List[str] = []
    contexts: int = 0
    questions_answered: int = 0
    answers_total: int = 0
    attempts_total: int = 0
    model: Optional[str] = None


# stage 5: answer each stage-4 question by iterating the model to convergence,
# writing the review shape for stage-6 curation.
def generate_answers(bento_path) -> AnswersReport:
    # gated on stage-4 questions existing (transitively stages 1-2). reads
    # anchors/paling/questions/, writes anchors/paling/review/. fail-closed if the
    # model is unavailable -- a missing model is fatal, not a question to drop.
    from paling import modelclient

    path = Path(bento_path).expanduser().resolve()
    verify = verify_bento(path)
    if not verify.valid:
        return AnswersReport(
            bento_id=path.name, generated=False,
            issues=["verify gate failed; fix the bento and re-verify"] + verify.issues)
    q_dir = path / "anchors" / "paling" / "questions"
    q_files = sorted(q_dir.glob("*.json")) if q_dir.is_dir() else []
    if not q_files:
        return AnswersReport(
            bento_id=path.name, generated=False,
            issues=["stage-4 questions not found; run `paling questions` first"])

    model = (verify.routing or {}).get("gap_generation", "flan-t5-large")

    out_dir = path / "anchors" / "paling" / "review"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contexts = answered = answers_total = attempts_total = 0
    issues = []
    for qf in q_files:
        try:
            cq = json.loads(qf.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("could not read %s (non-fatal): %s", qf.name, e)
            continue
        context = cq.get("context", "")
        entries = []
        for question in cq.get("questions", []):
            prompt = _ANSWER_PROMPT.format(question=question, context=context)
            seen = set()
            attempts = 0
            # iterate to convergence: regenerate until a pass adds no new answer.
            while attempts < _MAX_ANSWER_ATTEMPTS:
                try:
                    ans = modelclient.generate_seq2seq(
                        model, prompt, temperature=0.9, top_p=0.95, top_k=100).strip()
                except modelclient.ModelUnavailable as e:
                    issues.append(f"model {model!r} unavailable: {e}")
                    return AnswersReport(bento_id=path.name, generated=False,
                                         issues=issues, model=model)
                attempts += 1
                prev = len(seen)
                norm = ans.lower()
                if norm and norm not in seen:
                    seen.add(norm)
                if len(seen) == prev:
                    break
            attempts_total += attempts
            answers = sorted(seen)
            answered += 1
            answers_total += len(answers)
            entries.append(QuestionEntry(question=question, answers=answers, approved=False))
        review = ContextReview(context_id=cq.get("context_id", qf.stem),
                               source_doc=cq.get("source_doc", ""),
                               context=context, questions=entries)
        _safe_write_json(out_dir / f"{review.context_id}.json", review.model_dump())
        contexts += 1

    return AnswersReport(
        bento_id=path.name, generated=True, contexts=contexts,
        questions_answered=answered, answers_total=answers_total,
        attempts_total=attempts_total, model=model, issues=issues)
