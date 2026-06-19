# Stage 3 — Relationship Extractor (design)

Tracking: paling#18 (pipeline rebuild). Edge-projection sub-question: paling#10 (GATED — do not implement until the algorithm is agreed).

This is a **design doc only**. No paling implementation code is proposed to be written yet. It covers: (1) what the gizzard prior art actually does, (2) the substantial bugs found in it, (3) a clean reimplementation of paling stage 3 fitted to the bento/daemon/skill pattern and the ethic/concept/process/axiom/system ontology, and (4) open questions for Max — especially the projection algorithm.

---

## 0. Where stage 3 sits

Pipeline so far:

- **Stage 1 (verify)** — `verify_bento()` walks a bento, confirms it is processable (corpus + valid schema), writes `preflight/preflight.json`. Gate for everything downstream.
- **Stage 2 (taxonometry)** — `profile_bento()` profiles the corpus into per-document lexical signatures under `taxonometry/signatures/**/{stem}-taxonometry.json` plus a corpus rollup `taxonometry/corpus.json`. Model-free, deterministic. Gated on stage 1.
- **Stage 3 (this doc) — relationship extraction** — turn the corpus's natural language into a typed graph of ontology nodes (`ethic / concept / process / axiom / system`) and edges between them, plus graph metrics over that graph. Gated on stage 2.

The ontology (the structure of character) is the five node kinds plus the relationship edges between them. Stage 3 is the first stage that produces the *graph* the rest of the pipeline (gap generation, convergence Q/A, training) reasons over.

---

## 1. Functional spec of gizzard's relationship path

Prior art: `/Users/jane/work/archaea/wonder/tools/gizzard/src/gizzard/processor.py` (one 1317-line module), driven by `cli.py` and configured by `config/gizzard.yaml` + `config/schema.yaml`.

The config (`gizzard.yaml`) is the part worth keeping conceptually. It defines:
- `symbol_mapping.categories`: the **exact ontology** — `ethic:e, concept:c, axiom:a, process:p, system:s`. (The schema requires all five.)
- `relationship_notation`: `arrow: →`, `relationship_separator: ∧`, `alternative_separator: ∨`, `concept_separator: ⋅`.
- `example_transformations`: e.g. `"The ethic of care emphasizes maintaining and nurturing relationships..." -> "E:care->maintain,nurture;growth,support"`. This is the intended NL→ontology compression shape.

There are **two distinct relationship code paths** in gizzard, and they barely talk to each other. This is the key thing to separate.

### 1a. `RelationshipAnalyzer` — graph metrics over *already-noted* edges (PORT THIS)

`RelationshipAnalyzer` (processor.py:22-81) is pure graph analytics. It does **no NL understanding**:

- `build_relationship_graph(relationships)` (28-40): takes a list of `{source, target}` dicts and builds an adjacency list `graph` plus a `node_degrees` counter. Treats edges as **directed** for storage but bumps degree on **both** endpoints.
- `analyze_graph_metrics()` (42-81): returns `{density, avg_clustering, avg_degree}`.
  - density = `2m / (n(n-1))` (the *undirected* density formula).
  - avg_clustering = mean local clustering coefficient over nodes with ≥2 neighbours.
  - avg_degree = `sum(node_degrees) / n`.

**Inputs → outputs:** `List[{source,target}]` → `{density, avg_clustering, avg_degree}`. That's the whole job. This is the "port the existing path" half — the algorithmic intent is sound (graph metrics), only the implementation is buggy (see §2). Note it is invoked only from `analyze_model_compatibility` (867), which feeds a `_estimate_model_performance` heuristic that scores `gpt-4/gpt-3.5/claude/gemini` — dead weight for paling; drop it.

### 1b. Where NL actually becomes nodes+edges in gizzard (the part to REBUILD)

There are **two** unrelated extraction routines, and the one wired into the run pipeline is the weaker of the two.

**Routine A — `extract_relationships(text, title)` (282-375).** The "real" NL extractor on paper. Steps:
1. Explicit wiki-links: `[[Target]]` → `{source: title, target, type: links_to}`.
2. Semantic patterns: split into sentences; for each, scan for regex indicator groups (`requires|needs|...` → `requires`; `relates to|...` → `relates_to`; `extends|...`; `contrasts with|...`; `part of|...`; `similar to|...`; `influences|...`). On a hit, grab capitalized noun-phrase chunks before/after the indicator and emit `{source, target, type}`.
3. Proximity heuristic: for a hard-coded `special_terms` set (`metareal, orthoreal, ..., sigil, kernel, ethic`), within a 10-word window emit `related_concept` edges.
4. Dedup by `source-type-target`.

**This routine is never actually called by the kernel pipeline.** `process_file` (377) calls it, but `process_file` itself is dead — nothing in `process_kernel`/`process_kernel_data` invokes it. (`cli.py` only calls `process_kernel`.)

**Routine B — title-mention extraction inside `process_kernel_data` (624-751).** This is what actually runs:
1. First pass: read every sigil `.md`, extract its `# Title`, collect `sigil_titles` (lowercased).
2. Second pass: for each doc, lowercase its content; for every *other* sigil title that appears as a substring, add `relationship_map[title].add(other_title)`. **Untyped** — every edge is just "X mentions Y."
3. Reduce content (`clean_content`, 453 — stopword/suffix stripping + crude long-sentence compression).
4. Format edges token-efficiently as strings `source→t1,t2,t3` via `clean_term` (753 — lowercases, abbreviates realm names, replaces `of/and/the`).

So gizzard's *operative* node-typing is: **node = whole markdown document**, typed by `identify_category` (241), which guesses the ontology bucket by checking whether the literal word `ethic`/`concept`/`axiom`/`process`/`system` appears in the file path. Edges are **untyped co-mention** between documents. The richer typed-edge logic in Routine A is orphaned.

**Net:** gizzard's genuine, shippable contribution is the *config ontology* + the *graph-metrics* path (1a). Its NL→graph extraction (1b) is shallow (document-granular nodes, substring co-mention edges, path-string typing) and partly dead code. paling's stage 3 must **rebuild** the NL→{ethic/concept/process/axiom/system}+edges extraction, and may **port the intent** of the graph metrics.

---

## 2. Bug catalog

All references `processor.py` unless noted. "Consequence" is the observable effect.

1. **processor.py:53-54 — density double-counts and mislabels directed/undirected.** `m = sum(len(neighbors) ...)` counts **directed** edges, then density multiplies by 2 (`2.0*m`) as if `m` were undirected edge count. With directed storage this overcounts density by ~2× and can exceed 1.0. Consequence: density metric is wrong/unbounded.
2. **processor.py:67 — clustering coefficient uses directed adjacency asymmetrically.** `n2 in self.graph.get(n1, [])` only sees edges stored in n1's out-list, but neighbours were collected from a single direction, so triangles are under/over-counted depending on edge direction. Consequence: `avg_clustering` is not a meaningful clustering coefficient.
3. **processor.py:39-40 vs 53 — degree and edge accounting disagree.** `node_degrees` increments **both** endpoints (undirected intent) while `graph` is a **directed** adjacency list; `avg_degree` (75) and density (53) therefore use inconsistent edge models. Consequence: metrics are internally inconsistent.
4. **processor.py:331-333 — `extract_relationships` inverts its own guard and drops most edges.** `if source != title and target != title: continue` skips any relationship where the current title is on *neither* side — but the comment says "skip if neither is the current concept," and the very next block (336) assumes the title may be the target. The condition discards exactly the cross-concept edges it should keep, then the swap logic only fires for the survivors. Consequence: semantic edges are almost never emitted; Routine A is nearly inert even when called.
5. **processor.py:251 & 254 — article/aux-verb stripping regex is unanchored and substring-matching.** `re.sub(r'the|a|an', '', text)` removes the letters `a` from inside every word ("**a**nd"→"nd", "c**a**re"→"cre"), and `r'is|are|was|were'` mangles "th**is**", "f**are**". Consequence: `reduce_content` corrupts content; downstream tokens are garbage. (`clean_content`, the routine actually used, avoids this by matching whole words — but `reduce_content` is still wired into `process_file` and the example transforms.)
6. **processor.py:275 — suffix stripper is unanchored over the whole word set.** `re.sub(r'ing$|ed$|s$', ...)` strips trailing `s` from every plural/3rd-person AND from legitimate words ("axi**s**"→"axi", "ethic**s**"→"ethic but also "proces**s**"→"proces"). Consequence: term identity is destroyed, breaking later substring co-mention matching (Routine B relies on exact title substrings).
7. **processor.py:241-246 — `identify_category` matches on substring of the whole path/content and returns the *first* category whose name appears.** Iterating a dict and returning on first `category in content.lower()`; a file under `.../concept/` that merely contains the word "ethic" in prose is mis-typed. Also returns `None` (skipping the file entirely, 393-395) if none of the five literal words appear in the path. Consequence: silent doc loss + wrong ontology buckets.
8. **processor.py:438 — `all_relationships` is fed bare target *strings*, but every consumer expects dicts.** `self.all_relationships.extend(rel['target'] for rel in relationships)` appends strings; `analyze_framework_statistics` (1182) does `if isinstance(rel, dict)` and so silently skips all of them. Consequence: `relationship_types` and `most_connected_concepts` are always empty.
9. **processor.py:667-695 + 707-723 — `relationship_map` is global but re-emitted per file inside the loop.** Inside the second pass, `formatted_relationships` is rebuilt from the *entire accumulating* `relationship_map` on every iteration and attached to each document's `content` entry (722). Consequence: each document's `relationships` field contains every relationship discovered *so far across all docs*, not its own — O(n²) duplication and incorrect per-doc attribution.
10. **processor.py:855 — `print_token_stats` divides by zero when no tokens.** `total_reduction = (total_original - total_processed) / total_original` with no guard. Consequence: crash on an empty/all-skipped corpus.
11. **processor.py:1121 — `datetime` referenced but only imported inside the `try`.** In `get_git_metadata`, the `except` branch calls `datetime.utcnow()` but `from datetime import datetime` happens at 1099 inside the `try`; if the failure occurs before that import line executes, the `except` raises `NameError`. Consequence: the error handler can itself crash. (Also `datetime.utcnow()` is deprecated.)
12. **processor.py:873 — graph built from formatted *strings*, not dicts.** `build_relationship_graph(processed_kernel['relationships'])` is passed the `"source→t1,t2"` **strings** from `process_kernel_data` (749), but `build_relationship_graph` expects `{source,target}` dicts (34: `if isinstance(rel, dict)`). Consequence: the graph is always empty → all graph metrics are 0.0 in the actual pipeline.
13. **processor.py:96-97 & 346-349 — ontology/term sets are hard-coded Wonder vocabulary** (`metareal, Rokolisk, Cinder, ...`). Not a logic bug but a fatal generality bug: the extractor only "sees" relationships near Wonder-specific words. Consequence: zero transfer to any other corpus; the special-term proximity path emits nothing for a general bento.
14. **cli.py — `validate` constructs `GizzardProcessor(None, schema_path)`** then calls `processor.validate_kernel(kernel_path)` passing a **path**, but `validate_kernel` (554) expects a kernel **dict** and runs jsonschema on it. Also `GizzardProcessor.__init__` opens `config_path` unconditionally (111), so `None` raises immediately. Consequence: `gizzard validate` is entirely broken.

Summary: the only relationship metric that runs in the real pipeline (Routine B) produces untyped, mis-attributed, duplicated edges; the typed extractor (Routine A) is dead and self-sabotaging; and the graph metrics it all feeds compute on an empty graph (#12). The *ideas* (five-bucket ontology, typed edges, density/clustering) are worth carrying; almost none of the code is.

---

## 3. Clean reimplementation: paling stage 3

### 3.1 Contract (matches the stage-1/stage-2 pattern)

**`bento.py`** gains:

```
def extract_relationships(bento_path) -> dict
```

Mirrors `profile_bento`:
1. `report = verify_bento(path)`; if not `report["valid"]` → return `{"bento_id", "extracted": False, "issues": [...]}`.
2. **Gate on stage 2**: require `taxonometry/corpus.json` to exist (stage 2 ran). If missing → `extracted: False, issues: ["taxonometry gate failed; run profile first"]`. This is the stage-2→stage-3 dependency, analogous to how stage 2 gates on stage 1's validity.
3. Read `schema/schema.json` `routing` to choose the extraction model (see §3.4).
4. Run extraction (§3.3) over `raw_data/**/*.md`, reusing stage-2 signatures (rare_terms) as candidate node lexicon hints.
5. Write outputs (§3.2). Return a summary dict `{bento_id, extracted: True, nodes, edges, by_kind, graph_metrics, untyped_documents}`.

**`daemon.py`** gains:

```
@app.post("/bento/{bento_id}/extract")
```
- 404 if bento dir missing (same as verify/profile).
- Call `bento.extract_relationships(path)`; if `not report["extracted"]` → `HTTPException(409, detail=report["issues"])` (same 409-on-gate-failure shape as `/profile`).
- `producer.emit(bento_id, "extract", BanchanState.IN_PROGRESS)` on success.
- Return the report.

**`skill/paling`** gains a verb between `profile` and `prepare`:
```
paling extract <bento>    extract ontology nodes + relationship edges
```
mapped to `curl -fsS -X POST "$base/bento/$1/extract"`. Usage text updated.

**Tests** (`tests/test_bento_api.py`, same style — `TestClient`, temp `_BENTOS_ROOT`, stubbed `emit`):
- `test_extract_valid_bento`: create → ingest a 2-doc corpus (one clearly an *ethic*, one a *process*) → verify → profile → extract; assert 200, `extracted is True`, expected node kinds present, at least one edge, and that `relationships/graph.json` + `relationships/nodes.jsonl` exist on disk.
- `test_extract_ungated_bento_conflicts`: create → (no profile) → extract → 409. (And/or verify-but-not-profile → 409.)
- `test_extract_missing_bento`: `/bento/ghost/extract` → 404.
- Keep extraction **deterministic in tests** by exercising the model-free fallback path (§3.4), exactly as stage 2 tests run model-free.

### 3.2 Output location and shape

Write under a new **`relationships/`** dir the stage owns and rebuilds each run (mirroring `taxonometry/signatures/` — clean-dir-then-write so reruns are idempotent and never merge stale graph state). Rationale for a top-level `relationships/` rather than `anchors/paling/`: `anchors/` holds *curated* steering anchors (owner + paling), whereas this is a *derived, regenerable* artifact — same category as `taxonometry/`. (Open question 4 lets Max overrule this.)

```
relationships/
  nodes.jsonl        # one node per line
  edges.jsonl        # one edge per line
  graph.json         # rollup: counts by kind, graph metrics, untyped docs
```

**Node** (`nodes.jsonl`):
```json
{
  "id": "ethic/care",
  "kind": "ethic",                 // ethic|concept|process|axiom|system
  "label": "the ethic of care",
  "source_doc": "ethic-of-care.md",
  "evidence": ["Care is what makes alignment safe."],
  "rare_terms": ["metareal", "cocreation"],   // carried from stage-2 signature
  "confidence": 0.82
}
```
- `id` = `{kind}/{slug(label)}`, stable across runs for the same label.
- `kind` is the ontology bucket. Document-level nodes are allowed (one per doc, like gizzard Routine B), but the design intent is **finer-grained nodes** where the extractor can name a concept inside a doc.

**Edge** (`edges.jsonl`):
```json
{
  "source": "ethic/care",
  "target": "concept/convergence",
  "type": "requires",   // requires|relates_to|extends|contrasts_with|part_of|similar_to|influences|mentions
  "evidence": "Without care, convergence becomes performance.",
  "confidence": 0.6,
  "extractor": "model"  // "model" | "comention" | "wikilink"
}
```
- Edge `type` vocabulary carried from gizzard Routine A's pattern set (the one sound piece of its taxonomy), plus `mentions` as the untyped fallback (gizzard Routine B's co-mention, kept but explicitly labelled low-value via `extractor: "comention"`).
- `wikilink` for explicit `[[...]]` if present in source markdown.

**`graph.json`** (the rollup; reuses the ported metrics):
```json
{
  "bento_id": "...",
  "nodes": 41,
  "edges": 117,
  "by_kind": {"ethic": 9, "concept": 18, "process": 7, "axiom": 4, "system": 3},
  "by_edge_type": {"requires": 30, "relates_to": 52, "mentions": 35},
  "graph_metrics": {"density": 0.07, "avg_clustering": 0.21, "avg_degree": 5.7},
  "untyped_documents": ["misc-notes.md"]
}
```

`graph_metrics` is the **ported** `RelationshipAnalyzer` path — clean reimplementation with the §2 bugs fixed: choose one edge model (recommend **undirected** for character-graph cohesion metrics), compute density as `2m/(n(n-1))` over *undirected* `m`, compute clustering over a symmetric adjacency, guard all divisions, and feed it **dicts** not strings (fixes #1, #2, #3, #12).

### 3.3 Extraction algorithm (the genuinely new work)

Two layers, both emit into the same node/edge schema:

**Layer 1 — deterministic / model-free (always runs; the test path).**
- **Node typing.** For each document, assign an ontology `kind`. Do **not** copy gizzard's path-substring guess (#7). Instead: (a) prefer an explicit front-matter / schema hint if present; (b) else score the document against a small keyword/cue lexicon per kind (ethic: "ought/duty/care/obligation"; process: "step/iterate/converge/method"; axiom: "always/never/by definition/given that"; system: "component/interface/pipeline/state"; concept: default). Ties/empties → `concept` (never drop the doc — fixes #7's silent loss). Record low confidence so Layer 2 / humans can revisit.
- **Edges.** (i) explicit `[[...]]` wikilinks; (ii) co-mention edges between documents whose labels/titles appear in each other (gizzard Routine B, but typed `mentions`, attributed **per-doc correctly** — fixes #9); (iii) the typed-indicator regex from Routine A with the **guard bug fixed** (#4) and operating on whole-word boundaries (fixes #5/#6) — but only as `relates_to`/`requires`/etc. candidates with modest confidence.
- Reuse stage-2 `rare_terms` from each doc's signature as the candidate vocabulary for intra-doc concept nodes (better than gizzard's hard-coded Wonder term list — fixes #13).

**Layer 2 — model-assisted NL→ontology (the real extraction; runs when a model is routed).**
- Prompt the routed model per document (or per chunk) to: (1) name the ethic/concept/process/axiom/system units the passage expresses, and (2) name the directed relationships between them, in the typed vocabulary. Output strict JSON conforming to the node/edge schema. This is the analogue of gizzard's *intended* `"E:care->maintain,nurture"` compression, done by a model instead of brittle regex.
- Layer 2 nodes/edges carry `extractor: "model"` and higher confidence; Layer 1 results are merged in and deduped (by node `id`, by `source-type-target` for edges — keep gizzard's dedup key, it's the one clean bit of Routine A).

Determinism note: Layer 2 should run at low/zero temperature for reproducibility; tests exercise Layer 1 only (no MLX load), exactly like stage 2.

### 3.4 Model routing (which model does NL→ontology)

`schema.json` routing today: `{gap_generation: flan-t5-large, summarization: mistral}`.

The NL→ontology bucketing + typed-edge extraction is a **structured-extraction / classification** task over prose, not free generation. Recommendation: **route Layer 2 to `mistral` (the `summarization` route), not flan-t5-large.** Reasons:
- flan-t5-large is a seq2seq instruction model tuned for short targeted generations (it's the gap/question generator in `exhaustive-flan.py`); it is weak at emitting long, *strict* structured JSON and at holding a five-way taxonomy + edge list in one pass.
- mistral (decoder LM, larger context, used for summarization) is better suited to "read this passage, emit a JSON object of typed nodes and edges" — closer to a comprehension/structuring task than a generation task.
- Keeping flan reserved for stage-4 gap/Q-A generation (its existing job) avoids overloading one route with two very different prompt regimes.

Concretely: stage 3 reads `routing` and uses the **summarization** model for Layer 2; if no model is available/loadable it falls back to Layer 1 (model-free), matching stage 2's "falls back to fast lexical heuristics" behavior. This keeps a clean routing story and leaves the final call to Max (open question 3) since it's a routing-policy decision, not a hard technical constraint.

### 3.5 What is "port" vs "new"

- **Port (intent, clean rewrite):** graph metrics (`density/avg_clustering/avg_degree`) from `RelationshipAnalyzer`; the typed-edge vocabulary; the dedup key; the five-bucket ontology from `gizzard.yaml`.
- **New:** all NL→node/kind typing (lexical cue scoring + model extraction); correct per-doc edge attribution; reuse of stage-2 signatures as the term lexicon; the bento/daemon/skill wiring; JSONL node/edge outputs.
- **Drop entirely:** `ModelContextAnalyzer`, `analyze_model_compatibility`, `_estimate_model_performance`, `generate_model_profile`, all the gpt-4/claude/gemini scoring, the YAML refined-kernel writer, `reduce_content`/`clean_content` content-compression (that's a different concern from edge extraction and is where most §2 corruption lives).

---

## 4. Open design questions for Max

These are deliberately **not** decided here.

1. **(paling#10 — GATED) The edge-exploration / projection algorithm.** Once we have a typed node/edge graph, the downstream "edge exploration / lexical cartography" step wants a 2-D projection for clustering + rendering (`cartographer.py`'s `TopologyBanchan` already hard-codes UMAP→HDBSCAN→KNN). The gated decision is *what projection to standardize on*:
   - **UMAP** (current `cartographer.py` choice): preserves global + local structure, fast, but stochastic and sensitive to `n_neighbors`/`min_dist`; non-deterministic unless seeded; can manufacture apparent clusters.
   - **t-SNE**: strong local-neighborhood separation, but destroys global distances, `O(n²)`-ish, perplexity-sensitive, poor for "how far apart are two ethics" questions.
   - **PCA / spectral / graph-Laplacian layout**: deterministic, cheap, interpretable axes; weaker at non-linear structure.
   - **Force-directed graph layout** (operate on the actual edge graph, not on embeddings): most faithful to *our* typed edges since the graph is the primary object, but layouts aren't stable across runs and don't give a metric space.
   - Cross-cutting questions: do we project **node embeddings** or lay out the **edge graph** directly? Is the projection an *artifact for humans* (rendering only) or does any downstream stage consume coordinates? What seed/determinism guarantee do we need for reproducible bentos? **Recommend not implementing any projection in stage 3** — stage 3 emits the graph; projection is a separate gated banchan. Decision needed before `cartographer.py` is wired in.

2. **Node granularity.** Document-level nodes (one node per `.md`, gizzard's operative model) vs. sub-document concept nodes (model names concepts inside a passage). Finer nodes give a richer graph but cost a model pass and risk fragmentation/dupes. How fine does Max want to go for v1?

3. **Routing policy.** Confirm mistral (summarization route) for Layer 2 extraction vs. flan-t5-large (gap route), or introduce a dedicated `extraction` key in `schema.json` `routing`? Adding a third route is cleaner but changes the schema contract used by stage-1 verify.

4. **Output location.** Top-level `relationships/` (proposed, parallels `taxonometry/`) vs. under `anchors/paling/` (treats the extracted graph as paling-authored steering material). Affects how stage 4 and the anchor-curation flow consume it.

5. **Confidence / human-in-the-loop.** Should low-confidence nodes/edges be written but flagged (`approved: false`-style, like the `*-review.json` gold files), creating a review surface, or filtered out? The review-JSON precedent suggests a "write everything, mark approved" convention may be the house style.

6. **Edge directionality for metrics.** The character graph is conceptually directed (`requires`, `part_of`), but cohesion metrics (density/clustering) are cleanest undirected. Store directed edges, compute metrics undirected? (That's the §2-#1/#3 fix and the proposed default — confirm.)
