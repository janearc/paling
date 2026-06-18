# Gizzard Defect Catalog — Full Deconstruction

This is the complete, line-by-line defect audit of the legacy `gizzard` tool that produced
semantically-compressed YAML "pico kernels." Gizzard is being deconstructed and reimplemented
cleanly in **paling**; this catalog is the exhaustive record of what is being thrown away, so
nothing defective gets carried forward.

**Audited files**
- `/Users/jane/work/archaea/wonder/tools/gizzard/src/gizzard/processor.py` (1316 lines)
- `/Users/jane/work/archaea/wonder/tools/gizzard/src/gizzard/cli.py` (146 lines)
- supporting config: `config/gizzard.yaml`, `config/schema.yaml`

**Provenance:** written by an earlier-generation model (Claude in Cursor, ~2024). It "worked"
for its demo purpose but is poorly organized and full of defects.

**Legend**
- Each entry: `file:line` — defect — *consequence* — `[CATEGORY]`
- Entries tagged **[NEW]** were NOT in the original 14-bug pass (which only covered the
  relationship-extraction path). Entries tagged **[ORIG]** correspond to one of the original 14.
- Categories: `VOCAB` (fabricated/hardcoded vocabulary), `DEAD` (dead/unreachable code/config),
  `LOGIC` (wrong condition / wrong variable / off-by-one / inverted guard),
  `CRASH` (unguarded division / index / key / exception-handler failure),
  `FLATUS` (junk/cruft in the supposedly-compressed output),
  `REGEX` (unanchored / substring-corrupting / greedy patterns),
  `PORT` (hardcoded paths / layout / vocabulary assumptions),
  `SERDE` (compression/serialization correctness),
  `THEATER` (fabricated "analysis" that computes nothing real).

---

## Summary

- **Total defects catalogued: 58**
- **New (beyond the original 14): 44**

The original 14 were scoped to the relationship-extraction path. This pass confirms all of them
and adds 44 more across vocabulary fabrication, dead code, broken entrypoints, config/schema
mismatch, "analysis theater," and serialization junk.

---

## The kill chain (why most of the "analysis" computes nothing)

The individual defects matter less than how they chain. The tool's elaborate "model
compatibility" output is computed end-to-end from data that is **structurally guaranteed to be
empty**, and nobody noticed because the output looks busy. The trace:

1. The path that actually runs is `process_kernel_data`. It writes `processed['relationships']`
   as **formatted strings** like `"source→t1,t2"` (`:715`, `:746`).
2. `analyze_model_compatibility` feeds those strings into `build_relationship_graph` (`:873`),
   which only ingests items where `isinstance(rel, dict)` (`:34`). Strings aren't dicts, so the
   graph is **always empty** → `density`/`avg_clustering`/`avg_degree` are always `0.0` (#31).
3. Every `_estimate_model_performance` score (`:934`) therefore reduces to just its context
   terms; the entire relationship half of the weighting is dead weight, pinned at zero. Branches
   like gpt-4's `overall_score > 0.8` (`:1071`) can essentially never fire.
4. `generate_model_profile` (`:948`) then prints per-model "Relationship Graph Analysis:
   density 0.000, clustering 0.000" every run, decorated with **hardcoded** context windows
   (`:957–998`) and invented per-model personality prose (`:955–1010`).
5. The whole `model_analysis` block is attached to the kernel (`:599`) and then **silently
   dropped** by `write_output` (`:790`) — so it's computed from zeros and thrown away.

A parallel dead path: `analyze_framework_statistics` reads `self.all_relationships` (`:1182`),
which is only populated by `process_file` (`:438`) — a function the kernel pipeline never calls.
So those statistics are empty too, and even if populated they'd be bare strings failing the same
`isinstance(rel, dict)` check (#24–26).

Net: two independent, structurally-empty data sources feed an elaborate analysis-and-reporting
layer. The signal that made gizzard *work* came from the corpus and the kernel idea — never from
this analysis. Everything below is the itemized version of that story.

---

## A. Fabricated / Hardcoded Vocabulary  `[VOCAB]` / `[PORT]`

Verified against the real corpus under `/Users/jane/work/archaea/wonder`: the actual sigil
layout is `sigil/core`, `sigil/metareal`, `sigil/primitive`, `sigil/skillsets`. The terms
`parareal`, `hyperreal`, `hyporeal` appear **ZERO** times anywhere in the real corpus. `metareal`,
`orthoreal`, `Rokolisk`, `Cinder` do exist; `orthoreal` exists as a term but there is no
`sigil/orthoreal` directory.

1. **processor.py:96** — preserve-regex hardcodes fabricated `parareal|hyperreal|hyporeal` (plus real `metareal|orthoreal|Rokolisk|Wonder|Cinder|sigil|kernel|ethic`). *Term-preservation metric counts terms that never occur; vocabulary baked into code.* `[VOCAB]` **[ORIG]**
2. **processor.py:216–218** — `sigil_dirs` lists `sigil/parareal`, `sigil/hyperreal`, `sigil/hyporeal` (and `sigil/orthoreal`) as filesystem paths that can never resolve; real layout is `core/metareal/primitive/skillsets`. *Default file discovery walks non-existent dirs and silently finds nothing; the real `core`, `primitive`, `skillsets` trees are never scanned.* `[VOCAB]` `[PORT]` **[ORIG]**
3. **processor.py:346–349** — `extract_relationships` "special_terms" set hardcodes `parareal`, `hyperreal`, `hyporeal`. *Proximity-relationship inference keys on terms that never appear.* `[VOCAB]` **[ORIG]**
4. **processor.py:456–460** — `clean_content` "special_terms" set again hardcodes `parareal`, `hyperreal`, `hyporeal`. *Preserve-during-cleaning logic gated on fake terms; third duplicated copy of the term list.* `[VOCAB]` **[ORIG]**
5. **processor.py:765–769** — `clean_term` does `replace('parareal','para')`, `replace('hyperreal','hyper')`, `replace('hyporeal','hypo')`. *No-op string replacements for terms that never occur; cruft in the abbreviation path.* `[VOCAB]` **[ORIG]**
6. **processor.py:988, 1005–1006** — model-profile prose hardcodes Wonder-specific claims ("understanding metareal concepts", "struggle with metareal ambiguity", "Less experienced with Wonder-specific concepts"). *Fabricated qualitative claims baked into output; not portable; not measured.* `[VOCAB]` `[THEATER]` **[NEW]**
7. **processor.py:649, 658** — kernel name hardcoded to `"cinder_picokernel"` and identity name to `"Cinder"` regardless of input kernel. *Every output is mislabeled "Cinder"; the configured kernel name/identity is ignored.* `[VOCAB]` `[PORT]` **[NEW]**
8. **The three term lists (lines 96, 346–349, 456–460) are duplicated, not shared** — divergent maintenance; line 96 has 11 terms, 456–460 adds 13 logic glyphs. *Any vocabulary change must be made in 3+ places; they already disagree.* `[VOCAB]` **[NEW]**

**Complete list of fabricated/unused vocabulary terms found, with every reference:**
- `parareal` — processor.py:96, :216, :347, :457, :767 (fabricated; 0 occurrences in corpus)
- `hyperreal` — processor.py:96, :217, :347, :457, :768 (fabricated; 0 occurrences in corpus)
- `hyporeal` — processor.py:96, :218 (path `sigil/hyporeal`), :347, :457, :769 (fabricated; 0 occurrences in corpus)
- `orthoreal` — processor.py:96, :215 (path `sigil/orthoreal`, no such dir), :347, :457, :766 (term exists in corpus but no matching sigil directory)
- `Rokolisk` — processor.py:96, :348, :458 (real term, but never actually used in any logic path beyond term-matching theater)

---

## B. Broken / Dead Entrypoints & Config  `[DEAD]` / `[CRASH]`

9. **processor.py:1287–1288, 1309** — `main()` loads config from `control/gizzard-processing.yaml`, `control/gizzard-schema.yaml`, `control/preserve_terms.yaml`. **No `control/` directory exists**; the real files are `config/gizzard.yaml` and `config/schema.yaml`. *The entire `main()` / `argparse` entrypoint is dead — it always prints "Configuration file not found" and returns.* `[DEAD]` `[PORT]` **[NEW]**
10. **processor.py:1277–1317 vs cli.py** — two competing entrypoints (`argparse` `main()` and the typer `app`). Only the typer CLI is wired up (`pyproject` console-script); `main()` is unreachable dead code. *Dead, divergent second entrypoint.* `[DEAD]` **[NEW]**
11. **cli.py:138** — `validate` calls `GizzardProcessor(None, schema_path)`; the constructor immediately does `open(None, 'r')` → `TypeError`. *`gizzard validate` always crashes before doing anything.* `[CRASH]` **[ORIG]**
12. **cli.py:139** — `processor.validate_kernel(kernel_path)` is passed a `Path`, but `validate_kernel` expects an already-parsed dict and runs `jsonschema.validate(instance=<Path>, ...)`. *Even if the constructor didn't crash, validation validates a Path object, not the kernel contents.* `[LOGIC]` **[NEW]**
13. **cli.py:138** — `validate` never loads/parses the kernel YAML at all (no `yaml.safe_load`). *No actual validation logic exists.* `[LOGIC]` **[NEW]**
14. **processor.py:111–118 / cli.py:90** — config and schema are both loaded as `yaml.safe_load(f)['gizzard_processing']` and the *config* is jsonschema-validated against the *schema*. This couples the tool to one exact YAML shape; any reshaping of config breaks startup. *Startup is brittle and tied to a fixed config dialect.* `[SERDE]` **[NEW]**
15. **processor.py:114–115** — schema file is required to also be nested under a `gizzard_processing` key, an odd convention that buries the JSON-schema one level deep. *Confusing, non-standard schema packaging.* `[SERDE]` **[NEW]**
16. **cli.py:18–19** — `PACKAGE_DIR = Path(__file__).parent.parent.parent.parent`; `CONFIG_DIR = PACKAGE_DIR / "config"`. Four `.parent` hops from `src/gizzard/cli.py` lands on the gizzard root, but this is fragile to any packaging/install layout (e.g. site-packages). *Config resolution breaks when installed as a wheel.* `[PORT]` **[NEW]**
17. **cli.py:26** — `version("wonder")` looks up a distribution named `wonder`, but the package/tool is `gizzard`. *Version lookup always falls into the bare `except` and reports "development".* `[LOGIC]` **[NEW]**
18. **cli.py:28** — bare `except:` swallows all errors (including `KeyboardInterrupt`/`SystemExit`). *Hides real failures.* `[CRASH]` **[NEW]**

---

## C. Relationship Extraction Logic  `[LOGIC]`

19. **processor.py:332–333** — inverted guard: `if source != title and target != title: continue` skips exactly the relationships where the title participates, i.e. it drops the relationships it means to keep. *Almost all semantic relationships are discarded.* `[LOGIC]` **[ORIG]**
20. **processor.py:336–337** — after the inverted guard, the swap `if target == title: source, target = target, source` makes the source = title, producing self-referential / reversed edges. *Direction of surviving edges is wrong.* `[LOGIC]` **[ORIG]**
21. **processor.py:324–325** — noun-phrase "extraction" is a capitalized-word regex `\b[A-Z][a-z]+...`; it picks up any capitalized word (sentence-initial words, proper nouns) as a "concept." *Garbage sources/targets; not actual NLP.* `[LOGIC]` `[REGEX]` **[NEW]**
22. **processor.py:355–364** — proximity inference appends a relationship for every pair of special-term-containing words within 10 words, with `target = words[j]` (the raw word, possibly with punctuation). *Explosive, low-quality edges; targets carry trailing punctuation.* `[LOGIC]` **[NEW]**
23. **processor.py:356, 359** — `any(term.lower() in word.lower() ...)` is a substring test, so `kernel` matches "kernels", but also `ethic` matches "ethical", "aesthetic", etc. *False-positive term matches.* `[REGEX]` `[LOGIC]` **[NEW]**
24. **processor.py:438** — `self.all_relationships.extend(rel['target'] for rel in relationships)` stores bare target **strings**, but `analyze_framework_statistics` (line 1182) later does `if isinstance(rel, dict)`. *Relationship stats silently see only strings → all relationship metrics are empty/zero.* `[LOGIC]` `[SERDE]` **[ORIG]**
25. **processor.py:1182–1192** — because of #24, the `isinstance(rel, dict)` branch is never taken; `relationship_types`, `concept_connections`, `most_connected_concepts` are always empty. *Whole relationship-statistics block is dead at runtime.* `[DEAD]` `[THEATER]` **[ORIG]**
26. **processor.py:438** — `process_file()` populates `all_relationships`, but `process_file` itself is never called by the kernel pipeline (the pipeline uses `process_kernel_data`). So `all_relationships` is always empty even of strings. *`total_relationships` is always 0.* `[DEAD]` **[NEW]**
27. **processor.py:687–694** — relationship extraction here is "does any other sigil title appear as a substring of this content," a completely different (and cruder) mechanism than `extract_relationships`. *Two incompatible relationship models; the typed one (`extract_relationships`) is unused in the kernel path.* `[DEAD]` `[LOGIC]` **[ORIG]**
28. **processor.py:691** — `other_title in content_lower` substring test: a short title (e.g. "care") matches inside unrelated words. *Spurious relationships.* `[LOGIC]` `[REGEX]` **[NEW]**
29. **processor.py:707–723** — the per-file loop iterates the *entire accumulated* `relationship_map` (all sources so far) on every file and rebuilds `formatted_relationships` from scratch, attaching the whole global map to each content item. *O(n²) work; every content item's `relationships` field contains the global relationship list, not its own.* `[LOGIC]` `[FLATUS]` **[ORIG]**
30. **processor.py:719–723 vs 737–749** — `formatted_relationships` is computed twice (per-file and once at the end). The per-file copies (#29) are redundant duplicates of the final global list. *Massive duplication in output.* `[FLATUS]` **[NEW]**

---

## D. Graph / Metrics Math  `[CRASH]` / `[LOGIC]` / `[THEATER]`

31. **processor.py:873** — `build_relationship_graph(processed_kernel['relationships'])` is fed the **formatted strings** (`"source→a,b,c"`), not dicts. `build_relationship_graph` only handles dicts (line 34 `isinstance(rel, dict)`), so the graph is always empty. *All graph metrics (density, clustering, degree) are silently 0.0.* `[LOGIC]` `[THEATER]` **[ORIG]**
32. **processor.py:52–54** — density uses the **undirected** formula `2m / (n(n-1))` while the graph is built **directed** (only `graph[source].append(target)`, line 38). *Density is inconsistent with the graph model and can exceed 1.0.* `[LOGIC]` **[ORIG]**
33. **processor.py:39–40** — degree bookkeeping increments both `node_degrees[source]` and `node_degrees[target]` (undirected) while edges are stored directed. *`avg_degree` double-counts relative to the stored edge set; mixed directed/undirected accounting.* `[LOGIC]` **[ORIG]**
34. **processor.py:70** — clustering uses `max_possible = len(neighbors) * (len(neighbors) - 1)` (directed) but counts neighbor-edges via `n2 in self.graph.get(n1, [])` which only sees out-edges. *Clustering coefficient is systematically wrong/under-counted.* `[LOGIC]` **[ORIG]**
35. **processor.py:65–68** — clustering inner double-loop is O(neighbors²) and re-scans `self.graph.get(n1, [])` (a list, linear membership test) — O(degree³) overall. *Quadratic/cubic blow-up on dense nodes.* `[LOGIC]` **[NEW]**
36. **processor.py:855** — `print_token_stats` computes `total_reduction = (total_original - total_processed) / total_original` with no guard; if no files processed, `total_original == 0`. *ZeroDivisionError on empty corpus.* `[CRASH]` **[ORIG]**
37. **processor.py:425** — `reduction_ratio = (original_tokens - processed_tokens) / original_tokens` (in `process_file`) with no guard for `original_tokens == 0` (empty/title-only file). *ZeroDivisionError.* `[CRASH]` **[NEW]**
38. **processor.py:867–895** — `analyze_model_compatibility` loops over four models but recomputes `yaml.dump(processed_kernel)` and re-analyzes identical content each iteration; graph_metrics computed once but reused. *Wasteful, and the per-model "analysis" is identical input → the differentiation is fake (see #40).* `[THEATER]` **[NEW]**
39. **processor.py:897–946 / 948–1093** — the entire model-compatibility / profile subsystem (`_estimate_model_performance`, `generate_model_profile`, `_generate_recommendations`, hardcoded per-model weights and context windows) is invented scoring with no empirical basis, fed by metrics that are all 0.0 (#31). *"Analysis theater": elaborate output computed from zeros.* `[THEATER]` `[DEAD]` **[NEW]**
40. **processor.py:957, 971, 984, 998** — hardcoded context-window sizes (gpt-4 8192, gpt-3.5 4096, claude 100000, gemini 32768) and ModelContextAnalyzer's fixed 8000-token base (line 94). *Stale, wrong numbers baked into "compatibility" output; misleading.* `[THEATER]` **[NEW]**
41. **processor.py:599** — `processed['model_analysis'] = model_analysis` injects the entire fabricated model-analysis block into the kernel dict, but `write_output` (line 790) never copies `model_analysis` into the final YAML. *Computed, attached, then dropped — pure wasted work; or, if surfaced, it's flatus.* `[DEAD]` `[FLATUS]` **[NEW]**

---

## E. Content Reduction / Regex Corruption  `[REGEX]` / `[LOGIC]`

42. **processor.py:251** — `re.sub(r'the|a|an', '', text, ...)` is unanchored: the bare `a` matches inside *every word containing "a"*, and `the`/`an` match substrings. *Words are gutted ("data"→"dt", "name"→"nme"); catastrophic corruption.* `[REGEX]` **[ORIG]**
43. **processor.py:254** — `re.sub(r'is|are|was|were', '', ...)` unanchored: `is` matches inside "this", "basis", "vision"; `are` inside "share", "aware". *Pervasive mid-word corruption.* `[REGEX]` **[NEW]**
44. **processor.py:275** — `re.sub(r'ing$|ed$|s$', '', word)` strips a trailing `s` from any plural/possessive and from words legitimately ending in s. *"axis"→"axi", "process"→"proces"? (only one s removed), "ethics"→"ethic" — semantic terms mangled.* `[REGEX]` **[ORIG]**
45. **processor.py:520** — `re.sub(r'(ing|ed|ly|ment|ness|tion|sion)$', '', word)` naive suffix-stripping: "tion"→ strips from "relation"→"rela", "mention"→"men". *Destroys meaning of domain words.* `[REGEX]` **[NEW]**
46. **processor.py:126–132** — `self.patterns` dict is built in `__init__` (mapping regexes to symbols) but is **never used anywhere**. *Dead field; the real reduction uses inline regexes (#42–44).* `[DEAD]` **[NEW]**
47. **processor.py:515–517** — "skip redundant pairs": if a word is in {completely, totally, ...} it sets `skip_next=True`, deleting the *following* word regardless of what it is. *Arbitrary deletion of meaningful words after an intensifier.* `[LOGIC]` **[NEW]**
48. **processor.py:527–544** — long-sentence "summarization" keeps first 3 + special terms + last 3 words joined with no grammar. *Produces incoherent fragments presented as compressed content.* `[FLATUS]` **[NEW]**
49. **processor.py:476–477, 552** — sentences with `< 3` words are dropped entirely, and final join drops any cleaned sentence with `< 3` words. *Short but meaningful statements (definitions, axioms) silently discarded — lossy compression that loses signal.* `[LOGIC]` **[NEW]**
50. **processor.py:480, 492, 531** — "preserve special terms" uses substring containment (`term.lower() in word.lower()`), so "ethic" preserves "aesthetic"/"ethical" and "kernel" matches inside other tokens. *Wrong words preserved; the preserve list misfires.* `[REGEX]` **[NEW]**
51. **processor.py:241–246** — `identify_category` returns the prefix of the first category whose **name** appears in the content string (here the file path). Order depends on dict iteration; returns `None` if no category name is in the path. *In `process_file` (line 393), a `None` category drops the whole document.* `[LOGIC]` **[ORIG]**

---

## F. Crash / Robustness  `[CRASH]`

52. **processor.py:1121** — in the `get_git_metadata` except handler, `datetime.utcnow()` is referenced but `datetime` was imported *inside the try block* (line 1099). If the failure occurs before that import (or in a fresh call), `datetime` is undefined → `NameError` inside the error handler. *Exception handler itself crashes.* `[CRASH]` **[ORIG]**
53. **processor.py:583** — `kernel_name = next(iter(kernel))` assumes the kernel YAML is a non-empty dict whose first key is the kernel name; on an empty file or a list-typed YAML this raises `StopIteration`/`TypeError`. *Crash on malformed kernel; the broad `except` at 620 masks it as a generic message.* `[CRASH]` **[NEW]**
54. **processor.py:1012** — `chars = model_characteristics[model]`; if `models` (line 877) ever includes a model not in the dict, `KeyError`. *Brittle coupling between two hardcoded model lists.* `[CRASH]` **[NEW]**
55. **processor.py:620–622, 733–735, 642–644** — broad `except Exception` blocks that print and `continue`/return False, swallowing real bugs (e.g. the StopIteration above, file errors). *Failures look like "no files processed" instead of surfacing.* `[CRASH]` **[NEW]**
56. **processor.py:661** — `kernel_data.get("actions", "").split(".")` assumes `actions` is a string; the schema (validate_kernel_schema line 1266) says string, but if a list is provided this raises `AttributeError`. *Crash on alternate-shaped input; also produces empty-string actions filtered later.* `[CRASH]` **[NEW]**

---

## G. Output-Quality / Serialization "Flatus"  `[FLATUS]` / `[SERDE]`

57. **processor.py:380, 388, 394, 397, 401, 427, 672** — numerous `print(...)  # Debug` statements (file paths, titles, full processed content) dumped to stdout on every run. *Debug noise pollutes CLI output; the typer CLI wraps this in a status spinner so it interleaves badly.* `[FLATUS]` **[NEW]**
58. **processor.py:790–804, 815** — `write_output` embeds `git` metadata and a full `framework_statistics` block (file counts, token metrics, relationship metrics) into the "compressed" kernel YAML — most of which is zero/empty due to #24/#25/#31. *The "pico kernel" is bloated with empty/meaningless statistics and a celebratory `✨ Success!` line is printed mid-data.* `[FLATUS]` **[NEW]**

---

## H. Config / Schema Mismatch & Dead Config  `[DEAD]` / `[LOGIC]`

59. **processor.py:1152** — category detection in `analyze_framework_statistics` hardcodes `['ethic', 'concept', 'axiom', 'process', 'primitive']`, but the configured categories (config/gizzard.yaml:11–15) are `ethic, concept, axiom, process, **system**`. `primitive` is not a category; `system` is missing. *Files in `system` are never counted; `primitive` matches nothing the rest of the code knows about.* `[LOGIC]` `[VOCAB]` **[NEW]**
60. **config/gizzard.yaml** — many keys are required by schema but never read by code: `core_principles.*`, `pattern_recognition.*` (incl. `cross_reference_symbol`), `special_cases.*` (incl. `optional_brackets`), `usage_guidelines.*`, `processing_order`, `example_transformations`, `relationship_notation.concept_separator`, `content_reduction.convert_sentences`, `content_reduction.hierarchy_prefix`. *Large dead-config surface; the schema enforces structure for fields the processor ignores.* `[DEAD]` **[NEW]**
61. **config/schema.yaml:117–126** — `processing_order` (an array) declares `required: [...]`; `required` is only meaningful for objects in JSON Schema, so it is silently ignored. *Schema gives false assurance it validates processing-order contents.* `[SERDE]` **[NEW]**
62. **processor.py:124** — `self.categories` is read but `identify_category` is only used by the dead `process_file` path (#26); the live `process_kernel_data` path never calls `identify_category`. *Category mapping is effectively dead in the real pipeline.* `[DEAD]` **[NEW]**

---

## I. Miscellaneous  `[DEAD]` / `[LOGIC]`

63. **processor.py:135, 235–239, 1144** — `preserve_terms` is loaded only by the dead `main()` path (line 1311); the typer CLI never calls `load_preserve_terms`, so `preserve_terms` is always the empty set and `reduce_content`'s preserve check (line 271) never fires. *Preserve-terms feature is entirely inert in the live pipeline.* `[DEAD]` **[NEW]**
64. **processor.py:85–106** — `ModelContextAnalyzer.relationship_count`/`complexity_score` are initialized to 0 and never updated, yet emitted as `relationship_density` and `complexity_score`. *Always-zero fields presented as analysis.* `[THEATER]` **[NEW]**
65. **processor.py:223–227** — `resolve_path` treats any path starting with `/` as absolute and otherwise joins WONDER_ROOT, but kernel paths from the CLI are already absolute (`resolve_path=True` in cli.py:54). *Inconsistent path handling between CLI and internal use; Windows paths unhandled.* `[PORT]` **[NEW]**
66. **processor.py:638, 683** — title regex `^#\s+(.+)$` with `re.MULTILINE` + `re.match` only matches if the file *starts* with `# `; front-matter or leading blank lines defeat it, silently falling back to filename. *Inconsistent titles; relationship matching (which keys on titles) degrades.* `[LOGIC]` **[NEW]**
67. **processor.py:759–774** — `clean_term` order-dependent replacements: `' of '`→`'/'` runs after the `ethic of ` prefix rule, and ` and `/` the ` collapse semantics. Applying `.replace(' the ', '/')` mangles any term containing " the ". *Unpredictable term normalization.* `[LOGIC]` **[NEW]**
68. **processor.py:801, 661** — `actions` produced by `split(".")` then filtered `if action` (line 801); a description with no periods becomes one giant "action," and trailing empty splits create empties. *Action list is a poor split of a prose string, not real actions.* `[LOGIC]` `[FLATUS]` **[NEW]**

---

## Reconciliation with the original 14

The original (relationship-path) pass is fully represented here:
density/clustering math (#32–34), directed-vs-undirected (#32–33), inverted edge guard (#19–20),
unanchored/word-corrupting regexes (#42, #44), `identify_category` returning None and dropping
docs (#51), relationship stats getting bare strings (#24–25), graph built from formatted strings
→ 0.0 metrics (#31), O(n²) per-doc relationship rebuild (#29), divide-by-zero on empty corpus
(#36), NameError-prone except handler (#52), broken `cli.py validate` (#11), hardcoded Wonder
vocabulary (#1–5). Everything tagged **[NEW]** is beyond that original scope.
