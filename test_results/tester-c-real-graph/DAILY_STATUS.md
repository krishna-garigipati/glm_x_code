# DAILY STATUS — tester-c-real-graph (2026-09-25)

## UPDATE (2026-09-25, IS-11 conjunction expansion lock, branch dharani)
- The corpus gained two IS-11 coordination sentences (47 sentences total):
  "Fish include salmon and tuna." / "Salmon and tuna are fish."
- Rebuilt the real graph: **72 nodes / 49 edges** pre-store (70 stored after embedding merge).
  New per-conjunct edges (edges_dump.tsv): `fish --example_of--> salmon`,
  `fish --example_of--> tuna`, `salmon --is_a--> fish`, `tuna --is_a--> fish`.
  unmapped raw relations still = [].
- Pre-registered 3 new goldens (trc22 "What is a salmon?" -> fish; trc23 "What is a tuna?" -> fish;
  trc24 "What is an example of a fish?" -> salmon) BEFORE running the evaluator.
  Result: **24/24 PASS**, all `heuristic_used=False`, `template_matched=True`.
- Gate evidence: tester-b-real-graph rebuilt+rerun 18/18 PASS (corpus unchanged, graph byte-same);
  corpus_500 old-vs-new diff gate 0 diffs on non-coordinated sentences; no duplicate edges.
- Stale counts in this file below were from the original 45-sentence run; see golden.md + build_stats.json
  for the current 47-sentence/49-edge rebuild.

## What this run did
- Built tester-c's real graph with the **full fixed kg_builder pipeline** from a single
  plain-text corpus: `datasets/health_science_source.txt` (Health & Science).
- Full pipeline: spaCy `en_core_web_sm` (DocumentProcessor) -> rule-based TripleExtractor
  -> zero-heuristic RelationMapper -> EntityResolver -> GraphBuilder -> SQLiteGraphStore.
- Raw connector labels normalised to the 16 canonical relations via `REL_MAP`
  (identity entries for canonical names; unmapped raw relations = []).
- The corpus deliberately includes the exact phrasings that IS-09 / IS-10 target, making
  this build a regression gate for those fixes (see golden table + edge gate below).

## Graph (built by kg_builder, not curated)
- 45 sentences -> 45 edges (1 edge per sentence; **nothing dropped**) -> 69 nodes in
  graph_data. SQLite write applies the store's add_dataset embedding merge
  (cos ~ 1.0 singular/plural collapse, e.g. "the lungs" -> "the lung") -> **67 stored
  nodes / 45 edges** in `tester_c_real_graph.db`. `edges_dump.tsv` is written from the
  pre-store graph_data (shows pre-merge labels, e.g. "the lungs").
- All 16 canonical relations present:
  is_a 4, has_property 4, causes 4, caused_by 4, example_of 4, part_of 3, synonym 3,
  linguistic_maps 3, follows 2, precedes 2, contradicts 2, associated_with 2,
  temporal_coincident 2, spatial_near 2, antonym 1.

## Evaluation (pre-registered goldens)
- `tester_c_real_graph_golden.md` frozen before execution: 21 goldens = 16 canonical
  relations direct/mirror + 5 dedicated IS-09/IS-10 probe rows (trc05/06/11/16/19/20/21).
- Runner: `tester_c_real_graph_runner.py` (mirror of tester-b-real-graph: hops + object + chain).
- Result: **21/21 PASS** — `heuristic_used=False` and `template_matched=True` for every
  question → all answers came from the real graph, no fallbacks, no breakdowns.
  See `tester_c_real_graph_results.txt` and `tester_c_real_graph_<id>.json` per question.
- Mirrors verified on a KB-built graph: trc17/trc18 (expected `causes`, walked over
  `caused_by`); trc03 (expected `causes`, walked `caused_by`) — same class as tester-b trg17/18.

## Bug-fix confirmation (the point of this run)
| Bug | Pre-fix behaviour | Now (this run) |
|-----|-------------------|----------------|
| IS-09 NER whole-sentence collapse | "Spring follows winter." tagged as one DATE span -> sentence DROPPED, 0 triples | Edge `spring --follows--> winter` extracted (trc05 PASS) + `autumn --precedes--> winter` (trc06 PASS) |
| IS-09 bare-pobj rescue | "Bonjour translates to hello." ("hello"=INTJ/pobj, no chunk) -> 0 triples | `bonjour --linguistic_maps--> hello` (trc16 PASS) + `ciao`->`goodbye` (trc19 PASS) |
| IS-10 partitive object | "Heart is part of the circulatory system." -> junk object, edge came out is_a | `the heart --part_of--> the circulatory system` (trc11 PASS) + stomach (trc20) + lung (trc21) |
| IS-10 "is a type of" | "Insulin is a type of hormone." -> connector "is"/junk object | `insulin --example_of--> hormone`, `vitamin d --example_of--> a nutrient` (edges gate OK; trc10 PASS via `hormones --example_of--> insulin`) |
| Whole pipeline integrity | — | 45 sentences -> 45 edges (1:1, zero drops), 16/16 relations, unmapped = [],

Confirmed fixed and no regressions: the exact sentences that previously produced ZERO
triples now produce correctly-mapped canonical edges and answer through glmx_ask.

## Artifacts (all under test_results/tester-c-real-graph/)
- `datasets/health_science_source.txt` (47-sentence corpus incl. IS-11 coord lock)
- `build_real_graph.py`, `tester_c_real_graph.db`, `build_stats.json`, `edges_dump.tsv`
- `tester_c_real_graph_golden.md` (pre-registered, frozen), `tester_c_real_graph_runner.py`
- `tester_c_real_graph_results.txt` (24/24) + 24 per-question JSONs
- this status note

## Not committed / pushed
- Entire `test_results/tester-c-real-graph/` directory is uncommitted (branch dharani).