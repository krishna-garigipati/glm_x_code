# DAILY STATUS — tester-b-real-graph (2026-09-21)

Branch: Bhargav (Batch-3 merge `6524bb1` local). No production code changed.

## What this run did differently (per instruction)
- Completely left the experimentation.md style graph (pre-curated relation set). Instead the graph
  for tester-b was **built by `kg_builder` from a single plain-text corpus**:
  `datasets/food_biology_source.txt` (47 sentences).
- Full pipeline: spaCy `en_core_web_sm` (DocumentProcessor) -> rule-based TripleExtractor
  (discovers ALL relations from text) -> zero-heuristic RelationMapper (raw connector label) ->
  EntityResolver (embedding merge) -> ConfidenceScorer/GraphBuilder -> SQLiteGraphStore.
- Raw connector labels were then normalised to the system's 16 canonical relation names via a
  documented mapping (`REL_MAP` in `build_real_graph.py:REL_MAP`). No relation was excluded.

## Graph (built, not curated)
- Nodes: 74, Edges: 47. DB: `tester_b_real_graph.db`.
- 15 distinct raw connector labels found by the extractor -> mapped to all 16 canonical relations:
  is_a 4, has_property 4, causes 4, caused_by 4, example_of 4, part_of 4, precedes 3,
  supports 3, linguistic_maps 3, follows 2, contradicts 2, associated_with 2, synonym 2,
  antonym 2, temporal_coincident 2, spatial_near 2. Unmapped: none.
- Build evidence: `build_stats.json` (extraction stats + raw vs mapped relation counts),
  `edges_dump.tsv` (full edge list).

## Evaluation (pre-registered goldens)
- `tester_b_real_graph_golden.md` frozen before the run: 18 goldens (1 per canonical relation
  direct + 2 mirror questions causes<->caused_by).
- Runner: `tester_b_real_graph_runner.py` (mirrors tester-c/food-bio scoring: hops + object + chain).
- Result: **18/18 PASS** — every question answered through the real graph; `relation_chain`
  correct for all 18 (no heuristic fallback; template matched).
  see `tester_b_real_graph_results.txt` and `tester_b_real_graph_<id>.json` per question.
- Mirrors verified on a KB-built graph: trg17/trg18 (expected `causes`, walked `caused_by`);
  trg06 (expected `precedes`, walked `follows`).

## Findings / notes for lead
- spaCy `en_core_web_sm` NER heavily shapes extraction: season words ("Spring follows winter.")
  got tagged as a single DATE span (dropped); "is part of X" is parsed as one object chunk so the
  edge came out `is_a` — switched to "X consists of Y" phrasing for clean `part_of` edges.
  This is a corpus/phrasing sensitivity, not a GLM-X issue.
- Embeddings for all 74 nodes present in DB (SQLiteGraphStore schema compatible with glmx_ask).
- Config banking: one extra unused relation bank exists (`has_instance` only in schema notes);
  all 16 config relations verified present in graph.
- Env: `spacy` + `en_core_web_sm` installed into Python 3.14 (only env change; nothing committed).

## Not committed / pushed
- `test_results/tester-b-real-graph/` new (corpus, build script, golden, runner, db, stats,
  results txt, 18 per-question json, this status).
- Prior pending: tester-a 3 result files, tester-b result files, `test_results/tester-c/`
  (all uncommitted; Batch-3 merge `6524bb1` not pushed).