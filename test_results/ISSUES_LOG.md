# Issues found so far — GLM-X testing log

Compiled: 2026-09-21 · Branch `Bhargav` (local Batch-3 merge `6524bb1`, not pushed)
Scope: tester-a (nature/weather), tester-b (food/bio small + medium), tester-c (health/science),
       tester-b-real-graph (kg_builder-built graph), decoder/walker gates, tooling.

Severity legend: BLOCKER / REGRESSION / MISMATCH / FAIL(bug) / COSMETIC / ENV-WORKFLOW

---

## Issue index

| id | severity | title | status |
|----|----------|-------|--------|
| IS-01 | REGRESSION | N1 gate expects 32 relation_phrases, Batch-3 produces 33 (`has_part`) | open, needs lead |
| IS-02 | FAIL(bug) | fbs14 strict golden mismatch (contextful honest answer) | open, needs lead |
| IS-03 | FAIL(bug) | fbm19 strict golden mismatch (contextful honest answer) | open, needs lead |
| IS-04 | MISMATCH | tester-a results header vs committed DB (12/5 vs 14/4) | open, reconcile |
| IS-05 | MISMATCH | tester-a dataset JSON is a 55-triple superset of its 14-edge DB | open, reconcile |
| IS-06 | COSMETIC | decoder inverts phrasing on reversed/mirrored walks | open, cosmetic |
| IS-07 | ENV-WORKFLOW | spacy + en_core_web_sm missing from python envs (installed into Py3.14) | resolved |
| IS-08 | ENV-WORKFLOW | KGBuilder sqlite rebuild onto stale DB -> UNIQUE nodes.label collision | fixed |
| IS-09 | ISSUE-FOUND | spaCy NER collapses whole sentences to one span (seasons, NORP) -> triples dropped | fixed |
| IS-10 | ISSUE-FOUND | "is part of X" parsed as one object chunk -> part_of lost (became is_a) | fixed |
| IS-11 | ISSUE-FOUND | "X and Y" coordinated NP merges into one chunk (example_of targets) | worked around |
| IS-12 | WORKFLOW | tester-a/b/c results + Batch-3 merge uncommitted / not pushed | open |
| IS-13 | INFO | no graph/chart/visualization produced by the testers (data only) | info |

Resolved earlier:
| id | severity | title | resolution |
|----|----------|-------|-----------|
| IS-14 | BLOCKER | tester-b-001 (relation-name mismatch, P-blocker) | fixed by Batch-3 `6524bb1` |
| IS-15 | FIXED | heuristic_fallback never True when graph lacks the real answer | grounding gate added in `glmx_ask.py` (below) |

---

## Detail

### IS-01 · N1 gate regression (relation_phrases 32 vs 33) — REGRESSION
- `decoder/tests/test_all.py:1090` expects **32** relation phrases; the Batch-3 merge added
  `has_part`, the model produces **33**.
- Impact: any N1 gate run on the merged Batch-3 code fails before testing starts.
- Needs: lead must sync the expected count in the N1 test (or drop `has_part`), and re-run N1.

### IS-02 · fbs14 strict-criteria mismatch — FAIL(bug)
- food_bio_small Q "Which food is rich in iron?" → walk/relation correct, but the honest-by-relation
  answer is the contextful form "iron-rich foods"; strict golden (`iron`) rejects it.
- File: `test_results/tester-b/food_bio_small_fbs14.json`.
- Needs: lead decision — relax strictness to relation/object semantic match, or accept context suffix.

### IS-03 · fbm19 strict-criteria mismatch — FAIL(bug)
- Same family as IS-02 on food_bio_medium (correct relation/walk, strict golden mismatch).
- File: `test_results/tester-b/food_bio_medium_fbm19.json` (medium results).
- Needs: lead decision, same options as IS-02.

### IS-04 · tester-a results header vs committed DB — MISMATCH
- `test_results/tester-a/nature_weather_small_results.json` header: `edges: 12`, relations include
  `caused_by` (5 relations). Committed `datasets/nature_weather_small.db` actually holds
  **14 edges / 4 relations** (`associated_with, causes, follows, precedes`), no `caused_by`.
- The DB is authoritative; the run summary header is stale. Reconcile or regenerate.

### IS-05 · tester-a dataset JSON superset of DB — MISMATCH
- `test_results/tester-a/datasets/nature_weather_small.json` lists **55 triples**; the built DB only has
  14 edges. The JSON looks like the intended-but-not-fully-inserted source set.

### IS-06 · decoder phrase inversion on reversed walks — COSMETIC
- trg06 "What precedes summer?" → correct path `summer -> spring`, chain `['precedes']`, but decoded
  text is "summer follows spring."; trg11 part_of "the circulatory system is part of the heart.";
  trg10 "fish is an example of salmon." Correct gold/path, wrong surface phrasing. Decoder renders the
  forward phrase of the edge label regardless of traversal direction.

### IS-07 · missing spaCy — ENV-WORKFLOW → resolved (fixed 2026-09-24, branch dharani)
- kg_builder pipeline needs `spacy` + `en_core_web_sm`; no repo python env had it.
  Installed into Python 3.14 (env-only, nothing committed). Reruns need the same env.
- Fix: spaCy import made lazy in `document_processor.py` (clear pip error if the package is
  missing; the auto-download of `en_core_web_sm` on OSError is kept), and the dead eager
  `import spacy` in `triple_extractor.py` removed. `import kg_builder` no longer hard-fails
  when spaCy is absent.

### IS-08 · KGBuilder sqlite rebuild collision — ENV-WORKFLOW → fixed (fixed 2026-09-24, branch dharani)
- `build_graph_store(store_type="sqlite", db_path=...)` constructs `SQLiteGraphStore(db_path)` which
  auto-loads the existing file, then `add_dataset`/`save_state` merge onto the old rows →
  `sqlite3.IntegrityError: UNIQUE constraint failed: nodes.label` when rebuilding.
- Workaround in `build_real_graph.py`: delete stale `*.db*` files before building.
- Fix: `build_graph_store` now opens `SQLiteGraphStore(db_path, load=False)`; stale rows are never
  merged into memory and `save_state` rewrites all tables. No file deletion required, so the
  Windows file-lock (previous store's connection still open) is not an issue. Verified: building
  the same path twice in one process → 74 nodes / 47 edges both times, no IntegrityError.

### IS-09 · spaCy NER collapses sentences to one span — ISSUE-FOUND → fixed (fixed 2026-09-24, branch dharani)
- "Spring follows winter." → whole sentence tagged a single DATE entity (one span) → extractor drops it.
  "Bonjour translates to hello." → `Bonjour` tagged NORP, only one span → dropped.
- Workaround: non-seasonal pairs / words spaCy doesn't tag (verified via extractor probe).
- Fix (`triple_extractor._extract_spans`): skip NER spans that cover the entire sentence
  (the DATE/NORP collapse; a full-sentence span previously subsumed everything else via dedup),
  and rescue bare `pobj` tokens the noun-chunker misses (`hello` is INTJ).
  "Spring follows winter." → (Spring, follows, winter); "Bonjour translates to hello." → (Bonjour, translates to, hello).

### IS-10 · "is part of X" parsed as one object chunk — ISSUE-FOUND → fixed (fixed 2026-09-24, branch dharani)
- spaCy parses "Heart is part of the circulatory system." with object NP "part of the circulatory
  system" → edge came out `(heart, is_a, part of the circulatory system)`; no `part_of`.
- Workaround: phrase as "X consists of Y" → clean `(x, part_of, y)` edges. (Real-graph corpus.)
- Fix (`triple_extractor`): when the connector is a bare copula and the object chunk starts with a
  partitive phrase, the object is stripped to its inner noun and the raw connector is relabelled to
  the REL_MAP key — "is part of" → `part_of`, "is a type of"/"is an example of" → `example_of`.
  "Heart is part of the circulatory system." → (Heart, is part of, the circulatory system).

### IS-11 · coordinated NP "A and B" merges into one chunk — ISSUE-FOUND
- "Fish include salmon and tuna." → object span "salmon and tuna" (single chunk) → messy `example_of`
  target. Workaround: one instance per sentence ("Fish include tuna.").

### IS-12 · uncommitted / not pushed — WORKFLOW
- Local merge `6524bb1` (Batch-3) not pushed to `origin/Bhargav`.
- Uncommitted: tester-a 3 files (DAILY_STATUS, golden, results.json), tester-b files
  (BUG_tester-b-001.md, DAILY_STATUS.md, baseline_2026-09-18.txt, results txt, 15 small jsons),
  `?? test_results/tester-c/` (untracked), `?? test_results/tester-b-real-graph/` (untracked).

### IS-13 · testers produce data, not charts — INFO
- No matplotlib/plotly charting in any tester/lead module; only `model_training/training/plotting.py`
  (training curves) exists. testers output graph DBs + results JSON/TXT; lead recomputes from `.db`.

### IS-14 · tester-b-001 (resolved) — BLOCKER → fixed
- P-blocker relation-name mismatch previously filed; fixed by Batch-3 `6524bb1`.

### IS-15 · `heuristic_fallback_used` never True when the graph lacks the real answer — FIXED (2026-09-21)
- Symptom: a question whose relation chain exists in the graph but the anchored node cannot complete
  it still produced a confident (possibly wrong) answer with `heuristic_used=False`;
  extractor-fallback plans also walked confidently instead of being honest.
- Root cause: `heuristic_fallback_used` was set only when the question text could not be mapped to a
  relation (`g2p_planner.py:177`); it never checked graph answer availability. The old honesty gate
  (`honest_by_relation`) only inspected hop-1 relation presence and skipped entirely for heuristic plans.
- Fix (`scripts/glmx_ask.py`): new `_answer_grounded(anchor_id, chain)` — BFS over ACTUAL graph
  adjacency consuming the whole relation chain in order (mirror relations via `INVERSE_REL_LABEL`
  allowed per hop, like the walker's reverse edges). The Step-4 honesty gate now runs for every plan
  (heuristic or not):
  - chain grounds   -> `answer_grounded=True`, pipeline answers normally
  - chain fails     -> `honest_by_relation=True`, `heuristic_fallback_used=True`
                       (via `dataclasses.replace`), honest "no relation" answer emitted
- New output fields: `answer_grounded`, `answer_grounded_depth` (also in the entity-not-found branch).
- Regression check: tester-b-real-graph 18/18, tester-c 12/12, tester-b small 14/15 (fbs14),
  medium 19/20 (fbm19) — all unchanged. Probes: `gold`/`has_property`, `fish`/`example_of`,
  `the heart`/`part_of` grounded True; `salmon`/`causes`, `night`/`synonym` grounded False.

---

## Current scorecards (reference)

- tester-b small: 14/15 (IS-02) · medium: 19/20 (IS-03)
- tester-a rerun: 5/10 (relation availability in old graph)
- tester-c small: 12/12
- tester-b-real-graph (kg_builder-built, all 16 relations + mirrors): 18/18

## Open action items

1. Lead: align N1 expected count (IS-01).
2. Lead: decide strict-criteria policy for contextful honest answers (IS-02/03).
3. Tester/lead: reconcile tester-a header + dataset JSON vs DB (IS-04/05).
4. Lead: confirm cosmetic decoder phrasing acceptable (IS-06).
5. Tester: decide whether to commit/push pending results + Batch-3 merge (IS-12).