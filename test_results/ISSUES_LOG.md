# Issues found so far — GLM-X testing log

Compiled: 2026-09-21 · Branch `Bhargav` (local Batch-3 merge `6524bb1`, not pushed)
Scope: tester-a (nature/weather), tester-b (food/bio small + medium), tester-c (health/science),
       tester-b-real-graph (kg_builder-built graph), decoder/walker gates, tooling.

Severity legend: BLOCKER / REGRESSION / MISMATCH / FAIL(bug) / COSMETIC / ENV-WORKFLOW

---

## Issue index

| id | severity | title | status |
|----|----------|-------|--------|
| IS-01 | REGRESSION | N1 gate expects 32 relation_phrases, Batch-3 produces 33 (`has_part`) | resolved 2026-09-25 (lead accepted 33) |
| IS-02 | FAIL(bug) | fbs14 strict golden mismatch (contextful honest answer) | resolved 2026-09-21 (runner prefix-match) |
| IS-03 | FAIL(bug) | fbm19 strict golden mismatch (contextful honest answer) | resolved 2026-09-21 (runner prefix-match) |
| IS-04 | MISMATCH | tester-a results header vs committed DB (12/5 vs 14/4) | resolved 2026-09-25 (rebuilt 37/53 golden+db from dharani) |
| IS-05 | MISMATCH | tester-a dataset JSON is a 55-triple superset of its 14-edge DB | resolved 2026-09-25 (rebuilt 37/53 golden+db from dharani) |
| IS-06 | COSMETIC | decoder inverts phrasing on reversed/mirrored walks | open, cosmetic |
| IS-07 | ENV-WORKFLOW | spacy + en_core_web_sm missing from python envs (installed into Py3.14) | resolved 2026-09-25 (lazy import + clear guidance from dharani) |
| IS-08 | ENV-WORKFLOW | KGBuilder sqlite rebuild onto stale DB -> UNIQUE nodes.label collision | resolved 2026-09-25 (load= param from dharani) |
| IS-09 | ISSUE-FOUND | spaCy NER collapses whole sentences to one span (seasons, NORP) -> triples dropped | resolved 2026-09-25 (span-collapse fix from dharani) |
| IS-10 | ISSUE-FOUND | "is part of X" parsed as one object chunk -> part_of lost (became is_a) | resolved 2026-09-25 (pobj rescue from dharani) |
| IS-11 | ISSUE-FOUND | "X and Y" coordinated NP merges into one chunk (example_of targets) | resolved 2026-09-25 (coordination split from dharani) |
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

### IS-02 · fbs14 strict-criteria mismatch — RESOLVED
- food_bio_small "What follows photosynthesis?" (dead-end question): walk/relation correct and honest,
  but the exact-string golden check failed because `render_no_relation` appends context
  ("Closest concepts I have: ..." + "(No <rel> relation found.)").
  Code: `test_results/tester-b/food_bio_runner.py` (missing_relation check).
- Fix: the `missing_relation` edge-case check now uses `.strip().startswith(FALLBACK_SENTENCE)` while
  keeping `heuristic_used` + empty-walk assertions. Strict-chain/object checks for real answers untouched.
- Result: tester-b small 14/15 -> **15/15**.

### IS-03 · fbm19 strict-criteria mismatch — RESOLVED
- Same exact-sentence strictness on food_bio_medium "What organism lives close to plankton?"
  (`food_bio_medium_runner.py`, missing_relation check); same prefix-match fix.
- Result: tester-b medium 19/20 -> **20/20**.

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

### IS-07 · missing spaCy — ENV-WORKFLOW
- kg_builder pipeline needs `spacy` + `en_core_web_sm`; no repo python env had it.
  Installed into Python 3.14 (env-only, nothing committed). Reruns need the same env.

### IS-08 · KGBuilder sqlite rebuild collision — ENV-WORKFLOW
- `build_graph_store(store_type="sqlite", db_path=...)` constructs `SQLiteGraphStore(db_path)` which
  auto-loads the existing file, then `add_dataset`/`save_state` merge onto the old rows →
  `sqlite3.IntegrityError: UNIQUE constraint failed: nodes.label` when rebuilding.
- Workaround in `build_real_graph.py`: delete stale `*.db*` files before building.

### IS-09 · spaCy NER collapses sentences to one span — ISSUE-FOUND
- "Spring follows winter." → whole sentence tagged a single DATE entity (one span) → extractor drops it.
  "Bonjour translates to hello." → `Bonjour` tagged NORP, only one span → dropped.
- Workaround: non-seasonal pairs / words spaCy doesn't tag (verified via extractor probe).

### IS-10 · "is part of X" parsed as one object chunk — ISSUE-FOUND
- spaCy parses "Heart is part of the circulatory system." with object NP "part of the circulatory
  system" → edge came out `(heart, is_a, part of the circulatory system)`; no `part_of`.
- Workaround: phrase as "X consists of Y" → clean `(x, part_of, y)` edges. (Real-graph corpus.)

### IS-11 · coordinated NP "A and B" merges into one chunk — ISSUE-FOUND
- "Fish include salmon and tuna." → object span "salmon and tuna" (single chunk) → messy `example_of`
  target. Workaround: one instance per sentence ("Fish include tuna.").

### IS-07/08/09/10/11 · kg_builder fixes merged from dharani (2026-09-25) — RESOLVED
- Merged `dharani@4bbeb76` practical fixes into `Bhargav` (cherry-picked hunks):
  - IS-07: `kg_builder/document_processor.py` — lazy spaCy import with clear pip-install guidance.
  - IS-08: `SQLiteGraphStore.__init__(db_path=None, load=True)` + `pipeline` passes `load=False` so
    rebuilds start from a fresh schema instead of merging onto stale rows.
  - IS-09: `kg_builder/triple_extractor.py` — whole-sentence span collapse fix (seasons/NORP).
  - IS-10: copular partitive object rescue (`part_of`) via pobj handling.
  - IS-11: coordinated NP split (`salmon and tuna` → separate example_of targets).
- Regression gate: new `test_results/tester-c-real-graph/` suite (IS-09/10 phrasing: "Spring follows
  winter.", "Bonjour translates to hello.", "The heart is part of...", "Insulin is a type of...").
- NOT merged: dharani's regenerated `tester-b-real-graph` results (they predate IS-06 canonicalization).

### IS-12 · uncommitted / not pushed — WORKFLOW
- Local merge `6524bb1` (Batch-3) not pushed to `origin/Bhargav`.
- Uncommitted: tester-a 3 files (DAILY_STATUS, golden, results.json), tester-b files
  (BUG_tester-b-001.md, DAILY_STATUS.md, baseline_2026-09-18.txt, results txt, 15 small jsons),
  `?? test_results/tester-c/` (untracked), `?? test_results/tester-b-real-graph/` (untracked).

### IS-13 · testers produce data, not charts — RESOLVED (2026-09-25)
- No matplotlib/plotly charting in any tester/lead module; only `model_training/training/plotting.py`
  (training curves) exists. testers output graph DBs + results JSON/TXT; lead recomputes from `.db`.
- Resolution: `test_results/plot_results.py` — shared scorecard generator that scans the per-question
  JSONs every runner writes (`_golden_id`, `_golden_chain`, `_golden_hops`, `_golden_gold`, `_pass`,
  `honest_no_relation`, …) and renders, per suite, a per-question PASS/FAIL/honest bar + per-relation
  accuracy, plus a combined overview.
- Generated artiffs: `<suite>_scorecard.png` in each tester dir + `test_results/scorecard_overview.png`.
- Usage: `python test_results/plot_results.py [--only real-graph] [--no-overview]`.
- Note: `tester-b-real-graph/render_graph.py` (matplotlib+networkx → PNG/SVG of the full KG) already
  covered graph visualization and remains as-is.

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

- tester-b small: 15/15 · medium: 20/20
- tester-a rerun: 5/10 (relation availability in old graph)
- tester-c small: 12/12
- tester-b-real-graph (kg_builder-built, all 16 relations + mirrors): 18/18

## Open action items

1. Lead: confirm cosmetic decoder phrasing acceptable (IS-06).
2. Tester: decide whether to commit/push pending results + Batch-3 merge (IS-12).
3. run tester-c-real-graph + tester-a on this merge to regen results/config ints (2026-09-25 merge).