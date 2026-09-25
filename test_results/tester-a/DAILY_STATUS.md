# Tester-A Daily Status

## 2026-09-25 (tester-a) — IS-04/05 reconcile on dharani
- Root cause: nature_weather_small.db (committed) was a stale manual 14-edge/4-relation subset of
  its own 53-triple source JSON — not reproducible, header/status counts disagreed everywhere.
- Fix: `datasets/build_nature_weather_small.py` (committed, mirrors tester-b harness pattern) now
  builds the DB deterministically from the JSON: 37 nodes, 53 edges, 9 canonical relations
  (antonym, associated_with, caused_by, causes, follows, has_property, is_a, precedes,
  temporal_coincident). B4/B7 + Section 5.4 band asserts OK.
- Re-ran the frozen 10 golden questions (`nature_weather_small_runner.py`) against the rebuilt DB:
  **7/10 (was 5/10)**; chain exact 7/10. q5 (antonym) now answers concretely.
- Remaining fails are dataset/planning facts, not regressions:
  - q2/q9 mirror-chain inversion (planned causes vs caused_by), object still reached on q9.
  - q7: `rain` has no `is_a` edge in the authored JSON → honest gate correct.
  - q8: the JSON has no `part_of` triples at all → unanswerable as authored; honest gate fires.
- No golden edits (questions untouched). Regenerated `nature_weather_small_results.json`,
  golden Run-Results table/summary, `lead/unified_golden_results.*`, `lead/domain_matrix.*`.

## 2026-09-21 (tester-a) — re-run of nature_weather_small on Bhargav (Batch-3 code)

### Datasets Run
- nature_weather_small.db (14 nodes, 12 edges; relations actually in graph:
  causes, follows, precedes, associated_with)

### Golden Pass Rate (re-run, grading: hops + expected-answer-node)
- **5/10 PASS** (q1, q3, q4, q6, q9). Chain exact: 6/10.
- Updated `nature_weather_small_results.json` + golden Run-Results table + summary.

### Gates (shared system, run by tester-b today, relevant for the gate clause)
- G1 221/102, G3 8/8, G4 OK; G2 has 1 regression (decoder 12.4 relation_phrases 32 vs 33,
  caused by Batch-3 `has_part` template — lead to sync `test_all.py:1090`).

### Notes
- Batch-3 exact-label anchoring: q2/q7 anchor `rain` and walk forward `causes` -> flood.
  Expected semantics (causes of rain = cloud/storm, is_a = water/weather) need `caused_by` /
  `is_a`, which are not edges in this graph -> object fail.
- Batch-3 relation-honesty gate now fires on q5/q8/q10 (anchor lacks the asked relation):
  honest "I don't have a relation ..." sentence replaces the old nearest-relation answers.
- Prior 2026-09-18 record: 10/10 under "chain within available relations + template matched"
  grading (no answer-semantics check). No golden edits; new grading documented in the golden file.

---

## 2026-09-18 (tester-a)

### Datasets Run
- nature_weather_small (14 nodes, 12 edges, 3 relations)

### Golden Pass Rate
- 10/10 (all template_matched=True, heuristic_used=False)

### Open Blockers
- None

### Open P-bugs
- None

### Gate Suite Today
- G1: 213 passed / 102 skipped
- G2: 195 passed / 2 skipped
- G3: 8/8 OK

### Changes Made / Fixed
- Created tester-a branch
- Completed onboarding: toy_eval 7/7 PASS, A/B 21/21 tobacco
- Created nature_weather_small.json (53 triples, 5 domains)
- Built nature_weather_small.db (14 nodes, 12 edges, 3 relations)
- Ran 10 golden questions: 10/10 PASS (template_matched=True, heuristic=False)
- Pre-registered golden questions in nature_weather_small_golden.md

### Notes / Needs Lead Decision
- Chain extraction limited by available graph relations (only 3 relations in this small graph)
- Edge case "missing relation" not yet tested
- Ready for lead review of dataset + goldens + results