# tester-c: Health & Science - Daily Status

Canonical tester-c set (Section 13.2): `supports`, `contradicts`, `spatial_near`,
`linguistic_maps` + 2 from tester lists A/B (chosen: `causes` from A, `part_of` from B).

## 2026-09-21 — Small dataset (Level 1)

Dataset: `datasets/health_science_small.db` — 40 nodes, 23 edges, 6 relations
[causes, contradicts, linguistic_maps, part_of, spatial_near, supports].
Built via `datasets/build_health_science_small.py` (tester-b harness pattern;
add_dataset without embeddings, post-hoc embeddings, no label collisions).

Goldens: 12 pre-registered 2026-09-21 in `health_science_small_golden.md`
(2 per assigned relation; all hops=1; hsc09/hsc10 are cause-source questions
anchored on the effect with the walk across the caused_by mirror).

Run: `python test_results/tester-c/health_science_small_runner.py`

Result: **12/12 PASS** (hops + object + chain all correct, every question)

| id   | q                              | chain            | path (answer)        | conf   |
|------|--------------------------------|------------------|----------------------|--------|
| hsc01 | What supports health?          | supports         | health -> exercise   | .906   |
| hsc02 | What is supported by sunlight? | supports         | sunlight -> vitamin_d| .913   |
| hsc03 | What contradicts health?       | contradicts      | health -> smoking    | .920   |
| hsc04 | What does sugar contradict?    | contradicts      | sugar -> healthy_teeth | .883  |
| hsc05 | What is near the heart?        | spatial_near     | heart -> lungs       | .903   |
| hsc06 | What is located near the liver?| spatial_near     | liver -> stomach     | .896   |
| hsc07 | What is the term for cure?     | linguistic_maps  | cure -> treat        | .865   |
| hsc08 | What do you call a doctor?     | linguistic_maps  | doctor -> physician  | .877   |
| hsc09 | What leads to fever?           | causes           | fever -> virus       | .866   |
| hsc10 | What causes insomnia?          | causes           | insomnia -> stress   | .862   |
| hsc11 | What is the heart a part of?   | part_of          | heart -> circulatory_system | .891 |
| hsc12 | What is the stomach a part of? | part_of          | stomach -> digestive_system | .885 |

Notes
- All 6 assigned relations answered correctly by the batch-3 pipeline (walker
  exact-rel preference + decoder phrases: "is near", "relates to", "is part of").
- hsc09/hsc10 walk the `caused_by` mirror edge but keep plan chain `[causes]` and
  answer "fever is caused by virus." (matches fbs13 pattern).
- hsc01 anchor `health` has both a supports and a contradicts neighbor; walker
  exactly preferred `supports` -> exercise. No cross-relation leakage.
- No honest-fallback goldens here (synonym absent in graph; a "what is another
  word for health?" probe would hit the fbs14-class strict-criteria mismatch,
  already reported for tester-b).

Artifacts: health_science_small_results.txt, health_science_small_<hscNN>.json (12),
health_science_small_golden.md, datasets/health_science_small.db,
health_science_small_runner.py.

Medium (100-1,000 nodes) not yet built -> pending user go-ahead.