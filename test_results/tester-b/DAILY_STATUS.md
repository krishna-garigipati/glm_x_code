# Tester-B Daily Status

## 2026-09-21 (tester-b) — re-run on Bhargav after merging lead Batch-3 (6524bb1)
Datasets run:  food_bio_small (51n/39e) + food_bio_medium (196n/181e) — full gate + golden re-run
Golden pass rate: small 14/15 (was 12/15); medium 19/20 (was frozen at fbm09)
Open blockers:  none — tester-b-001 RESOLVED by Batch-3 (see BUG_tester-b-001.md)
Open gate issue: N1 — G2 regression (test 12.4) for lead, see below
Gate suite (re-run after code change, Section 6):
  - G1 pytest g2p/walker/decoder + pipeline relations : 221 passed, 102 skipped (baseline 213/102) — green
  - G2 decoder.tests.test_all : 194 passed, 2 skipped, 1 FAILED — REGRESSION
      test 12.4 relation_phrases count: expected 32, got 33. Batch-3 added `has_part`
      template to decoder/config_decoder.yaml but decoder/tests/test_all.py:1090 still
      expects 32. Not edited (tester role).
  - G3 validate_configs : 8/8 OK
  - G4 demo questions : all 3 match baseline (is_a/antonym/associated_with, heuristic=False)
Changes made / fixed:  none (observational re-run after lead code change; no code edits; no new files)
Small dataset re-run (food_bio_runner.py overwrote existing food_bio_small_results.txt + fbs JSONs):
  14/15 PASS (was 12/15)
  - fbs07 (example_of) now PASS — deterministic (Batch-3 top-k tie-break + chain-preference)
  - fbs12 (opposite of sweet) now PASS — exact-label anchor picks sweet instead of sugar
  - fbs14 still FAIL (strict criteria): behavior improved; system now emits an honest no-relation
      answer: "I don't have a relation in my knowledge graph that fully answers this question.
      Closest concepts I have: photosynthesis. (No causes relation found.)" — no bare echo, no
      fabricated edge. The runner check requires heuristic_used=True AND the exact bare Section 6.4
      sentence; the system answers via the new honest_by_relation gate and appends context.
      CRITERIA MISMATCH vs behavior — flagged (S13.5: no golden edit; lead's call).
Medium dataset re-run (in-process across the 20 pre-registered goldens; NO new result files):
  19/20 PASS, 0 crashes (was frozen at fbm09 via tester-b-001)
  - fbm09 snow now answers has_property -> cold — P-blocker verified gone
  - fbm20 corn/Corn distinct (106/107), "corn is a grain." — PASS
  - fbm19 still FAIL (strict criteria) — same honest_no_relation-with-context class as fbs14:
      chain=['part_of'] (expect ['has_property']), honest_by_relation=True,
      ans "I don't have a relation... Closest concepts I have: plankton. (No part_of relation found.)"
Needs lead decision:
  - N1 GATE REGRESSION G2/12.4: relation_phrases 32 vs 33 caused by Batch-3 has_part template;
      sync decoder/tests/test_all.py:1090 (32 -> 33) or revert. Not touched by tester.
  - fbs14/fbm19 criteria: honest no-relation behavior now matches Section 6.4 INTENT but the runner's
      strict check (heuristic_fallback_used=True AND exact bare sentence) doesn't match the new answer
      shape (context suffix, honest_by_relation path). Recommend accepting honest-by-relation answers
      (compare without trailing context) — lead's call per S13.5.

---

## 2026-09-18 (tester-b) — late evening
Datasets run:  food_bio_medium (own-domain Medium; 196 nodes, 181 edges, 9 relations)
Golden pass rate: BLOCKED at fbm09 (P-blocker tester-b-001) — fbm01-08 executed, run incomplete
Open blockers:  tester-b-001 (P-blocker) — see BUG_tester-b-001.md
Open P-bugs:    P1/P2/P3 from Small run (recorded, no code change) + tester-b-001
Gate suite today: baseline recorded (baseline_2026-09-18.txt) — G1 213/102, G2 195/2, G3 8/8, G4 OK.
                  Matches lead reference. Not re-run (no code change yet).
Changes made / fixed:  none (observational runs + bug report only)
Notes / needs lead decision:
  - P-BLOCKER tester-b-001: walker crashes on the Medium graph. Tier1 top_k=64 gate prunes
    boundary seed embeddings from resonated.node_embeddings (64 of 71 nodes); glmx_ask passes
    the partial dict to a walker with no embedding provider -> EmbeddingLookupError on start-node
    lookup. Reproduced 2/2 via documented CLI. Experimentation on Medium frozen per Section 8.1
    until lead fixes or re-scopes.
  - Medium golden set (20 questions) fully pre-registered and FROZEN in
    food_bio_medium_golden.md; fbm19 missing-relation probe design-time verified
    (clause best-matches absent spatial_near @ 0.670); fbm20 duplicate labels corn/Corn
    verified distinct at build (cos=1.0000, would merge if embeddings passed to add_dataset).
  - Medium build asserts passed: 196 nodes (100-1,000), 9 relations (Section 5.4: is_a+antonym,
    causes+caused_by, part_of+has_property), B4 label lookup OK, B7 subset-of-16 OK,
    plankton isolated (no edges).

---

## 2026-09-18 (tester-b) — evening
Datasets run:  food_bio_small (own-domain Small; 51 nodes, 39 edges, 7 relations)
Golden pass rate: 12/15 (80%) — first own-domain run; goldens pre-registered in
                   food_bio_small_golden.md BEFORE execution
Open blockers:    none
Open P-bugs:      P1 (seed anchoring: "opposite of sweet" -> sugar node), P2 (borderline
                  stochastic object selection on example_of), P3 (over-confident relation
                  match on absent relation -> no honest fallback)  [details below]
Gate suite today: not run (no code change yet)
Changes made / fixed:  none (observational run; no system edits)
Notes / needs lead decision:
  - fbs07 is NON-DETERMINISTIC: same question flipped to PASS on a diagnostic re-run.
    Softmax sampling at temperature 0.1 gives animal 73.9% vs robin 26.0% from the bird
    seed. First-run FAIL stands (no cherry-picking per 13.5).
  - P3 DECISION NEEDED: extractor never heuristic-falls-back on an absent relation; a
    confident wrong match (0.83) beats the Section 6.4 honest fallback. Finding recorded;
    NO code change made (tester role). Options for lead: raise similarity_threshold, or
    require the matched relation to have edges in the walked subgraph else force
    heuristic_fallback, or accept behavior. tester-b recommendation: subgraph-edge check.

### Small dataset run (own-domain) — 2026-09-18
- Built `test_results/tester-b/datasets/food_bio_small.db` via `build_food_bio_small.py`
  (mirrors lead's `build_toy_eval.py`). Output: 51 nodes, 39 edges, 7 relations.
- Relation set: antonym, causes, example_of, has_property, is_a, part_of, synonym (>=4; is_a
  and antonym present for the 5.4 band; causal causes present).
- Edge cases verified at build: `rice`/`Rice` remain distinct nodes (B4 label lookup OK,
  B7 subset-of-16 OK, cos(rice,Rice)=1.0000 would have auto-merged if embeddings had been
  passed to add_dataset — avoided by setting store._embeddings post-hoc, lead pattern).
- Runner `food_bio_runner.py` executed the 15 pre-registered goldens through
  `scripts/glmx_ask`; result file `food_bio_small_results.txt`; per-question full JSON
  written as `food_bio_small_<id>.json` (Section 5.7 naming).

#### Per-question results (12/15 PASS; 3 FAIL)
- PASS  fbs01-06, fbs08-11, fbs13, fbs15      (all hops+object+chain correct)
- NOTE  fbs15 (pass) planned chain was `has_property` (extractor matched "what is" instead
        of "is a", conf 0.8563) but the walker still reached grain via is_a. Chain
        expectation recorded as [is_a] in the golden; pass judged on hops+object per lead
        grading (hop/object only), deviation logged.
- FAIL  fbs07  Give me an example of a bird    object=animal, gold robin. Hop/chain correct.
               Borderline discrimination + stochastic softmax (see P2).
- FAIL  fbs12  What is the opposite of sweet?  target seed resolved to `sugar` (id 49), not
               `sweet`; sugar's only edge is sugar->tooth decay (causes) -> wrong answer.
               Hop(1) and planned chain [antonym] correct (see P1).
- FAIL  fbs14  What follows photosynthesis?    extractor confidently matched `causes`
               (conf 0.8342, heuristic_fallback=False) for an absent relation -> empty walk
               -> decoded bare echo "photosynthesis" instead of the honest no-relation
               sentence. Isolated seed mechanics worked (0 hops) (see P3).

#### Section 8 failure categorization (system observations, no golden edits)
- P1 seed anchoring: target-entity = argmax cos(question, node) picks `sugar` for
  "opposite of sweet". Single-sense anchoring without verification.
- P2 borderline + non-determinism: two near-equal candidates from bird seed (animal is_a
  score ~0.183 vs robin example_of ~0.174); softmax T=0.1 sampling flipped the outcome
  across runs. Recording/audit/replay (roles B: Cross-Tester Audit) should flag this.
- P3 over-confident extractor instead of honest fallback: clause "follows
  photosynthesis" -> causes (0.8342). Honest no-relation branch
  (glmx_ask.py:623, `heuristic_fallback_used and not walk.path_edges`) never fires
  because the extractor rarely heuristic-falls-back. This is the exact tester-B
  "missing relation -> honest fallback" (13.3/B3) probe: system answers with a bare
  echo rather than the Section 6.4 sentence — no fabricated edge, but not the
  documented honest fallback either.

#### Environment notes (unchanged from morning)
- Python 3.14.4, sentence-transformers 5.5.1, numpy/yaml/pytest OK
- spaCy not installed -> `scripts/ingest.py` / `kg_builder` dormant; `.db` built directly
  via `SQLiteGraphStore.add_dataset` + real SBERT embeddings (lead pattern).
- Working on `maharshi`; no tester-b branch yet (Section 13.1).

---

## 2026-09-18 (tester-b) — morning
Datasets run:  (onboarding only — no own-domain dataset yet)
Golden pass rate: n/a (own-domain goldens not yet run)
Open blockers:  none
Open P-bugs:    none
Gate suite today: not re-run (no code change yet)
Changes made / fixed:  none
Notes / needs lead decision:

### Mandatory onboarding (Section 13.4) — PASSED
- `python test_results/lead/datasets/build_toy_eval.py`  → 21 nodes, 17 edges OK (SBERT BAAI/bge-small-en-v1.5, now cached)
- `python test_results/lead/toy_eval_runner.py --out test_results/tester-b/onboard_2026-09-18.txt`  → **7/7 PASS**
  (see `onboard_2026-09-18.txt`)
- `python test_results/lead/toy_eval_walker_ab.py`  → weight=0.0: tobacco 11/21, fire 10/21 (coin flip, matches lead baseline);
  weight=1.0: tobacco 21/21 (matches lead baseline P1: 21/21)

Environment notes:
- Python 3.14.4, sentence-transformers 5.5.1, numpy/yaml/pytest OK
- spaCy **not installed** → `scripts/ingest.py` JSON-triples path (`kg_builder`) is dormant.
  Per lead precedent, own-domain `.db` files are built directly via `SQLiteGraphStore.add_dataset`
  + real SBERT embeddings (the exact pattern in `test_results/lead/datasets/build_toy_eval.py`),
  then exercised through the full pipeline via `scripts/glmx_ask --db` (Section 5.2 requirement).
- HF cache runs in degraded non-symlink mode (warning only, harmless).
- Working on `maharshi`; no tester-b branch yet (Section 13.1).