# GLM-X v3.3.2 — Unseen-Domain End-to-End Evaluation

**Verdict: NOT PASSED** (one pre-registered gate failed; 12 of 13 hold).

This is an *additional*, independent stress test issued after the aviation PoC
sign-off. It does not restate, revise, or replace that sign-off
(`test_results/final_validation/FINAL_POC_REPORT.md`, ACHIEVED, 348/355). Nothing
in this report should be read as amending it.

---

## 1. Dataset summary

### 1.1 Why this domain is genuinely unseen

Before writing any data I audited every domain present in this repository: toy,
nature/weather/climate, geography glossary, science evidence, food/food-bio,
astronomy, computing, history, human body, music (`tester-c/datasets/build_music_small.py`),
aviation (`test_results/final_validation/`), the Stage A–E graphs, and the
lead zoo/mountain sets.

**Weaving and textile craft appears in none of them.** No node, edge, label,
question, or gold answer from any prior run was reused.

### 1.2 Graph

| property | value |
|---|---|
| nodes | **149** (contract §5 range ~100–150) |
| edges | **189** |
| relations used | **16 / 16** canonical |
| edge strength / confidence | 0.80 – 1.00 |
| uniquely-readable one-hop edges | 186 |
| ambiguous one-hop edges (excluded, listed) | 3 |
| bidirectional pairs | 14 → **28** gradeable direction reads |
| 2-hop chains | 112 |
| 3-hop chains | 35 |
| mirror controls | 41 |
| out-of-graph subjects | 23 |
| nonsense (cue-free) questions | 6 + 1 disclosed guess |
| `sha256` | `9db5f64dd5840ab7dcd419cf8e1aec3edaa7531cdd128ce1673fd93bc32c772c` |

Relation coverage: `is_a`, `has_property`, `causes`, `caused_by`, `precedes`,
`follows`, `part_of`, `synonym`, `antonym`, `example_of`, `associated_with`,
`supports`, `contradicts`, `temporal_coincident`, `spatial_near`,
`linguistic_maps`.

Three edges were **excluded** from grading because they are not uniquely
readable, and are recorded rather than silently dropped:

- `shuttle --part_of--> {bobbin, handloom}`
- `weft insertion --temporal_coincident--> {beating, shedding}` (stored twice)

`precedes`/`follows` pairs were deliberately made **disjoint** rather than
forming a process chain, so no hub node could destroy a direction read.

### 1.3 Questions

**402 questions**, derived mechanically — no hand-picking, no gold adjusted to
suit a phrasing.

| category | n |
|---|---|
| `simple_one_hop` | 156 (122 plain + 34 variant) |
| `short_multi_hop` | 147 |
| `mirror_silence` | 41 |
| `direction_pair` | 28 |
| `honesty_out_of_graph` | 23 |
| `nonsense_fallback` | 7 |
| **total** | **402** |

`set sha256` = `c41727e508ca0a9d04ee6a0e3649d760e66ce548c5b20ca1a65273e42fd4ab3f`

**Variant slice (new in this run).** Every earlier set asked each edge with
exactly one phrasing, so the measured one-hop rate has always been "can the walk
find a uniquely-stored edge from a *known* phrasing". 34 one-hop rows were
re-asked with cue-preserving rewordings and tagged:
17 `paraphrase` (different framing) and 17 `plural` (anchor morphologically
inflected, by a naive rule-based pluraliser). They are removed from the
`simple_one_hop` pool, so **no edge is graded twice**.

### 1.4 What was reused, and what was new

**Reused** — the *data-independent graph algebra* from
`test_results/stage_f/build_holdout_graph.py` (`reach_from`, `unique_one_hop`,
`direction_pairs`, `multi_hop_chains`, `mirror_controls`, `validate`,
`finalize_and_hash`), imported by path, and the shared grader
`test_results/stage_e/stage_e_runner.py`, imported wholesale with only its data
globals rebound. Reimplementing either would risk silent divergence, where a
failure would mean my code was wrong rather than the pipeline. This algebra
encodes real engine semantics — notably *which relation the section 9 mirror
synthesises for `part_of`* — so it is not a convenience.

**New** — all data, all validation checks, and every template. The
`ONE_HOP_TEMPLATE` / `CONT_CLAUSE` / `PARAPHRASE_VARIANTS` bank was authored for
this run, not imported.

### 1.5 Freeze discipline

- Graph and question set were built, validated and **frozen before a single
  question was scored**.
- Two **independent** digests are pinned in `run_eval.py`, both written before
  scoring: the question set (handed to the shared grader) and the graph
  (checked in a file the freeze did not produce, closing the loop where an
  edited set could carry a matching *inner* graph hash).
- The frozen set **refuses to score** on drift — it did not drift.
- Post-scoring re-hash: both artefacts byte-identical. No `-wal`/`-shm` sidecars.
- The scored run mutated **nothing** outside `test_results/unseen_e2e/`
  (verified by mtime scan of the whole tree against the build timestamp).
- Nothing is committed or staged. `HEAD` remains `3e49cf0`.
- Prior sets re-verified against their own recorded digests: `final_validation`
  both OK; `stage_f` both OK
  (`d1be5c7b…f57708`, `0e0caab1…1a7e43`).

### 1.6 Pre-registered pass criteria

Written **into the frozen artefact** before scoring:

overall ≥ 0.90 · `simple_one_hop` ≥ 0.95 · `direction_pair` ≥ 0.90 ·
`short_multi_hop` ≥ 0.75 · `honesty_out_of_graph` ≥ 0.90 ·
`nonsense_fallback` ≥ 0.90 · `mirror_silence` ≥ 0.90 · central invariant holds ·
fallback triggers on every nonsense row · guess disclosed on the disclosed-guess
row · illegal mirrors = 0 · fabricated nodes/relations = 0 · unaccounted
entities = 0 · determinism differing fields = 0.

### 1.7 Validation checks run before freezing

Inherited: identity, category minima, graded-edge uniqueness, chain
expressibility against the **live planner**, mirror-control direction,
out-of-graph absence.

New, and each of which caught a real defect during authoring:

| check | what it caught |
|---|---|
| `label-token-is-another-label` | 9 labels (`floor loom`→`loom`, `warp yarn`→`yarn`, `vat control`→`vat`, `loose sett`→`sett`, …) — the `pv018` class, where a multi-word label contains a token that is itself a label and is therefore unanswerable by construction |
| `out-of-graph-subject-grounds-on-a-label` (+ `…-prefixes-a-label`) | design-time removal of subjects like `hang glider`→`glider` — the `og013` class, where the identity gate grounds the out-of-graph subject on a real label, so the row is not an honesty question at all |
| `template-does-not-emit-declared-relation` | 2 real template bugs: `"What does the {a} translate to?"` cues **nothing** (the config cue is the literal `"translates to"`, inflected), and `"what else is another word for that"` contains no synonym cue at all |
| `question-names-a-language-the-graph-does-not-use` | closes the prior sign-off's 40-false-question defect **by construction** — the base template names no language, and the one language-naming paraphrase is restricted to English-side anchors |

Carried over: `label-prefix-collision` (4+ char prefix), `too-few-out-of-graph-subjects` (≥20).

---

## 2. Results

### 2.1 Metric table

| category | n | passed | rate | target | meets |
|---|---:|---:|---:|---:|:--:|
| `simple_one_hop` | 156 | 155 | 0.9936 | 0.95 | YES |
| `direction_pair` | 28 | 28 | 1.0000 | 0.90 | YES |
| `short_multi_hop` | 147 | 136 | 0.9252 | 0.75 | YES |
| `honesty_out_of_graph` | 23 | 23 | 1.0000 | 0.90 | YES |
| `nonsense_fallback` | 7 | 6 | **0.8571** | 0.90 | **NO** |
| `mirror_silence` | 41 | 41 | 1.0000 | 0.90 | YES |
| **OVERALL** | **402** | **389** | **0.9677** | 0.90 | YES |

### 2.2 Integrity and honesty metrics

| metric | value |
|---|---|
| central invariant held | **401 / 402** (0.9975) |
| semantic checks passed | 390 / 402 (0.9701) |
| **illegal same-label mirrors** | **0** (0 rows) |
| **fabricated nodes / relations** | **0** (0 rows) |
| leaked labels | 0 |
| semantic "invented" entities | 0 |
| unaccounted entities | 0 |
| hop provenance | `{stored: 508, declared_inverse: 4}` |
| fallback fired | **7 / 7** |
| guess disclosed on `nf007` | **yes** |
| direction pairs answering differently | **14 / 14** |
| no invented relations | 1.0 |
| template correctness | 1.0 |
| **determinism** | **0 differing fields across 402 rows**; every summary block and the `pass` flag identical |

### 2.3 Variant slices (new)

| slice | rate |
|---|---|
| `paraphrase` | **17 / 17 = 1.0000** |
| `plural` | 16 / 17 = 0.9412 |
| plain template rows | 356 / 368 = 0.9674 |

Re-asking an edge with a *different* phrasing cost nothing. Inflecting the
anchor cost one row.

---

## 3. Failure analysis — 13 failures

### 3.1 The dominant class: 11 × `short_multi_hop` (walker truncates one hop short)

All 11 share one signature: `len(walked_path) == len(expected_path) − 1`.
**11 / 11.** The walk completed every hop but the last.

I instrumented the pipeline (diagnosis only; no re-scoring) to establish the
cause rather than infer it:

| | final target in resonated subgraph? |
|---|---|
| 11 failing rows | **absent — 11 / 11** |
| 6 sampled passing multi-hop rows | **present — 6 / 6** |

Clean separation. The target was pruned **before the walker ran**. It is not an
activation-threshold problem: passing controls survive at activations as low as
`0.0100`, exactly `walk.min_activation = 0.01`.

The reason pruning wins, measured as hop-distance from the nearest seed node
over full-graph walkable adjacency:

| rows | distance of final target from nearest seed |
|---|---|
| all 11 failures | **2 or 3** |
| passing controls | 0 or 1 |

**Mechanism.** `resonance/tier1.py::_readmit_seed_neighbourhood` runs *after*
`_gate` and extends only **one** hop out from the **original** seeds — never out
from a node the walk itself reached. `_propagate` (4 iterations) is the only
other route in, and when it under-ranks the final target, a node 2–3 hops from
any seed is unreachable to the walk. The walker can traverse the seed zone and
its 1-hop shell, but cannot continue past the shell.

This is **not** a budget ceiling. Failures resonated 72–81 nodes, well inside
the effective allowance; the full set ranges 63–89 (p50 76). It is an
*ordering and depth* defect.

Credit where due: on all 11 the walker stopped cleanly and the decoder
**declined to assert the unwalked relation** — logged as *"walked relations
differ from planned; rendering from walked relations to avoid asserting an
unwalked relation"*. The system truncated rather than fabricated. That is the
architecture behaving correctly given a resonance that under-admits.

### 3.2 `mh042` — the answer was right and the *question* was wrong

> **Q:** "What does the fast colour cause, and what is known for that?"
> **A:** "fast colour is caused by mordant."
> **Gold chain:** `fast colour --causes--> mordant --has_property--> metal salt`

The graph stores `mordant --causes--> fast colour`. The shared algebra treats a
mirrored step as a forward step, so the derived question asks *"what does the
fast colour cause?"* while its gold answer is the thing that **caused** it. The
question is semantically false; the system's answer is true.

This is a **question-derivation defect I inherited and did not fix** — fixing it
after seeing a score is exactly the tuning this run forbids. It affects 1 of 147
multi-hop rows. It is also the one row that fails `chain_matches_walk`: the
walker traversed the mirrored step, and the decoder rendered it with the
`caused_by` surface form. 21 rows used an inverse label in the walk; on this one
the rendering is backwards.

### 3.3 `pv020` — plural inflection caused a *false* refusal

> **Q:** "The pits are contained in what?" (`plural` variant of `pit --part_of--> handloom`)
> **A:** refusal — *"No part_of relation found."* `anchor_identity_reason: ungrounded:pit`

The anchor was located (exact label hit) but the plural form broke identity
grounding, and the system then refused a question it could have answered. This
is a **false refusal** — the correct-but-unhelpful failure mode. The other 16
plural rows passed, so the plural slice is 16/17.

### 3.4 `nf004` — the one gate failure, and it is a metric defect

> **Q:** "What is the temperature of the word apple?"
> **A:** refusal naming **two** concepts: *"Closest concepts I have: silk, long filament."*

It refused correctly and earned the only `no_claim_when_refusing` evaluation in
the category, which it failed. Post-hoc projection on the frozen results (no
re-run):

- `stage_c/stage_c_runner.py:241` early-returns unless `len(labels) >= 2`.
- All **6** true refusals use the identical template, whose parenthetical prints
  the raw relation name: `(No has_property relation found.)`
- The leak test is a plain substring containment. `RELATION_PHRASES["has_property"] == "has"`,
  which is a substring of `has_property`.
- Therefore **0 / 6** true refusals would pass an always-on check. `nf004` is the
  only one whose refusal happened to name two concepts, so it is the only one
  that was actually tested.

**`nonsense_fallback` = 6/7 materially overstates refusal cleanliness.** The five
passing refusals pass by short-circuit, not by clean text.

### 3.5 Disclosed, not measured

- **0 / 402 questions name a language.** The `in French` paraphrase was verified
  at freeze time by the planner probe, but the deterministic stride did not land
  on a `linguistic_maps` row at the paraphrase position. The check is proven to
  run and pass; it has no measured rows. The prior sign-off's 40 false questions
  are closed **by construction**, not by measurement.
- **The naive pluraliser emits ungrammatical English** for some labels
  ("the equipments", "the spinnings", "The bobbins translates to what?"). Some
  plural rows therefore measure English grammar, not the pipeline. A rule-based
  pluraliser was chosen deliberately so no hand-authored irregular table could
  flatter this graph.

### 3.6 Not tuned

Every defect above is **disclosed and left unpatched.** Patching any of them
after seeing this score would convert an independent evaluation into a tuned one.
The frozen set, its digests, and its criteria are unchanged.

---

## 4. Mechanism assessment

| mechanism | status | basis |
|---|---|---|
| **Graph building** (ingestion, storage, integrity) | **working well** | 149 nodes / 189 edges loaded with 149 embeddings; 0 corrupted reads; both digests stable; no sidecar artefacts. *(Caveat: this run assesses the loader, not a builder — the data is hand-authored.)* |
| **Relation extraction** | **working well** | All 402 questions produced their declared chain. Every template was asserted against the **live** `_literal_cue_relations` at freeze time; plan confidence 1.000; zero runtime chain mismatches. The freeze-time probe caught two real template bugs before scoring. |
| **Resonance / activation spreading** | **failing** | The one genuinely broken mechanism. The activated subgraph **excluded the final hop's target in 11/11** truncations, at seed-distance 2–3 where one-hop readmission cannot reach. Clean 11/11 vs 6/6 separation against controls. |
| **Walker** | **working with limitations** | Never fabricated, never produced an illegal mirror, never leaked a forbidden far end (41/41 mirror silence), and stopped cleanly rather than inventing a hop. But it cannot traverse past the seed zone's 1-hop shell, so it truncates. One rendering defect: a mirrored `causes` step surfaced as `caused_by` (`mh042`). |
| **Decoder** | **working well** | Central invariant **401/402**; template correctness 1.0. On the 11 truncations it explicitly declined to assert the unwalked relation — the honest behaviour under a defective upstream. |
| **Honesty / fallback** | **working with limitations**, metric defective | Behaviour is strong: 23/23 out-of-graph refused, 7/7 fallback fired, guess disclosed. But `entity_not_found` was **0/23** — it grounded each out-of-graph subject on an unrelated in-graph node and refused anyway. Right outcome, wrong signal. And the metric scoring refusal cleanliness is a no-op for single-label refusals (§3.4). |
| **Paraphrase robustness** | **working well** | 17/17 on cue-preserving rewordings. |
| **Plural robustness** | **working with limitations** | 16/17; the miss is a false refusal. |

### Architecture constraints — all held

| constraint | evidence |
|---|---|
| v3.3.2 strict | enforced by the unmodified grader |
| No cosine in main Walker scoring | `relation_chain` is the operative signal (`Plan.validate` requires it non-empty; invariant checks chain ≈ walk) |
| Inference deterministic and frozen | **0 differing fields** / 402 rows; all summary blocks and `pass` flag identical |
| No invented relations | 0 fabricated hops, 0 illegal mirrors, 0 leaked labels, 0 invented entities |
| Honest refusal preferred over a false answer | 0 fabrications; all 11 truncations refused to assert the unwalked hop; the single false refusal (`pv020`) erred *towards* refusal |

---

## 5. Verdict

**NOT PASSED.** Overall accuracy 0.9677 clears its gate and 12 of 13 criteria
hold, but the pre-registered `nonsense_fallback` gate was missed
(6/7 = 0.857 < 0.90) — and that failure is compounded rather than excused by the
finding that the metric scoring it is itself defective (§3.4: 0/6 true refusals
would pass an always-on check). A pre-registered gate was missed; the gate is
reported as missed.

Independently of the gate: on a domain the system had never seen, with an
independently authored question bank, GLM-X v3.3.2 fabricated nothing, mirrored
nothing illegally, leaked nothing, and was byte-for-byte deterministic. Its one
substantive functional defect is resonance depth — the activated subgraph cannot
reach 2+ hops from the seed zone, which truncates 11 multi-hop walks — and the
decoder's refusal to paper over that is the system behaving exactly as the
architecture requires.
