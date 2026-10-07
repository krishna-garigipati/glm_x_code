# GLM-X v3.3.2 — Final PoC Sign-off Report

**Verdict: ACHIEVED.** 348 / 355 = **0.980** overall on a single, frozen,
never-before-scored evaluation set in a new domain. Every category target met,
every mechanism target met, determinism exact, zero illegal mirrors, zero
fabricated knowledge.

This report also discloses **two defects the exercise itself found** — a
contract deviation in the resonance node budget, and 11.3% of the frozen
questions being semantically false phrasings. Neither changes the verdict; both
are stated plainly rather than buried, and neither was patched after scoring,
because patching after scoring is tuning.

---

## 1. What was frozen, and when

| Artefact | SHA-256 |
|---|---|
| `test_results/final_validation/final_eval.db` (graph) | `c5f27fd40bda293b5d1fd4084aa525ea6be06cc1421a91c3aba167f713e7f95f` |
| `test_results/final_validation/final_questions_frozen.json` (355 questions) | `358c1eb1083c785d5cf34caa1d478122cec6721b674459ea4519bf0ae4fce715` |

Both digests were read once at freeze time and pinned in
`run_eval.py`, which refuses to score on drift. Both were **re-verified after
scoring and are unchanged**; no `-wal` / `-shm` sidecars exist.

Order of operations, strictly: build graph → validate → derive questions →
preflight → freeze → **score once** → re-run for determinism → this report.
Nothing in the set was edited after any score was seen.

Graph: **149 nodes, 171 edges, all 16 canonical relations used.** 167 uniquely
readable one-hop edges, 4 ambiguous (excluded from grading and reported, not
hidden), 14 bidirectional pairs → 28 gradeable direction reads, 94 two-hop and
23 three-hop chains, 42 mirror controls, 22 out-of-graph subjects.

---

## 2. Why this set can be believed

**The domain is new, and that was checked by audit rather than taste.** Every
domain already present in this repository was enumerated first: toy,
nature/weather/climate, geography glossary, science evidence, food and
food-bio, astronomy, computing, history, human body, music
(`tester-c/datasets/build_music_small.py`, which already carries
guitar→chitarra and violin→violino), the Stage A–E graphs, and the lead
zoo/mountain material. An earlier draft of this file used **music** and was
discarded on finding that overlap. The final domain is **aviation**, which
appears in no dataset, stage or probe in this repository.

**The questions are derived, not chosen.** Stage F's derivation is run
unchanged against the aviation graph, and `ONE_HOP_TEMPLATE` / `CONT_CLAUSE` are
*imported* from `test_results/stage_e/generate_questions.py` rather than copied,
so if the templates were edited to flatter this graph the diff here would show
it. No gold answer was adjusted to suit a phrasing.

**The set is preflighted mechanically against the live planner.** Cue
recoverability, category minima, graded-edge uniqueness, chain expressibility,
mirror-control direction, and out-of-graph absence are all checked
programmatically. `build_graph.py` refuses to build and `generate_questions.py`
refuses to freeze if any check fails — both did, repeatedly, during construction.

**Two validation checks were added, both forced by the two fixes:**
`too-few-out-of-graph-subjects` (≥ 20, because the identity gate now carries the
honesty category and at 9 questions one question is worth 11 points, too coarse
to resolve a 90% target) and `label-prefix-collision` (no label may be a 4+
character prefix of another, because the identity gate grounds on exactly that).
Nothing inherited was filtered, suppressed or weakened.

---

## 3. Results

### 3.1 Categories

| Category | n | pass | rate | target | ok |
|---|---|---|---|---|---|
| `simple_one_hop` | 139 | 138 | 0.9928 | ≥ 0.95 | **YES** |
| `direction_pair` | 28 | 28 | 1.0000 | ≥ 0.90 | **YES** |
| `short_multi_hop` | 117 | 112 | 0.9573 | ≥ 0.70–0.80 | **YES** |
| `honesty_out_of_graph` | 22 | 21 | 0.9545 | ≥ 0.90 | **YES** |
| `nonsense_fallback` | 7 | 7 | 1.0000 | must trigger | **YES** |
| `mirror_silence` | 42 | 42 | 1.0000 | ≥ 0.90 | **YES** |
| **OVERALL** | **355** | **348** | **0.9803** | ≥ 0.90 | **YES** |

### 3.2 Mechanisms

| Check | Result | Target |
|---|---|---|
| central invariant hold rate | **355 / 355 = 1.000** | ≥ 0.90 |
| semantic checks (path order, template glue, hop provenance) | 350 / 355 = 0.9859 | parity |
| `no_invented_relations` | **1.000** | ≥ 0.98 |
| illegal or fabricated hops | **0** | 0 |
| unaccounted entities | **0** | 0 |
| `template_correctness` | **1.000** | ≥ 0.95 |
| direction pairs answering differently | **14 / 14** | all |
| fallback trigger rate | **7 / 7** | all |
| hop provenance | 420 stored, 3 `declared_inverse`, 0 unattributed | — |

The 3 `declared_inverse` hops are the sanctioned `has_part` mirror. No edge was
ever mirrored under a forward relation label: all 42 mirror-silence controls
refused correctly, including controls whose far end is a high-degree node.

`relation_chain` remained the operative signal throughout — grading compares the
walked chain and path, never just the answer string.

### 3.3 Paraphrase / plural robustness (new in this set)

Every previous stage asked its one-hop questions with exactly one phrasing per
relation, so `simple_one_hop` had only ever measured *"can the walk find a
uniquely-stored edge from a known phrasing"*. A deterministic slice of the
one-hop pool is re-asked here with cue-preserving rewordings, tagged
`variant_kind`. The slice is **removed from** the `simple_one_hop` pool, so no
edge is graded twice under two phrasings.

| Variant | n | pass | rate |
|---|---|---|---|
| `template` (Stage E bank, unchanged) | 104 | 104 | 1.000 |
| `paraphrase` (different framing, same literal cue) | 23 | 23 | 1.000 |
| `plural` (anchor morphologically inflected) | 12 | 11 | 0.917 |

The system is insensitive to rewording and almost entirely insensitive to
inflection. The single failure is analysed in §5.2 and is a genuine defect, not
a scoring artefact.

### 3.4 Determinism

The scored run and an independent re-run agree on **0 differing fields across
all 355 rows**, excluding only wall-clock `time_seconds`. Every summary block
(`categories`, `mechanisms`, `pass`, `stage`) is identical. Inference is
deterministic and frozen by default, as required.

---

## 4. How the two fixes were attributed

For the record, because the earlier stages could not attribute anything: Stages
A–E all sat at 100% (78/78, 88/88, 195/195, 56/56, 208/208) and therefore had
**no headroom** — "all stages identical to baseline" proved nothing about either
fix. The only real attribution evidence is a 2×2 controlled ablation on the
previous 147-node held-out graph (Change A = resonance seed-neighbourhood
readmission; Change B = entity-identity gate), both switches verified to fire and
to be orthogonal:

| | Change B on | Change B off |
|---|---|---|
| **Change A on** | 255 / 257 (verdict **passes**) | 254 / 257 (verdict fails) |
| **Change A off** | 250 / 257 (verdict **passes**) | 249 / 257 (verdict fails) |

- A alone fixes **5**: `oh029 oh052 oh097 oh100 mh006`
- B alone fixes **1**: `og002`
- A + B together fix all **6**
- neither fixes **2**: `mh012`, `mh053`
- **0 regressions** from either change
- cell 4 reproduced the frozen record exactly: identical failure set, and
  identical pass/fail for all 257 rows, which also proves the other working-tree
  removals (adjacency rescue, chain lift) had zero effect on that benchmark

A is the larger contributor to raw accuracy (249→254) but does **not** flip the
verdict; B alone flips it (249→250), because honesty at 8/9 = 0.889 was the only
sub-target missed. That ablation ran on an already-seen set, so it is
attribution evidence, not proof. **This run is the proof.**

---

## 5. All seven failures

Failure attribution: `walker_wrong_path: 5`, `wrong_anchor: 2`. Nothing is
papered over and no honesty failure was waived.

### 5.1 Five — final hop target pruned by the node budget

| id | expected path | walked | missing target |
|---|---|---|---|
| `mh009` | airstrip → runway → marked threshold | airstrip → runway | `marked threshold` |
| `mh025` | flight engineer → pilot → instrument rating | flight engineer → pilot | `instrument rating` |
| `mh077` | tail fin → glider → ridge lift | tail fin → glider | `ridge lift` |
| `mh078` | tail fin → glider → wooden | tail fin → glider | `wooden` |
| `mh103` | ingegnere → flight engineer → pilot → instrument rating | …→ pilot | `instrument rating` |

All five are the same shape: the planner produced the **correct declared chain**
and the walker completed every hop but the last, because the last hop's target
was not present in the resonated subgraph.

Cause, identified: readmission is scoped **one hop from the seeds**
(`resonance/tier1.py::_readmit_seed_neighbourhood`). When the hop-1 endpoint
(`runway`, `pilot`, `glider`, `flight engineer`) is not itself a seed, its own
outgoing edges remain prunable by the tier-1 node budget. This is the already
documented scoping limit, and **this run reproduces it independently on a third
graph** — the same class as Stage F's `mh012` and `mh053`, which still failed
with readmission fully enabled. Widening readmission to k-hop would void the
contract's node budget, so the limitation is documented rather than engineered
away. It is a real, quantified capability ceiling: chains whose second target
lies outside the seed neighbourhood are unreachable.

### 5.2 Two — wrong anchor

**`og013` — "What is the hang glider part of?"** anchored on `glider` and
answered (`glider` → `tail fin`, a real `declared_inverse` edge) instead of
refusing. The identity gate grounds on token overlap and `hang glider` contains
the label `glider`. This is precisely the defect class the fix targets, and it
is **not fully fixed**: a multi-word out-of-graph subject whose tokens include a
real label is still mis-grounded, producing a confident answer about an entity
the question never named.

**`pv018` — "What is known for the instrument landing systems?"** anchored on
`landing` rather than `instrument landing system`, then refused. Refusing an
answerable question is still a failure. Cause: plural inflection broke the
multi-word label match, and the single-word label `landing` won on an exact
token hit. The other 11 plural questions passed, so this is narrow — it fires
only when the anchor is a multi-word label containing a token that is *itself* a
label. Note this reveals a **gap in my own added validation**: the
`label-prefix-collision` check compares label-vs-label, but this failure is
label-vs-its-own-constituent-token. A stricter set would refuse to build on that
pattern.

---

## 6. Defect 1 — the resonance node budget no longer holds at 64

**Contract §7 fixes `tier1.top_k: 64`.** Measured on this run:
`n_resonated_nodes` min 36, median 71, **max 84**, with **313 / 355 rows (88.2%)
exceeding 64.**

Cause, confirmed in source: in `resonate()` the order per iteration is
`_propagate(...)` — which normalises, snapshots to `pre_gate`, then applies
`_gate` (the `top_k` cut) — and *then* `_readmit_seed_neighbourhood(...)`, which
adds seed neighbours back **with no subsequent re-cut**. The effective ceiling is
therefore `top_k + |initial_seeds|`. `scripts/glmx_ask.py:902` seeds with
`top_k=20`, giving 64 + 20 = **84** — which is exactly the observed maximum.

This is a bounded, seed-count-sized overshoot, and it is the price of the fix
that recovered 5 questions on the previous set. But it is a **literal deviation
from §7**, and `configs/config_resonance.yaml` still carries the comment
*"hard-codes TierConfig(top_k=64, max_iterations=4) from contract section 7"*.
This must be reconciled by one of two routes, and it is not reconciled here:
either amend §7 to state the real bound (`top_k + |initial_seeds|`), or re-apply
the budget after readmission. It was left alone deliberately: changing either the
code or the config after seeing this run's scores is exactly the tuning this
exercise forbids.

---

## 7. Defect 2 — 11.3% of the frozen questions are semantically false

**40 of 355 questions ask for a Spanish translation** ("What do you call the
ala in Spanish?", and 39 more) while the graph's `linguistic_maps` pairs are
English↔Italian (`ala`↔wing, `ingegnere`↔flight engineer, `motore`↔jet engine,
`timone`↔rudder, `pista`↔runway).

Cause: `ONE_HOP_TEMPLATE["linguistic_maps"]` in
`test_results/stage_e/generate_questions.py` hardcodes the phrase *"in Spanish"*.
Importing that bank verbatim was the correct anti-tuning choice — it is exactly
what makes the set trustworthy — but it silently conflicts with having chosen
Italian for this domain. My own paraphrase variants were written to say "in
Italian"; the inherited one-hop and multi-hop templates were not, and I did not
check them for domain-neutral *wording*.

**Why preflight did not catch it:** `_literal_cue_relations` is a pure regex over
the descriptor bank, so it verifies that the *relation* is recoverable. The cue
for `linguistic_maps` is "what do you call" — the language word is not part of the
cue. Preflight can prove a question is *answerable*; it cannot prove a question
is *true*. That is a real gap in the preflight design, stated here rather than
patched.

**Effect on the verdict: none.** 39 of the 40 passed anyway — the engine returns
the stored target and ignores the language word. The one failure, `mh103`,
failed for the unrelated and independently verified budget cause in §5.1, not
because of the phrasing. The set was **not** re-frozen, because re-deriving a set
after observing its scores is the definition of tuning.

---

## 8. Known residual limitations

1. **Two-hop targets outside the seed neighbourhood are unreachable** (§5.1).
   Quantified here at 5/117 multi-hop; measured before at 2/257. Systematic, not
   a one-off, and reproduced on three separate graphs.
2. **The identity gate grounds on token overlap, not entity identity.** An
   out-of-graph multi-word subject containing a real label (`hang glider` ⊃
   `glider`) still yields a confident answer (`og013`, 1/22 honesty).
3. **Plural inflection can break a multi-word label match** when a constituent
   token is itself a label (`pv018`, 1/12 plural).
4. **The cue bank matches literal substrings**, so some phrasings cue `[]` and
   fall through to `default_chain: [has_property]`.
5. **`has_part` has no cue phrase**, so `part_of` cannot be asked in reverse.
6. **§7's `top_k: 64` is no longer the true bound** (§6).

---

## 9. Record-keeping

### 9.1 Removed test file — deliberate, and why

`scripts/tests/test_glmx_ask_rescue.py` was **deleted**. It asserted
contract-non-compliant behaviour: it required `_rescue_expected_relation_neighbors`
to bypass the resonance gate and mirror edges under the forward relation label.
That mechanism was **not** restored.

- Preserved copy: `C:\Users\mahar\AppData\Local\Temp\opencode\test_glmx_ask_rescue.py.REMOVED_COPY`
- 3873 bytes, 119 lines, SHA-256 `C07AEF1079AF0AA81CED980A935D4C51C39414D87173C8FD63A112F73F03D850`
- The only surviving reference in source is a comment at
  `scripts/glmx_ask.py:1157–1172` recording the removal rationale.

### 9.2 Pre-existing unit-test defects deliberately not fixed

Out of scope for this exercise; noted so they are not mistaken for regressions.

- `graph/tests/test_all.py::TestSerializer::test_unavailable_lz4_raises` —
  `DID NOT RAISE SerializationFailedError` at `:520`; lz4 fallback warning at
  `serializer.py:40`
- `graph/tests/test_all.py::TestGraphStoreEdgeCases::test_get_node_increments_use_count` —
  `assert 10 == 6` at `:627`

`pytest` after the deletion: **711 passed, 107 skipped, 2 failed** — only the two
above. Mutation batteries for Stage E and Stage F each caught 35/35 mutations
with `anti_vacuity` true and exit 0, reports byte-identical to their `.PREV`
backups.

### 9.3 The prior frozen record is intact

`test_results/stage_f/` was not modified by this exercise. Re-verified after this
run:

| Artefact | SHA-256 | status |
|---|---|---|
| `holdout_eval.db` | `d1be5c7b…f57708` | unchanged |
| `holdout_questions_frozen.json` | `0e0caab1…a7e43` | unchanged |
| `STAGE_F_REPORT.md` | `22430cca…a9b6` | unchanged |

Stage F's verdict stands as the historical record: **249 / 257 = 0.969, NOT YET
ACHIEVED**, on honesty at 8/9 = 0.889. It was not edited.

### 9.4 Changes made for this set

- `test_results/final_validation/` — new: `build_graph.py`,
  `generate_questions.py`, `run_eval.py`, plus the two frozen artefacts and their
  digest files.
- `test_results/stage_e/stage_e_runner.py` — **one additive change**: the
  per-row projection now carries `variant_kind = q.get("variant_kind")`. It
  reads an optional key and adds a column; it cannot change any score, and Stage
  E's frozen results are unaffected. No grading logic was touched, and
  `CATEGORY_TARGETS` was deliberately left alone — editing it would have
  retroactively changed what the frozen Stage E number means.
- No change to any contract file. §6 and §7 above are **disclosures, not
  amendments**.

---

## 10. What this sign-off does and does not claim

**It claims.** On one frozen, mechanically derived, never-before-scored
evaluation set in a domain with no prior presence in this repository, GLM-X
v3.3.2 scores **348/355 = 0.980** overall with every category and every §15
mechanism target met, determinism exact, `relation_chain` operative, inference
frozen and deterministic, zero illegal same-label mirroring across 42 controls,
zero invented relations and zero unaccounted entities across 355 questions, and
both correctness targets reached — the honesty gate refuses 21/22 out-of-graph
subjects *with a real relation cue present*, so the refusals are attributable to
entity identity rather than to cue absence.

**It does not claim.** That the two-hop scoping ceiling is gone (it is not, and
§5.1 quantifies it); that the identity gate is complete (§5.2); that §7's
`top_k: 64` holds as written (§6 it does not); or that the frozen set is
entirely well-formed — 11.3% of its questions are misphrased, harmlessly but
really (§7).

**It does not claim** any of the seven stages' earlier "100%" results mean
anything about the fixes, because those stages had no headroom (§4).

**Final answer: the PoC is ACHIEVED**, with §6 and §7 carried forward as
mandatory follow-ups before any claim is made about graphs larger than this one,
and with §8's limitations accepted as the system's measured current ceiling
rather than argued away.