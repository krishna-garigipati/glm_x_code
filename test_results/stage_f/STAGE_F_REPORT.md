# Stage F — Held-Out Frozen PoC Evaluation

**Final report for GLM-X v3.3.2.** New graph, new question set, both frozen before
scoring. This is the only score that counts as the headline result; Stage E is
closed and superseded.

---

## 1. Verdict

> ## ❌ PoC NOT YET ACHIEVED
>
> GLM-X v3.3.2 passes **every** §16 matrix target **except one**: it scores
> **8/9 = 0.889** on `honesty_out_of_graph` against a **≥ 90%** target. Overall
> accuracy is **249/257 = 0.969**, all three §15 mechanisms pass, and every
> §17 forbidden action is clean — but a mandatory gate was missed on data it had
> never been tuned against, so the honest verdict is **not yet achieved**.

This is a **1-question miss in a 9-question category**, and §16 asks for exactly
that many questions minimum — the category is under-powered. It is not, however,
a rounding artefact. It is a **real, newly discovered defect** (§5), and it is
a defect in the one place the contract cares most about: refusing to confabulate
about entities the system does not know.

**Decision, stated plainly:** the run was scored once, against a set frozen
before scoring, and the set was **not** edited afterwards. Fixing the defect and
re-scoring the same set would be peeking; the correct process is a **new frozen
set (Stage G)**. See §11.

---

## 2. What was built and frozen

A genuinely new controlled graph and a new matching question set, in a domain
disjoint from every earlier stage (food / cooking / nutrition vs. Stages A–E).

| | |
|---|---|
| Nodes | **147** (contract band 100–150, §5 workflow step 5) |
| Edges | **161**, all **16/16** canonical relations present |
| strength / confidence | **0.86 – 0.98** (inside the required 0.8–1.0 band) |
| One-hop edges | 147 unique; **14 skipped as genuinely ambiguous** (interior nodes of the preparation chain — the ambiguity is real, not an authoring slip), leaving **129** emitted as `simple_one_hop` questions |
| Direction pairs | **9 forward/reverse pairs → 18 gradeable pair reads** (all 9 verified to answer *differently*) |
| Multi-hop | **60 two-hop + 5 three-hop** chains = 65 |
| Mirror controls | 29 must-silence reads |
| Questions | **257**, generated mechanically |

Coverage of the one-hop pool by relation: `is_a` 43, `linguistic_maps` 12,
`spatial_near` 12, `synonym` 10, `part_of` 9, `has_property` 9, `antonym` 8,
`associated_with` 6, `example_of` 6, `temporal_coincident` 6, `contradicts` 3,
`supports` 3, `follows` 1, `precedes` 1 — **14 of 16 relations** appear in the
one-hop pool, with `causes`/`caused_by` reserved for the direction pairs.

### Frozen digests — recorded before any scoring

| Artefact | SHA-256 |
|---|---|
| `holdout_eval.db` (graph) | `d1be5c7b5538b2c8337f1fa61f6257cd7cf1732fb8702b40a110acd9a5f57708` |
| `holdout_questions_frozen.json` (set) | `0e0caab131c8894f9514a5691029297f048552ec23b411de33c6219e301a7e43` |
| Question-set version | `1.0.0` |

Both digests were re-verified **after** scoring and are unchanged. Stage E's
artefacts (`5bc235f8…`, `8454dc23…`) were not touched.

### Guarding the generator against tuning

The single biggest way a held-out set becomes worthless is if its generator was
tuned against the engine it is meant to judge. That was blocked structurally,
not by promise:

- `test_results/stage_f/generate_questions.py` **imports** Stage E's
  `ONE_HOP_TEMPLATE` and `CONT_CLAUSE` by explicit path
  (`importlib.util.spec_from_file_location`) instead of copying them. Same
  question style, provably unedited for this graph.
- All 257 questions are derived **mechanically** from graph structure. None were
  hand-picked after seeing behaviour.
- The Stage F runner is a **binding, not a copy**: it loads Stage E's grader by
  path and rebinds only data/display globals. Grading logic is shared verbatim,
  so Stage F is scored by the same judge that scored Stage E.

### Build-time guards that actually fired

The builder refused to freeze until these passed; four real defects were caught
and fixed **before** the freeze, not after:

- `_pct` overflow — strength spread ran to 2.18 (divisor error; fixed → 0.86–0.98)
- label `sugar cause claim` collided with the `cause` cue → renamed
- node count 158 > 150 → trimmed 11 nodes
- two authoring bugs (a leftover `if False else`, a deleted loop body)

Label-cue traps, `_QUESTION_FILLER` traps, a non-ASCII allowlist, digit ban,
lowercase enforcement, unused-concept detection, edge-endpoint declaration,
relation canonicality, symmetric-storage-both-ways, node-count band and
strength/confidence band are all enforced at build time. Preflight passed on the
first attempt.

---

## 3. Full metric table — every contract target

### §16 recommended PoC test matrix

| Category | n | Pass | Rate | Target | Min | Verdict |
|---|---:|---:|---:|---|---:|---|
| `simple_one_hop` | 129 | 125 | **0.969** | ≥ 95% | 30 | ✅ **PASS** |
| `direction_pair` | 18 | 18 | **1.000** | ≥ 90% | 10 | ✅ **PASS** |
| `short_multi_hop` | 65 | 62 | **0.954** | ≥ 70–80% | 8 | ✅ **PASS** |
| `honesty_out_of_graph` | 9 | 8 | **0.889** | ≥ 90% | 8 | ❌ **FAIL** |
| `nonsense_fallback` | 7 | 7 | **1.000** | must trigger | 5 | ✅ **PASS** |
| `mirror_silence` | 29 | 29 | **1.000** | no illegal mirroring | — | ✅ **PASS** |
| **OVERALL** | **257** | **249** | **0.969** | **≥ 90%** | — | ✅ **PASS** |

All six category minima are met. `nonsense_fallback` triggered **7/7**.

### §15 mechanisms

| Mechanism | Rate | Target | Verdict |
|---|---:|---|---|
| `no_invented_relations` | **1.000** | ≥ 98% | ✅ PASS |
| `template_correctness` | **1.000** | ≥ 95% | ✅ PASS |
| `central_invariant.hold_rate` | **257/257 = 1.000** | ≥ 90% | ✅ PASS |
| `semantic_checks` (Stage C parity) | 254/257 = 0.988 | — | informational |

### §17 forbidden actions / architecture integrity

| Check | Result | Verdict |
|---|---|---|
| Illegal same-label mirrors | **0** | ✅ clean |
| Fabricated nodes | **0** | ✅ clean |
| Fabricated relations | **0** | ✅ clean |
| Unaccounted entities | **0** | ✅ clean |
| Activation-rescue path reinstated | **no** (deliberately absent, `scripts/glmx_ask.py:1061-1076`) | ✅ clean |
| Illegal mirroring reintroduced | **no** | ✅ clean |
| Direction pairs answering identically | **0 of 9** (9/9 differ) | ✅ clean |
| Learning enabled during evaluation | **no** (`_no_learning = True`, seed 0) | ✅ clean |

Hop provenance across the whole set: **`stored: 278`, `declared_inverse: 1`** —
every single hop traces to either a stored edge or a contract-declared inverse.

---

## 4. Failure analysis

**8 failures out of 257.** §16 line 499 requires separating failures caused by
missing graph data from failures caused by code, so each was attributed by
reading the actual resonated subgraph the walker is allowed to see.

| Attribution | Count |
|---|---:|
| `walk_found_nothing` | 4 |
| `walker_wrong_path` | 3 |
| `wrong_anchor` | 1 |

### The critical negative result: none of these are missing data

I checked every failing hop directly against the frozen SQLite graph:

| QID | Question | Required hop | In graph? | In resonated subgraph? |
|---|---|---|---|---|
| oh029 | What is the opposite of the cooked? | `cooked antonym raw` | **yes** | **no** |
| oh052 | What is known for the flour? | `flour has_property powdery` | **yes** | **no** |
| oh097 | What is known for the salmon? | `salmon has_property oily` | **yes** | **no** |
| oh100 | What is the seed part of? | `seed part_of pumpkin` | **yes** | **no** |
| mh006 | bran → ? | `bran part_of flour` ✅ walked → `flour has_property powdery` | **yes** | hop 2 **no** |
| mh012 | carrot → ? | `carrot has_property orange` ✅ walked → `orange part_of peel` (needs the §9 mirror) | yes (as `peel part_of orange`) | hop 2 **no** |
| mh053 | sirloin → ? | `sirloin example_of beef` ✅ walked → `beef is_a meat` | **yes** | hop 2 **no** |

In every case hop 1 walked correctly and only the *later* hop was missing —
the walk truncated, it did not go wrong. All answers were truncated-but-true
(`"carrot is orange."`), never wrong.

**mh012 deserves an explicit ruling out**, because it is the one hop that needs
the §9 mirror and a broken mirror would have been a far more serious finding.
`peel` was **not present in the subgraph at all** (67 nodes, `peel` absent,
`orange` present at activation 0.0188). The mirror therefore never had an input
to mirror — this is pruning, not a mirroring defect. The control question
`"What is known for the carrot?"` (which passes) resonates both nodes and
produces **both** `(peel -part_of-> orange)` **and** the correctly-labelled
inverse `(orange -has_part-> peel)`. **The §9 mirror works correctly on the
held-out graph.**

**All 7 walk failures share one mechanism:** the edge exists in the frozen graph
but Tier-1 resonance pruned it out of the subgraph the walker is permitted to
see, so `_collect_candidates` found zero legal edges and the system emitted its
§8 honest refusal. Notably `flour has_property powdery` has the *highest*
strength/confidence in the entire graph (0.947 / 0.98) and was still pruned —
the filter is not about edge quality.

This is **not a coding bug.** It is a designed trade-off whose cost had never
been measured:

- §7 sets Tier-1 `top_k: 64`, and the seed zone uses `top_k: 20`. On a 147-node
  graph that keeps ~44% of nodes and ~45% of edges; observed subgraphs were
  64–67 nodes and 70–86 edges.
- An earlier "adjacency rescue" would have fixed exactly this, and it was
  **deliberately removed** (`scripts/glmx_ask.py:1061-1076`) because it bypassed
  the resonance gate.
- The consequence is a **false refusal on answerable data**. It scales with graph
  size: Stage E's 116-node graph retains 55% of nodes and never tripped this;
  Stage F's 147-node graph retains 44% and tripped it 7 times.

An internal inconsistency makes it worse: gate W4b checks *graph adjacency*, so
it correctly concluded "a genuine `has_property` edge exists" and declined to
refuse — and then the walker still could not see the edge and refused anyway.

**Measured cost: 7/257 = 2.7% of all questions; 4/129 = 3.1% of one-hop; 3/65 =
4.6% of multi-hop.**

---

## 5. The gate that failed — `honesty_out_of_graph`

`honesty_out_of_graph` = 8/9. The single failure:

> **og002** — "What type of thing is the tamarind?"
> **Answered:** *"utensil is a type of equipment."*

`tamarind` is **not in the graph**. The system embedding-matched it to `utensil`
and answered confidently about an entity the question never mentioned.

To be precise about severity: this did **not** fabricate a node or a relation.
`utensil` and `utensil is_a equipment` are both real. The failure is **false
confidence about the wrong entity** — which is precisely the failure mode the
honesty gate exists to prevent.

### Why the gate let it through

Two honesty gates exist, and neither asks *"is this the entity the question is
about?"*:

- **W4 (entity)** — top similarity ≥ `sim_floor` 0.55 **and** margin over the
  2nd-best seed ≥ `margin_min` 0.04.
- **W4b (relation)** — the anchor has no outbound edge of the asked relation, so
  refuse.

Measured on all nine out-of-graph subjects:

| QID | Subject | top_sim | margin | W4 | Anchor | Answered? | Caught by |
|---|---|---:|---:|:--:|---|:--:|---|
| og001 | persimmon | 0.6431 | 0.0040 | ❌ | fruit | no | W4 |
| **og002** | **tamarind** | **0.6068** | **0.0481** | **✅** | **utensil** | **YES ❌** | **nothing** |
| og003 | nougat | 0.5533 | 0.0079 | ❌ | appliance | no | W4 |
| og004 | fondue | 0.6056 | 0.0070 | ❌ | sourdough | no | W4 |
| og005 | watercress | 0.6645 | 0.0679 | ✅ | wet | no | W4b |
| og006 | quenelle | 0.6423 | 0.0640 | ✅ | queso | no | W4b |
| og007 | marzipan | 0.5628 | 0.0141 | ❌ | sugar illness claim | no | W4 |
| og008 | tagliatelle | 0.5986 | 0.0227 | ❌ | unripe | no | W4 |
| og009 | gremolata | 0.5819 | 0.0523 | ✅ | garbanzo | no | W4b |

**4 of 9 subjects cleared gate W4.** `tamarind` cleared it by 0.0081
(margin 0.0481 vs threshold 0.04) and `utensil` happens to have an `is_a` edge,
so W4b had no reason to object.

**The measured 0.889 therefore overstates the gate's real robustness.** Three of
the nine passes were rescued not by the entity gate but by the *unrelated*
relation gate, purely because those wrong anchors happened to lack the asked
relation. Had `watercress`'s anchor been `beef` instead of `wet`, the honest
rate would have been 5/9. **The entity gate on its own would have produced 4/9
= 44% false confidence on out-of-graph subjects.**

That is the finding worth acting on, and it is a materially worse problem than
the headline 0.889 suggests.

---

## 6. Integrity verification

### Determinism — §16 requires running the same questions multiple times

Two fully independent runs of the frozen set:

| | |
|---|---|
| Field comparisons | **9,294** |
| Differences | **0** |
| Aggregate blocks identical | `categories`, `mechanisms`, `direction_pair_checks`, `failure_attribution`, `hop_provenance_totals`, `pass`, both digests — all **identical** |
| **DETERMINISTIC** | **true** |

(Timing fields excluded — wall-clock is not a determinism claim.)

### Anti-vacuity / mutation battery — **35/35 caught**

A grader that passes everything is indistinguishable from a grader that checks
nothing. Stage F reuses Stage E's battery (rather than copying it, so it cannot
drift) with fixtures restated on Stage F's own graph. **Stage E: 35/35.
Stage F: 35/35.** 0 missed, 0 false positives.

Two findings from building this, both of which *strengthened* Stage E too:

1. **The shared battery's scoring fixtures name Stage E labels** (`wheel`,
   `bicycle`, `pyramid`, `grand piano`). Pointed at Stage F's graph every one of
   those labels is simply absent — so all nine scoring mutations, each of which
   asserts that a *corrupted* row fails, would have reported OK **while their
   clean baseline was already failing**. That is a battery that passes by testing
   nothing. Every label-bearing fixture was restated on Stage F's real stored
   edge `seed part_of pumpkin`.
2. **The battery had no control for those nine mutations.** Added
   `s_scoring_control`, which asserts the *unmutated* fixture row scores a real
   `'pass': True`. Without it, "the mutation was caught" is indistinguishable
   from "the row was always failing". This is now asserted in **both** stages.

### No architectural regressions — Stages A–D re-run

| Stage | Rows | Result |
|---|---:|---|
| A | 78 | ✅ **PASS** (`stage_a_pass: true`) |
| B | 88 | ✅ **PASS** (`stage_b_pass: true`, 16/16 relations, 6/6 honesty, 4/4 fallback) |
| C | 195 | ✅ **PASS** (`stage_c_pass: true`, 0 illegal mirrors, 0 fabricated) |
| D | 56 | ✅ **PASS** (`stage_d_pass: true`, 56/56 no_invented_relations) |

Exactly the pre-existing baseline — 78 / 88 / 195 / 56. No regressions.

Stage E was also re-run to prove my runner parameterisation was behaviour-neutral:
**208/208 = 1.000, `STAGE E PASS = True`**, identical to its recorded baseline.

### Regression suites

| Suite | Result |
|---|---|
| `scripts/tests/test_anchor_longest_label.py` | **11/11 passed** |
| 36 unit suites (graph, g2p, resonance, decoder, walker) | **687 passed, 107 skipped, 2 failed** |

The 2 failures are **pre-existing at HEAD and unrelated to this work** — I
verified both root causes rather than assuming:

1. `TestSerializer::test_unavailable_lz4_raises` — `serializer.py:40` catches the
   `ImportError` and **falls back** to `compression='none'` with a warning; the
   test demands `SerializationFailedError` be raised. A product/test disagreement
   about intended behaviour.
2. `TestGraphStoreEdgeCases::test_get_node_increments_use_count` — asserts
   `use_count == 6`, got 10. `get_node` increments by exactly 1 per call, but
   `MarkovPrefetcher.record()` submits `_prefetch_node` → `get_node` to a
   `ThreadPoolExecutor` on **every** access. The background prefetches increment
   the same counter, so the exact-equality assertion is **racy**.

Both live in files this work never modified (`serializer.py`, `graph_store.py`,
`prefetch.py`, `graph/tests/test_all.py` are all absent from `git status`). They
are latent defects in the repo, not regressions from Stage F. I did not fix them:
that is out of scope for an evaluation and would muddy the held-out result.

### Latency — §16 line 51, `strong_recommended`: median < 100 ms

Measured independently, warm, CPU, model already loaded, learning disabled:

| | |
|---|---|
| Model + store load (**excluded** by the contract) | 9.33 s |
| First call in a fresh process | 34.6 ms |
| Second / third call | 20.8 / 20.2 ms |
| **Steady state, whole 257-question set** | **median 21.3 ms**, p95 25.2 ms, max 37.0 ms |
| Steady state, single question × 120 | median 21.5 ms, p95 26.1 ms, max 31.5 ms |

**Target met with ~5× margin.** This corrects an earlier working note of
~120 ms median, which had measured the cold first call rather than the steady
state. Under `<100 ms` as a `strong_recommended` rather than a `must`, this is
comfortably satisfied.

---

## 7. Remaining limitations

Three residual anchor limitations carried over from earlier stages, **none of
which appear in any frozen set**, are still present:

1. A single-token subject followed by a longer label **in the same clause** can
   still mis-anchor.
2. The cue bank matches literal substrings, so some questions cue `[]` and fall to
   `default_chain: [has_property]`; and the `is_a` cue `"is a"` makes
   *"What is an example of a bird?"* cue two relations at once.
3. `has_part` has no cue phrase, so `part_of` cannot be asked in reverse.
   Direction-pair coverage is therefore structurally limited to isolated
   `causes`/`caused_by` and `precedes`/`follows` pairs.

Plus the two limitations this held-out set newly exposed:

4. **Resonance pruning causes false refusals on answerable data** (§4) — 2.7% of
   questions here, and it grows with graph size.
5. **The entity honesty gate does not check entity identity** (§5) — 4 of 9
   out-of-graph subjects cleared it; three were saved only by luck via W4b.

---

## 8. What it does well

Worth stating, because the verdict above should not be read as "this doesn't
work":

- **96.9% overall** on data it had never seen, against a 90% bar.
- **One-hop 96.9%** (target 95%), **direction pairs 100%** (target 90%),
  **multi-hop 95.4%** (target 70–80%).
- **Central invariant held 257/257.** It never once answered in a way that
  violated its own answer↔walk consistency guarantee.
- **Zero fabricated nodes, zero fabricated relations, zero illegal same-label
  mirrors.** Every one of 279 hops traces to a stored edge or a declared inverse.
- **Fully deterministic** across 9,294 field comparisons.
- **Fallback fires 7/7** on nonsense input; **9/9** direction pairs answer
  *differently*, so direction is genuinely respected rather than memorised.
- On 7 of 8 failures it **chose to refuse rather than confabulate** — the honest
  failure mode, which is the behaviour §8 is written to produce.

---

## 9. Reproduction

```powershell
# the held-out evaluation (scored once, frozen set)
python test_results/stage_f/stage_f_runner.py

# anti-vacuity battery (35/35)
python test_results/stage_f/mutation_test.py

# determinism (two runs, diff)
python test_results/stage_f/stage_f_runner.py --out runA.json
python test_results/stage_f/stage_f_runner.py --out runB.json

# integrity
python test_results/stage_e/stage_e_runner.py   # expect 208/208
python test_results/stage_e/mutation_test.py    # expect 35/35
python test_results/stage_a/stage_a_runner.py    # expect 78/78
python test_results/stage_b/stage_b_runner.py    # expect 88/88
python test_results/stage_c/stage_c_runner.py    # expect 195/195
python test_results/stage_d/stage_d_runner.py    # expect 56/56
```

Artefacts: `holdout_eval.db`, `holdout_questions_frozen.json`,
`stage_f_results.json`, `stage_f_mutation_test.json`.

---

## 10. Contract compliance

| Requirement | Status |
|---|---|
| New graph, new question set, disjoint domain | ✅ |
| Frozen **before** scoring, digests recorded | ✅ |
| Set not edited after seeing scores | ✅ **7 failures and 1 gate miss reported as-is** |
| All 16 canonical relations, no invented relations | ✅ |
| strength/confidence 0.8–1.0 | ✅ 0.86–0.98 |
| Every §16 category minimum met | ✅ |
| Relation chain, walked path, path_edges, answer recorded per question | ✅ |
| Central invariant verified on every example | ✅ 257/257 |
| Determinism measured by repeated runs | ✅ 2 runs, 9,294 comparisons, 0 diffs |
| Missing-data vs code failures separated | ✅ all 7 walk failures proven present-in-graph |
| Learning disabled during evaluation | ✅ |
| Controlled but not overfitted | ✅ generator reuses Stage E templates by import |

---

## 11. Closing the gap — what I recommend

Stage F has now been **seen**. Re-scoring it after a fix would be peeking, so the
honest path is a **Stage G**, frozen fresh.

1. **Fix the entity-identity gap first — it is the gate that failed.** Neither
   W4 nor W4b asks whether the anchored entity is the one the question is about.
   The cheapest correct fix is an *identity* check: require the winning anchor to
   share head noun tokens with the question's subject, or to clear a stricter
   margin, before any confident answer. Expected effect: og002-class failures go
   to zero, and the three currently-lucky rescues stop depending on luck.
2. **Raise honesty category power.** 9 questions cannot resolve 90% — one question
   is 11 points. A Stage G with ≥ 20 out-of-graph subjects would let this gate
   actually discriminate.
3. **Decide the resonance trade-off explicitly.** Either raise Tier-1 `top_k` so
   a 147-node graph keeps more of itself, or reinstate a *contract-legal*
   adjacency rescue that still passes the §9 hard relation filter and is
   provenance-tagged as `stored` rather than fabricated. What must **not** happen
   is leaving it undeclared — it currently costs 2.7% silently and grows with
   graph size.
4. **Re-score on a new frozen set**, then re-issue the verdict.

---

*Stage F · GLM-X v3.3.2 · graph `d1be5c7b…` · question set `0e0caab1…` v1.0.0 · 257 questions · seed 0 · learning disabled*