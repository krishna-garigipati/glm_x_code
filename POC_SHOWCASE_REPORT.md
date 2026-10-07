# GLM-X — Proof-of-Concept Showcase Report

**System:** GLM-X v3.3.2 — deterministic graph-reasoning question answering
**Document:** `POC_SHOWCASE_REPORT.md` (single master document)
**Repository:** `glm_x_code`

---

## Status banner

| | |
|---|---|
| **PoC claim** | **CLAIMED, with stated limits** — see Part 6 |
| Evaluations run | 10 scored runs over **9 graph contents** and 10 frozen question sets — 2,563 question-runs. *Ten graph **files** exist: the showcase graph was rebuilt between run 1 and run 2, and a controlled rebuild reproduced its content hash exactly (Part 1.1), so it is counted once.* |
| Central invariant | **2,100 / 2,101 = 0.99952** held *(denominator = the 9 valid runs; the disclosed run 1 is excluded — Part 3)* |
| Fabricated nodes or relations | **0**, every run, every stage |
| Verdicts | **7 PASS / 3 NOT PASSED** — the 3 failures are reported unchanged |
| Final new-data demo | **PASS**, 445/462 = 0.9632 (pottery & ceramics, frozen before scoring) |
| Defects known and unpatched | **11** — see Appendix A |

---

## How to read this document

Three conventions are used throughout, and they are not decoration:

1. **Every number is read from a result file in this repository.** No number in
   this document is estimated, rounded up, or carried over from memory. Where a
   figure could not be measured, it says *not measured*.
2. **Failures are reported at the same prominence as successes.** Two prior
   verdicts are **NOT PASSED** and are printed in the same table as the passing
   ones. The prior aviation sign-off is *referenced, not restated or amended* —
   it lives at `test_results/final_validation/FINAL_POC_REPORT.md` and is not
   reopened here.
3. **Nothing was patched after a score was seen.** Where a defect was found by an
   evaluation, it is listed in Appendix A and left in the code. The single
   exception to "nothing changed between runs" is disclosed in full in Part 4.3,
   because hiding it would be worse than the defect itself.

**Terminology.** A *question* is one row of a frozen set. A *hop* is one edge
traversal. A **verdict** is the PASS/NOT PASSED flag on a whole run, decided by
pre-registered criteria inside the frozen artefact.

---

# Part 1 — System overview

## 1.1 What GLM-X is

GLM-X answers natural-language questions by walking an **explicit knowledge
graph**. Every fact it asserts comes from an edge it actually traversed. It has
no parametric world knowledge, it calls no external model at inference, and when
it cannot walk to an answer it says so rather than inventing one.

The operative idea is a single sentence:

> ### `relation_chain` ≈ `walked path` ≈ `generated sentence`

- **`relation_chain`** — the ordered list of relations the *plan* stage extracted
  from the question's literal cue words (`["part_of", "is_a"]`).
- **`walked path`** — the sequence of edges the *walker* actually traversed
  (`path_labels` / `path_edges`), each hop carrying a `provenance` of `stored`,
  `declared_inverse`, or — never observed — `fabricated`.
- **`generated sentence`** — the decoder's output, rendered *from the walked
  relations, not from the plan*.

That last clause is the load-bearing one. `scripts/glmx_ask.py:1234`:

```
[6/6] Decoder: walked relations %s differ from planned %s;
      rendering from walked relations to avoid asserting an unwalked relation
```

When the walk falls short of the plan, the decoder **renders the shorter truth**
instead of the planned claim. The system is built so that its worst failure mode
is *an incomplete honest sentence*, never *a fluent false one*.

## 1.2 The six-stage pipeline

Stages are enumerated by `steps_log` in `GLMXPipeline.ask()`
(`scripts/glmx_ask.py`, method starts line 891):

| # | `steps_log` key | Stage | Implementation | Key constants (source) |
|---|---|---|---|---|
| 1 | `1_encode` (l.898) | **Encode** — SBERT embed the question | `configs/config_core.yaml:84` | model `BAAI/bge-small-en-v1.5` (`config_core.yaml:84`); 384-dim, hard-coded at `glmx_ask.py:407` |
| 2 | `2_subgraph` (l.1024) | **Anchor** — exact-label match + seed subgraph | `get_subgraph_by_embedding_similarity` | seeds `top_k=20` (`glmx_ask.py:902`); anchor floor `0.55`, margin `0.04` (`config_orchestrator.yaml:8-9`); `entity_not_found < 0.25` (`:21`, used `glmx_ask.py:653`) |
| 3 | `3_resonance` (l.1055) | **Spread / Resonance** — activation spreading | `resonance/tier1.py::resonate` | contract §7: `top_k=64`, `max_iterations=4` (**hard-coded** `glmx_ask.py:288-291`; YAML copy `config_resonance.yaml:39-40` — the YAML itself warns the ask path does not read it, `:36-38`); tier2 `top_k=1024`, `max_iterations=8` (hard-coded `glmx_ask.py:316-319`, YAML `:72-73`) |
| 4 | `4_plan` (l.1063) | **Plan** — extract `relation_chain` | `g2p/g2p_planner.py::QueryRelationExtractor` | cue bank `extraction.relation_variants`; `max_chain_length: 3` (`config_g2p.yaml:45`); `default_chain: [has_property]` (`:57`) |
| 5 | `5_walk` (l.1201) | **Walk** — traverse the resonated subgraph | `walker/graph_walker.py` | `min_activation: 0.01` (`config_walker.yaml:28`) |
| 6 | `6_decode` (l.1263) | **Decode** — render the sentence | `decoder/template_decoder.py` | `CHAIN_TEMPLATES` (25) / `RELATION_PHRASES` (33) from `decoder/config_decoder.yaml` |

A seventh key, `6b_reinforce` (l.1358), exists and is **gated off** in inference
— see component 10.

> **Honest note on the log labels.** The counter is `[2/6] … [6/6]`. There is no
> `[1/6]` log line: stage 1 (the SBERT encode) times into `steps_log["1_encode"]`
> but emits nothing to the log. That is a minor observability gap in the
> pipeline's own logging, not a missing stage — the timing key is written at
> `glmx_ask.py:898`.

Below is a **verbatim capture** from a live run of this demo's frozen set
(question `mh001`, run against `showcase_eval.db`; the database sha256 and the
results file were both verified unchanged afterwards):

```
[2/6] Anchor: exact_hit=True (in seed zone) -> trusted
[2/6] Embedding path: | GraphStore: 118 nodes, 20 seeds| Target entity: [4]
[3/6] Resonance: 76 nodes, energy=2.7258, tier=tier1, 4 iterations
[4/6] Plan: chain=['linguistic_maps', 'is_a'], heuristic_fallback=False, confidence=1.0000
[5/6] Walker: 2 steps, chain_used=['linguistic_maps', 'is_a'], confidence=0.8763,
      path=['argile', 'clay', 'ceramic material']
[6/6] Decoder: template_ok=True, answer_len=55,
      answer_start='argile relates to clay, and clay is a ceramic material.'
```

Every figure in that log matches the row frozen in `showcase_results.json`
exactly: `n_resonated_nodes` 76, `plan_confidence` 1.0, `walk_confidence`
0.8763, `n_walk_steps` 2, `path_labels` `["argile","clay","ceramic material"]`,
and the full answer text. The `[6/6] Decoder: walked relations … differ from
planned …` line at `glmx_ask.py:1234` is **absent** here — it only prints when
the two disagree, which on this row they did not.

## 1.3 What is *not* in the pipeline

- **No external LLM.** Inference uses a local sentence-transformer
  (`BAAI/bge-small-en-v1.5`, configured at `config_core.yaml:84`,
  `config_g2p.yaml:29`, `model_training/config.yaml:2`) and a string-template
  decoder. There is no network call, no API client, and no generative model in
  the answering path.
- **No cosine similarity in the main walker scoring.** The walker scores by
  graph traversal against the plan's `relation_chain`; activation similarity
  enters only at encode/anchor, before the walk. This is enforced, not merely
  intended: the scoring formula is hard-required to contain
  `strength * confidence * target_activation * relation_bias`
  (`walker/graph_walker.py:126`, required terms `:135`) and the loader
  **rejects** any formula containing `target_similarity`, `cosine` or
  `intent_bias` (`:141-147`).
- **No learning at inference** — component 10.
- **No invented relations.** Measured at 0 in every run; see Part 3.

## 1.4 What is *not* claimed

GLM-X does not do open-domain QA, does not read text at inference, does not
generalise beyond the graph it was given, and does not currently complete every
multi-hop walk it plans (Part 5, component 3). Its value is a *provably honest
traversal*, not knowledge.

---

# Part 2 — Component-by-component showcase

Each component below is described in the same order: **Purpose · How it works ·
Mechanisms · Evidence · Measured behaviour · Limitations.** All measured
behaviour is from `test_results/showcase_demo/showcase_results.json` (the
pottery demo, Part 4) unless another source is named.

---

## Component 1 — Graph Store

**Purpose.** Persist nodes, edges and embeddings so that a graph can be frozen,
hashed, re-loaded and proven unchanged across a run.

**How it works.** `graph/graph_component_implementation/sqlite_graph_store.py`
implements `SQLiteGraphStore`. It is loaded with `load_state(path)` (l.212) and
saved with `save_state(path)` (l.152).

Observed schema on the frozen demo database:

| table | columns | rows |
|---|---|---|
| `nodes` | `id, label, node_type, activation, use_count, create_time, sense_id` | 147 |
| `edges` | `source_id, target_id, relation, strength, confidence` | 193 |
| `embeddings` | `node_id, vector` | 147 |
| `metadata` | `key, value` | 10 |

`metadata` keys observed: `contract`, `dataset_name`, `edge_count`, `embed_model`,
`embedding_dim`, `frozen`, `merge_journal`, `node_count`, `saved_at`, `version`.

**Mechanisms.**

- **Every node carries an embedding**, 147/147 — no node exists that cannot be
  reached by the encode stage.
- **`protected_labels`** (`protect_labels`, l.369) marks labels that must not be
  merged away, so an evaluation's anchor vocabulary cannot be silently absorbed.
- **`merge_journal`** metadata records merge operations — the graph's edit history
  is auditable rather than implicit in the final edge set.
- **`frozen`** metadata is the flag the evaluators pin against.

**Evidence.** The demo graph's two identities are separated deliberately:

- **Content hash** (sha256 over sorted `id→label` and sorted
  `(source_id, target_id, relation, strength, confidence)` tuples):
  `0d6edf6eb6d78ac45230abc484f37f214ffb26bccbbe460babe6a92155680c48`
- **File sha256**: `e465e0707ae9b0f78dff617f365cd03ad15dbd274af7ee5c9af1d86cc7378d2e`

**Measured behaviour.**

- The content hash of the **frozen graph and of a fresh, independent rebuild are
  identical**: `0d6edf6e…` on both. The rebuild was run to a *temp path* with
  `DB_PATH` redirected, so none of the frozen artefacts were touched, and
  `graph_content_hash.py` (Appendix B) was used to compare. The graph is
  content-reproducible.
- File bytes are **not** reproducible across rebuilds — `saved_at` metadata and
  per-node `create_time` are wall-clock stamps. Three builds on record give three
  different file hashes: `e12d7610…` (run 1), `e465e070…` (run 2, the frozen
  one), `995475b5…` (the controlled probe). The probe differed from the frozen
  file in exactly one respect: the `create_time` sample moved
  (`1791372915…` → `1791378324…`); `activation_sum 73.5` and `use_count_sum 0`
  were unchanged.
- *Honest limits of the above:* run 1's database was overwritten by the rebuild,
  so **its** content hash cannot be re-measured today — only its file hash and
  its node/edge/relation counts survive (147 / 193 / 16-of-16, identical to the
  frozen set, Part 4.3). So the claim is "content hash reproduced on the two
  builds still measurable", not "on all three".
- Consequently the pin that actually protects the evaluation is the *file* hash,
  written once at freeze time and checked by `run_eval.py` **before** the grader
  is entered and **after** the run completes. On this demo the hash was
  `e465e070…` before and after a 27-question diagnostic re-run — **unchanged**.

**Limitations.**

- Byte-level non-reproducibility means a rebuilt graph cannot be substituted for
  a frozen one without detection (good) but also cannot be rebuilt identically
  (an inconvenience). The workaround is "build once, never rebuild", which is
  what the demo does.
- `activation`, `use_count` and `create_time` are mutable runtime columns living
  in the same table as the frozen content; they are excluded from the content
  hash by design, but a naive `sha256(db)` would include them.

---

## Component 2 — Encode

**Purpose.** Turn a question and a node label into the same vector space so an
anchor can be found by similarity.

**How it works.** A local sentence-transformer embeds the question (`1_encode`,
`glmx_ask.py:898`); nodes are embedded at graph-build time and stored in the
`embeddings` table. Anchoring then does an exact-label lookup first and falls
back to nearest-neighbour similarity.

**Mechanisms.**

- **Exact-label-first.** `anchoring.prefer_exact_label: true` and
  `exact_label_trusted: true` (`config_orchestrator.yaml:22-23`). A literal match
  is trusted *only if it is inside the seed zone* — `glmx_ask.py:926` logs
  `exact_hit=True (in seed zone) -> trusted`, and `:930` logs
  `exact_hit=True but node outside seed zone -> audit`. This prevents a lucky
  string match outside the retrieved neighbourhood from hijacking the answer.
- **Similarity fallback** with a floor of `0.55` and a margin-over-second of
  `0.04` (`config_orchestrator.yaml:8-9`), consumed at `glmx_ask.py:916-917`.
- **`entity_not_found`** fires when `anchored_sim < 0.25`
  (`config_orchestrator.yaml:21`, `glmx_ask.py:653`).
- **Seed priming** floors the target entity at `0.9` and the top-3 seeds at `0.8`
  (`config_orchestrator.yaml:26-30`).

**Evidence.** The audit line's format is `glmx_ask.py:938-939`:
`[2/6] Anchor audit: anchored_sim={:.4f} margin={:.4f} (floor={sim_floor},
min_margin={margin_min})`, with `sim_floor=0.55` and `margin_min=0.04` supplied
from `glmx_ask.py:916-917`. Its inputs are recorded per row in
`selected_anchor.entity_top_sim` / `selected_anchor.anchor_margin`, so the
auditable evidence is in the results file rather than only in the log.

**Measured behaviour.**

- `exact_label_hit` on **416 / 462 = 0.9004** of all rows.
- On those 416 exact hits, `entity_top_sim` spans **0.7093 – 0.9423 (median
  0.8421)** — every one of them above the `0.55` floor. `anchor_margin` is `null`
  on exact hits, since no second candidate competes.
- Of the **46** rows that were *not* an exact label hit, **44 passed** — i.e.
  similarity anchoring is not a fallback that mostly fails: it is 44/46.
- **4 rows** scored `entity_top_sim < 0.55`; **25 rows** scored
  `anchor_margin < 0.04`.
- `anchor_identity_grounded` on **436 / 462**.
- The 26 ungrounded rows break down as **23 `honesty_out_of_graph`** + 2
  `nonsense_fallback` + 1 `simple_one_hop` — the honesty subjects dominate, and
  are *supposed* to be ungrounded (component 7).

**Limitations.**

- The identity gate grounds on **token overlap, not entity identity**. An
  out-of-graph multi-word subject containing a real label still resolves to a
  real node — measured on the aviation set (`og013`, 1/22 honesty) and repeated
  here: on all 23 honesty rows the anchor landed on an in-graph node with reason
  `ungrounded:<label>` (component 7).
- `entity_not_found` fired **0 / 23** on the honesty rows: the signal meant to
  say "this entity does not exist" never fired, even though the system then
  refused correctly. Right outcome, wrong signal — carried as Appendix A-9.

---

## Component 3 — Spread / Resonance

**Purpose.** Decide, before the walk, *which part of the graph the walker is
allowed to see*.

**How it works.** `resonance/tier1.py::Tier1Resonance.resonate(q_emb, store,
seed_nodes)` returns `(subgraph, history)`. Activation is propagated for
`max_iterations=4`, gated by a `top_k` cut, and then seed neighbours are
readmitted. The walker may only traverse nodes in the returned subgraph.

**Mechanisms.**

- **Contract §7 budget**: `top_k: 64`, `max_iterations: 4`. The value the ask
  path actually uses is the one **hard-coded** at `glmx_ask.py:288-291`
  (`TierConfig(..., top_k=64, max_iterations=4)`), not the YAML — a distinction
  the YAML itself flags: *"the ask path does NOT read this value … Keep in sync
  with that, or the two silently diverge"* (`config_resonance.yaml:36-38`).
  The two currently agree (`:39-40`).
- **`_gate`** (`tier1.py:407-419`) keeps the top `top_k` nodes by activation:
  `dict(sorted_nodes[: self._tier.top_k])`.
- **`_readmit_seed_neighbourhood`** (`tier1.py:216`) runs *after* `_gate` and
  puts a seed's immediate neighbours back **with no subsequent re-cut**. Its
  own docstring states the scope limit: *"readmission is ONE-HOP from the seeds.
  It cannot rescue a second hop, because when the walk's hop-1 endpoint is not
  itself a seed, that endpoint's own neighbours stay prunable."* (`:239-241`)
- **`_propagate`** normalises, snapshots to `pre_gate`, then applies `_gate`.
  Readmission reuses `pre_gate` values verbatim — no activation is invented
  (`tier1.py:250-252`).
- **Tier 2** runs only when Tier 1 energy is low — the actual condition is
  `if self.tier2 is not None and resonated.activation_energy <
  self.tier2_energy_threshold` (`glmx_ask.py:1034`, documented at `:1029-1030`).
  Its budget is likewise hard-coded at `glmx_ask.py:316-319`:
  `top_k=1024`, `max_iterations=8`, analogy enabled (matching
  `config_resonance.yaml:72-73`).
  On this demo the sampled run logs show `tier=tier1` with Tier 2 not invoked.
  *Not verified per-row:* `tier` is not a field in the results JSON, so this
  report does not claim all 462 rows resolved at tier 1 — only the ones observed
  in the logs.

**Evidence — measured, not inferred.** A diagnosis-only probe (no re-scoring, no
writes; the results JSON was read and never modified) instrumented
`Tier1Resonance.resonate` to capture the node set handed to the walker, then
replayed the 15 truncated multi-hop rows and 12 passing controls:

| | final hop target present in resonated subgraph |
|---|---|
| 15 failing multi-hop rows | **absent — 15 / 15** |
| 12 passing multi-hop controls | **present — 12 / 12** |

Distance of that final target from the nearest seed, over undirected stored-graph
adjacency:

- failures: `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]` — **all ≥ 2**
- controls: `[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]` — **11 of 12 at ≤ 1** (the
  twelfth is 2)

Database sha256 was `e465e070…` **before and after** the probe — unchanged.

This is a **second, independent instrumentation of the same root cause** — the
first was on the weaving set (11/11 absent, 6/6 present,
`UNSEEN_E2E_REPORT.md` §3.1). Two different graphs, two different question sets,
the same instrument, the same clean separation.

**Measured behaviour (budget).**

| metric | value |
|---|---|
| `n_resonated_nodes` min / p50 / max / mean | 66 / 74 / 86 / 74.1 |
| rows exceeding the contract's `top_k: 64` | **462 / 462 = 1.0000** |
| effective ceiling observed | **86** |

**Limitations — both are Appendix A entries, both left unpatched.**

- **A-3 (the genuinely broken mechanism).** The resonance subgraph excludes the
  final hop's target whenever that target sits 2–3 hops from every seed, because
  readmission is one-hop from the *original* seeds and `_propagate` under-ranks
  it. This is an **ordering-and-depth** defect, not a budget ceiling: the 17
  failing rows resonated **71–80 nodes** — comfortably inside the observed
  66–86 spread, so they were not starved of budget. It is the single largest
  source of error in the demo — **15 of 17 failures**.
- **A-1 (budget overrun).** The contract's `top_k: 64` is not the true bound; the
  effective ceiling is `top_k + |initial_seeds|`. On aviation this was 313/355
  (88.2%) rows over 64, max 84; here it is 462/462, max 86.

**Credit where due.** In all 15 truncations the walker stopped cleanly and the
decoder declined to assert the unwalked relation. The system truncated rather
than fabricated.

---

## Component 4 — Plan

**Purpose.** Translate the question into an ordered `relation_chain` — the
contract the walk will be held to.

**How it works.** `g2p/g2p_planner.py::QueryRelationExtractor.extract()` matches
the question against a **literal descriptor bank**: `extraction.relation_variants`
in `configs/config_g2p.yaml`. `_literal_cue_relations` is pure regex over that
bank and loads no model (`stage_d_runner.py::load_planner` docstring).

**Mechanisms.**

- One cue bank per canonical relation; 16 canonical relations
  (`antonym, associated_with, caused_by, causes, contradicts, example_of,
  follows, has_property, is_a, linguistic_maps, part_of, precedes, spatial_near,
  supports, synonym, temporal_coincident`).
- `max_chain_length: 3` (`config_g2p.yaml:45`) caps an ordered chain at 3 hops.
- `collapse_consecutive_repeats: true` (`:46`).
- `clause_split` on `which/what/how/why/when/where/because/since/", "/" and "`
  (`:60-74`) — note `"that"` is deliberately excluded, with the reason written in
  the config (`:62-65`).
- **Cue absence → fallback.** `fallback.enabled: true`,
  `fallback.flag: "heuristic_fallback_used"`, `fallback.default_chain:
  [has_property]` (`config_g2p.yaml:54-58`). The contract's `preferred_trigger`
  is reproduced in the config at `:51-53` (the config notes it is guidance
  "implemented in `g2p/g2p_planner.py` as cue-absence, not as a config enum",
  `:49-50`) together with the contract's own warning, transcribed here exactly as
  the config spells it: *"Do not rely only on similarity score
  threshold - scores of real and nonsense questions overlap."*

**Evidence.** Every template in the demo's question bank was asserted against the
**live** `_literal_cue_relations` at freeze time, both as a first clause and as a
continuation clause behind a `causes` first clause, and every generated row was
re-checked. A template that stops emitting its own declared relation refuses the
freeze. The freeze-time probe caught real template bugs before scoring.

**Measured behaviour.**

- **Plan confidence 1.0000** on every non-fallback row — all **455** of them,
  min = max = 1.0000.
- **Plan confidence 0.6000** on all 7 fallback rows, and on no others.
- Relation coverage: **all 16 canonical relations** appear in stored edges, in
  the `rel` field, and in at least one `relation_chain`. `simple_one_hop` covers
  **12** of the 16 (smallest cells `contradicts` 3/3, `supports` 3/3,
  `example_of` 5/5, largest `is_a` 75/75); the remaining four — `causes`,
  `caused_by`, `follows`, `precedes` — are exercised through `direction_pair`,
  which scored **28/28**.

**Limitations.**

- **The cue bank matches literal substrings**, so some natural phrasings cue `[]`
  and fall through to `default_chain: [has_property]`. Disclosed in the aviation
  report §8.4.
- **`has_part` has no cue phrase**, so `part_of` cannot be asked in reverse
  (aviation §8.5). The reverse exists only as a runtime mirror label for walking.
- **Preflight can prove a question is *answerable*, not that it is *true*.** This
  is a real gap in the preflight design and it produced 40 semantically false
  questions in the aviation set (Appendix A-2). The demo closes it by
  construction — its one language-naming template is restricted to English-side
  anchors and checked on every generated row — but that is a construction fix,
  not a general preflight capability.

---

## Component 5 — Walk

**Purpose.** Traverse the resonated subgraph along the planned chain and produce
`path_labels` / `path_edges`, each hop tagged with provenance.

**How it works.** `walker/graph_walker.py`, invoked at `glmx_ask.py:1200` (logged at `:1202` as
`[5/6] Walker: …`). The
walker steps from the anchor following the plan's relations, subject to
`min_activation` and to membership in the resonated subgraph.

**Mechanisms.**

- **`min_activation: 0.01`** (`config_walker.yaml:28`) — a candidate below this
  is not walkable even if present.
- **Declared inverses only.** `walker/graph_walker.py:33-34`:
  `"part_of": "has_part"`, `"has_part": "part_of"` (plus `causes↔caused_by`,
  `precedes↔follows`). Mirroring is permitted *only* for these pairs — enforced
  by `MIRROR_PERMITTED` in `stage_c_runner.py:77`.
- **`has_part` is a runtime-only label.** It is never stored — the runtime mirror
  synthesises it at walk time (contract §9). Two guards exist where mirroring
  happens. `validate()` appends `("relation-not-canonical", "has_part is a
  runtime mirror label")` (`stage_f/build_holdout_graph.py:679-680`), and the
  showcase, aviation and weaving builders inherit that check because they import
  that module as their algebra and call `B.validate()` (`showcase_demo/build_graph.py:69-73`,
  `:464`). The build path then raises `REFUSING TO FREEZE: has_part is a runtime
  mirror label` (`stage_f/build_holdout_graph.py:845`,
  `stage_e/build_poc_graph.py:797`,
  `stage_d/build_honesty_fallback_graph.py:562`,
  `stage_c/build_direction_multihop_graph.py:656`). Stages A and B carry no such
  guard of their own — which is why the claim below rests on **measurement**:
  `SELECT DISTINCT relation FROM edges` returns **no `has_part`** in any of the
  27 SQLite databases in the repository (26 have an `edges` table; 0 contain it).
- **Hop provenance** is recorded per hop as `stored` / `declared_inverse` /
  `illegal_same_label_mirror` / `fabricated`.

**Evidence.** `hop_provenance` totals for the demo: `{"stored": 633,
"declared_inverse": 13}` — **zero** `illegal_same_label_mirror`, **zero**
`fabricated`. `walk_used_inverse_label` on 48 rows (48/48 passed).

**Measured behaviour.**

| measure | value |
|---|---|
| fabricated hops (whole run) | **0** |
| illegal same-label mirror hops | **0** |
| leaked forbidden labels | **0** |
| silent wrong answers (neither claim nor refusal) | **0** |
| mirror-silence rows honoured (forbidden far end never named) | **39 / 39 = 1.000** |
| direction pairs whose two ends answered differently | **14 / 14 = 1.000** |
| rows where the walk completed every step it committed to, among failures | 2 |
| rows where the walk **stopped short** of its own chain, among failures | **15** |

**Limitations.**

- **The walk cannot traverse past the seed zone's one-hop shell.** This is the
  resonance defect surfacing (component 3), not an independent walker fault: the
  walker stopped rather than inventing, and the decoder then rendered the shorter
  truth. Still, from the user's point of view, **7.5% of multi-hop questions
  (15/200) got a partial answer.**
- **One rendering defect:** a mirrored `causes` step surfaced as `caused_by` in
  the weaving set (`mh042`, Appendix A-4). Not reproduced on this demo.
- **`expected_path_order` was the only semantic check to fail** besides
  `no_claim_when_refusing`: 15 failures, exactly the 15 truncations.

---

## Component 6 — Decode

**Purpose.** Turn the walked path into an English sentence that asserts exactly
what was walked.

**How it works.** `decoder/template_decoder.py::TemplateDecoder`.
`render_chain()` (l.128) first tries `_select_chain_template` — an **exact match
on the walked chain** — and falls back to `_render_chain_path` (l.203) if there
is none. Refusals go through `render_no_relation()` (l.147).

**Mechanisms.**

- `CHAIN_TEMPLATES` (25 entries, loaded from `decoder/config_decoder.yaml` by
  `load_chain_templates()`, `stage_b_runner.py:66`) keyed by exact relation tuple.
  Present keys include
  `('part_of',) → "{node0} is part of {node1}."`,
  `('part_of','is_a') → "{node0} is part of {node1}, which is {node2}."`,
  `('has_property',) → "{node0} is {node1}."`,
  `('causes','part_of') → "{node0} {relation0} {node1} because it is part of {node2}."`.
- `RELATION_PHRASES` — **33** entries from the same file
  (`load_relation_phrases()`, `stage_b_runner.py:77`). Verified values:
  `part_of → "is part of"`, `is_a → "is a"`, `causes → "causes"`,
  `has_property → "has"`, **`has_part → "has part of"`**.
- **Refusal text** (`render_no_relation`, l.147-158):
  `"I don't have a relation in my knowledge graph that fully answers this
  question."` + optional `" Closest concepts I have: {names}."` + optional
  `" (No {relations} relation found.)"`.
- **The decoder renders from walked relations, not from the plan**
  (`glmx_ask.py:1232` — `decode_chain = list(walk.path_edges) or chain`, then
  passed at `:1237`), which is what makes the central invariant checkable.

**Evidence.** Central invariant held **462 / 462 = 1.000**, with **zero** failed
checks across all six sub-checks. `template_correctness` = **1.000** against a
target of 0.95. `template_matched` false on **0** rows.

**Measured behaviour.**

| measure | value |
|---|---|
| central invariant held | **462 / 462** |
| `chain_matches_walk`, `final_node_stated`, `no_unwalked_claim` failures | **0** |
| `refusal_is_explicit`, `refusal_offers_no_fabricated_concept`, `guess_disclosed` failures | **0** |
| template correctness | **1.000** (target 0.95) |
| rows with a *surface-form grammatical* defect | **21 / 462 = 0.0455 — all 21 passed** |

**Limitations — a new finding from this evaluation, Appendix A-8.**

The decoder's surface forms are mechanically derived from `_relation_phrases`
inserted into `"{node0} {phrase} {node1}"` (`template_decoder.py:214-215`), and
two of those phrases do not survive the insertion:

1. **`has_part → "has part of"` produces ungrammatical sentences.** Because
   `has_part` has **no** `CHAIN_TEMPLATES` key (verified: only `part_of` keys
   exist), every walk that uses the inverse label takes the `_render_chain_path`
   path:
   > *"bowl **has part of** rim, and rim is a component."*

   **12 such answers, 12/12 passed.**

2. **`is_a → "is a"` is inserted before a vowel-initial label with no article
   adjustment:**
   > *"terracotta **is a** earthenware"*, *"kiln **is a** equipment"*

   **9 such answers, 9/9 passed.**

The grader does not catch either, because the semantic checks test *label
ordering and provenance*, not English grammar. **21 broken sentences were scored
as correct.** This is a genuine quality gap in the decoder-plus-grader pair, it
was found by this evaluation, and it is **not patched** — see Appendix A-8.

---

## Component 7 — Honesty gates

**Purpose.** Guarantee that when the graph does not support an answer, the system
refuses instead of guessing.

**How it works.** Four gates are computed inside `GLMXPipeline.ask()`
(`scripts/glmx_ask.py`); nothing else in the repository sets them.

| gate | set at | condition |
|---|---|---|
| `entity_not_found` | `:653` | `anchored_sim < 0.25` (`config_orchestrator.yaml:21`) |
| `honest_by_entity` (W4) | `:918-941`, threshold at `:937` | `entity_top_sim < sim_floor or margin < margin_min`, with `sim_floor=0.55`, `margin_min=0.04` (`:916-917`) |
| `honest_by_relation` (W4b) | `:1099` | the anchor has no edge of the asked relation — logged `[4/6] Honesty: anchor {id} has no '{rel}'` |
| `honest_by_identity` (W4c) | `:960` | the anchor was **not grounded in the text of the question** — logged `[2/6] Identity: anchor NOT grounded in the question` |

`anchor_identity_grounded` / `anchor_identity_reason` are the auditable
projections of W4c.

**Mechanisms.** A refusal is *explicit text*, not an empty string: it names the
relations it could not find and lists the closest concepts it does hold. It
carries **no relation claim** — that is exactly what `no_claim_when_refusing`
tests, and what `claim_text()` in `stage_d_runner.py:220-231` separates from the
disclosure tail so that naming a node while declining to answer is not mistaken
for asserting it.

**Evidence.**

| gate | demo run |
|---|---|
| out-of-graph rows refused | **23 / 23 = 1.000** |
| `honest_by_identity` | **23 / 23** |
| `honest_by_entity` | 22 / 23 |
| `honest_by_relation` | 0 / 23 |
| `entity_not_found` | **0 / 23** |
| `anchor_identity_grounded` | 0 / 23 |
| `exact_label_hit` | 0 / 23 |
| refusal rows across the whole run | 69 (39 mirror + 23 honesty + 6 nonsense + 1 one-hop) |
| refusal rows that passed | **67 / 69** |

Every refusal reason was `ungrounded:<label>` — e.g. `ungrounded:figurine` ×4,
`ungrounded:stoneware` ×2, `ungrounded:tableware` ×2, and one each for
`stoker, argile, trimming, shaft, glazing, handicraft, porous body, flux mineral,
object, engobe, rim, teapot, jug, grog, round disc`.

**Measured behaviour.** **22 / 23 = 0.9565** against a target of 0.90 — meets
target. The one failure is `og018`, and it is a **grader** failure, not a system
failure (Appendix A-7): the refusal was correct, and `no_claim_when_refusing`
flagged the substring `is` inside "th**is** question" of the standard refusal
boilerplate.

**Limitations.**

- **Right outcome, wrong signal.** `entity_not_found` was **0/23**. Each
  out-of-graph subject grounded on an unrelated in-graph node and was refused
  anyway — by the identity gate, not the entity gate. The system behaved; the
  telemetry did not. Carried unchanged from the weaving run (Appendix A-9).
- **The refusal-cleanliness metric is defective.** Disclosed on weaving
  (Appendix A-6) and seen again here in its chain-template variant (A-7). The
  row-level field `template_fragments_leaked` is `null` on **56 of the 69**
  refusals — computed on only **13**, and those 13 are *exactly* the refusals
  that walked a path (set identity verified: `walked >= 1 edge` and
  `fragments computed` are the same 13 rows). Of those 13, **11 had an empty
  fragment list** and **2 had `["is"]`** — and the flag itself was a false
  positive both times (`pv002` had also genuinely failed A-5; `og018` had not).
  So the check that is supposed to prove a refusal claims nothing never ran on
  the 56 pure refusals, and the only two times it ran with a non-empty fragment
  it flagged both.
- **The refusal boilerplate can contradict its own walk** (Appendix A-11): 13 of
  the 69 refusals say *"(No `is_a` relation found.)"* while their own
  `path_edges` contains `is_a`.
- Both are **left unpatched**.

---

## Component 8 — Heuristic fallback

**Purpose.** When the question contains no relation cue at all, still produce an
answer — but label it as a guess rather than as graph-derived fact.

**How it works.** Cue absence (not a similarity threshold — see the contract
warning quoted at `config_g2p.yaml:50-53`) triggers
`fallback.enabled`, sets `heuristic_fallback_used`, and applies
`default_chain: [has_property]`. The decoder then prefixes the answer with the
disclosure string.

**Mechanisms.**

- **Disclosure text**, `config_orchestrator.yaml:17`:
  `"Based on a heuristic guess (no relation cue matched):"` — rendered as a
  prefix on answers produced from `default_chain`, and explicitly **not** on
  honesty-gate refusals, because *"a refusal is not a guess"* (`:14-15`).
- **Plan confidence drops to 0.6000** on the fallback path — a visible,
  machine-readable downgrade rather than a silent one.
- **`early_chain: ["has_property"]`** (`config_orchestrator.yaml:46`) is the
  zero-anchor guard's chain.
- Two rescue mechanisms that would have bypassed the resonance gate — `chain_lift`
  and `rescue` — were **removed**, with the measurement recorded in the config
  (`config_orchestrator.yaml:38-43`): *"Measured across Stage A (78) and Stage B
  (88): firing on 1 question and changing 0 answers, so they bought no recall
  while weakening a real gate."*

**Evidence — the two distinct behaviours, both required by the contract.**

*Cue-free nonsense → refusal with disclosure of what IS held:*
```
Q:  How many edges does the number four have?        [nf001, cue-free]
chain=['has_property']  anchor='four' (exact_label)  plan_conf=0.6  walk: 0 steps
A:  I don't have a relation in my knowledge graph that fully answers this
    question. Closest concepts I have: four. (No has_property relation found.)
```
*(Note: `four` is a real graph node, node id 53 — the French counterpart of
`kiln` in the stored `linguistic_maps` relation (`build_graph.py:391`) — so the
anchor resolved legitimately; the walk then found no `has_property` edge and
refused.)*

*Disclosed guess → fallback fires AND the guess is labelled:*
```
Q:  How wooden is the workbench?                    [nf007, disclosed guess]
chain=['has_property']  anchor='workbench' (exact_label)  plan_conf=0.6
path = workbench --has_property--> wooden            provenance: stored
A:  Based on a heuristic guess (no relation cue matched): workbench is wooden.
```

**Measured behaviour.**

| measure | value |
|---|---|
| fallback fired | **7 / 7 = 1.000** (target 0.90) |
| `heuristic_used` on any row outside `nonsense_fallback` | **0** |
| plan confidence on fallback rows | **0.6000**, and only those rows |
| disclosure present on the row that requires it | **1 / 1** |
| `nonsense_fallback` category | **7 / 7 = 1.000** |

**Limitations.**

- Disclosure is required on **one** row by the pre-registered criteria
  (`guess_disclosure_required_on_disclosed_guess_row: True`), because the other
  six are cue-free refusals, not guesses. The contract does not ask a refusal to
  disclose a guess it did not make.
- The demo's `nonsense_fallback` is the category that produced **run 1's NOT
  PASSED verdict** — for a reason that was my harness's fault, not the system's.
  That is disclosed in full in Part 4.3.

---

## Component 9 — Determinism controls

**Purpose.** Make a run repeatable so that a score means something.

**How it works.** Seed control, frozen artefacts, two independent SHA-256 pins,
and a grader imported by file path rather than copied.

**Mechanisms.**

- **`seed: 0`**, applied as `random.seed(args.seed)` and `np.random.seed(args.seed)`
  (`stage_e_runner.py:656-657`) and `pipeline._seed = args.seed` (`:684`).
- **Two independent pins, both written before scoring.** `FROZEN_SET_SHA256` is
  handed to the shared grader; `FROZEN_GRAPH_SHA256` is checked against the
  database *before the grader is entered*. The second pin exists because the
  graph hash also lives **inside** the question file — pinning only the set would
  let an edited set verify itself.
- **Grader imported, not copied.** `run_eval.py` loads
  `stage_e/stage_e_runner.py` by explicit file path and rebinds only data
  locations. Copying a runner is how two "independent" evaluations stop measuring
  the same thing.
- **Post-run integrity.** The graph hash is re-verified after scoring.
- **`pipeline._measure = True`** records per-row instrumentation without changing
  the decision path.

**Measured behaviour — the demo's determinism re-run.**

| measure | value |
|---|---|
| rows compared | 462 |
| row fields | 47 total, **46 compared** (`time_seconds` excluded — see below) |
| **field-value differences** | **0** |
| total comparisons | **462 × 46 = 21,252** |
| leaf-level re-check (dicts expanded, lists atomic, `time_seconds` excluded) | **462 × 66 = 30,492** comparisons — 66 being the union of leaf paths across all rows — **0** differing |
| summary block identical | **True** — all **17** non-`rows` top-level keys equal |
| pass flag identical | **True** (`True` / `True`) |
| category rates identical | **True** — all six rates matched exactly |

**The one excluded field, disclosed.** `time_seconds` is wall-clock latency and
cannot be deterministic. Including it: **462 × 47 = 21,714 comparisons, 339
differing — all of them `time_seconds`, none elsewhere.** That is the only field
that moved. The comparison above excludes exactly this one field and nothing
else.

The rerun is preserved as `test_results/showcase_demo/determinism_rerun.json`
(sha256 `59f6c8543ec0d7271457d056070d1dd572290652182574321d0162c1ead30825`) so the
claim can be re-derived rather than taken on trust. The in-place frozen
`showcase_results.json` was never overwritten by the rerun.

**Limitations.**

- Determinism is demonstrated **between runs of the same frozen set**, not
  between *builds* of the graph (component 1): rebuilds differ at byte level due
  to wall-clock timestamps.
- The determinism criterion (`determinism_differing_fields_must_equal: 0`) is
  pre-registered inside the frozen question file, so it cannot be relaxed after
  the fact.

---

## Component 10 — Learning (offline only)

**Purpose.** Provide a training-time capability that is structurally excluded from
inference.

**How it works.** `learning/` contains `engine.py`, `hebbian.py`,
`eligibility.py`, `compression.py`, `reward.py`, `persistence.py`, `audit.py`,
`config.py`, `types.py`, plus its own tests. The contract's §11 requirement is
stated in the source at `glmx_ask.py:1334-1335`, verbatim:

```
        # Contract v3.3.2 section 11 (LEARN): "active_only_in: training / offline
        # mode" and "inference_behaviour: Completely frozen and deterministic".
```

**Mechanisms — the gate.**

`self._no_learning: bool = True` (`glmx_ask.py:445`) — **the default is off.**
It guards two sites:

- **`:1271`** — the REINFORCE + Evolutionary-Controller update:
  `logger.debug("[--no-learning] skipping REINFORCE + ES update (deterministic
  mode)")`
- **`:1341`** — `LearningEngine.process_feedback`, with the rationale written in
  the source at `:1336-1340`:
  > *"The LearningEngine call below applies Hebbian updates to the live graph and
  > can call graph.add_edge(); running it unguarded mutates the graph during
  > inference, which breaks section 12 determinism and lets the system invent
  > relations."*

Every evaluation runner sets `pipeline._no_learning = True` explicitly
(`stage_e_runner.py:685`), and Stage A's result file records
**`learning_enabled: false`**.

**Evidence.** The two protections are the same gate, so there is no path by which
a Hebbian update can add an edge during a scored run. The empirical consequence
is the one every run reports: **0 fabricated relations.**

**Measured behaviour.**

- `learning_enabled = false` recorded in `test_results/stage_a/stage_a_results.json`.
- `use_count` sum on the frozen demo graph: **0** — the graph was not mutated by
  any run.
- `activation` sum: `73.5`, identical across rebuilds.
- Fabricated nodes/relations: **0** in every run (Part 3).

**Limitations.**

- The claim is *structural plus empirical*, not a proof: it rests on there being
  exactly two call sites and one gate. That is verifiable by grep and was verified
  (`glmx_ask.py` mentions `_no_learning` at `:445` (definition), `:1271` and
  `:1341` (the two guarded call sites), `:1339` (inside the rationale comment)
  and `:1549` (CLI wiring)), but a future call site added without the gate would
  not be caught by any test in this repository.
- The offline learning machinery itself is not evaluated anywhere in this report;
  only its *exclusion* is.

---

# Part 3 — Evaluation evidence

## 3.1 Master metric table

Every figure is read from the named result file. **All ten runs are shown,
including the three that did not pass.**

| # | Run | File | Domain | n | Passed | Rate | Verdict |
|---|---|---|---|---:|---:|---:|---|
| 1 | Stage A | `stage_a/stage_a_results.json` | toy graph | 78 | 78 | **1.0000** | PASS |
| 2 | Stage B | `stage_b/stage_b_results.json` | relation coverage | 88 | 88 | **1.0000** | PASS |
| 3 | Stage C | `stage_c/stage_c_results.json` | direction + multi-hop | 195 | 195 | **1.0000** | PASS |
| 4 | Stage D | `stage_d/stage_d_results.json` | honesty + fallback | 56 | 56 | **1.0000** | PASS |
| 5 | Stage E | `stage_e/stage_e_results.json` | held-out | 208 | 208 | **1.0000** | PASS |
| 6 | Stage F | `stage_f/stage_f_results.json` | final holdout | 257 | 249 | 0.9689 | **NOT PASSED** — honesty 8/9 = 0.889 < 0.90 |
| 7 | Aviation PoC | `final_validation/final_results.json` | aviation | 355 | 348 | **0.9803** | PASS |
| 8 | Weaving (unseen) | `unseen_e2e/unseen_results.json` | weaving | 402 | 389 | 0.9677 | **NOT PASSED** — fallback 6/7 = 0.857 < 0.90 |
| 9 | Showcase run 1 | `showcase_demo/run1_stale_nonsense/…run1.json` | pottery | 462 | 444 | 0.9610 | **NOT PASSED** — fallback 6/7 = 0.857 < 0.90 |
| 10 | Showcase run 2 | `showcase_demo/showcase_results.json` | pottery | 462 | 445 | **0.9632** | **PASS** |

**Totals:** 2,563 question-runs; 2,101 excluding the showcase re-run.
**Verdicts: 7 PASS, 3 NOT PASSED.**

> The aviation PoC sign-off is **referenced, not restated**. Its report and
> verdict stand unchanged at `test_results/final_validation/FINAL_POC_REPORT.md`
> (ACHIEVED, 348/355 = 0.9803). This document does not amend it.
>
> The weaving verdict is reported **side by side** with the passing verdicts,
> exactly as it is written in `UNSEEN_E2E_REPORT.md`: NOT PASSED, 389/402 =
> 0.9677, `nonsense_fallback` 6/7 = 0.857 < 0.90.

The verdicts above are reproduced, not re-adjudicated. A later cross-run check
**does** bear on *why* weaving's gate was missed — Appendix A-6 and A-7 show that
the one row behind the `6/7` was scored down by a grader substring false
positive, while five of the six passing refusals were never checked at all.
Neither observation is acted on; both are disclosed, and the published verdicts
stand exactly as printed.

## 3.2 Category table — per run

Targets are the shared grader's `CATEGORY_TARGETS`, unchanged.

| Category | Target | A | B | C | D | E | F | Aviation | Weaving | **Showcase (run 2)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `simple_one_hop` | ≥0.95 | 40/40 | 49/49 | — | 24/24 | 93/93 | 125/129 | 138/139 | 155/156 | **164/165** |
| `direction_pair` | ≥0.90 | 13/13 | 14/14 | 101/101 | 7/7 | 20/20 | 18/18 | 28/28 | 28/28 | **28/28** |
| `short_multi_hop` | ≥0.75 | 10/10 | 15/15 | 65/65 | — | 54/54 | 62/65 | 112/117 | 136/147 | **185/200** |
| `honesty_out_of_graph` | ≥0.90 | 9/9 | 6/6 | 7/7 | 8/8 | 9/9 | ❌ **8/9** | 21/22 | 23/23 | **22/23** |
| `nonsense_fallback` | ≥0.90 | 6/6 | 4/4 | 3/3 | 6/6 | 6/6 | 7/7 | 7/7 | ❌ **6/7** | **7/7** |
| `mirror_silence` | ≥0.90 | — | — | 11/11 | — | 26/26 | 29/29 | 42/42 | 41/41 | **39/39** |
| **Overall** | ≥0.90 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9689 | 0.9803 | 0.9677 | **0.9632** |

Bold marks the **Showcase (run 2)** column; ❌ marks the only two cells below
target in the whole table (Stage F honesty, Weaving fallback). Run 1 is not a
column here — it is disclosed in full at §4.3. Stage A–D use their own internal
matrices (`mechanism_matrix`, `per_category`) rather than the Stage E schema; the
cells above are their equivalent counts.

*The table covers the six categories shared by every run, so two Stage A–D
categories appear only in §3.4 and are not columns here: Stage C's
`forbidden_reverse` (**8/8**) and Stage D's `missing_relation` (**11/11**).
Adding them makes C = 195 and D = 56 exactly.*

## 3.3 Contract §15 mechanism metrics

| Mechanism | Target | Stage E | Stage F | Aviation | Weaving | **Showcase** |
|---|---:|---:|---:|---:|---:|---:|
| `decode.no_invented_relations` | ≥0.98 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| `decode.template_correctness` | ≥0.95 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| `central_invariant.hold_rate` | ≥0.90 | **1.0000** | **1.0000** | **1.0000** | 0.9975 | **1.0000** |
| `semantic_checks` | — | 1.0000 | 0.9883 | 0.9859 | 0.9701 | 0.9632 |
| `fallback_trigger_rate` | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `direction_pair_distinct_answers` | — | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rows w/ illegal **or** fabricated hops | 0 | **0** | **0** | **0** | **0** | **0** |
| rows w/ unaccounted entities | 0 | **0** | **0** | **0** | **0** | **0** |
| direction pairs answering differently | — | 10/10 | 9/9 | 14/14 | 14/14 | **14/14** |

**The two invariants that never moved:** `no_invented_relations` = 1.0000 in
every run, and `template_correctness` = 1.0000 in every run.

## 3.4 Stage B / C / D detail (their own schema)

| | Stage B | Stage C | Stage D |
|---|---|---|---|
| questions | 88 | 195 | 56 |
| overall | 88/88 = 1.0 | 195/195 = 1.0 | 56/56 = 1.0 |
| central invariant | 88/88 = 1.0 | 195/195 = 1.0 | 56/56 = 1.0 |
| relation coverage | **16/16 canonical relations exercised** | — | — |
| direction accuracy | 14/14 | **101/101** | inverse-direction 7/7 all refused, 0 backwards reads |
| multi-hop | 15/15 (2-hop 14/14, 3-hop 1/1) | **65/65** (2-hop 63/63, 3-hop 2/2) | — |
| mirror silence | — | **11/11**, 0 backwards reads | — |
| forbidden reverse | — | **8/8 all refused, 0 fabricated claims** | — |
| missing relation | — | — | **11/11 all refused** |
| honesty | 6/6 all refused | 7/7 | 8/8 all refused |
| fallback | 4/4 all refused or disclosed | 3/3 | 6/6 all triggered; disclosed-guess 1/1 OK |
| hop provenance | — | stored 229, declared_inverse 5, **illegal 0, fabricated 0** | stored 28, declared_inverse 0, **illegal 0, fabricated 0** |
| no invented relations | — | — | **clean 56/56, illegal 0, fabricated 0** |

## 3.5 What the table does and does not show

**Does show:** a graph traversal that never invents a fact, never leaks a
forbidden edge, never confuses the two directions of a relation, and whose
central invariant — planned chain, walked path and rendered sentence all
agreeing — held in **2,100 of 2,101** cases.

**Does not show:** that every planned walk completes. Three of ten runs missed a
target, and the causes are not the same:
- Stage F missed on **honesty** (8/9).
- Weaving and Showcase-run-1 missed on **nonsense_fallback** (6/7 each), and
  **both misses are grader-side, not system-side**: run 1's `nf002` and weaving's
  `nf004` were both *correct* refusals scored down by the same substring
  false positive (A-7) — in run 1's case on top of my stale input (Part 4.3).
  Separately, five of weaving's six passing refusals were **never checked at all**
  (A-6), so this metric's verdict is unreliable in both directions at once.
- **34 multi-hop failures across Stages F, Aviation, Weaving and Showcase — and
  all 34 stopped short of their own plan** (Stage F 3/3, Aviation 5/5, Weaving
  11/11, Showcase 15/15). The resonance cause was instrumented and confirmed on
  two of those sets (Weaving 11/11, Showcase 15/15); on Stage F and Aviation the
  truncation signature is measured but the cause was not re-instrumented, so
  those 8 are attributed by signature rather than by probe.

---

# Part 4 — Final new-data demo

## 4.1 Domain: pottery and ceramics

The domain was chosen by first **proving its absence**: a repository-wide search
across every dataset, stage, probe and result file confirmed that no pottery or
ceramics content existed anywhere before this demo was built. It shares no
graph, no node, no edge, no question and no out-of-graph subject with Stage A–F,
the aviation PoC, or the weaving set.

Artefacts: `test_results/showcase_demo/`.

| Artefact | Value |
|---|---|
| `showcase_eval.db` | 147 nodes, 193 edges, **16/16 canonical relations** |
| graph content hash | `0d6edf6eb6d78ac45230abc484f37f214ffb26bccbbe460babe6a92155680c48` |
| graph file sha256 | `e465e0707ae9b0f78dff617f365cd03ad15dbd274af7ee5c9af1d86cc7378d2e` |
| question set sha256 | `5ff31283180093d5ee956d8d5855009de3e0fc153f2d621463e3dc42f96d08da` |
| questions frozen | **462** |

**Build validations** (all enforced by refusing to freeze): naming discipline
that avoids `label-token-is-another-label` and `label-prefix-collision` across
every node and every property value; 23 out-of-graph subjects (≥20 required);
6 cue-free nonsense questions verified cue-free against the live descriptor
bank; 14 bidirectional pairs; 39 mirror controls; 0 ambiguous one-hop edges;
200 multi-hop chains (140 two-hop + 60 three-hop). `validate()` returned an
empty problem list immediately before the build.

Five naming collisions were caught by the validator and fixed *before any
question existed* — `pottery tool`→`ceramic tool`, `pottery ware`→`ceramic ware`
(prefix collision with `potter`), `foot powered`→`rotating platform`,
`over-firing`→`overheating`, `red earthenware`→`baked earth`.

**What is reused, and what is new.** The graph algebra and question derivation are
imported by path from `test_results/stage_f/` — they are data-independent
(uniquely-readable edges → questions, sibling ids for direction pairs, forbidden
far ends for mirror controls). **Every template, every node, every edge and every
question is new.** The `linguistic_maps` base template names no language at all,
so it cannot assert a translation the graph lacks — the defect that shipped 40
false questions in the aviation set.

**Pass criteria were written into the frozen question file before scoring**
(`pass_criteria` block), including `overall_min 0.90`, `simple_one_hop_min 0.95`,
`honesty_out_of_graph_min 0.90`, `nonsense_fallback_min 0.90`,
`illegal_mirrors_must_equal 0`, `fabricated_nodes_or_relations_must_equal 0`,
`determinism_differing_fields_must_equal 0`.

## 4.2 Freeze discipline

```
build_graph.py       → frozen, sha256 e465e070…        (never rebuilt after)
generate_questions.py → frozen, sha256 5ff31283…       (pre-registered criteria)
run_eval.py          → verifies BOTH pins before entering the grader
                     → the one and only scored run
                     → verifies the graph hash again after scoring
```

No external LLM was used at any point — only `BAAI/bge-small-en-v1.5` and the
template decoder.

## 4.3 The two-run disclosure

**This is the one place where something changed after a score was seen, and it is
stated in full.**

**Run 1** (set `bce0183d…`, graph `e12d7610…`): **444/462 = 0.9610, NOT PASSED** —
`nonsense_fallback` 6/7 = 0.857.

**The defect was mine, not the system's.** `build_graph.py` rebound
`CONCEPTS`, `EDGES`, `OUT_OF_GRAPH_SUBJECTS`, `SPANISH_ANSWERS` and `DB_PATH`
into the imported Stage F module — but **omitted `NONSENSE_CUE_FREE`**. The
fallback category therefore scored Stage F's food-domain nonsense list.

A provenance check after run 1 proved the damage:

> **All 6 of those nonsense questions are byte-identical to questions already
> scored in Stage F, the aviation PoC, and the weaving set.** They appear in
> `stage_f/holdout_questions_frozen.json`,
> `final_validation/final_questions_frozen.json` and
> `unseen_e2e/unseen_questions_frozen.json`.

So run 1's fallback category **was not new data at all**, which directly
contradicts the demo's own requirement.

**And run 1's NOT PASSED verdict had a second cause, which is disclosed here
because it is easy to mistake for the first.** The one nonsense row that scored
down — `nf002`, *"What colour is the sound of a doorbell?"* — did **not** score
down because the answer was wrong. Its answer was a correct refusal. It scored
down because of **A-7**, the `no_claim_when_refusing` substring false positive:

| | run 1 (`nf002`) | the same question in its home set (Stage F) |
|---|---|---|
| graph | **pottery** (wrong domain for this question) | food/holdout (its own) |
| anchor | `inflection:door~doorbell` → `door`, sim 0.6956, margin 0.0308 | `orange`, exact label |
| walk | **1 step** — `door --has_property--> hinged panel` (stored) | **0 steps** |
| refusal text | correct refusal | correct refusal |
| `template_fragments_leaked` | **`["is"]`** → check ran → `no_claim_when_refusing: false` | **`null`** → check never ran |
| score | **FAIL** | PASS |

The stale question put a *walkable* row into the set, and A-7 only ever runs on
rows that walked — which is why the same question passed at home and failed
here. Run 1 had **14** walked refusals and **3** flagged (`og018`, `nf002`,
`pv002`); run 2 has **13** and **2** (`og018`, `pv002`).

**Counterfactual, stated plainly:** had A-7 not fired, run 1 would have gained
**two** rows — `nf002` (only `no_claim_when_refusing` in its `why`) and `og018`
(likewise) — giving `nonsense_fallback` **7/7**, `honesty_out_of_graph` **23/23**,
total **446/462**, and **PASSed**. The third flagged row, `pv002`, would *still*
have failed: its `why` carries `refused_an_answerable_question` as well, a
genuine false refusal (A-5), so A-7 was not load-bearing there. Net: run 1's NOT
PASSED verdict was produced by *both* my stale input and the known grader defect
working together — the stale row supplied the casualty, and A-7 is what scored
it down. This is disclosed rather than resolved, because A-7 is deliberately
unpatched.

**The correction.** One line was added — `B.NONSENSE_CUE_FREE = NONSENSE_CUE_FREE` —
restoring the list authored before any scoring. Then the graph was rebuilt, the
set re-frozen, and **run 2 was scored**.

**Proof that nothing else moved, obtained *before* run 2 was scored:**

| check | result |
|---|---|
| qid sets identical | **True** |
| questions byte-identical | **456 / 462** |
| questions whose payload differs | **6** |
| categories of differing rows | `{nonsense_fallback}` only |
| fields that moved | **`["question"]` only** |
| differing questions outside `nonsense_fallback` | **0** |
| category counts | **identical** |
| graph nodes / edges / relations | **147 / 193 / 16 — identical** |
| graph **content** hash | `0d6edf6e…` — **identical on the frozen graph and on an independent controlled rebuild** (excludes the wall-clock columns; re-derivable with `graph_content_hash.py`, Appendix B) |
| graph **file** hash | `e12d7610…` → `e465e070…` — differs, only because of wall-clock `saved_at`/`create_time` stamps (component 1) |

**What did NOT change:** the graph, the templates, the pass criteria, the grader,
the honesty gates, the resonance configuration, and every threshold. No mechanism
was tuned, no criterion relaxed, no failure hidden.

**Both runs are preserved and both are reported here and in Part 3.** Run 1's
results and set are in `test_results/showcase_demo/run1_stale_nonsense/`.

**My own judgement of this:** the correction was necessary because leaving a
known-broken binding in place would have made the deliverable's central claim —
*new data* — false. But because `nonsense_fallback` was run 1's only failing
gate, the appearance of tuning is unavoidable, which is exactly why both runs are
published rather than one.

## 4.4 Results — run 2

| Category | n | Passed | Rate | Target | OK |
|---|---:|---:|---:|---:|:--:|
| `simple_one_hop` | 165 | 164 | **0.9939** | 0.95 | ✅ |
| `direction_pair` | 28 | 28 | **1.0000** | 0.90 | ✅ |
| `short_multi_hop` | 200 | 185 | **0.9250** | 0.75 | ✅ |
| `honesty_out_of_graph` | 23 | 22 | **0.9565** | 0.90 | ✅ |
| `nonsense_fallback` | 7 | 7 | **1.0000** | 0.90 | ✅ |
| `mirror_silence` | 39 | 39 | **1.0000** | 0.90 | ✅ |
| **OVERALL** | **462** | **445** | **0.9632** | 0.90 | ✅ |

### Requested headline measures

| Measure asked for | Result |
|---|---|
| **Overall** | **445 / 462 = 0.9632** — PASS |
| **One-hop** | **164 / 165 = 0.9939**; baseline (canonical phrasing) **129 / 129 = 1.0000**; re-asked variants **35 / 36 = 0.9722** |
| **Direction** | **28 / 28 = 1.0000**, 14 pairs, **14/14 answered differently** |
| **Multi-hop** | **185 / 200 = 0.9250**; 2-hop **132 / 140 = 0.9429**, 3-hop **53 / 60 = 0.8833** |
| **Honesty** | **22 / 23 = 0.9565**; 23/23 refused; 0 fabricated concepts named in any refusal |
| **Fallback** | **7 / 7 = 1.0000**; fired on every nonsense row; disclosure present 1/1; 0 firings outside the category |
| **Central invariant** | **462 / 462 = 1.0000**; 0 failures in any of the 6 sub-checks |
| **Determinism** | **0 differing field-values across 462 × 46 = 21,252 comparisons** (the 46 excludes only `time_seconds`, wall-clock latency; including it, all 339 differences are `time_seconds` and none elsewhere); summary block and pass flag identical |
| **No fabricated knowledge** | **0** fabricated hops, **0** illegal mirror hops, **0** leaked labels, **0** unaccounted entities, **0** silent wrong answers |
| **Robustness slice** | paraphrase **18 / 18 = 1.0000**; plural **17 / 18 = 0.9444** |

**Variant slice** (same gold edges, different phrasing — the slice that exists
because every earlier set measured each edge with exactly one phrasing):

| phrasing | n | passed | rate |
|---|---:|---:|---:|
| canonical (baseline) | 129 | 129 | **1.0000** |
| `paraphrase` rewording | 18 | 18 | **1.0000** |
| `plural` inflection | 18 | 17 | 0.9444 |
| **all re-asked variants** | **36** | **35** | **0.9722** |

**Hop provenance:** `{"stored": 633, "declared_inverse": 13}` — illegal 0,
fabricated 0.

**Failure attribution** (all 17): `walker_wrong_path: 15`, `wrong_anchor: 1`,
`walk_found_nothing: 1`.

### Where the 17 failures come from

| cause | n | classification |
|---|---:|---|
| Walk stopped before completing its own planned chain | **15** | genuine system limitation — resonance pruning (A-3) |
| Plural anchor caused a refusal of an answerable question (`pv002`) | **1** | genuine system limitation (A-5 pattern) |
| Correct refusal flagged by a substring check (`og018`) | **1** | **grader false positive** (A-7) |
| Fabricated / illegal / silent-wrong | **0** | — |

**Zero** of the 17 failures involved a false claim.

## 4.5 Twelve full traces

Format: question → `relation_chain` → anchor → `path` → `path_edges` → answer.
All from `showcase_results.json`; `inv` = central invariant, `pc`/`wc` =
plan/walk confidence.

---

### Trace 1 — simple fact (`is_a`) · `oh002` ✅

```
Q     : What type of thing is the architectural ceramic?
chain : ["is_a"]
anchor: architectural ceramic   (exact_label)
path  : architectural ceramic → ceramic ware
edges : ["is_a"]                provenance: [stored]
answer: architectural ceramic is a type of ceramic ware.
tpl   : {node0} is a type of {node1}.
reson : 82n/114e   pc=1.0000   wc=0.9145   inv=True   gold=ceramic ware ✓
```

---

### Trace 2 — simple fact (`has_property`) · `oh010` ✅

```
Q     : What is known for the burner?
chain : ["has_property"]
anchor: burner   (exact_label)
path  : burner → gas flame
edges : ["has_property"]        provenance: [stored]
answer: burner is gas flame.
tpl   : {node0} is {node1}.
reson : 74n/116e   pc=1.0000   wc=0.8600   inv=True   gold=gas flame ✓
```

---

### Trace 3 — simple fact (`part_of`) · `oh007` ✅

```
Q     : What is the bat part of?
chain : ["part_of"]
anchor: bat   (exact_label)
path  : bat → wheel
edges : ["part_of"]             provenance: [stored]
answer: bat is part of wheel.
tpl   : {node0} is part of {node1}.
reson : 71n/104e   pc=1.0000   wc=0.8818   inv=True   gold=wheel ✓
```

---

### Trace 4 — direction pair, end A (`causes`) · `dp001f` ✅

```
Q     : What does the air bubble cause?
chain : ["causes"]
anchor: air bubble   (exact_label)
path  : air bubble → blowout
edges : ["causes"]              provenance: [stored]
answer: The reason is that air bubble causes blowout.
tpl   : The reason is that {node0} {relation0} {node1}.
reson : 70n/86e    pc=1.0000   wc=0.9473   inv=True   gold=blowout ✓
```

---

### Trace 5 — direction pair, end B (`caused_by`) · `dp001r` ✅
*Same stored edge, opposite question — the pair must differ, and does.*

```
Q     : What was the blowout caused by?
chain : ["caused_by"]
anchor: blowout   (exact_label)
path  : blowout → air bubble
edges : ["caused_by"]           provenance: [stored]
answer: blowout is caused by air bubble.
tpl   : {node0} is caused by {node1}.
reson : 73n/106e   pc=1.0000   wc=0.9473   inv=True   gold=air bubble ✓
```

*Ends: `dp001f` answers `blowout`, `dp001r` answers `air bubble` — **different**,
as required. Same structure verified for `dp002f`/`dp002r`
(`follows`/`precedes`, `burnishing`↔`stamping`), one of 14/14 pairs.*

---

### Trace 6 — 2-hop multi-hop · `mh001` ✅

```
Q     : The argile translates to what, and what type of thing that is?
chain : ["linguistic_maps", "is_a"]
anchor: argile   (exact_label)
path  : argile → clay → ceramic material
edges : ["linguistic_maps", "is_a"]   provenance: [stored, stored]
answer: argile relates to clay, and clay is a ceramic material.
reson : 76n/103e   pc=1.0000   wc=0.8763   inv=True   steps=2   gold=ceramic material ✓
```

*`argile` is the French word for `clay`; the pair is stored as English↔French and
the template names no language, so the claim is true in either direction.*

---

### Trace 7 — 3-hop multi-hop · `mh141` ✅

```
Q     : What is another word for the baked earth, and what type of thing that
        is, and what is known for that?
chain : ["synonym", "is_a", "has_property"]
anchor: baked earth   (exact_label)
path  : baked earth → terracotta → earthenware → low fired
edges : ["synonym", "is_a", "has_property"]   provenance: [stored, stored, stored]
answer: baked earth is synonymous with terracotta, and terracotta is a
        earthenware, and earthenware has low fired.
reson : 78n/116e   pc=1.0000   wc=0.8957   inv=True   steps=3   gold=low fired ✓
```

*Three hops, all stored, invariant held, gold node reached. Note the answer also
contains the article defect **"a earthenware"** — see component 6 / Appendix A-8.
It was scored as correct because the grader checks ordering, not grammar.*

---

### Trace 8 — honesty refusal · `og001` ✅

```
Q     : What type of thing is the raku?              ("raku" is not in the graph)
chain : ["is_a"]
anchor: figurine   reason=ungrounded:figurine   exact_label_hit=False
path  : figurine                                    edges: []
answer: I don't have a relation in my knowledge graph that fully answers this
        question. Closest concepts I have: figurine. (No is_a relation found.)
reson : 73n/98e    pc=1.0000   wc=0.0000   steps=0   inv=True
```

*The subject `raku` is out-of-graph. The anchor resolved to `figurine` but was
flagged **ungrounded**, the walk produced **no edge**, and the system refused
while naming what it does hold. No relation was asserted; `gold=None` by design.*

---

### Trace 9 — fallback, disclosed guess · `nf007` ✅

```
Q     : How wooden is the workbench?                  (cue-free by construction)
chain : ["has_property"]   ← default_chain, heuristic path
anchor: workbench   (exact_label)
path  : workbench → wooden
edges : ["has_property"]        provenance: [stored]
answer: Based on a heuristic guess (no relation cue matched): workbench is wooden.
reson : 74n/98e    pc=0.6000   wc=0.9800   inv=True
```

*The fallback fired (no relation cue), the answer was produced from
`default_chain`, and the **disclosure prefix is present**. Plan confidence is
0.6000 — the guess is machine-readable as a guess.*

---

### Trace 10 — cue-free fallback → refusal · `nf001` ✅

```
Q     : How many edges does the number four have?     (cue-free)
chain : ["has_property"]   ← default_chain, heuristic path
anchor: four   (exact_label — "four" IS a graph node: French for kiln)
path  : four                                     edges: []
answer: I don't have a relation in my knowledge graph that fully answers this
        question. Closest concepts I have: four. (No has_property relation found.)
reson : 74n/95e    pc=0.6000   wc=0.0000   steps=0   inv=True
```

*Cue-free, so fallback fired (confidence 0.6000); the walk then found no
`has_property` edge on `four` and the system refused rather than guessing.
Fallback and honesty gate both active on the same row, correctly sequenced.*

---

### Trace 11 — mirror silence · `ms001` ✅

```
Q     : What is the architectural ceramic an example of?   ← forbidden direction
chain : ["example_of"]
anchor: architectural ceramic   (exact_label)
path  : architectural ceramic                          edges: []
answer: I don't have a relation in my knowledge graph that fully answers this
        question. Closest concepts I have: architectural ceramic.
        (No example_of relation found.)
reson : 82n/106e   pc=1.0000   wc=0.0000   steps=0   inv=True
```

*The graph stores `tile --example_of--> architectural ceramic`. Asking the
reverse must **not** be synthesised — `example_of` is absent from
`INVERSE_RELATION_LABELS` (`walker/graph_walker.py:28-36`, which declares only
`causes`/`caused_by`, `follows`/`precedes`, `part_of`/`has_part` and
`is_a`), so there is no inverse label to synthesise it under. It was not: no
edge walked, no far end named. **39/39 mirror controls passed, 0
illegal mirror hops.***

---

### Trace 12 — the honest failure · `mh002` ❌

```
Q     : The argile translates to what, and what is known for that?
chain : ["linguistic_maps", "has_property"]     ← plan (2 hops)
anchor: argile   (exact_label)
path  : argile → clay                            ← walk stopped after hop 1
edges : ["linguistic_maps"]         provenance: [stored]
answer: argile relates to clay.
        ↑ states ONLY what was walked — the unwalked hop is NOT claimed
reson : 79n/107e   pc=1.0000   wc=0.8709   steps=1/2   inv=True
gold  : plastic body   ✗
failed: expected_path_order   attribution: walker_wrong_path
```

**Why it failed.** Diagnosis-only instrumentation: the final target
(`plastic body`, the `has_property` value of `clay`) was **absent from the
subgraph resonance handed the walker** — 1 of the 15/15 — at undirected distance
2 from the nearest seed, outside the one-hop readmission shell.

**Why it matters anyway.** The plan promised two hops, the walk delivered one, and
the decoder **refused to assert the second**. The sentence is *incomplete and
true*, not complete and false. That is the architecture working as specified under
a defective upstream — the same behaviour on all 15 truncations.

---

## 4.6 What the demo establishes

1. A brand-new domain, frozen before scoring, passes every pre-registered gate.
2. The central invariant held on **462/462** questions with zero check failures.
3. Zero fabricated knowledge: **0** fabricated hops, **0** illegal mirrors,
   **0** leaked labels, **0** unaccounted entities, **0** silent wrong answers.
4. Re-running the identical frozen set reproduces **exactly** — 0 differences in
   21,252 field comparisons.
5. Re-asking the same edges with different phrasing costs almost nothing:
   35/36 (0.9722) versus 129/129 (1.0000) canonical.
6. **The known resonance limitation reproduced independently**: 15/15 truncations
   with the target pruned, 12/12 controls with it present.
7. Two defects *of the evaluation harness and the grader* were found and are
   disclosed rather than hidden (Part 4.3, Appendix A-7 and A-10).

---

# Part 5 — Mechanism health status table

| # | Mechanism | Status | Basis (measured) |
|---|---|---|---|
| 1 | **Graph building / storage** | **Working well** | 147 nodes, 193 edges, **147/147 embeddings**; 16/16 relations; **content hash `0d6edf6e…` reproduced exactly by a fresh controlled rebuild to a temp path** (content identical; file hash changed `e465e070…`→`995475b5…` only because `create_time` moved), frozen artefacts untouched; file hash verified unchanged before and after a 27-run diagnostic. *Caveat: this assesses the **loader and store**, since the data is hand-authored.* |
| 2 | **Encode / anchoring** | **Working well** | `exact_label_hit` 416/462 (0.9004); of 46 non-exact rows **44 passed**; `anchor_identity_grounded` 436/462; floors `0.55`/`0.04` and `entity_not_found < 0.25` all live in config, not buried in code. |
| 3 | **Spread / Resonance** | **Failing — the one genuinely broken mechanism** | Final hop target **absent 15/15** on truncations, **present 12/12** on controls; all failures at seed-distance **2–3**, outside the one-hop readmission shell (`tier1.py:239-241` states this limit). Also overruns the contract budget: **462/462 rows > `top_k: 64`**, max 86 (A-1). |
| 4 | **Plan / relation extraction** | **Working well** | Plan confidence **1.0000** on every non-fallback row; **0.6000** on exactly the 7 fallback rows; every template asserted against the **live** cue matcher at freeze time; all 16 relations exercised. |
| 5 | **Walker** | **Working with limitations** | **0 fabricated hops, 0 illegal mirrors, 0 leaked labels, 39/39 mirror silence, 14/14 direction pairs differ.** But **15/200 multi-hop rows stopped short** of their own plan (7.5%). Across four sets there were **34 multi-hop failures and all 34 stopped short** (Stage F 3/3, Aviation 5/5, Weaving 11/11, Showcase 15/15); the resonance cause was instrumented and confirmed on two of them (Weaving 11/11, Showcase 15/15). |
| 6 | **Decoder** | **Working well, with a surface-form defect** | Central invariant **462/462**, template correctness **1.0000**, `template_matched` false on 0 rows, and it **declined to assert the unwalked relation on all 15 truncations**. Defect: **21/462 (4.55%) sentences are grammatically broken and all 21 scored correct** (A-8). |
| 7 | **Honesty gates** | **Working, telemetry partly wrong** | **23/23** out-of-graph refused, **22/23** = 0.9565 ≥ 0.90, `honest_by_identity` 23/23, 0 fabricated concepts named in any refusal. But `entity_not_found` fired **0/23** — right outcome, wrong signal (A-9); the refusal-cleanliness check ran on only **13 of 69** refusals, and the two times it produced a non-empty fragment it flagged both **falsely** (A-7 — `og018`'s refusal was correct; `pv002`'s row failed anyway, for A-5); and **13/69** refusal sentences assert *"No `<rel>` relation found"* while their own walk contains that relation (A-11). |
| 8 | **Heuristic fallback** | **Working well** | Fired **7/7**; 0 firings outside the category; disclosure present **1/1** where required; confidence correctly downgraded to **0.6000**; the two bypass mechanisms (`chain_lift`, `rescue`) were **removed** after measuring them at 1 firing / 0 changes. |
| 9 | **Determinism controls** | **Working well** | **0** differing field-values across **462 × 46 = 21,252** comparisons; summary and pass flag identical; two independent SHA-256 pins verified before *and* after scoring; grader imported by path, not copied. |
| 10 | **Learning (offline only)** | **Correctly excluded** | `_no_learning` defaults **True** (`:445`) and gates both mutation sites (`:1271`, `:1341`); `learning_enabled: false` recorded in Stage A; graph `use_count` sum **0**; **0 fabricated relations in every run**. Not itself evaluated. |
| 11 | **Paraphrase robustness** | **Working well** | **18/18 = 1.0000** on cue-preserving rewordings. |
| 12 | **Plural robustness** | **Working with limitations** | **17/18 = 0.9444**; the single miss is a **false refusal on an answerable question** (`pv002`), the same pattern as weaving `pv020` (A-5). |

## Architecture constraints — all held on this demo

| Constraint | Status | Evidence |
|---|---|---|
| v3.3.2 strict; `relation_chain` is the operative signal | ✅ | Plan stage emits and records it on all 462 rows |
| No cosine in the main walker scoring | ✅ | Walker scores by traversal against the chain; similarity enters only at encode/anchor — the loader rejects any formula containing `cosine`/`target_similarity`/`intent_bias` (`graph_walker.py:141-147`) |
| Deterministic and frozen by default | ✅ | 0/21,252 field differences |
| No invented relations | ✅ | 0 fabricated, 0 illegal same-label mirror |
| Honest refusal when the graph cannot answer | ✅ | 23/23 out-of-graph, 39/39 mirror silence |
| Guess disclosure (§8) | ✅ | 1/1, prefix present |
| Central invariant `chain ≈ path ≈ sentence` | ✅ | **462/462** |
| Learning inactive at inference (§11) | ✅ | `_no_learning=True` at both sites; `use_count` 0 |
| Node budget `top_k: 64` (§7) | ❌ | **462/462 rows over budget**, max 86 — A-1, disclosed, unpatched |

---

# Part 6 — The precise PoC claim

## What is claimed

> **GLM-X is a working proof of concept that answers natural-language questions
> over an explicit knowledge graph by walking it, and that — measured across ten
> evaluations over nine frozen graphs totalling 2,563 question-runs — never once
> fabricated a node or a relation, never leaked a forbidden edge, and produced a
> sentence whose claims were licensed by the path it actually walked in
> 2,100 of 2,101 cases (0.99952).**
>
> **Specifically, on a domain never previously seen by the system — pottery and
> ceramics, frozen before scoring — it passed every pre-registered gate at
> 445/462 = 0.9632, with the central invariant at 462/462 = 1.0000, zero
> fabricated knowledge, and exact reproducibility across a full re-run
> (0 differences in 21,252 field comparisons).**

## What is claimed *precisely*

The claim is about **honest traversal**, not about knowledge or fluency:

1. **`relation_chain` ≈ `walked path` ≈ `generated sentence`** is enforced, not
   aspirational: six sub-checks, **2,100/2,101** held, and the one miss is
   disclosed in `UNSEEN_E2E_REPORT.md`.
2. When the walk is shorter than the plan, **the decoder renders the shorter
   truth** — observed on all 15 truncations of this demo.
3. When the graph cannot answer, **the system refuses in explicit text** and
   names no relation — 23/23 out-of-graph, 39/39 mirror silence, 0 fabricated
   concepts in any refusal.
4. When a question contains no relation cue, **the system says the answer is a
   guess** — 7/7 fired, disclosure 1/1, confidence downgraded to 0.6000.
5. The whole thing is **reproducible and auditable**: frozen artefacts, two
   independent digests verified before and after, a grader imported rather than
   copied.

## What is explicitly *not* claimed

| Not claimed | Why |
|---|---|
| **That every planned walk completes.** | **15/200 multi-hop rows truncated (7.5%)** in this demo; **34/34** multi-hop failures across four sets stopped short. Root cause instrumented on two sets (Weaving 11/11, Showcase 15/15) and left unpatched. |
| **That the evaluation harness is perfect.** | Two defects were mine or the grader's: a missing binding that made run 1's fallback category non-new (Part 4.3), and a substring check producing false positives — **2 in this demo** (`og018`, `pv002`) and **1 in the weaving run** (`nf004`), where it is the whole of that run's gate miss (A-7). |
| **That the answers are well-formed English.** | **21/462 sentences (4.55%) are grammatically broken and all scored correct** (A-8). |
| **That honesty telemetry is trustworthy.** | `entity_not_found` fired **0/23** on rows that should have triggered it (A-9). |
| **That the contract's node budget holds.** | **462/462 rows exceed `top_k: 64`**, max 86 (A-1). |
| **That all ten runs passed.** | **7 PASS, 3 NOT PASSED.** Stage F, Weaving and Showcase-run-1 are printed as failures in Part 3. |
| **Open-domain capability.** | The system has no knowledge outside the graph it is given. |
| **That no defects were tuned away.** | Eleven are known and unpatched; the one input correction made after a score is disclosed in full in Part 4.3. |

## The one-sentence version

**A deterministic, non-LLM graph walker that provably does not invent facts,
passes 7 of 10 evaluations including a fully held-out domain, and whose one
serious mechanical defect — the resonance stage pruning the far end of multi-hop
questions — is isolated, reproduced on two independent domains at 15/15 and
11/11, and deliberately left unpatched.**

---

# Appendix A — Disclosed defects register

**Every defect below is disclosed and left unpatched.** Patching any of them
after seeing a score would convert an independent evaluation into a tuned one.
The frozen sets, their digests and their criteria are unchanged.

## Carried from the aviation PoC — `FINAL_POC_REPORT.md` §6–7

| ID | Defect | Measurement | Status |
|---|---|---|---|
| **A-1** | **Contract §7's `top_k: 64` is not the true bound.** `_gate` cuts to `top_k`, then `_readmit_seed_neighbourhood` re-adds seed neighbours with no re-cut → ceiling is `top_k + \|initial_seeds\|`. | Aviation: **313/355 (88.2%)** rows over 64, max 84. **Showcase: 462/462 (100%)** over 64, max 86. | Unpatched; reconcile by amending §7 or re-applying the budget after readmission |
| **A-2** | **11.3% of the aviation frozen questions were semantically false** — 40/355 asked for a *Spanish* translation while the graph stored *English↔Italian*. `ONE_HOP_TEMPLATE["linguistic_maps"]` hardcodes *"in Spanish"*. | 40/355; 39 passed anyway (the engine ignores the language word). | Unpatched; the showcase closes it **by construction** (language-neutral template + freeze-time check) |

## Carried from the weaving unseen run — `UNSEEN_E2E_REPORT.md` §3

| ID | Defect | Measurement | Status |
|---|---|---|---|
| **A-3** | **Resonance prunes the final hop's target.** `_readmit_seed_neighbourhood` extends only **1 hop from the original seeds**, never from a node the walk reached; `_propagate` under-ranks targets at seed-distance 2–3. An **ordering-and-depth** defect, not a budget ceiling. | Weaving: absent **11/11**, controls present **6/6**. **Showcase: absent 15/15, controls present 12/12** — independent reproduction. Distances: failures 2–3, controls 0–2. | Unpatched; widening readmission to k-hop would void the contract's node budget |
| **A-4** | **A mirrored step was derived as a forward step**, producing a semantically false *question* (`mh042`): asked *"What does the fast colour cause, and what is known for that?"* while the graph stores the reverse — `mordant --causes--> fast colour`, walked as `fast colour --caused_by--> mordant` — and the row's own `expected_chain` carries the corrupted forward derivation (`["causes", "has_property"]`). The **system's answer was true**: *"fast colour is caused by mordant."* | **1 row of 402** (itself a `short_multi_hop` row); also the only `chain_matches_walk` miss in ten runs — the sole central-invariant failure of 2,101. | Unpatched — question-derivation defect inherited from the shared algebra |
| **A-5** | **Plural inflection can break identity grounding → false refusal.** Anchor located, plural form ungrounded, system refused an answerable question. | Weaving `pv020` (plural slice 16/17). **Showcase `pv002` (plural slice 17/18)** — reproduced. | Unpatched |
| **A-6** | **`no_claim_when_refusing` is largely a no-op.** `stage_c_runner.py:241` reads `if not edges or len(labels) < 2: return {"checks": {}, "failed": [], "ok": True, ...}` — so a refusal with no walked edge, or fewer than two path labels, is **never checked at all** and is returned as passing. | Weaving: the check ran on **10 of 71** refusals and flagged exactly one row, `nf004`. The other five nonsense refusals carry `path_edges = []` and `semantic_checks = {}` — **they passed by short-circuit** (the prior weaving report's *"the five passing refusals pass by short-circuit, not by clean text"* reproduces exactly). Its companion figure, **"0/6 true refusals would pass an always-on check", does not re-derive from the code**: with the early return removed a 0-edge row has `key = ()`, finds no chain template, and yields `fragments = []`, so it still passes. That figure assumes fragments come from the relation the refusal *names*; under that reading all six would flag — but every such mention sits inside a **negation** ("No has_property relation found"), so the flag would itself be a false positive. Showcase: the check ran on only **13 of 69** refusals — precisely those with ≥1 walked edge (set identity verified) — so **56 were never checked**. | Unpatched. The prior report's verdict on the metric (*"materially overstates refusal cleanliness"*) is quoted, not adopted — the re-derivation above, and A-7 for what actually flagged `nf004`, do not support it |

## Found by this evaluation — showcase demo

| ID | Defect | Measurement | Status |
|---|---|---|---|
| **A-7** | **`no_claim_when_refusing` false positives (chain-template form).** `CHAIN_TEMPLATES[('has_property',)] = "{node0} is {node1}."` yields fragment `is`, recorded in the row field `template_fragments_leaked` and matched by **plain substring** against "th**is** question" inside the standard refusal boilerplate — which is present in **all 69** refusals. | The fragment check ran on **13 of 69** refusals (the 13 that walked ≥1 edge — set identity verified); **56 `null`, 11 empty, 2 = `["is"]`**. The two `["is"]` rows are `og018` and `pv002`, and they are **not** equivalent: `og018`'s *only* failure reason was the substring hit, so **a correct refusal was scored down by the grader alone** — with A-7 removed it passes. `pv002` tripped A-7 too, but it had *already* failed `refused_an_answerable_question` (A-5 — it walked `bat --has_property--> round disc` and then refused anyway), so A-7 was **not load-bearing** for it. The other 67 passed, 56 of them with the check never running. **The defect is not confined to this demo:** in the **weaving** run the same fragment flagged **`nf004`**, whose `why` is the single entry `semantics:['no_claim_when_refusing']` (`template_used = "{node0} is {node1}."`, `template_fragments_leaked = ["is"]`, verified in `unseen_results.json`) — which makes A-7 **load-bearing for weaving's verdict as well**, since `nf004` is the whole of that run's `nonsense_fallback` **6/7 = 0.857 < 0.90**. The prior weaving report attributes the flag to a different substring (`has` ⊂ `has_property`); the fragment that actually fired is `is`. Counterfactual, stated but not applied: without A-7 that row passes, the gate reads **7/7**, overall **390/402 = 0.9701**, and the sole failed criterion clears — the published **NOT PASSED, 389/402 = 0.9677** is reported unchanged, as is the aviation sign-off. | Unpatched. Root cause shared with A-6: over-generic fragments under a substring test |
| **A-8** | **Decoder surface forms are ungrammatical and the grader cannot see it.** `_relation_phrases` values are inserted verbatim into `"{node0} {phrase} {node1}"` (`template_decoder.py:214-215`): `has_part → "has part of"` (and `has_part` has **no** `CHAIN_TEMPLATES` key, so it always takes this path) gives *"bowl **has part of** rim"*; `is_a → "is a"` gives *"terracotta **is a** earthenware"*. | **21/462 = 4.55%** of answers broken — 12 `has part of` + 9 `is a <vowel>` (no row carries both) — **21/21 passed.** Pre-exists in every earlier set, counted with the same two patterns and de-duplicated per row: **Stage C 17, Stage E 13, Stage F 6, Aviation 19, Weaving 4** (`has part of` alone: 5, 7, 1, 2, 4). | Unpatched; the grader checks label ordering and provenance, not English |
| **A-9** | **`entity_not_found` never fires on out-of-graph subjects.** Each subject grounds on an unrelated in-graph node and is refused by the *identity* gate instead. Right outcome, wrong signal. | Showcase **0/23**; weaving **0/23**; aviation `og013` answered confidently (1/22). | Unpatched |
| **A-10** | **Run 1's fallback category was not new data** — `B.NONSENSE_CUE_FREE` was never rebound, so all 6 questions were byte-identical to Stage F / aviation / weaving. A defect in **my harness**, found by provenance check, corrected by restoring the authored list. | 6/6 nonsense rows differed between runs; **456/462 questions byte-identical**; 0 changes outside `nonsense_fallback`; graph content hash unchanged. **Second cause, also disclosed:** run 1's single `nonsense_fallback` failure (`nf002`) was scored down by **A-7**, not by a wrong answer — it walked 1 edge on the pottery graph, so the fragment check ran, whereas at home in Stage F it walked 0 and the check never ran. Without A-7, run 1 would have been 7/7 on that gate, **446/462** overall (A-7 also un-fails `og018`) — and PASSed. | **Corrected and fully disclosed in Part 4.3**, with both runs preserved and both scores published |
| **A-11** | **The refusal boilerplate can contradict its own walk.** The refusal text is rendered by `render_no_relation()` (`template_decoder.py:147-158`) with an optional *"(No {relations} relation found.)"* clause chosen from the **planned** chain — not from whether an edge was actually walked, and not from the real reason for refusal. On these rows the real reason was `honest_by_identity` (ungrounded anchor), yet the sentence asserts relation *absence*. | **13 / 69 refusals (18.8%)** contain *"(No `<rel>` relation found.)"* while their own `path_edges` contains exactly that relation — `og002 og004 og005 og006 og007 og008 og009 og010 og013 og018 og019 og021 pv002` (relations `is_a` ×9, `has_property` ×2, `part_of` ×1, `has_part` ×1). **0 of the 13 were scored down by it** — the 2 that failed both trip A-7's substring instead (`pv002` carries A-5 as well, a genuine false refusal). | Unpatched. Two readings are defensible: false if read as a claim about the *anchor the system named*, true if read as a claim about the *out-of-graph subject the question asked about*. The text does not say which, which is the defect |

## Not a defect of the system

- **Run 1 → run 2** was an input-specification error in the test harness, not a
  change to any mechanism. No threshold, template, gate, configuration value or
  grading rule was altered. See Part 4.3 for the pre-scoring proof.

---

# Appendix B — Artefact index

## This demo — `test_results/showcase_demo/`

| File | SHA-256 / role |
|---|---|
| `build_graph.py` | builds + validates + freezes the pottery graph |
| `showcase_eval.db` | graph — content `0d6edf6e…`, file `e465e0707ae9b0f78dff617f365cd03ad15dbd274af7ee5c9af1d86cc7378d2e` |
| `graph_content_hash.py` | computes the **content** hash above (excludes wall-clock columns); `python -B graph_content_hash.py showcase_eval.db` → `0d6edf6eb6d78ac45230abc484f37f214ffb26bccbbe460babe6a92155680c48` |
| `frozen_graph_digest.txt` | `e465e070…  showcase_eval.db` |
| `generate_questions.py` | derives 462 questions; asserts every template against the live planner |
| `showcase_questions_frozen.json` | set — `5ff31283180093d5ee956d8d5855009de3e0fc153f2d621463e3dc42f96d08da` |
| `frozen_questions_digest.txt` | `5ff31283…  showcase_questions_frozen.json` |
| `run_eval.py` | pins both digests; verifies before *and* after scoring |
| `showcase_results.json` | **the scored run** — `4be9108beb7939cdb8e53645ce06b3a7cb6a49bf85377bc00e923fba7bd04360` |
| `determinism_rerun.json` | independent re-run — `59f6c8543ec0d7271457d056070d1dd572290652182574321d0162c1ead30825`; 0/21,252 field differences vs the scored run |
| `run1_stale_nonsense/showcase_results_run1.json` | run 1 — `740097e5b76f3c2f1ddc594516810c3e2943736966c8bbb1d05238e9d9e00f41` |
| `run1_stale_nonsense/showcase_questions_frozen_run1.json` | run 1 set — `bce0183d456e8fd179270d2c13ab374e495a9576e225a160f3d430ed42c90c55` |

## Prior evaluations

| Run | Result file | Verdict |
|---|---|---|
| Stage A | `test_results/stage_a/stage_a_results.json` | PASS |
| Stage B | `test_results/stage_b/stage_b_results.json` | PASS |
| Stage C | `test_results/stage_c/stage_c_results.json` | PASS |
| Stage D | `test_results/stage_d/stage_d_results.json` | PASS |
| Stage E | `test_results/stage_e/stage_e_results.json` | PASS |
| Stage F | `test_results/stage_f/stage_f_results.json` | **NOT PASSED** |
| Aviation PoC | `test_results/final_validation/final_results.json` + `FINAL_POC_REPORT.md` | PASS (referenced, not restated) |
| Weaving unseen | `test_results/unseen_e2e/unseen_results.json` + `UNSEEN_E2E_REPORT.md` | **NOT PASSED** |

## Where each component lives

| Component | Primary source |
|---|---|
| Graph store | `graph/graph_component_implementation/sqlite_graph_store.py` |
| Encode / anchor | `scripts/glmx_ask.py` (`1_encode` l.898, `2_subgraph` l.1024), `configs/config_orchestrator.yaml` |
| Resonance | `resonance/tier1.py`, `resonance/tier2.py`, `configs/config_resonance.yaml` |
| Plan | `g2p/g2p_planner.py`, `configs/config_g2p.yaml` |
| Walk | `walker/graph_walker.py`, `configs/config_walker.yaml` |
| Decode | `decoder/template_decoder.py`, `CHAIN_TEMPLATES`/`RELATION_PHRASES` from `decoder/config_decoder.yaml` |
| Honesty gates | `scripts/glmx_ask.py` `:653`, `:916-941`, `:960`, `:1099` |
| Fallback | `configs/config_g2p.yaml:48-58`, `configs/config_orchestrator.yaml:11-17` |
| Determinism | `test_results/stage_e/stage_e_runner.py:656-688`, `test_results/showcase_demo/run_eval.py` |
| Learning | `learning/`, gated at `scripts/glmx_ask.py:445`, `:1271`, `:1341` |

---

*End of report.*
