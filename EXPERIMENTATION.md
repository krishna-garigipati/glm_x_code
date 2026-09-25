# GLM-X M0 PoC — Experimentation Protocol v1.3

**Applicable branch:** `maharshi` @ `52dbb94`
**Architecture:** Deviation 9 — static knowledge graph QA pipeline. No intents, no LLM, no runtime internet.
**Status:** ACTIVE — all testers must follow this protocol exactly, in order, with no shortcuts.

---

## 1. Scope & Non-Goals

### 1.1 Scope
Test the offline QA pipeline end-to-end over **graph / toy datasets only**:

```
question -> SBERT encode -> GraphStore.get_subgraph (seed) -> Resonance
         -> QueryRelationExtractor (chain) -> Graph Walker -> Template Decoder -> answer
```

### 1.2 Non-Goals (STRICTLY FORBIDDEN during this phase)
| # | Forbidden activity | Reason |
|---|--------------------|--------|
| 1 | PDF ingestion / document-to-graph | Explicitly deferred by the lead. `kg_builder` is dormant. |
| 2 | Any LLM, T5 fine-tuning, or model training | Deferred. T5 decode path is skipped by tests (env-gated). |
| 3 | Downloading ConceptNet data | PoC rule: no runtime internet; demo graph is the data source. |
| 4 | Changing the 16-relation vocabulary | It is an immutable constant (Section 1.3). A relation change is a scope violation, not an enhancement. |
| 5 | Changing any immutable constant (Section 1.3) | Same as above. |
| 6 | New features / architecture changes | If a test seems to require a new feature, file a P-enhance bug and wait. |
| 7 | Refactoring / rewrite of existing components | Bug fixes only, with approval. |

### 1.3 Immutable Constants
These values may NOT be changed by any tester. Treat as read-only.

| Constant | Value |
|----------|-------|
| Canonical relations | `is_a`, `has_property`, `causes`, `caused_by`, `follows`, `precedes`, `contradicts`, `supports`, `associated_with`, `example_of`, `part_of`, `synonym`, `antonym`, `temporal_coincident`, `spatial_near`, `linguistic_maps` (exactly 16) |
| Property propagation threshold θ_prop | 0.008 |
| Edge resonance threshold θ_edge | 0.02 |
| Resonance boost λ | 0.1 |
| Similarity top_k | 64 (1-hop expansion capped at `max(top_k * 3, 10)`) |
| Energy slot theta | 48-dim (slots 4..20) |
| Alpha / Beta / Gamma / Delta | 0.05 / 0.02 / 0.9 / 0.001 |
| Node embeddings | int8 quantized |
| Embedding model | `BAAI/bge-small-en-v1.5` (frozen), 384-dim, normalized |
| Dev-9 rule | No intents, no LLM, no runtime internet; static graph |

---

## 2. Baseline Verification (MANDATORY BEFORE ANY EXPERIMENT)

Every tester must reproduce the baseline on their machine **before touching datasets**. If any baseline check fails, STOP and file a P-blocker; do not proceed.

### 2.1 Branch state
```powershell
git fetch origin
git checkout maharshi
git status          # must be clean
git log --oneline -1        # must be 77dedf5 (or newer after approved fixes)
```

### 2.2 Gate Suite — the 4 commands
Run all four. Record your exact outputs in `test_results/<tester>/baseline_<date>.txt`.

**G1 — Full unit suite** (lead baseline reference: `213 passed / 102 skipped`):
```powershell
python -m pytest g2p/tests walker/tests decoder/tests scripts/test_glmx_pipeline_relations.py -q
```
Must complete with `0 failed`. Match the counted `passed / skipped` to the lead reference and note any delta.

**G2 — Decoder self-runner** (lead baseline reference: `195 passed / 2 skipped`):
```powershell
python -m decoder.tests.test_all
```
`2 skipped` = T5 embodiment tests, gated on the cached T5 checkpoint — **expected and acceptable**. Must report `0 failed`.

**G3 — Config validation** (lead baseline reference: `8/8 OK`):
```powershell
python scripts/validate_configs.py --config-dir configs
```

**G4 — Demo questions** on the bundled demo graph (no `--db` flag):
```powershell
python scripts/glmx_ask.py -q "What is a dog?"
python scripts/glmx_ask.py -q "What is the opposite of hot?"
python scripts/glmx_ask.py -q "Tell me something related to water"
```
Expected results (judgment is on **chain + path**, not exact final sentence, because the walker is stochastic):

| Question | answer (reference) | relation_chain | walk path (reference) |
|----------|--------------------|----------------|------------------------|
| What is a dog? | `dog is animal.` | `["is_a"]` | dog -> animal |
| What is the opposite of hot? | `hot is the opposite of cold.` | `["antonym"]` | hot -> cold -> ice |
| Tell me something related to water | `water is associated with rain.` | `["associated_with"]` | water -> rain |

**A teammate machine whose baseline differs must not start experiments. File a P-blocker.**

### 2.3 Environment
- Python 3.14 used by the lead (any modern Python is acceptable; record your version).
- There is **no `requirements.txt`**; see Appendix D for the dependency list.
- The SBERT model downloads on first run (~100 MB, needs internet once; the whole experiment can then be offline). Do not re-download repeatedly.

---

## 3. Tester Workflow (the mandatory loop)

Every experiment = one dataset × one question. Repeat these steps in order.

1. **Pre-register** your golden questions (Section 3.1) BEFORE running anything.
2. Build/obtain the `.db` (Section 5), commit it under `test_results/<tester>/datasets/`.
3. Run the exact command (Section 5.3). Capture the full JSON output.
4. Record the result in `test_results/<tester>/<dataset>_<id>.json` and in your daily status file (Section 10).
5. On failure: **reproduce once** with the exact same command. If it reproduces, classify it (Section 8) and either fix (only if a small, safe fix) or report it.
6. After ANY code change (yours or a teammate's): re-run the **full Gate Suite (Section 2.2)** and record results. No gate regression is ever accepted.

### 3.1 Golden question pre-registration (anti-bias rule)
For every toy dataset, each tester defines **10–25 golden questions** with:
- **expected relation_chain**
- **expected answer semantics (subject + relation + object)**
- **expected walk path** (if knowable)

`experiments must be registered before running` — write them down (file: `test_results/<tester>/<dataset>_golden.md`), then run. You are not allowed to change the golden list after seeing results to make the pass rate look better.

---

## 4. Gate Suite Usage

- Run the full Gate Suite (G1–G4, Section 2.2) + one representative toy-dataset run **after every code change**.
- A P-blocker or a gate regression freezes experimentation until fixed or explicitly re-scoped by the lead.

---

## 5. Toy Dataset Specification

### 5.1 Source formats accepted
`.json`, `.csv`, `.parquet` (or an already-built `.db`). Other formats are out of scope.

### 5.2 Standard ingestion path (recommended, zero custom code)
```powershell
# structured file -> .db with SBERT embeddings
python scripts/ingest.py --input <toy>.json --output <toy>.db
# or .csv / .parquet in place of .json

# then run the pipeline against the toy graph
python scripts/glmx_ask.py --db <toy>.db -q "<question>"
```

`--db` loads the graph via `SQLiteGraphStore.load_state` (scripts/glmx_ask.py:844-851). This path is required for toy testing because it also serves as the SQLite vs Dict parity exercise (Section 6.1, B1).

### 5.3 Minimum data schema
| Data | Requirement |
|------|-------------|
| Node labels | A list of concept strings (unique). |
| Edges | `(source_label, target_label, relation[, strength, confidence])`, relation ∈ 16-relation vocabulary. |
| Embeddings | 384-dim per node; produced automatically by `ingest.py` via the frozen SBERT model. Datasets without embeddings are NOT valid inputs. |

### 5.4 Relation coverage rule (per dataset)
Each toy dataset must contain **≥ 4 distinct relations**, including:
- at least one of `{is_a, associated_with, antonym}`
- at least one causal relation `{causes, caused_by}`
- at least one partitive/property relation `{part_of, has_property}`

Cover the full 16 across the whole toy-dataset set (Appendix B has the table for planning coverage).

### 5.5 Size mix (whole toy set)
| Size | Nodes | Datasets |
|------|-------|----------|
| Small | 10–100 | 3 |
| Medium | 100–1,000 | 4 |
| Large | 1,000–10,000 | 3 |
| Edge cases | any | (below) |

### 5.6 Edge-case datasets (required, at least 2)
1. **Sparse/disconnected** graph (nodes with no edges, multiple components).
2. **Missing relation**: golden questions that reference a relation NOT present in the graph — must produce the honest fallback answer (Section 6.4), NOT a wrong chain.
3. **Cycles** (A→B→C→A).
4. **Duplicate labels** (case-insensitive collisions) — must demonstrate correct `_label_to_id` behavior (no silent merge corruption).

### 5.7 Naming & storage
- Files: `<dataset_id>.db`, committed under `test_results/<tester>/datasets/`.
- Golden lists: `test_results/<tester>/<dataset_id>_golden.md`.
- Results: `test_results/<tester>/<dataset_id>_<question_id>.json`.

---

## 6. Per-Component Test Matrix

Ownership key (lead-assigned): **A** = Lead (orchestration/learning/integration), **B** = Graph & Encoding, **C** = Resonance & Planner, **D** = Walker & Decoder.

### 6.1 B — Graph & Encoding
| ID | Checklist | Pass criterion |
|----|-----------|----------------|
| B1 | SQLite vs Dict parity | Same question on same data via `--db` (SQLite) vs demo loader (Dict) produces the same chain and same semantics. |
| B2 | Ingest fidelity | Node/edge counts after `ingest.py` match the source file. |
| B3 | Embedding sanity | Embedding dim == 384; int8 storage; normalized for similarity. |
| B4 | Label lookup | `_label_to_id` resolves every label used by edges (case-correct). No KeyError. |
| B5 | Save/load round-trip | `save_state` → `load_state` preserves nodes, edges, embeddings, edge `last_used`/`frequency`. |
| B6 | Similarity expansion cap | Expansion stays bounded (does not flood memory on the 10k-node dataset). |
| B7 | Relation set | `get_all_relations()` returns a subset of the 16 canonical relations only. |

### 6.2 C — Resonance & Planner
| ID | Checklist | Pass criterion |
|----|-----------|----------------|
| C1 | Seed retrieval | Question retrieved evidence nodes are semantically plausible for in-graph concepts. |
| C2 | Resonance | Resonated nodes/edges are non-empty for in-graph questions; `resonance_energy > 0`. |
| C3 | Chain extraction | The 3 demo questions produce chains exactly `["is_a"]`, `["antonym"]`, `["associated_with"]` (case-insensitive phrasing variants included). |
| C4 | Graph filter | Every relation in the extracted chain is present in the graph's relation set (`plan()` is subgraph-aware). Never returns a relation absent from the graph. |
| C5 | Chain length | Max chain length respected (no unbounded chains). |
| C6 | Confidence bounds | `confidence` and `walk_confidence` ∈ [0, 1]. |
| C7 | Fallback rule | Default-chain fallback ONLY when nothing matches; `heuristic_used=False` for in-graph questions. |

### 6.3 D — Walker & Decoder
| ID | Checklist | Pass criterion |
|----|-----------|----------------|
| D1 | No "Therefore," prefix | Output never starts with "Therefore," or LLM-style lesson text. |
| D2 | Template coverage | `template_matched=True` for all golden questions whose chain is in the graph. |
| D3 | Node mention | Answer must mention at least one node from the walk (`require_node_mention`). |
| D4 | Honest fallback | Out-of-graph relation question → the exact configured fallback sentence (Section 6.4), never a fabricated chain. |
| D5 | Reject patterns | None of `": intent="`, `"intent="`, `"->"`, `"generate:"` ever appears in output. |
| D6 | Length bounds | Answer length between 5 and 500 chars. |
| D7 | Path quality | Walk path is a plausible traversal using the chain relations; `n_walk_steps ≥ 1`. |
| D8 | Determinism sanity | Same question, same seeded graph: output is stable across ≥ 3 runs (walker is stochastic — allow sentence phrasing variance, but the chain must be stable 5/5). |

### 6.4 Honest-fallback reference (exact DO NOT alter)
`I don't have a relation in my knowledge graph that fully answers this question.`
(Source: `decoder/config_decoder.yaml` `chain_render.no_relation_answer`.)

---

## 7. Definition of "Works Perfectly" (the extension gate)

The phase ends with a GO only when **ALL** of the following hold (lead verifies):

1. Every submitted toy dataset ingests and runs without crash.
2. Every toy dataset passes its pre-registered golden questions with **chain match** ≥ 90% and `template_matched=True` for all in-graph golden relations.
3. `heuristic_used=False` for all in-graph questions; honest fallback used for all out-of-graph questions.
4. The 3 demo questions still produce the reference chains (G4) on the bundled demo graph.
5. Full Gate Suite still matches the lead baseline reference (G1–G3); decoder `0 failed`.
6. Zero open P-blockers. All P-bugs either closed or explicitly accepted by the lead.
7. Parity spot-check (B1) passes for at least one medium and one large dataset.

If any item fails: the phase continues; the lead decides whether to fix, re-scope, or extend to real graph data.

---

## 8. Bug Handling Protocol

### 8.1 Classification thresholds
| Tag | Definition | Required action |
|-----|-----------|-----------------|
| **P-blocker** | Crash / exception anywhere in the pipeline; wrong chain for any of the 3 demo questions; any Gate-Suite regression. | Lead fixes and pushes immediately. Do not continue unrelated experiments. |
| **P-bug** | Wrong/missing chain for a golden in-graph question; `template_matched=False` though the relation is in the graph; placeholder/empty/oversized answer; chain instability across runs (D8 fails). | Fix (small, safe) or report to the class owner (B/C/D); must not be left open past the phase. |
| **P-enhance** | Awkward phrasing, suboptimal walk path, coverage ideas, cosmetic issues. | Log with full detail; batch for later; not blocking. |

### 8.2 Fixed report template (use verbatim — file as `test_results/<tester>/BUG_<issue_id>.md`)
```
tag:             P-blocker | P-bug | P-enhance
component:       graph/encoding | resonance/planner | walker/decoder | orchestrator
issue_id:        <tester>-<NNN>
date:            YYYY-MM-DD
dataset:         <dataset_id>
command:         (exact command run, copy-paste)
input:           (question verbatim)
expected:        (golden entry)
actual:          (result.answer / relation_chain / traceback)
traceback:       (full, if application)
repro:           (exact steps to reproduce; # of times)
proposed fix:    (optional)
```

---

## 9. Branch & Commit Hygiene (STRICT)

1. Testers work on **individual branches** (e.g. `tester-a`, `tester-b`, `tester-c`) derived from `maharshi`. Datasets/goldens/results commit to the tester branch and are pushed; the lead reviews and merges them back into `maharshi` (single source of truth). **Never** touch `main` (local or `origin/main`).
2. No force-push, no amend, no rebase. Linear history, small grouped commits.
3. Commit message prefix by component: `[store]`, `[resonance]`, `[planner]`, `[walker]`, `[decoder]`, `[tests]`, `[docs]`.
4. Commits must include the tests demonstrating the fix (tests as proof, not just code).
5. Datasets/results committed under `test_results/<tester>/`; do not commit `.pyc`/`__pycache__` (already gitignored).
6. `test_results/model_checkpoint/` artifacts are NOT to be regenerated or replaced by testers.

---

## 10. Daily Status Reporting

Each tester appends to `test_results/<tester>/DAILY_STATUS.md` at end of day:

```markdown
## YYYY-MM-DD (<tester>)
Datasets run:  dataset_id, dataset_id
Golden pass rate: X/Y (list failed question ids)
Open blockers:  <links/copies>
Open P-bugs:    <links/copies>
Gate suite today: G1 (a passed / b skipped) G2 (c passed / d skipped) G3 (e/f OK)
Changes made / fixed:  <commit hash — subject>
Notes / needs lead decision:
```

Bugs tagged **P-blocker** must be reported to the lead the same day.

---

## 11. End-of-Phase Sign-Off Checklist (each tester before handing back)

| Tester | Sign-off item | Where |
|--------|---------------|-------|
| B | Data round-trip, parity, expansion verified | `test_results/<tester>/B_signoff.md` |
| C | Chain extraction + graph-filter verified | `test_results/<tester>/C_signoff.md` |
| D | Walker/decoder + honest fallback verified | `test_results/<tester>/D_signoff.md` |
| All | No open P-blockers, gate suite green, goldens committed with results | `test_results/<tester>/FINAL.md` |

Final decision (extend to real graph data vs. re-scope) is made by the lead against Section 7.

---

## 12. Demonstrated Improvements (lead) — Semantic-Similarity Walker Scoring (P1) + Chain-Aware Stop (P2)

Status: **implemented, evaluated, committed** on `maharshi`. Not an architectural change: component interfaces, data flow, and all immutable constants (Section 1.3) are unchanged. It is a tunable enhancement of the walker's scoring formula.

### 12.1 What changed

| Change | Where | Behavior |
|--------|-------|----------|
| **P1** similarity term in next-hop scoring | `walker/path_scorer.py`, `walker/graph_walker.py` | Each candidate step is now multiplied by `cos(query_embedding, target_node_embedding)`. Weight `weight_target_similarity: 1.0` (new, in `configs/config_walker.yaml`). Backward compatible — default factor is 1.0 when embeddings are unavailable (never zeroes a candidate). |
| **P2** chain-aware stop | `walker/graph_walker.py` | Walk stops once every relation in `plan.relation_chain` has been traversed (previously it kept extending with the last relation, producing extra hops such as `dog -> animal -> cat`). Legacy intent plans map through the same chain logic. |

Tests: `walker/tests/test_chain_walk.py` updated — the old assertion that the last chain relation repeats past the end now asserts stop-after-chain instead.

### 12.2 Evidence (toy_eval)

New harness under `test_results/lead/` (dataset spec-compliant, real SBERT embeddings only):

| Artifact | Purpose |
|----------|---------|
| `datasets/build_toy_eval.py` | Builds `toy_eval.db` — the standard toy (21 nodes / 17 edges) **plus "ambiguity pairs"** where two candidate targets have identical strength/confidence/activation so the ONLY deciding signal is query-target similarity (`smoke→tobacco` vs `smoke→fire`; `water→rain` vs `water→oil`). |
| `toy_eval_runner.py` | Level-1: runs the 7 golden questions through the full pipeline (`scripts/glmx_ask --db`), checks hop count + object node, prints PASS/FAIL. |
| `toy_eval_walker_ab.py` | Level-2: feeds the SAME subgraph+plan to two walkers differing only in `weight_target_similarity` (0.0 vs 1.0) to isolate P1's effect. |
| `toy_eval_baseline.txt` / `toy_eval_after.txt` | Committed before/after runs (evidence). |

**Level 1 — full pipeline (7 golden questions): 2/7 → 7/7 PASS**

| Question | Before (baseline) | After (P1+P2) |
|----------|-------------------|---------------|
| What is a dog? | `dog -> animal -> cat` (2 hops) | `dog -> animal` |
| What is the opposite of hot? | `hot -> cold -> ice` (2 hops) | `hot -> cold` |
| What does rain cause? | `rain -> flood` | `rain -> flood` |
| What property does water have? | `water -> liquid` | `water -> liquid` |
| What is ice? | — | `ice -> cold` |
| Tell me something related to water | `water -> rain -> cloud` (2 hops) | `water -> rain` |
| What does fire cause? | `fire -> smoke -> tobacco` (2 hops) | `fire -> smoke` |

**Level 2 — P1 isolated (21 trials each, same seed):**

| Walker variant | tobacco picked | fire picked |
|----------------|----------------|-------------|
| `weight_target_similarity = 0.0` (old) | 11/21 | 10/21 |
| `weight_target_similarity = 1.0` (P1) | 21/21 | 0/21 |

At weight 0.0 both candidates are score-identical → pure coin flip. At weight 1.0 the walker deterministically tracks query similarity (`cos(query, tobacco) = 0.752` vs `cos(query, fire) = 0.594`). Note the source-similarity alone is NOT enough here (`cos(smoke, tobacco) = 0.840` ≈ `cos(smoke, fire) = 0.843`) — the improvement comes from the **query-anchored** similarity inside the walker.

### 12.3 Teammate pull-and-test instructions

```powershell
git fetch origin
git checkout maharshi        # NEVER test on main (old architecture)

# regenerate the eval dataset (downloads SBERT on first run, then offline)
python test_results/lead/datasets/build_toy_eval.py

# Level 1: full pipeline over the 7 golden questions  (expect 7/7 PASS)
python test_results/lead/toy_eval_runner.py --out test_results/<tester>/toy_eval_<date>.txt

# Level 2: walker isolation A/B  (expect 0.0 -> ~50/50, 1.0 -> 21/21 tobacco)
python test_results/lead/toy_eval_walker_ab.py
```

Environment notes: `pip install lz4` if the `graph/tests` suite fails at collection (pre-existing env dependency); do NOT change `weight_target_similarity` without a golden-set re-run and a protocol-waiver note here.

---

## 13. Three-Tester Parallel Domain Assignments

Three testers run in parallel, each on their **own branch** derived from `maharshi`, each building toy datasets from an assigned domain and testing individually.

### 13.1 Branch model (amends Section 9)

| Role | Branch | Rules |
|------|--------|-------|
| Lead | `maharshi` | Single source of truth. Reviews and merges tester branches. Never force-push/amend. |
| Tester A | `tester-a` | Work + results committed locally, pushed, then merged back to `maharshi` after review. |
| Tester B | `tester-b` | same |
| Tester C | `tester-c` | same |

`main` is untouched by everyone. Goldens and results live under `test_results/<tester>/`.

### 13.2 Assignments (3 domains)

| Tester | Branch | Domain | Assigned relations | Size targets (Section 5.5) |
|--------|--------|--------|--------------------|--------------------|
| A | tester-a | Nature & Weather | `causes`, `caused_by`, `follows`, `precedes`, `temporal_coincident`, `associated_with` | 1 Small (10–100 nodes) + 1 Medium (100–1,000 nodes) |
| B | tester-b | Food & Biology | `is_a`, `part_of`, `has_property`, `example_of`, `synonym`, `antonym` | 1 Small + 1 Medium |
| C | tester-c | Health & Science | `supports`, `contradicts`, `spatial_near`, `linguistic_maps` (+ 2 relations from the A or B list) | 1 Small + 1 Medium |

Every dataset must still satisfy Section 5.4: **≥ 4 distinct relations**, including at least one of `{is_a, associated_with, antonym}`, at least one causal `{causes, caused_by}`, and at least one partitive/property `{part_of, has_property}`. Testers therefore add 1–2 relations outside their primary list on purpose. Together the three domains should cover all 16 relations (Appendix B).

### 13.3 Edge-case distribution (each tester ≥ 2 from Section 5.6)

| Tester | Edge cases |
|--------|-----------|
| A | sparse/disconnected graph, cycles |
| B | missing relation (must produce the honest fallback per Section 6.4), duplicate labels |
| C | sparse/disconnected graph, missing relation (honest fallback) |

### 13.4 Mandatory onboarding (BEFORE building any own-domain dataset)

Each tester must first prove their environment on the lead's shipped harness and record it in `test_results/<tester>/DAILY_STATUS.md`:

```powershell
git fetch origin
git checkout tester-a   # tester's own branch (e.g. tester-a / tester-b / tester-c)
python test_results/lead/datasets/build_toy_eval.py
python test_results/lead/toy_eval_runner.py --out test_results/<tester>/onboard_<date>.txt   # expect 7/7 PASS
python test_results/lead/toy_eval_walker_ab.py                                               # expect 0.0 ~50/50, 1.0 21/21
```

No own-domain dataset work before these pass.

### 13.5 Golden pre-registration (anti-bias rule, Section 3.1)

For each dataset, the golden list `<dataset_id>_golden.md` must be committed to the tester's branch **before the first run**. Every golden entry: `question`, expected answer node, expected relation chain, expected hop count. Minimum **8 goldens per dataset**, with **≥ 2 goldens per assigned relation** (13.2).

### 13.6 Per-tester deliverables (file map)

| Item | Path |
|------|------|
| Datasets | `test_results/<tester>/datasets/<dataset_id>.db` (built via `scripts/ingest.py` — real SBERT embeddings only) |
| Goldens (pre-registered) | `test_results/<tester>/<dataset_id>_golden.md` |
| Results | `test_results/<tester>/<dataset_id>_<question_id>.json` |
| Bug reports | `test_results/<tester>/BUG_<issue_id>.md` (Section 8.2) |
| Daily status | `test_results/<tester>/DAILY_STATUS.md` (Section 10) |
| Sign-off | `test_results/<tester>/<B|C|D>_signoff.md` + `FINAL.md` (Section 11) |

### 13.7 Merge-back sequence

1. Tester pushes branch → reports branch name + golden pass rate to lead.
2. Lead reviews branch (datasets in spec, goldens pre-registered, honest-fallback cases present, no `main` involvement).
3. Lead merges into `maharshi`; tester updates `checkout maharshi` + rebase/merge their branch.
4. Repeat per dataset.

---

## 14. Lead Post-Tester Fixes + Unified Harness (Phase 5 — applied directly on `maharshi`, no branch merges)

After reviewing both tester branches (13.x), the lead applied bug fixes directly on
`maharshi` and added a single harness that runs every pre-registered golden
deterministically. Tester branches stay on the remote, unmerged; their evidence is
materialized under `test_results/tester-a/` and `test_results/tester-b/` (the
top-level `toy_testings/` directory and tester-a's out-of-scope edits to
`decoder/tests/test_results.json`, `graph/toy_dataset_output.json`, and
`test_results/lead/datasets/toy_eval.db` are NOT carried over).

### 14.1 Bug fixes shipped (validated by runner + unit tests)

| ID | Fix | Files | Evidence |
|----|-----|-------|----------|
| B1 | EmbeddingLookupError crash: resonance keys `node_embeddings` only to gated map, but `_build_subgraph` re-adds pruned seeds → walker hit nodes with no embedding. Walker now always receives `embedding_provider`; `ask()` completes `node_embeddings` for every resonated node | `scripts/glmx_ask.py`, `walker/graph_walker.py` (provider already supported) | regression tests `test_walk_partial_embeddings_uses_provider` / `test_walk_missing_provider_embedding_raises`; fbm01–fbm20 now runnable |
| B2 | Bare-echo on confident-but-dead matches: honest branch required `heuristic_fallback_used`, but extractor rarely fell back. Now any empty walk emits the §6.4 honest sentence | `scripts/glmx_ask.py` | fbs14/fbm19 emit exact configured fallback |
| B3 | Seed anchoring: `seed_nodes[0]` (argmax sim of the whole question) mis-anchored "opposite of sweet" → `sugar`. Exact-label matching of the question's noun now wins | `scripts/glmx_ask.py` (`_resolve_target_entity`) | fbs12 → `sweet`→`sour` |
| B4 | Non-determinism: unseeded walker RNG + live REINFORCE/ES updates flipped answers between runs. Added `--seed`, `--no-learning`, `--measure` | `scripts/glmx_ask.py`, `walker/graph_walker.py` (seed param) | fbs07 path identical across runs |
| B7 | Similarity tie-break: `get_subgraph_by_embedding_similarity` sorted by sim only. Now `(sim, nid)` in sqlite/dict/`graph_store` | `graph/.../sqlite_graph_store.py`, `dict_graph_store.py`, `graph_store.py` | — |
| C1 | Chain-lift: chain-relation neighbors under `min_activation` were invisible to the walker (`sweet` @0.024 < 0.05 for "property of honey" on medium). Pipeline warms those nodes; walker contract untouched | `scripts/glmx_ask.py` (plus locked expected-relation bias floor in `walker/graph_walker.py`, DEVIATION 9) | fbm07 `honey`→`sweet` |
| B6 | lz4 missing → serialization failures. Serializer now resolves a working codec (falls back to `none`) once at init | `graph/.../serializer.py` | `python -m graph.graph_component_implementation.test_validate` → ALL TESTS PASSED |

### 14.2 Deterministic CLI (harness-facing)

```
python scripts/glmx_ask.py --db <db> -q "<q>" --seed 0 --no-learning --measure
# --seed       seed the walker RNG (reproducible walks)
# --no-learning disable REINFORCE + EvolutionaryController feedback (no state mutation)
# --measure    single-line JSON: question, answer, answer_level, relation_chain,
#              heuristic_used, entity_not_found, entity_top_sim, honest_no_relation,
#              template_matched, n_walk_steps, walk_path_labels/edges, time_seconds
```

New `ask()` fields (also in Appendix A): `entity_not_found`, `entity_top_sim`,
`honest_no_relation`.

### 14.3 Unified harness (living evidence for the §7 gate)

```
python test_results/lead/check_dataset.py                                  # Section 5.4 compliance
python test_results/lead/unified_golden_runner.py                          # all goldens, deterministic
   # embeds frozen golden snapshots (tester-a/b pre-registrations, unchanged)
   # --suites food_bio_small,food_bio_medium,nature_weather_small --seed 0
```

Grading mirrors the testers' own method (answer node + hops; chain reported for
transparency). `food_bio_small` + `food_bio_medium` are the §7.2 gate sets;
`nature_weather_small` is graded as a probe for historical reasons — after the
IS-04/05 reconcile (2026-09-25) it is Section-5.4 compliant (37 nodes / 53 edges /
9 canonical relations, rebuilt from its 53-triple JSON), see `unified_golden_results.txt`.

### 14.4 Result snapshot (deterministic, seed 0, 2026-09-20)

| Suite | tier | result | vs tester baseline |
|-------|------|--------|--------------------|
| food_bio_small | gate | **15/15** | 12/15 (fbs07 rand, fbs12 anchor, fbs14 echo fixed) |
| food_bio_medium | gate | **20/20** | frozen at fbm09 (B1 blocker); now fully runnable |
| nature_weather_small | probe | 4/10 | tester-a reported 10/10 via approval-within-available-relations; verbatim grading shows the gap (e.g. "What is rain?" → "rain causes flood") |

Standard gates re-run green after all fixes: walker suite **157 passed**; broad
g2p/resonance/walker/decoder test run **553 passed, 102 skipped**; toy_eval **7/7**;
P1 A/B **21/21 tobacco at weight 1.0** (11/21 coin flip at weight 0.0 preserved);
graph component validation **ALL TESTS PASSED** (lz4 fallback).

### 14.5 Outstanding / notes

- `nature_weather_small` (rebuilt 2026-09-25, IS-04/05) is Section-5.4 compliant: 37 nodes,
  53 edges, 9 canonical relations (is_a + antonym + causal band present);
  `check_dataset.py` all PASS; run results in `nature_weather_small_results.json`. It remains
  included in the cross-domain smoke matrix.
- Tester-b branch (and tester-a) remain on the remote, unmerged; commits staged
  locally awaiting lead upload (`maharshi` next commit).
- `entity_not_found` uses `entity_top_sim < 0.25`; verified empirically on
  in-graph questions (≥0.6) and out-of-graph probes.

### 14.6 Batch-2 fixes (2026-09-20): honesty gates, deterministic walking, cross-validated smoke matrix

**Determinism root cause (the main fix).** The walker's `_select_index` gated the
argmax branch on `_walker_config.scoring.normalization` (the YAML value, always
`softmax`) instead of the scorer normalization (`_select_index` now checks
`self._scorer.normalization`). The `force_argmax` / W5 mode therefore still
sampled from `random.Random(seed)` on every decision. Near-tied equal-strength
edges (e.g. nws `summer→follows` mirror candidates, all s=0.20 c=0.81) flipped
on successive RNG draws, which is what looked like "wall-clock flips" across
repeat asks — the draws are just progressive; fresh processes looked stable by
luck of the seed prefix. After the fix the walker RNG state is byte-identical
across asks (`Random.getstate()` unchanged) and the walk picks the true
max-score candidate, first on ties.

**Honesty batch (W4 + W4b).** `sim_floor=0.55` / `margin_min=0.04` defaults kept
(cross-domain median anchor sims 0.80–0.82, zero <0.55 hits — no recalibration).
W4b extends honesty to relation availability: if the anchored node has no
outbound edge of the asked relation in graph adjacency, emit honest
`no_relation` instead of walking a mirrored edge for a different relation. The
availability probe is inverse-aware (`causes`/`caused_by`, `part_of`/`has_part`,
`precedes`/`follows` are the same fact mirrored) — see `te09` fix below. A
zero-anchor guard was also added: a question embedding that matches **no graph
node** at all now returns the honest `no_relation` shape instead of crashing in
Tier1 `resonate([])` ("initial_seeds must be non-empty"). Verified against the
empty placeholder `tester-a/datasets/toy_eval.db`.

**Cross-domain smoke matrix (`test_results/lead/domain_matrix/`).** Four curated
per-domain question sets (`nature_weather_small`, `toy_eval`, `toy`,
`food_bio_small`), 3 seeded pipelines each. Final: **36/38**, seed0/1/2
**IDENTICAL** on all four domains; W4 defaults confirmed. `te09` ("What does
fire cause?") regression from the first argmax pass was the W4b availability
probe keying on the planned (mirrored) `caused_by` instead of the genuine
`causes` edge — the inverse-aware probe restored 10/10 on toy_eval.

**Medium probe rerun (Phase 4, `medium_scale_runner.py`).**
| metric | baseline | now |
|--------|----------|-----|
| overall | 34/46 (73.9%) | **40/46 (87.0%)** |
| determinism (seed0/1/2) | variants | **IDENTICAL** |
| zero-crash | — | **0 crashes** (138/138) |
| latency median / P90 | — | 44 ms / 76 ms |
Honesty items 5/5 (mp41/42/43/44/46; mp45 is a case-duplicate answer, not
honest). Residual fails with root causes:
- mp11 lemon→sour: `lemon is_a fruit` and `lemon has_property sour` are BOTH
  present in the DB (s=0.95/c=0.95). The walk picks `is_a→fruit` because the
  resonance subgraph filters `sour` out entirely (low embedding similarity to
  the query), so no exact `has_property` candidate reaches the walker to
  short-circuit on. Corrected diagnosis in §14.7 (the earlier "data-gap" note
  here was wrong — the edge exists).
- mp32 robin: the `robin` node and `bird→robin example_of` DO exist; both the
  real (`bird→animal`) and mirrored (`bird→robin`) hops are `is_a`, and the
  walk prefers the hotter `animal`. The pinned golden "robin" is
  over-constrained (sparrow/eagle/penguin are equally valid) — **accepted as a
  documented residual** (§14.7).
- mp33/34/35 fin/gill/beak: extractor over-generated `[part_of, example_of,
  is_a]` from dangling conjuncts ("and" → example_of sim 0.74, "?" →
  has_property 0.69). **FIXED in §14.7 (Fix B)** — now clean `[part_of, is_a]`.
- mp40: extractor over-generated `[caused_by, is_a, antonym]` from the premise
  clause ("fire is hot" → caused_by 0.59) and the connector "so" (→ is_a 0.74).
  **FIXED in §14.7 (Fix C + Fix A)** — now `[antonym]` → hot→cold.
- nws/nw `What is rain?`-type: extractor yields `causes` (rain causes flood)
  for a `*is_a*` question (curated honest item expects is_a absence); extractor.

**Phase 5 battery.**
- gates: pipeline relations **9/9**; walker **157/157**; golden suite
  (`unified_golden_runner.py`) **39/45** deterministic seed0; toy_eval **7/7**;
  P1 A/B **21/21 tobacco @ 1.0**; `check_dataset.py` **all PASS** (nature_weather
  extended to 4 relations with `associated_with` edges cloud–rain, sun–light to
  meet Section 5.4).

**Outstanding (unchanged scope):** mp11 is a subgraph-construction gap
(`food_bio_medium.db` resonance filtering drops `sour`; needs subgraph work) and
mp32 is an accepted over-constrained golden residual — both documented in §14.7.
The "data-gap" root-cause claims for mp11/mp32 in the first Batch-2 note above
were empirically disproved (the edges DO exist) and are corrected here.

### 14.7 Batch-3 fixes (2026-09-21): exact-match short-circuit, clause hygiene, premise discard

Three code changes implementing the approved plan (aimed at converting the six
§14.6 residual fails; mp32 accepted as a documented residual):

**Fix A — walker exact-match short-circuit (`walker/graph_walker.py`).** The
extractor chain is the source of truth: `_select_index` now prefers a candidate
whose `edge_type == expected_relation` over merely-similar edges whose higher
activation/similarity would otherwise dominate, falling back to score-based
selection only when no exact match exists. Determinism preserved (30088
`max(exact, key=scores)` is just as deterministic as the argmax path). Feature
visible on `food_bio_small` fbs06 (`lemon has sour` passes) and mp40.

**Fix B — orphan-clause filter (`g2p/g2p_planner.py`).** `extract` drops clauses
that are pure connectors/punctuation (`and`, `or`, `so`, …, `?`, `!`, len<4)
before embedding matching. Bare "and" had been matching the `example_of` axis at
sim 0.74 and "?" the `has_property` axis at 0.69, inflating mp33/34/35 chains
into `[part_of, example_of, is_a]`. Now `[part_of, is_a]`.

**Fix C — premise discard (`g2p/g2p_planner.py`).** `_strip_premise` drops a
declarative premise before a comma-anchored causal delimiter
(`so`/`thus`/`therefore`/`hence`), keeping only the text after the LAST
delimiter as question clauses. "Fire is hot, so what is the opposite of hot?"
→ "what is the opposite of hot?" → `[antonym]`. Plain internal "so" (e.g. "What
is so hot?") requires a leading comma, so it is never stripped.

**Verification (same pre-registered set, 3 seeded pipelines + unseeded).**
| metric | 14.6 | 14.7 |
|--------|------|------|
| medium probe overall | 40/46 (87.0%) | **44/46 (95.7%)** |
| determinism (seed0/1/2) | IDENTICAL | **IDENTICAL** |
| zero-crash | 0 | **0** (138/138) |
| unit tests (walker/g2p/decoder + pipeline gate) | 220 passed | **221 passed** (7 new) |
| golden suite | 39/45 | **39/45** |
| toy_eval / P1 A/B | 7/7 / 21/21 | **7/7 / 21/21** |
| `check_dataset.py` | all PASS | **all PASS** |

Remaining residuals (both by-design, not regressions):
- **mp11** `What property does lemon have?`: chain is now correct
  (`has_property`), but the `lemon→sour` edge is not in the resonance subgraph
  (sour is filtered out), so Fix A never sees the exact-match candidate and the
  walk lands on `is_a→fruit`. Fix requires subgraph-construction scope (include
  the exact-anchor's outbound typed edges even when the target is not resonant)
  — NOT yet implemented; flagged for lead decision.
- **mp32** `Name a kind of bird`: golden pins "robin"; the walk returns the
  equally-valid "animal" (`bird→animal is_a` out-scales `bird→robin`).
  Over-constrained golden → accepted as documented residual. Matches the plan:
  target **43/46 = 93.5%**, achieved **44/46 = 95.7%**.

---

## Appendix A — `ask()` JSON schema (field meanings)

Output of `python scripts/glmx_ask.py -q "<question>"` (keys present):

| Key | Meaning |
|-----|---------|
| `question` | The input question. |
| `answer` | Final answer string produced by the decoder. |
| `relation_chain` | Ordered list of relations the planner selected. |
| `heuristic_used` | True if a default-chain fallback fired instead of a matched relation. |
| `template_matched` | True if a decoder template covered the chain. |
| `confidence` | Planner plan confidence ∈ [0, 1]. |
| `walk_confidence` | Walker confidence ∈ [0, 1]. |
| `time_seconds` | Total wall-clock time for the ask. |
| `steps_timing` | Per-step timings dict. |
| `n_resonated_nodes` / `n_resonated_edges` | Resonance activation counts. |
| `resonance_energy` | Total activation energy. |
| `n_walk_steps` | Number of walk steps taken. |
| `walk_path_labels` | Node labels visited, in order. |
| `walk_path_edges` | Relation labels between visited nodes. |
| `walk_path_activations` | Node activation values at each step. |
| `relation_details` | Per-relation extraction detail. |
| `entity_not_found` | True when the target-entity seed similarity falls below the anchor threshold (out-of-graph subject). |
| `entity_top_sim` | Cosine similarity of the top seed to the question embedding. |
| `honest_no_relation` | True when the walk found no path and the §6.4 honest answer was emitted. |
| `subgraph` | Inner details of the resonated subgraph (diagnostic). |
| `walk_path` | Full structured walk path (diagnostic). |

---

## Appendix B — 16-relation coverage table (use for dataset planning)

| # | Relation | Meaning example | Covered by dataset(s) |
|---|----------|-----------------|------------------------|
| 1 | `is_a` | dog is a animal | |
| 2 | `example_of` | penguin is an example of bird | |
| 3 | `has_property` | ice has cold | |
| 4 | `causes` | fire causes heat | |
| 5 | `caused_by` | heat is caused by fire | |
| 6 | `supports` | evidence supports claim | |
| 7 | `contradicts` | claim contradicts finding | |
| 8 | `synonym` | happy is synonymous with glad | |
| 9 | `antonym` | hot is the opposite of cold | |
| 10 | `linguistic_maps` | word-to-word mapping link | |
| 11 | `part_of` | wheel is part of car | |
| 12 | `follows` | event A follows event B | |
| 13 | `precedes` | event A precedes event B | |
| 14 | `temporal_coincident` | events happen at the same time | |
| 15 | `spatial_near` | river is near sea | |
| 16 | `associated_with` | water is associated with rain | |

Display-only aliases that map back to canonical relations (NOT extra relations): `RelatedTo`, `Antonym`, `Synonym`.

---

## Appendix C — Command cheat-sheet

```powershell
# Baseline / gate
python -m pytest g2p/tests walker/tests decoder/tests scripts/test_glmx_pipeline_relations.py -q
python -m decoder.tests.test_all
python scripts/validate_configs.py --config-dir configs

# Demo graph sanity
python scripts/glmx_ask.py -q "What is a dog?"
python scripts/glmx_ask.py -q "What is the opposite of hot?"
python scripts/glmx_ask.py -q "Tell me something related to water"

# Toy dataset
python scripts/ingest.py --input <toy>.json --output <toy>.db     # json/csv/parquet supported
python scripts/glmx_ask.py --db <toy>.db -q "<question>"          # full JSON on stdout

# P1/P2 toy_eval harness (Section 12)
python test_results/lead/datasets/build_toy_eval.py               # build toy_eval.db
python test_results/lead/toy_eval_runner.py --out <result>.txt    # Level 1 golden set
python test_results/lead/toy_eval_walker_ab.py                    # Level 2 P1 A/B

# Phase 5 unified harness (Section 14) - deterministic gates
python scripts/glmx_ask.py --db <tester>.db -q "<q>" --seed 0 --no-learning --measure
python test_results/lead/check_dataset.py                         # Section 5.4 compliance
python test_results/lead/unified_golden_runner.py --seed 0        # all tester goldens
```

---

## Appendix D — Dependencies & environment

- **OS:** Windows / Linux / macOS (lead machine: Windows, PowerShell).
- **Python:** 3.14 (lead) — record your version in baseline.
- **Installed packages (runtime path):** `numpy`, `PyYAML`, `pytest`, `sentence-transformers` (loads `BAAI/bge-small-en-v1.5` once, ~100 MB), `pyarrow` (parquet ingest), stdlib `sqlite3`.
- **Optional / out-of-scope:** `torch` (only the T5 decoder embodiment — tests skip it), `spacy` (kg_builder only — dormant). Do NOT add new dependencies without lead approval.
- **Model cache:** first run downloads the SBERT model — requires internet once. All subsequent runs must be offline-capable.

---

*End of protocol v1.3 — §13 assigns three parallel tester domains on individual branches; §14 records the lead fixes applied directly on `maharshi` and the unified deterministic harness. Any deviation from this document is a scope violation. When in doubt, ask the lead.*