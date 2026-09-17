# GLM-X M0 PoC — Experimentation Protocol v1.0

**Applicable branch:** `maharshi` @ `77dedf5`
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

1. All work happens on `maharshi` only. **Never** touch `main` (local or `origin/main`).
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
```

---

## Appendix D — Dependencies & environment

- **OS:** Windows / Linux / macOS (lead machine: Windows, PowerShell).
- **Python:** 3.14 (lead) — record your version in baseline.
- **Installed packages (runtime path):** `numpy`, `PyYAML`, `pytest`, `sentence-transformers` (loads `BAAI/bge-small-en-v1.5` once, ~100 MB), `pyarrow` (parquet ingest), stdlib `sqlite3`.
- **Optional / out-of-scope:** `torch` (only the T5 decoder embodiment — tests skip it), `spacy` (kg_builder only — dormant). Do NOT add new dependencies without lead approval.
- **Model cache:** first run downloads the SBERT model — requires internet once. All subsequent runs must be offline-capable.

---

*End of protocol v1.0. Any deviation from this document is a scope violation. When in doubt, ask the lead.*