# Phase G — Dynamic Cross-Domain Relation Lexicon (tester-c)

Goal: make the G2P query-relation extractor *dynamic* so a brand-new knowledge graph
gets correct relation extraction **without training anything**, by symbolically
accreting verified phrase→relation mappings and persisting them per-KG.

Hard constraints honoured: no LLM, no generative transformer, fully offline, frozen
`BAAI/bge-small-en-v1.5`. "Dynamic" here means accreted **data** (a phrase→relation
lexicon), never updated weights. The learning signal is golden/harness-supplied only;
production self-learning stays dormant.

---

## 1. What was built

| Piece | Location |
|---|---|
| `DynamicLexiconConfig` + `G2PConfig.dynamic_lexicon` | `g2p/config.py` |
| Layer 1 lexicon: `learn_success`, `get_learned_lexicon`, `load_learned_lexicon`, `canonical_clause` | `g2p/g2p_planner.py` |
| Layer 2 graph grounding: `_GROUND_TEMPLATES`, `_graph_grounded_relation`, `set_label_vocab` | `g2p/g2p_planner.py` |
| Config block (default `enabled: false`) | `configs/config_g2p.yaml` |
| Inert-by-default rehydration on load | `scripts/glmx_ask.py` (`load_models`) |
| 5 domain KGs | `test_results/tester-c/datasets/build_*.py` |
| cold/hot/reload harness + 30 frozen goldens | `test_results/tester-c/generation_probe_runner.py` |
| Unit tests (13 new) | `g2p/tests/test_relation_extractor.py` |

Five brand-new domains, all inside the canonical 16-relation vocabulary:

| Graph | Nodes | Edges | Relations |
|---|---|---|---|
| `astronomy_small.db` | 75 | 76 | 12 |
| `computing_small.db` | 95 | 81 | 11 |
| `music_small.db` | 100 | 73 | 11 |
| `human_body_small.db` | 94 | 80 | 13 |
| `history_civilizations.db` | 94 | 82 | 15 |

## 2. Headline result — cold → hot → reload

30 goldens (6 per domain: 4 taught / 2 held-out), same questions, three modes.

| Mode | Total | Taught (20) | Held-out (10) | Lexicon |
|---|---|---|---|---|
| **cold** (static bank only) | **24/30** | 18/20 | 6/10 | – |
| **hot** (taught in-process) | **25/30** | 19/20 | 6/10 | 4/domain |
| **reload** (fresh process, lexicon rehydrated from KG metadata) | **25/30** | 19/20 | 6/10 | 4/domain |

`reload == hot` exactly: the accreted lexicon round-trips through
`store.set_metadata("learned_lexicon", …)` + `save_state` and is rehydrated by
`load_learned_lexicon` in a fresh pipeline. Persistence is proven, not assumed.

## 3. An earlier bug that inflated the headline

The first probe run reported **cold 15/30 → hot 25/30** (+10). That baseline was
false. The cause was a literal-cue table (`_LITERAL_CUES`) that prepended a
hard-coded relation before clause matching. It was my own Phase G scaffolding —
`git show HEAD:g2p/g2p_planner.py` confirms HEAD had no such table.

The cue forced one relation per question regardless of the graph, which broke
queries such as "What is the moon a part of?" (cue `part_of` + clause `is_a`).
It also caused a hard **regression of the frozen anchor to 92/99** (sc05, sc06, wc05
flipped to FAIL) while breaking the pre-existing
`test_extract_ordered_chain_from_question` unit test.

The cue table provided **zero benefit** (cue-first and cue-free both scored the same)
and was removed entirely. Real numbers after the fix are the 24/30 → 25/30 in §2.

## 4. Regression gate — 94/99 → 95/99

`test_results/lead/unified_golden_runner.py --seed 0`, untouched, default config:

| | Before this work | After |
|---|---|---|
| Overall | 94/99 | **95/99** |

Fails before: nws02, nws07, nws08, sc08, gg11.
Fails after: nws02, nws07, nws08, gg11 — **sc08 fixed**.

sc08 ("What comes after interphase?") previously produced a spurious two-hop
`['follows','precedes']` chain from the cue table; cue-free it resolves to a single
correct hop. The cue table was the sole cause of that failure.

`dynamic_lexicon` ships **disabled** (`enabled: false`), so the frozen runner exercises
exactly the static path; Layer 2 cannot fire. This is verified by the runner, not assumed.

## 5. Honest analysis of the remaining 5 probe failures

The dynamic layer's measured lift on this suite is **+1**, not +10 — because once the
cue bug was removed the static descriptor bank is already strong (24/30). The 5
residuals are dominated by a *different* subsystem, entity anchoring, not relation
extraction:

| id | Question | Diagnosis | Subsystem |
|---|---|---|---|
| com04 (taught) | "What causes data loss?" | **Relation now correct (`causes`)**; anchor resolved to node `data` instead of `data loss` (token-substring collision) → walk had no `causes` edge → honest | anchoring |
| com05 | "What is a malware infection caused by?" | anchor `malware` ≠ `malware infection` (same collision) | anchoring |
| his06 | "Sparta was warlike, so what appears after the bronze age?" | embedding argmax anchored to `warlike` — a **premise** token | anchoring |
| ast04 | "Name the planet that lies closest to the sun." | 2 relations from 1 clause (`['is_a','spatial_near']`); `is_a` cleared the threshold first so Layer 2 never ran | extraction (Layer 2 gate) |
| ast05 | "Which planet is red?" | `is_a` chosen; `has_property` paraphrase unrecognised — a genuine **held-out generalization gap** | extraction (expected) |

**Next bottleneck is anchoring, not the extractor.** `_resolve_target_entity`
(`scripts/glmx_ask.py`) scans question tokens and takes the first token that is a node
label, so a token that is a *substring* of a longer label wins. Preferring the longest
label match is the concrete fix. It was deliberately **not** attempted here: it lives
outside Phase G and would put the 95/99 anchor at risk.

## 6. Part E — `src` NameError: not reproducible

Earlier hypothesis was that `process_feedback` raises `name 'src' is not defined` on
every ask. It does not. Evidence:

- a dedicated repro (`learn_diag.py`) on `astronomy_small.db` completed clean;
- a full-text scan of the complete probe run (30 asks) and the complete golden run
  (99 asks) for `feedback failed` / `name 'src'` / `Traceback` returned **zero
  matches**.

`_no_learning` gates only REINFORCE/ES (`glmx_ask.py:1022`); `process_feedback`
(`glmx_ask.py:1084`) runs unconditionally but mutates in-memory state only, never the
`.db`, so determinism is preserved. **Closed: no production code was changed for it.**

## 7. Verification

- `python -m pytest g2p/tests -q` → **57 passed, 102 skipped** (13 new tests covering
  the learn gate, margin dedup, persistence round-trip, learned-override of a static
  relation, Layer 2 thresholds, and the disabled-mode invariants).
- `python test_results/lead/unified_golden_runner.py --seed 0` → **95/99**.
- `python test_results/tester-c/generation_probe_runner.py` → cold 24/30, hot 25/30,
  reload 25/30.

## 8. Reproduce

```powershell
python test_results/tester-c/datasets/build_astronomy_small.py   # …and the other 4
python test_results/tester-c/generation_probe_runner.py
python test_results/lead/unified_golden_runner.py --seed 0
python -m pytest g2p/tests -q
```

## 9. Status

Extractor-side Phase G is **complete**: the lexicon accretes verified mappings,
persists per-KG, survives process restart, and is inert by default. The cross-domain
lift on this 30-question suite is small (+1) because the static bank is stronger than
previously believed; the honest next step for real generalization gains is fixing
longest-label anchoring, not more extractor machinery.
