# Resonance Engine Static Validation Report

**Date**: 2026-05-13
**Scope**: `Resonance/` folder vs `configs/config_resonance.yaml`, `configs/config_core.yaml`, `configs/dataclass_schema.yaml`
**Type**: Full static analysis (no code modifications)

---

## 1. Folder Structure Compliance

| Expectation | Status |
|---|---|
| All code inside `Resonance/` | ✅ PASS |
| No files outside `Resonance/` modified | ✅ PASS |
| `__init__.py` exports all public classes | ✅ PASS |
| `py.typed` marker present (PEP 561) | ✅ PASS |

**Files present (15)**:
`__init__.py`, `engine.py`, `tier1.py`, `tier2.py`, `energy.py`, `temporal.py`, `analogy.py`, `es_controller.py`, `config.py`, `config_loader.py`, `types.py`, `glmx_types.py`, `validation.py`, `py.typed`, `README.md`

**Class-name conformity**:
| Config class name | Implementation file | Status |
|---|---|---|
| `ResonanceEngine` | `engine.py:19` | ✅ |
| `Tier1Resonance` | `tier1.py:18` | ✅ |
| `Tier2Resonance` | `tier2.py:16` | ✅ |
| `EvolutionaryController` | `es_controller.py:21` | ✅ |
| `AnalogyFinder` | `analogy.py:14` | ✅ |

---

## 2. API Interface Contract

### Function signatures (from `config_resonance.yaml:153-188`)

| Function | Config Signature | Implementation | Status |
|---|---|---|---|
| `resonate` | `(query_embedding, graph, initial_seeds, tier=1) -> Subgraph` | `engine.py:60` | ✅ |
| `resonate_with_theta` | `(theta, query_embedding, graph, initial_seeds) -> Subgraph` | `engine.py:108` | ✅ |
| `get_theta` | `() -> np.ndarray` | `engine.py:123` | ✅ |
| `set_theta` | `(theta) -> None` | `engine.py:126` | ✅ |
| `propose_theta_mutation` | `() -> np.ndarray` | `engine.py:129` | ✅ |
| `update_es_with_reward` | `(reward, theta_used) -> None` | `engine.py:132` | ✅ |
| `compute_activation_energy` | `(subgraph) -> float` | `engine.py:135` | ✅ |
| `check_resonance_convergence` | `(activation_history, epsilon=0.001) -> bool` | `engine.py:138` | ✅ |
| `get_analogy_leaps` | `(target_node, graph, top_k=3) -> List[Tuple[int, float]]` | `engine.py:146` | ✅ |

### Input types
| Input | Config Declaration | Implementation | Status |
|---|---|---|---|
| `query_embedding` | `np.ndarray(384,)` | Validated at `validation.py:49-52` | ✅ |
| `graph` | `GraphStore instance` | Protocol at `types.py:46-50` | ✅ |
| `initial_seeds` | `List[int]` | Validated at `validation.py:66-72` | ✅ |

### Output type
| Output | Config Declaration | Implementation | Status |
|---|---|---|---|
| `Subgraph` | Full schema from `dataclass_schema.yaml` | `types.py:31-42` | ✅ |

---

## 3. Schema Conformity (dataclass_schema.yaml vs types.py)

### Subgraph
| Field | Schema Type | types.py | Status |
|---|---|---|---|
| `nodes` | `List[int]` | ✅ | ✅ |
| `node_activations` | `Dict[int, float]` [0.01, 1.0] | ✅ | ✅ |
| `edges` | `List[Tuple[int,int,str]]` | ✅ | ✅ |
| `edge_strengths` | `Dict[Tuple[int,int,str], float]` [0.0, 1.0] | ✅ | ✅ |
| `edge_confidences` | `Dict[Tuple[int,int,str], float]` [0.0, 1.0] | ✅ | ✅ |
| `seed_nodes` | `List[int]` | ✅ | ✅ |
| `tier_used` | `int` [1, 2] | ✅ | ✅ |
| `activation_energy` | `float` | ✅ | ✅ |
| `query_embedding` | `np.ndarray` float32 (384,) | ✅ | ✅ |
| `timestamp` | `float` | ✅ | ✅ |

### Node
| Field | Schema | types.py | Status |
|---|---|---|---|
| `id` | `int` | ✅ | ✅ |
| `label` | `str` | ✅ | ✅ |
| `node_type` | `str` | ✅ | ✅ |
| `embedding` | `np.ndarray` int8 (32,) | ✅ (typed as `np.ndarray` only) | ⚠️ |
| `activation` | `float` | ✅ | ✅ |
| `use_count` | `int` | ✅ | ✅ |
| `create_time` | `float` | ✅ | ✅ |
| `sense_id` | `Optional[int]` | ✅ | ✅ |

### Edge
| Field | Schema | types.py | Status |
|---|---|---|---|
| `source` | `int` | ✅ | ✅ |
| `target` | `int` | ✅ | ✅ |
| `relation_type` | `str` | ✅ | ✅ |
| `strength` | `float` [0.0, 1.0] | ✅ | ✅ |
| `confidence` | `float` [0.0, 1.0] | ✅ | ✅ |
| `last_used` | `float` | ✅ | ✅ |
| `frequency` | `int` | ✅ | ✅ |

---

## 4. Dependency Analysis

### Import graph (Resonance internal)
```
energy.py → types.py
temporal.py → (none)
analogy.py → config, types
tier1.py → config, energy, temporal, types, validation
tier2.py → analogy, config, tier1, types, validation
es_controller.py → config, validation
validation.py → types, config
engine.py → config, energy, es_controller, tier1, tier2, types, validation
config.py → yaml, numpy, pathlib (stdlib/external only)
config_loader.py → yaml, pathlib (stdlib/external only)
glmx_types.py → types
__init__.py → engine, es_controller, analogy, tier1, tier2, types
```

**Cycles**: ✅ NONE detected
**External deps**: `numpy`, `yaml` (both appropriate)

---

## 5. CRITICAL ISSUES

### [CRIT-1] None found
The implementation is functionally complete and all core logic matches the specification.

---

## 6. WARNINGS

### [WARN-1] Dead thread lock in Tier1Resonance
**File**: `tier1.py:34`  
**Issue**: `self._lock = threading.Lock()` is created but never acquired anywhere in the class.  
**Risk**: Low. No concurrency issue, just dead code.  
**Fix**: Remove `self._lock` and the `threading` import, or use the lock properly in `_initialize_activations`, `_propagate`, etc.

### [WARN-2] Hardcoded embedding dimension (384)
**Files**: `analogy.py:65`, `tier2.py:92`, `validation.py:18`, `validation.py:51`  
**Issue**: The 384-dim sentence-BERT embedding size is hardcoded instead of read from `config_core.yaml:dimensions.sentence_bert_dim`.  
**Risk**: Low (384 is stable per config), but creates a maintenance burden if the dimension changes.  
**Fix**: Either (a) make `sentence_bert_dim` available via `CoreConfig`, or (b) add a `Config` dataclass field for it.

### [WARN-3] Zero-embedding in Tier2 analogy mini-propagation
**File**: `tier2.py:92`  
**Issue**: `query_embedding=np.zeros((384,), dtype=np.float32)` is hardcoded as the query for the analogy mini-propagation. The actual query embedding from the caller is ignored.  
**Risk**: Low. Mini-propagation only explores graph topology from analogy nodes, so the embedding isn't directly used in propagation dynamics. But it's semantically incorrect in the `Subgraph` structure.  
**Fix**: Pass the actual `query_embedding` through to the mini-propagation call.

### [WARN-4] Private member access across class boundary
**File**: `engine.py:146-147`  
```python
def get_analogy_leaps(self, ...):
    return self._tier2._analogy_finder.get_analogy_leaps(...)
```
**Issue**: Accesses `Tier2Resonance._analogy_finder` which is a private attribute.  
**Risk**: Low. Breaks encapsulation but works.  
**Fix**: Add a public `get_analogy_finder()` method on `Tier2Resonance`, or expose the method directly on Tier2Resonance.

### [WARN-5] Dual config loading system
**Files**: `config.py` (dataclass-based) vs `config_loader.py` (AttrDict-based)  
**Issue**: Two separate configuration loading mechanisms exist. `engine.py` uses `config.py:load_configs()` while integration tests use `config_loader.py:load_yaml()` / `CoreConfig.from_yaml()`.  
**Risk**: Medium. Structural differences could lead to config interpretation mismatches between tests and runtime.  
**Fix**: Unify to use a single config loading approach. Either eliminate `config_loader.py` or make integration tests use `config.py`.

### [WARN-6] GraphStore protocol mismatch with config_graph.yaml
**File**: `types.py:46-50`  
**Issue**: The `GraphStore` Protocol doesn't include `relation_filter` parameter on `get_neighbors` (specified in `config_graph.yaml:116-118`). Also missing `get_subgraph_activated`.  
**Risk**: Medium. The Protocol is optional (`@runtime_checkable`), so missing methods won't cause runtime errors unless strict type checking is used.  
**Fix**: Add missing methods and parameters to the Protocol.

### [WARN-7] `int()` truncation for theta top_k
**File**: `engine.py:174`  
```python
top_k=int(theta[idx.top_k]),
```
**Issue**: `int()` truncates toward zero. If theta value is `63.999`, this becomes 63 instead of 64.  
**Risk**: Low. The bound clip in `es_controller.py:148-150` should keep it at valid integers, but floating-point noise could cause off-by-one.  
**Fix**: Use `round()` instead of `int()`.

### [WARN-8] No unit tests in Resonance folder
**Issue**: No test files under `Resonance/`. The only test file (`integration_tests/test_integration.py`) tests cross-component compatibility, not resonance-specific logic.  
**Impact**: Propagation logic, ES updates, temporal factors, energy computation, and config loading have no automated verification.  
**Fix**: Add unit tests for each module.

---

## 7. Configuration Loading Correctness

| Feature | Status |
|---|---|
| YAML `safe_load` used | ✅ |
| Path validation on config dir | ✅ |
| Relation bias value range [0, 2] validated | ✅ |
| All 16 relation types present and correct | ✅ |
| Theta dim (48) matches reserved_end | ✅ |
| Theta index mapping matches core relation order | ✅ |
| All 16 relation bias slots validated | ✅ |
| Propagation type validated against whitelist | ✅ |
| Normalization type validated against whitelist | ✅ |
| Gate type validated against whitelist | ✅ |
| Activation range consistency (min < max, default >= min) | ✅ |

---

## 8. Numerical Precision

| Config Requirement | Implementation | Status |
|---|---|---|
| `activation: float32` | `np.float32` in `_activation_vector`, `_to_float_embedding` | ✅ |
| `theta: float32` | `.astype(np.float32)` in `es_controller.py:36-37` | ✅ |
| `similarities: float32` | Cosine similarity returns Python `float` (not explicitly float32) | ⚠️ |

---

## 9. Algorithm Correctness

### Propagation types (`tier1.py:158-168`)
- [x] `wilson_cowan` → `delta = (1 - A) * input - lambda * A`
- [x] `simple_diffusion` → `delta = input - lambda * A`
- [x] `threshold` → `delta = input - lambda * A` if input >= threshold, else `delta = -lambda * A`

### Normalization (`tier1.py:179-196`)
- [x] `l1_norm` → divide by total sum
- [x] `budget_soft_cap` → scale if total > budget_max
- [x] `none` → pass-through

### Gating (`tier1.py:198-212`)
- [x] `top_k` → keep top K by activation
- [x] `threshold` → keep activations >= threshold
- [x] `none` → pass-through

### Energy formula (`energy.py:12-33`)
- [x] `E = sum(activation * (strength * confidence))` — matches config exactly

### Temporal factor (`temporal.py:10-39`)
- [x] `r(T) = 1 / (1 + gamma * log(1 + delta_t))`
- [x] `s(freq) = min(1.0, freq / frequency_threshold)`
- [x] Combined: `recency * frequency_scale`, clamped to [0, 1]

### Tier switching (`engine.py:60-106`)
- [x] Always run Tier1 first
- [x] If Tier1 energy >= `T_conf * |seeds| * A_max`, skip Tier2
- [x] If Tier2 energy >= Tier1 energy + `tier2_min_improvement`, use Tier2
- [x] Otherwise fall back to Tier1

### ES update (`es_controller.py:76-118`)
- [x] Collect `evaluation_window` samples
- [x] Normalize rewards (z-score)
- [x] Compute weighted delta: `mean(normalized * (theta - mu))`
- [x] `mu += lr * weighted_delta`
- [x] Anchor penalty: `mu -= anchor_lambda * (mu - initial_mu)`
- [x] Sigma decay: `sigma *= (1 - sigma_decay_beta)`

### Analogy leaps (`analogy.py:23-41`)
- [x] LSH index with configurable bands/tables
- [x] Fallback to brute-force if LSH unavailable
- [x] Jaccard overlap validation
- [x] Edge confidence filter

---

## 10. Risk Assessment

| Risk | Severity | Likelihood | Description |
|---|---|---|---|
| Dead thread lock in Tier1 | Low | Very Low | No functional impact |
| Hardcoded 384 dim | Low | Low | Breaks if SBERT model changes |
| Zero embedding in analogies | Low | Very Low | Mini-propagation doesn't use query embedding |
| Private member access | Low | Very Low | Encapsulation violation only |
| Dual config loading | Medium | Medium | Tests and runtime may diverge |
| No unit tests | Medium | High | Regression risk on any code change |
| Protocol mismatch | Low | Low | Protocol runtime_checkable, so no crash |
| Float int() truncation | Low | Low | Off-by-one in edge case |

**Overall Risk Level**: **LOW** — The implementation is structurally sound and matches the blueprint.

---

## 11. Summary

| Category | Count |
|---|---|
| ✅ Pass | 45+ |
| ⚠️ Warnings | 8 |
| 🔴 Critical | 0 |
| ❌ Missing (blocking) | 0 |

The Resonance implementation is **functionally complete** and **correctly implements** the configuration blueprint. All 9 API functions are implemented, all dataclasses match the schema, the algorithm (Tier1/Tier2 propagation, energy, temporal factor, ES controller, analogy) correctly follows the specification, and dependency management is clean with no circular imports.

The most impactful warning is **WARN-8** (no unit tests), which represents the highest risk for future regressions.
