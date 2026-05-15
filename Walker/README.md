# Graph Walker Component (Team D)

## Overview

The Graph Walker navigates a resonance-activated subgraph following a plan's intent sequence, producing a WalkResult with the traversed path, activations, and embeddings.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Public API exports |
| `graph_walker.py` | `GraphWalker` class — main walk logic, candidate collection, start selection, temperature, restarts |
| `config.py` | `CoreConfig`, `WalkerConfig` dataclasses + YAML loader |
| `models.py` | `Subgraph`, `Plan`, `WalkResult` dataclasses with `validate()` |
| `path_scorer.py` | `PathScorer` + `ScoredCandidate` — edge scoring (weighted product), softmax/rank/none normalization |
| `intent_bias.py` | `IntentBiasTable` — thread-safe intent-to-relation bias lookup |
| `eligibility.py` | `EligibilityTrace` dataclass + `compute_eligibility_trace()` — gamma-decayed credit assignment |
| `utils.py` | `softmax()`, `geometric_mean()` |
| `exceptions.py` | `WalkerError`, `ValidationError`, `EmbeddingLookupError` |

## Dependencies

- `configs/config_walker.yaml` — walker-specific settings
- `configs/config_core.yaml` — core system settings (activation range, walker defaults, relations)

## API

| Method | Signature |
|---|---|
| `walk` | `(subgraph: Subgraph, plan: Plan) -> WalkResult` |
| `set_temperature` | `(temperature: float) -> None` |
| `get_intent_bias` | `(intent_id: int, relation: str) -> float` |
| `update_intent_bias` | `(intent_id: int, relation: str, bias: float) -> None` |
| `compute_eligibility_trace` | `(walk: WalkResult, subgraph: Subgraph) -> Dict[str, float]` |
| `get_walk_confidence` | `(walk: WalkResult) -> float` |
| `next_possible_nodes` | `(current_node: int, subgraph: Subgraph, current_intent: int, visited=None) -> List[Tuple[int, float]]` |

## Walk Algorithm

1. **Validate** — Subgraph and Plan are validated
2. **Select start** — highest-activation unvisited seed node (or any unvisited node if no seeds remain)
3. **Step loop** (up to `max_steps`):
   - Check current activation ≥ `min_activation`; stop if below
   - Resolve current intent from plan sequence
   - Collect outgoing edges from current node, filtering by:
     - Cycles (if `allow_cycles=false`)
     - Backtrack (if `allow_backtrack=false`)
     - Target activation ≥ `min_activation`
   - If no candidates and `restart_on_dead_end` → restart at a new seed
   - Score candidates: `w_s * s * w_c * c * w_t * a * w_i * b`
   - Apply temperature: `score / temperature`
   - Normalize (softmax / rank / none)
   - Sample next node via RNG threshold
4. **Build WalkResult** — compute walk_confidence (geometric mean of confidences), validate, return

## Key Config Parameters

| Parameter | Default | Effect |
|---|---|---|
| `walk.temperature` | 0.1 | Lower = more deterministic, higher = more random |
| `walk.allow_cycles` | false | Revisit nodes already in path |
| `walk.allow_backtrack` | false | Go back to the immediately previous node |
| `walk.restart_on_dead_end` | true | Restart from a new seed when no candidates |
| `scoring.normalization` | softmax | softmax, rank, or none (deterministic max) |
| `scoring.weight_intent_bias` | 1.0 | How strongly intent biases steer edge selection |

## Test Suite

### 3 files, 148 tests, all passing

| File | Tests | Coverage |
|---|---|---|
| `tests/test_walker.py` | 83 | Unit tests: config loading, model validation, scorer math, eligibility trace, intent bias, walker construction, walk execution, exceptions |
| `tests/test_requirements.py` | 37 | Edge-case gaps: missing config fields, mismatched edge keys, WalkResult.build edge cases, zero-temperature, backtrack/cycle prevention, max-length bound, thread safety, score negative values |
| `tests/test_integration.py` | 28 | Behavioral/integration: deterministic reproducibility (same seed = same walk), normalization=none determinism, end-to-end path verification, 7 property-based invariants, intent bias steering, intent-guided walks, cycle/backtrack constraints, restart correctness, edge-case walks, post-walk invariants |

### Test Results

```
148 passed in 1.35s
```

### Spec Compliance Gaps (Known)

1. `scoring.formula` string is stored but not parsed — actual scoring is hardcoded as a weighted product
2. `compute_eligibility_trace` returns `Dict[str, float]` (formatted keys like `"1:3:is_a"`), not `Dict[Tuple[int,int,str], float]` as documented
3. Test fixture only uses 2 of 16 intent biases — runtime handles all 16 correctly
