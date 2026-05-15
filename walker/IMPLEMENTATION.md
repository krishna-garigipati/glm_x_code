# Graph Walker Component — Implementation Guide

## What It Does

The Graph Walker takes a resonated subgraph (activated nodes + edges) and a plan (intent sequence), and walks through the graph following intent-guided scoring to produce a path of nodes with activations, confidences, and embeddings.

**Pipeline position:**
```
Resonance Engine → Subgraph → Graph Walker → WalkResult → Decoder → Answer
                          ↗
                G2P Planner → Plan
```

---

## Files & Their Roles

### Core Implementation

| File | Lines | What It Contains |
|---|---|---|
| `graph_walker.py` | 343 | Main `GraphWalker` class — the entire walk algorithm |
| `config.py` | 161 | `WalkerConfig`, `CoreConfig`, and sub-config dataclasses + `load_yaml()` |
| `models.py` | 158 | `Subgraph`, `Plan`, `WalkResult` dataclasses with `validate()` and `build()` |
| `path_scorer.py` | 65 | `PathScorer` class + `ScoredCandidate` dataclass — edge scoring & normalization |
| `intent_bias.py` | 36 | `IntentBiasTable` — thread-safe bias lookup |
| `eligibility.py` | 37 | `EligibilityTrace` dataclass + `compute_eligibility_trace()` |
| `utils.py` | 32 | `softmax()` and `geometric_mean()` |
| `exceptions.py` | 13 | `WalkerError`, `ValidationError`, `EmbeddingLookupError` |

### Tests (3 files, 148 tests)

| File | Tests | What It Covers |
|---|---|---|
| `tests/test_walker.py` | 83 | Unit tests for every module |
| `tests/test_requirements.py` | 37 | Edge-case gap coverage |
| `tests/test_integration.py` | 28 | Behavioral & integration tests |

### Config Files (in `configs/`)

| File | Purpose |
|---|---|
| `config_walker.yaml` | Walker-specific settings (walk params, 16 intent biases, scoring, eligibility, path recording, debug) |
| `config_core.yaml` | System-wide settings (activation range, relations, walker defaults, learning rates, dimensions) |
| `dataclass_schema.yaml` | Type contracts for all inter-component data structures |

---

## Data Flow Through the Walker

```
                    ┌─────────────────────────────────┐
                    │         GraphWalker              │
                    │                                  │
Subgraph ──────────►│  1. Validate inputs              │
Plan ──────────────►│  2. Select start node (seed)     │
                    │                                  │
                    │  ┌─ Loop (max_steps) ──────────┐ │
                    │  │  3. Check activation >= min  │ │
                    │  │  4. Resolve current intent   │ │
                    │  │  5. Collect outgoing edges   │ │
                    │  │  6. Score each candidate     │ │
                    │  │  7. Apply temperature        │ │
                    │  │  8. Normalize probabilities  │ │
                    │  │  9. Sample next node via RNG │ │
                    │  │                             │ │
                    │  │  If dead end + restart_on:   │ │
                    │  │    → Restart at new seed     │ │
                    │  └─────────────────────────────┘ │
                    │                                  │
                    │  10. Build WalkResult            │
                    │  11. Compute walk_confidence     │
                    │  12. Validate & return           │
                    │                                  │
                    └──────────────┬──────────────────┘
                                   ▼
                            WalkResult
                         (path, edges, activations,
                          confidences, embeddings)
```

---

## Component Deep-Dive

### 1. Config System (`config.py`)

Two-level config with cross-validation:

```
WalkerConfig                    CoreConfig
├── walk: WalkConfig            ├── activation: ActivationConfig
│   ├── max_steps: int          │   ├── min: float (0.01)
│   ├── min_activation: float   │   └── max: float (1.0)
│   ├── temperature: float      ├── walker: WalkerCoreConfig
│   ├── temperature_range       │   ├── default_temperature
│   ├── allow_cycles: bool      │   ├── softmax_temperature_range
│   ├── allow_backtrack: bool   │   ├── max_steps
│   ├── restart_on_dead_end     │   └── min_activation_to_continue
│   └── restart_penalty         └── relations: Dict[int, str]
├── intent_biases: Dict
├── scoring: ScoringConfig      Cross-validation checks that
├── eligibility: EligConfig     walker config matches core config
├── path: PathConfig            on temperature, max_steps,
└── debug: DebugConfig          min_activation, temperature_range
```

Loaded from YAML:
```python
cfg = WalkerConfig.from_yaml("configs/config_walker.yaml")
core = CoreConfig.from_yaml("configs/config_core.yaml")
walker = GraphWalker(cfg, core)
```

### 2. Input Models (`models.py`)

**Subgraph** — the graph to walk:
```python
Subgraph(
    nodes=[1, 2, 3, 4, 5],
    node_activations={1: 0.5, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.9},
    edges=[(1, 3, "is_a"), (1, 4, "has_property"), (3, 5, "causes")],
    edge_strengths={(1,3,"is_a"): 0.8, ...},
    edge_confidences={(1,3,"is_a"): 0.9, ...},
    seed_nodes=[1, 2],
    tier_used=1,
    activation_energy=1.0,
    query_embedding=np.zeros(384),  # Sentence-BERT embedding
    timestamp=time.time(),
)
```

**Plan** — the intents guiding the walk:
```python
Plan(
    intent_sequence=[0, 4, 7],      # define → contrast → example
    plan_confidence=0.9,
    heuristic_fallback_used=False,
    intent_names=["define", "contrast", "example"],  # optional
)
```

**WalkResult** — what comes out:
```python
WalkResult(
    path=[1, 3, 5],                 # node IDs visited
    path_edges=["is_a", "causes"],   # edge types traversed
    path_activations=[0.5, 0.7, 0.9],
    path_confidences=[0.9, 0.8],
    path_embeddings=[array(...), ...],
    walk_confidence=0.85,            # geometric mean of confidences
    final_activation=0.9,
    steps_taken=2,
    plan_followed=plan,
    timestamp=time.time(),
    intent_sequence_used=[0, 4, 7],
)
```

### 3. GraphWalker Algorithm (`graph_walker.py`)

**`walk(subgraph, plan) -> WalkResult`:**

```
START:
  1. validate(subgraph) — checks activations in range,
     edges reference valid nodes, tier is 1 or 2, etc.
  2. validate(plan)     — checks intent IDs in [0,15],
     sequence length 1-8, confidence in [0,1], etc.
  3. start_node = highest-activated UNVISITED seed node
     (falls back to any unvisited node if all seeds visited)

LOOP (while len(path_edges) < max_steps):
  4. if current activation < min_activation: BREAK
  5. intent_id = plan.intent_sequence[
       min(steps_taken, len(plan)-1)]    // repeat last intent
  6. candidates = []
     for each edge (source, target, relation) in subgraph.edges:
       if source != current_node: continue
       if not allow_cycles and target in visited: continue
       if not allow_backtrack and target == previous: continue
       if target_activation < min_activation: continue
       bias = IntentBiasTable.get_bias(intent_id, relation)
       candidates.append(ScoredCandidate(...))

  7. if no candidates:
       if restart_on_dead_end:
         restart with new start node (unvisited seed)
         CONTINUE (reset path to [start_node])
       else: BREAK

  8. scores = [scorer.score(c) for c in candidates]
     // scorer.score = w_s*s * w_c*c * w_t*act * w_i*bias

  9. adjusted = [s / temperature for s in scores]
  10. probabilities = softmax(adjusted, softmax_temperature)
  11. next_idx = weighted_random_choice(probabilities, rng)
  12. append chosen node to path, edge to path_edges, etc.

END:
  13. walk_confidence = geometric_mean(path_confidences)
  14. validate WalkResult
  15. return WalkResult
```

**Key implementation details:**

- **Temperature** controls randomness: low temperature (0.05) ≈ argmax, high temperature (0.5) ≈ uniform
- **Intent sequence** cycles: `intent_id = plan.intent_sequence[min(steps, len-1)]` so after exhausting all intents, the last one repeats
- **Restart penalty** multiplies activation: `activation * 0.5` (default) so restart prefers strongly activated nodes less aggressively than initial start
- **RNG** is Python `random.Random` with optional seed for reproducibility

### 4. Scoring (`path_scorer.py`)

**Score formula:**
```
score = weight_strength * strength
      * weight_confidence * confidence
      * weight_target_activation * target_activation
      * weight_intent_bias * intent_bias
```

All weights default to 1.0. With `weight_intent_bias: 5.0`, intent biases dominate the score.

**Normalization modes:**
| Mode | Behavior |
|---|---|
| `softmax` | `exp(score/temp) / sum(exp(score/temp))` — probabilistic |
| `rank` | Inverse rank weighting — more robust to outliers |
| `none` | Raw scores used directly, max is always selected |

### 5. Intent Bias (`intent_bias.py`)

Thread-safe lookup table. Each intent has per-relation biases + a default:

```python
biases = {
    0: {"is_a": 1.5, "has_property": 1.2, "default": 0.5},
    4: {"contradicts": 2.0, "antonym": 1.8, "default": 0.3},
}

table = IntentBiasTable(biases)
table.get_bias(4, "contradicts")   # → 2.0
table.get_bias(4, "unknown_rel")   # → 0.3 (default)
table.get_bias(99, "anything")     # → 1.0 (unknown intent)
```

The 16 intents with full bias mappings are in `configs/config_walker.yaml`.

### 6. Eligibility Trace (`eligibility.py`)

Computes credit assignment for each edge traversed:

```
E_e = Σ(t=0..K) γ^t * A_src(t) * S * C * 1.0
```

Where γ (gamma, default 0.9) discounts earlier steps, A_src is the activation at the source node at step t, and S*C is the strength*confidence of the edge.

### 7. Utilities (`utils.py`)

**`softmax(scores, temperature)`:**
- Numerically stable (subtracts max score)
- Temperature must be positive
- Returns uniform distribution if all exp are zero

**`geometric_mean(values)`:**
- Returns 1.0 for empty input
- Returns 0.0 if any value ≤ 0
- Used to compute `walk_confidence` from path confidences

---

## 16 Intents and Their Biases

| ID | Intent | Strongly Biased Relations |
|---|---|---|
| 0 | define | is_a (1.5), has_property (1.2), example_of (1.0) |
| 1 | assert_fact | supports (1.3), causes (1.2), part_of (1.0) |
| 2 | explain_cause | causes (1.8), caused_by (1.5), follows (1.2) |
| 3 | explain_effect | causes (1.6), caused_by (1.4), precedes (1.2) |
| 4 | contrast | contradicts (2.0), antonym (1.8) |
| 5 | compare | synonym (1.5), associated_with (1.2) |
| 6 | list | part_of (1.4), has_property (1.2) |
| 7 | example | example_of (2.0), has_property (1.2) |
| 8 | conclude | supports (1.5), causes (1.3) |
| 9 | question | associated_with (1.2) |
| 10 | uncertain | contradicts (1.2) |
| 11 | clarify | associated_with (1.3) |
| 12 | summarize | part_of (1.3), supports (1.1) |
| 13 | elaborate | has_property (1.4), part_of (1.2) |
| 14 | transition | follows (1.3), precedes (1.2) |
| 15 | emphasize | supports (1.6), causes (1.4) |

---

## Usage Example

```python
from Walker import GraphWalker, WalkerConfig, CoreConfig, Plan, Subgraph
import numpy as np, time

# Load configs
walker_cfg = WalkerConfig.from_yaml("configs/config_walker.yaml")
core_cfg = CoreConfig.from_yaml("configs/config_core.yaml")

# Create walker with embedding provider
walker = GraphWalker(
    walker_cfg, core_cfg,
    embedding_provider=lambda node_id: np.zeros(32, dtype=np.int8),
    random_seed=42,
)

# Build subgraph
sg = Subgraph(
    nodes=[1, 2, 3, 4, 5],
    node_activations={1: 0.5, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.9},
    edges=[(1, 3, "is_a"), (1, 4, "has_property"), (3, 5, "causes")],
    edge_strengths={(1,3,"is_a"): 0.8, (1,4,"has_property"): 0.8, (3,5,"causes"): 0.8},
    edge_confidences={(1,3,"is_a"): 0.9, (1,4,"has_property"): 0.9, (3,5,"causes"): 0.9},
    seed_nodes=[1],
    tier_used=1,
    activation_energy=1.0,
    query_embedding=np.zeros(384, dtype=np.float32),
    timestamp=time.time(),
)

# Create plan
plan = Plan(
    intent_sequence=[0, 4, 7],
    plan_confidence=0.9,
    heuristic_fallback_used=False,
)

# Walk
result = walker.walk(sg, plan)

# Inspect result
print(f"Path: {result.path}")
print(f"Edges: {result.path_edges}")
print(f"Confidence: {result.walk_confidence}")

# Get eligibility trace
trace = walker.compute_eligibility_trace(result, sg)
print(f"Eligibility: {trace}")
```

---

## Test Suite Details

### Running Tests
```bash
cd GLM-X
python -m pytest Walker/tests/ -v
python -m pytest Walker/tests/test_walker.py          # 83 unit tests
python -m pytest Walker/tests/test_requirements.py     # 37 edge-case tests
python -m pytest Walker/tests/test_integration.py      # 28 behavioral tests
```

### Test Result
```
148 passed in 1.35s
```

### Test Coverage by Component

| Component | Original Tests | Gap Tests | Integration Tests | Total |
|---|---|---|---|---|
| Config loading | 5 | 3 | 0 | 8 |
| Subgraph validation | 7 | 2 | 0 | 9 |
| Plan validation | 9 | 0 | 0 | 9 |
| WalkResult validation | 9 | 0 | 0 | 9 |
| WalkResult.build | 0 | 3 | 0 | 3 |
| WalkResult.to_tuples | 2 | 1 | 0 | 3 |
| Utils (softmax, geo_mean) | 6 | 4 | 0 | 10 |
| PathScorer | 7 | 2 | 0 | 9 |
| IntentBiasTable | 8 | 2 | 0 | 10 |
| EligibilityTrace | 6 | 2 | 0 | 8 |
| GraphWalker construction | 8 | 1 | 0 | 9 |
| GraphWalker internals | 0 | 11 | 0 | 11 |
| GraphWalker walk execution | 16 | 6 | 28 | 50 |
| Exceptions | 4 | 0 | 0 | 4 |
| **Total** | **83** | **37** | **28** | **148** |

### Behavioral Invariants Tested

Every walk result must satisfy:
1. `len(path) > 0`
2. `len(path_edges) == len(path) - 1`
3. `len(path_activations) == len(path)`
4. `len(path_confidences) == len(path) - 1`
5. `len(path_embeddings) == len(path)`
6. `steps_taken == len(path_edges)`
7. `0.0 <= walk_confidence <= 1.0`
8. `0.01 <= each activation <= 1.0`
9. `0.0 <= each confidence <= 1.0`
10. Every node in path is in `subgraph.nodes`
11. Every edge traversed exists in `subgraph.edges`
12. `walk_confidence == geometric_mean(path_confidences)`
13. `intent_sequence_used == plan.intent_sequence`
14. `walk_confidence == geometric_mean(path_confidences)` (double-checked)
15. `path_activations[i] == subgraph.node_activations[path[i]]`

---

## Known Spec Compliance Gaps

| Gap | Description | Impact |
|---|---|---|
| `scoring.formula` string not parsed | Stored as string but `PathScorer.score()` hardcodes `w_s*s * w_c*c * w_t*a * w_i*b` | Changing formula in config has no effect |
| `compute_eligibility_trace` return type | Returns `Dict[str, float]` (formatted keys like `"1:3:is_a"`), spec says `Dict[Tuple[int,int,str], float]` | Callers must handle string keys |
| `EligibilityTrace` dataclass unused | Defined in `eligibility.py` and schema but `compute_eligibility_trace()` returns plain dict | No object wrapper for traces |
| `next_possible_nodes` extra param | Spec shows 3 params, code has optional 4th `visited` param | Extra functionality, not a bug |
| Config mismatch detection is restrictive | Enforces `temperature == default_temperature` at construction | Can't run walkers at different temps without separate core config |

---

## Edge Cases Handled by the Implementation

| Scenario | Behavior |
|---|---|
| Single node subgraph | Walk returns path=[node], steps=0 |
| All nodes below min_activation | Walk returns just the start node |
| Dead end (no outgoing edges) | Restarts at new seed if enabled, otherwise stops |
| All nodes visited (after restarts) | Stop, return last restart path |
| Self-loop edge `(1,1,"loop")` | Treated as candidate but blocked by no-consecutive-duplicates check |
| Plan intent out of bias table range | Default bias of 1.0 applied |
| Missing embedding for node | `EmbeddingLookupError` raised |
| Empty subgraph | `ValidationError` during input validation |
| Subgraph with extra nodes in seeds | Seed nodes added to node list automatically |
