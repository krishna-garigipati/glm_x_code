# GLM-X: Comprehensive Training & Testing Plan

> **Version:** 1.0  
> **Date:** 2026-05-15  
> **Scope:** Data preparation → Pre-training → Fine-tuning → Testing → Deployment  
> **Target hardware:** 8 GB RAM laptop (no discrete GPU required)

---

## Table of Contents

1. [Architecture Overview: What Needs Training](#1-architecture-overview)
2. [Phase 0: Environment Setup & Prerequisites](#2-phase-0-environment-setup)
3. [Phase 1: Real-World Training Data Pipeline](#3-phase-1-data-pipeline)
4. [Phase 2: IntentFFN Training](#4-phase-2-intentffn-training)
5. [Phase 3: Hyperparameter Sweep & Model Selection](#5-phase-3-hyperparameter-sweep)
6. [Phase 4: Micro-Decoder Template Expansion](#6-phase-4-template-expansion)
7. [Phase 5: T5-Small Fine-Tuning (Optional)](#7-phase-5-t5-fine-tuning)
8. [Phase 6: Full Pipeline Integration Testing](#8-phase-6-integration-testing)
9. [Phase 7: Evaluation, Benchmarking & Regression](#9-phase-7-evaluation)
10. [Phase 8: Quantization & Production Deployment](#10-phase-8-quantization)
11. [Resource Budget & 8 GB Laptop Feasibility](#11-resource-budget)
12. [Timeline & Milestones](#12-timeline)
13. [Appendices](#13-appendices)

---

## 1. Architecture Overview

### 1.1 System Components

```
  Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                    RESONANCE ENGINE                          │
│  Tier1 (fast, local) + Tier2 (slow, global)                 │
│  ES Meta-Controller adapts θ (~20 params) online             │
│  ⚠ NO PRE-TRAINING NEEDED — online adaptation only          │
└──────────────────────┬───────────────────────────────────────┘
                       │ Subgraph
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    G2P PLANNER                               │
│  ┌─────────────────┐  ┌──────────────────┐                   │
│  │ GraphToText      │  │ IntentFFN (220K)  ◄── TRAIN THIS    │
│  │ Encoder          │  │ 2-layer MLP       │                  │
│  │ (frozen rules)   │  │ 384→128→128→16   │                  │
│  └────────┬─────────┘  └────────┬─────────┘                  │
│           ▼                     ▼                            │
│  ┌─────────────────────────────────────┐                    │
│  │        BeamSearchDecoder             │                    │
│  │        (greedy/beam, frozen logic)   │                    │
│  └────────────────┬────────────────────┘                    │
│                   │ Plan (intent sequence)                    │
└───────────────────┼──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│                    GRAPH WALKER                              │
│  Rule-based personalized PageRank                            │
│  ⚠ NO PRE-TRAINING NEEDED — deterministic logic             │
└───────────────────┬──────────────────────────────────────────┘
                    │ Walk
                    ▼
┌──────────────────────────────────────────────────────────────┐
│                    MICRO-DECODER                             │
│  ┌────────────────┐  ┌──────────────────┐                    │
│  │ TemplateDecoder│  │ T5 Decoder (60M) ◄── OPTIONAL         │
│  │ 15 hand-       │  │ (t5-small)       │   FINE-TUNE        │
│  │ written rules  │  │                  │                    │
│  └───────┬────────┘  └────────┬─────────┘                    │
│          └──────────┬─────────┘                               │
│                    ▼                                          │
│          HybridDecoder (template + T5)                        │
│  ⚠ TEMPLATES: hand-written (expandable)                      │
│  ⚠ T5: pre-trained, optional fine-tune                       │
└───────────────────┬──────────────────────────────────────────┘
                    │ Answer
                    ▼
┌──────────────────────────────────────────────────────────────┐
│                    LEARNING ENGINE (HEBBIAN)                 │
│  Online weight updates only                                  │
│  ⚠ NO PRE-TRAINING NEEDED — online learning only            │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 What Needs Pre-Training

| Component | Params | Type | Training Data | Priority |
|---|---|---|---|---|
| **IntentFFN** | 220K (0.22 MB int8) | 2-layer MLP + 1-layer classifier | (Subgraph → intent_sequence) pairs | **REQUIRED** |
| **T5-small** (optional) | 60M | Transformer encoder-decoder | (prompt → text) pairs | Optional |
| **Templates** | 0 | Rule strings | None (write by hand) | Optional, expandable |
| Sentence-BERT | 22M | BERT variant | Frozen (pre-trained) | None |
| Resonance | ~20 float θ | ES controller | Online only | None |
| Walker | 0 | PPR algorithm | None | None |
| Hebbian | 0 | Rules | Online only | None |

---

## 2. Phase 0: Environment Setup

### 2.1 Prerequisites

```powershell
# Python version: 3.10+
python --version

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Core dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pyyaml sentence-transformers

# For T5 fine-tuning (optional)
pip install transformers

# For testing
pip install pytest pytest-cov
```

### 2.2 Verify Existing Tests Pass

```powershell
# G2P unit tests (1287 tests in test_g2p_all.py)
cd g2p\tests
python test_g2p_all.py --json

# Full integration suite (600+ lines, 25 tests)
cd integration_tests
python test_integration.py
```

**Exit criteria:** All 25 integration tests and all g2p unit tests pass.

### 2.3 Create Directory Structure for Training

```
glm_x_code/
├── training_and_testing_planning.md     ◄── THIS FILE
├── g2p/
│   ├── models/                          ★ Create
│   │   ├── intent_ffn_trained.pt        (output)
│   │   └── intent_ffn_quantized.pt      (output)
│   ├── data_generation/                 ★ Create
│   │   ├── __init__.py
│   │   ├── conceptnet_loader.py
│   │   ├── wikidata_loader.py
│   │   └── synthetic_augmentation.py
│   └── training/
│       ├── __init__.py
│       ├── hyper_sweep.py               ★ Create
│       ├── trainer.py                   ★ Enhanced from train.py
│       └── evaluate.py                  ★ Create
├── evaluation/                          ★ Create
│   ├── __init__.py
│   ├── plan_accuracy.py
│   ├── confidence_calibration.py
│   ├── end_to_end_benchmark.py
│   └── report_generator.py
├── configs/
│   ├── training/                        ★ Create
│   │   ├── sweep_configs/
│   │   └── best_config.yaml
│   └── dataloader_configs/
└── scripts/
    ├── full_training_pipeline.ps1       ★ Create
    └── run_all_tests.ps1                ★ Create
```

### 2.4 Install External Data Tools

```powershell
# ConceptNet downloader
pip install conceptnet-lite

# Wikidata extractor
pip install wikidata-client

# Data processing
pip install pandas tqdm networkx
```

---

## 3. Phase 1: Real-World Training Data Pipeline

### 3.1 Data Sources Overview

| Source | Size | Format | License | Coverage |
|---|---|---|---|---|
| **ConceptNet 5.7** | ~8M edges | CSV (start, end, relation, weight) | CC-BY-SA 4.0 | General knowledge, 100+ languages |
| **WordNet 3.0** | 155K words | RDF/OWL | Princeton University | English lexical database |
| **Wikipedia** (subset) | 6M articles | Raw text + entity links | CC-BY-SA 3.0 | Encyclopedic knowledge |
| **FEVER** | 185K claims | Claim + evidence + label | CC-BY-4.0 | Fact-checking with evidence chains |
| **HotpotQA** | 113K QA pairs | Question + supporting facts | CC-BY-SA 4.0 | Multi-hop reasoning |

### 3.2 Primary Source: ConceptNet-to-Intent Mapping

This is the **main data generation strategy**. ConceptNet's relation types already map naturally to GLM-X's 16 intents.

#### 3.2.1 Relation-to-Intent Mapping Table

```python
CONCEPTNET_RELATION_MAP = {
    # GLM-X intent 0: define
    "IsA":          0,  # "A cat is a mammal" → define
    "DefinedAs":    0,
    "Synonym":      0,  # # GLM-X intent 0

    # GLM-X intent 1: assert_fact
    "RelatedTo":    1,
    "AtLocation":   1,
    "CapableOf":    1,
    "UsedFor":      1,
    "HasProperty":  1,
    "Desires":      1,
    "CreatedBy":    1,
    "HasFirstAppearance": 1,
    "MotivatedByGoal":    1,

    # GLM-X intent 2: explain_cause
    "Causes":       2,
    "CausesDesire": 2,
    "HasSubevent":  2,  # causal subevent chain
    "Prerequisite": 2,

    # GLM-X intent 3: explain_effect
    "CausedBy":     3,
    "Entails":      3,

    # GLM-X intent 4: contrast
    "Antonym":      4,
    "NotDesires":   4,
    "NotCapableOf": 4,
    "NotHasProperty": 4,

    # GLM-X intent 5: compare
    "SimilarTo":    5,
    "EtymologicallyRelatedTo": 5,

    # GLM-X intent 6: list
    "PartOf":       6,
    "HasA":         6,
    "MemberOf":     6,
    "MadeOf":       6,

    # GLM-X intent 7: example
    "InstanceOf":   7,
    "ExampleOf":    7,

    # GLM-X intent 8-15: mapped from composite patterns
    #   derive from multi-relation subgraph structure
}
```

#### 3.2.2 Subgraph Construction Algorithm

```python
def build_training_sample(concept: str, graph: ConceptNetGraph, max_hops: int = 2):
    """
    Given a seed concept, extract an ego-graph and derive the intent sequence.

    Algorithm:
    1. BFS from seed concept up to `max_hops` depth
    2. For each traversed edge, record the relation type
    3. Map each relation type to its GLM-X intent ID
    4. The resulting subgraph = (nodes, edges, activations)
    5. The resulting intent_sequence = mapped intents in traversal order

    Returns: (Subgraph, intent_sequence)
    """
    visited_nodes = set()
    visited_edges = []
    intent_sequence = []

    queue = [(concept, 0)]  # (node, depth)
    while queue:
        node, depth = queue.pop(0)
        if node in visited_nodes or depth > max_hops:
            continue
        visited_nodes.add(node)

        for neighbor, relation, weight in graph[node]:
            if neighbor not in visited_nodes:
                intent_id = CONCEPTNET_RELATION_MAP.get(relation)
                if intent_id is not None:
                    visited_edges.append((node, neighbor, relation))
                    intent_sequence.append(intent_id)
                queue.append((neighbor, depth + 1))

    return visited_nodes, visited_edges, intent_sequence
```

#### 3.2.3 Training Data Generation Script

**File: `g2p/data_generation/conceptnet_loader.py`**

```python
"""
ConceptNet → GLM-X Training Data Generator

Usage:
    python -m g2p.data_generation.conceptnet_loader \
        --max-samples 50000 \
        --max-hops 2 \
        --output-dir g2p/data/
"""

import numpy as np
from typing import List, Tuple, Generator
from g2p.types import Subgraph

# (full implementation as described above)
# ≈ 200 lines of code
```

**Steps:**
1. Download ConceptNet CSV (~200 MB compressed): `conceptnet-lite download`
2. Parse into in-memory graph (~8M edges, ~2 GB RAM peak)
3. For each of 50K seed concepts, extract 2-hop subgraphs
4. Filter: reject subgraphs with <2 nodes or empty intent sequences
5. Filter: validate each sample via `Subgraph.validate()`
6. Split: 80% train / 10% val / 10% test
7. Save as serialized `.npy` + metadata `.json`

#### 3.2.4 Augmentation: Synthetic Variation

**File: `g2p/data_generation/synthetic_augmentation.py`**

Augment real data with controlled perturbations:

```python
def augment_subgraph(sg: Subgraph) -> List[Subgraph]:
    """Generate 4 variants of a subgraph for robustness."""
    variants = []

    # 1. Drop low-confidence edges
    edges_kept = [(s,t,r) for (s,t,r) in sg.edges
                  if sg.edge_confidences.get((s,t,r), 0) > 0.3]
    variants.append(replace_edges(sg, edges_kept))

    # 2. Add noise to activations (±10%)
    noisy_activations = {
        n: np.clip(v * np.random.uniform(0.9, 1.1), 0.01, 1.0)
        for n, v in sg.node_activations.items()
    }
    variants.append(replace_activations(sg, noisy_activations))

    # 3. Shuffle node order (tests permutation invariance)
    shuffled_nodes = list(sg.nodes)
    np.random.shuffle(shuffled_nodes)
    variants.append(replace_nodes(sg, shuffled_nodes))

    # 4. Truncate to first 5 nodes (tests minimum input resilience)
    variants.append(truncate_nodes(sg, 5))

    return variants
```

#### 3.2.5 Data Validation & Statistics

Each generated dataset must be accompanied by a report:

```json
{
    "dataset": "conceptnet_train_50k",
    "total_samples": 50000,
    "valid": 49823,
    "invalid": 177,
    "node_count": {
        "min": 2, "max": 31, "mean": 7.2, "median": 6
    },
    "intent_sequence_length": {
        "min": 1, "max": 8, "mean": 3.1, "median": 3
    },
    "intent_distribution": {
        "0 (define)": 0.15,
        "1 (assert)": 0.28,
        "2 (explain_cause)": 0.11,
        "3 (explain_effect)": 0.08,
        "4 (contrast)": 0.04,
        "5 (compare)": 0.06,
        "6 (list)": 0.12,
        "7 (example)": 0.05,
        "8-15 (other)": 0.11
    },
    "edge_types_seen": 18,
    "duplicate_subgraphs": 12
}
```

### 3.3 Secondary Source: Wikipedia Entity Extraction

**File: `g2p/data_generation/wikidata_loader.py`**

```python
"""
Wikipedia/Wikidata → GLM-X Training Data

For each Wikipedia article:
1. Extract first 3 sentences (summary)
2. Extract entity links (hyperlinks to other articles)
3. Build subgraph: article → relation → linked_entity
4. Map relation type (from Wikidata property) → intent_id
5. Label: the discourse structure of the summary

≈ 300 lines of code
"""
```

### 3.4 Dataset Versioning

Maintain a manifest at `g2p/data/manifest.json`:

```json
{
    "version": "1.0",
    "sources": {
        "conceptnet": {
            "version": "5.7",
            "samples_total": 50000,
            "samples_valid": 49823,
            "hash_sha256": "a1b2c3d4..."
        },
        "synthetic_augmented": {
            "base": "conceptnet",
            "augmentation": "4x_drop_noise_shuffle_truncate",
            "samples": 199292
        },
        "wikidata": {
            "version": "2026-05-01",
            "samples": 10000
        }
    },
    "train_val_test_split": [0.8, 0.1, 0.1],
    "total_available": 269292
}
```

---

## 4. Phase 2: IntentFFN Training

### 4.1 Current Training Code (Baseline)

**File: `g2p/g2p_planner.py`** lines 147-240 (method `train()`)

Current limitations:
- **Predicts only the FIRST intent** of the sequence (cross-entropy on `intent_seq[0]`)
- Uses only 2000 synthetic samples
- No learning rate scheduling
- No weight decay
- No model checkpointing
- No early stopping on best val loss (saves only internally)

### 4.2 Enhanced Training Pipeline

**File: `g2p/training/trainer.py`**

```python
"""
Enhanced IntentFFN Trainer

Upgrades from the baseline train():
1. Full sequence prediction (teacher forcing or sequence loss)
2. Learning rate scheduler (cosine decay + warmup)
3. Gradient clipping
4. Model checkpointing (best val loss → file)
5. TensorBoard / CSV logging
6. Weight decay regularization
7. Multi-class focal loss for imbalanced intents
8. Configurable via YAML
"""
```

#### 4.2.1 Loss Function Options

| Loss | Formula | When to Use |
|---|---|---|
| **CrossEntropy** (current) | `CE(y, ŷ)` | Balanced intent distribution |
| **Focal Loss** | `-(1-p_t)^γ * log(p_t)` | Imbalanced intents (γ=2) |
| **Sequence NLL** | `Σ log P(i_t | i_<t, graph)` | Full sequence prediction |
| **Margin Loss** | `max(0, margin - (p_correct - p_wrong))` | Confidence calibration |

**Recommended:** Focal Loss (γ=2.0) with α-weighting for first-intent prediction, then Sequence NLL for full sequences.

#### 4.2.2 Training Loop Pseudocode

```python
def train_enhanced(config: TrainingConfig, train_data, val_data):

    model = IntentFFN(config.ffn)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = FocalLoss(gamma=2.0, alpha=[...])  # per-class weights

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(config.epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)                       # (B, 16)
            loss = criterion(logits, batch_y)             # CE or focal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = evaluate(model, val_loader, criterion)

        # LR scheduling
        scheduler.step()

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            torch.save(best_state, "g2p/models/intent_ffn_best.pt")

        # Early stopping
        if epoch - last_improvement > config.patience:
            break

    # Load best model
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val_loss, "epochs": epoch}
```

#### 4.2.3 Full Sequence Prediction (Advanced)

Currently, only `intent_seq[0]` is used as the target. To predict full sequences:

```python
def train_full_sequence(model, encoder, training_data, config):
    """
    Teacher-forcing approach:
    - Encode subgraph → embedding
    - Feed embedding as initial hidden state to an LSTM head
    - At each step, use ground-truth previous intent as input
    - Predict next intent

    Model architecture change in IntentFFN:
        embedding (384) → MLP body (128) → classifier_input (128)
        classifier_input → LSTM(128, 16) → intent_logits per step
    """
    # Requires modifying IntentFFN to have an LSTM head
    # ≈ 350K total params after change
```

**However**, since beam search already handles sequence-level decoding well with per-step logits, the simpler single-intent prediction + beam search is the recommended default. Full sequence prediction is an optimization for phase 3.

### 4.3 Training Configuration YAML

**File: `configs/training/train_config.yaml`**

```yaml
# ── G2P Planner Training Configuration ──
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adamw"
  loss: "focal"                    # cross_entropy | focal | sequence_nll
  focal_gamma: 2.0
  gradient_clip_norm: 1.0

  # LR schedule
  scheduler: "cosine"              # cosine | step | plateau
  warmup_epochs: 5

  # Early stopping
  early_stopping_patience: 10
  early_stopping_metric: "val_loss"

  # Data
  train_test_split: 0.8
  validation_split: 0.1

  # Augmentation
  augment: true
  augmentation_multiplier: 4

  # Logging
  log_interval: 10
  save_best: true
  output_dir: "g2p/models/"

ffn:
  input_dim: 384
  hidden_dim: 128
  num_layers: 2
  activation: "relu"
  dropout: 0.1
  classifier_input_dim: 128
  classifier_hidden_dim: 64
  classifier_num_layers: 1
  output_dim: 16

sentence_bert:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  freeze: true                     # NEVER train Sentence-BERT
```

### 4.4 Evaluation During Training

Metrics tracked every epoch:

```python
training_metrics = {
    "train_loss": float,
    "val_loss": float,
    "train_accuracy_top1": float,   # Top-1 intent match
    "val_accuracy_top1": float,
    "train_accuracy_top3": float,   # Top-3 contains correct intent
    "val_accuracy_top3": float,
    "train_sequence_accuracy": float,  # Exact full sequence match
    "val_sequence_accuracy": float,
    "learning_rate": float,
    "gradient_norm": float,
}
```

---

## 5. Phase 3: Hyperparameter Sweep & Model Selection

### 5.1 Sweep Configuration

**File: `g2p/training/hyper_sweep.py`**

```python
"""
Hyperparameter sweep for IntentFFN.

Searches over:
- hidden_dim: [64, 128, 256]
- num_layers: [1, 2, 3]
- dropout: [0.0, 0.1, 0.2, 0.5]
- learning_rate: [1e-4, 3e-4, 1e-3, 3e-3]
- classifier_num_layers: [0, 1, 2]
- weight_decay: [0, 1e-5, 1e-4]

Total combinations: 4 × 3 × 4 × 4 × 3 × 3 = 1728
After pruning correlated: ~200 representative runs

Runtime on 8GB CPU laptop: ~2-4 minutes per run (10K samples, 50 epochs)
Total sweep time: ~8-13 hours (sequential) or ~3 hours (parallel, 4 workers)
"""
```

#### 5.1.1 Sweep Strategy

1. **Grid search** over primary 4 parameters (hidden_dim, num_layers, dropout, LR)
2. **Bayesian optimization** (via Optuna) for fine-tuning around best grid point
3. **Prune bad runs early** (if epoch 5 val_loss > threshold, kill run)

```powershell
# Install sweep tool
pip install optuna

# Run sweep
python -m g2p.training.hyper_sweep \
    --study-name "intentffn_v1" \
    --n-trials 200 \
    --data g2p/data/conceptnet_train_50k.npy \
    --output-dir g2p/models/sweep_results/
```

### 5.2 Model Selection Criteria

| Metric | Weight | Description |
|---|---|---|
| **Val Loss** | 0.4 | Primary optimization target |
| **Top-1 Accuracy** | 0.3 | Correct first intent |
| **Top-3 Accuracy** | 0.1 | Correct intent in top 3 |
| **Sequence Accuracy** | 0.1 | Exact intent sequence match |
| **Inference Latency** | 0.05 | ms per plan (on CPU) |
| **Model Size** | 0.05 | KB (prefer smaller) |

### 5.3 Best Model Report

```json
{
    "best_config": {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "classifier_num_layers": 1,
        "weight_decay": 1e-5
    },
    "val_metrics": {
        "loss": 0.42,
        "top1_accuracy": 0.87,
        "top3_accuracy": 0.96,
        "sequence_accuracy": 0.73,
        "latency_ms": 0.8
    },
    "selected_over": [
        {"hidden_dim": 256, "reason": "larger but 0.01% accuracy gain, 2x slower"},
        {"num_layers": 3, "reason": "overfits, val_loss higher"}
    ]
}
```

### 5.4 Logging & Visualization

```powershell
# Logs go to:
g2p/models/sweep_results/
├── optuna_study.pkl
├── sweep_summary.json
├── learning_curves/
│   ├── run_001.html
│   └── ...
├── parallel_coordinates.html
└── hyperparameter_importance.png
```

---

## 6. Phase 4: Micro-Decoder Template Expansion

### 6.1 Current Templates (15 patterns)

**File: `decoder/config_decoder.yaml`** lines 37-92

Currently covers:
- Single intents: define, assert, explain_cause, explain_effect, contrast, compare, list, example, conclude, uncertain, clarify, summarize, elaborate, transition, emphasize
- Composites: [define, assert], [assert, explain, example], [contrast, assert], [list, conclude], [assert, uncertain], [assert, contrast, assert]

**Missing (gaps to fill):**

| Intent Sequence | Count in Training Data | Template Exists? |
|---|---|---|
| [1] (assert) | 28% | Yes |
| [0] (define) | 15% | Yes |
| [6] (list) | 12% | No — needs "List: {node0}, {node1}, and {node2}" |
| [2] (explain_cause) | 11% | Yes |
| [3] (explain_effect) | 8% | Yes |
| [5] (compare) | 6% | Yes |
| [0, 1] (define + assert) | 5% | Yes |
| [1, 2] (assert + explain) | 4% | No |
| [7] (example) | 5% | Yes |
| [4] (contrast) | 4% | Yes |
| [1, 6] (assert + list) | 3% | No |
| [1, 2, 7] | 3% | Yes |
| [2, 8] (explain + conclude) | 2% | No |
| [1, 10] | 2% | Yes |
| [0, 6] (define + list) | 1.5% | No |
| [12] (summarize) | 1% | Yes |

### 6.2 Template Expansion Script

**File: `scripts/expand_templates.py`**

```python
"""
Auto-generate templates for top-K most common intent sequences.

1. Load training data intent sequences
2. Count frequency distribution
3. For top 50 sequences without templates, generate placeholder templates
4. Human review step (print to stdout for manual verification)
5. Append to config_decoder.yaml

Usage: python scripts/expand_templates.py --top-k 50
"""
```

**Target: expand from 15 to 50+ templates covering 95% of predicted sequences.**

### 6.3 Template Generation Rules

```python
TEMPLATE_PATTERNS = {
    # Single intent → single sentence pattern
    "single": {
        0: "A {node0} is {node1}.",
        1: "{node0} {relation0} {node1}.",
        2: "The reason is that {node0} {relation0} {node1}.",
        3: "As a result, {node0} {relation0} {node1}.",
        4: "Unlike {node0}, {node1} {relation0} {node2}.",
        5: "{node0} is similar to {node1}. Both {relation0} {node2}.",
        6: "The key points are: {node0}, {node1}, and {node2}.",
        7: "For instance, {node0} {relation0} {node1}.",
        8: "In conclusion, {node0} {relation0} {node1}.",
        9: "Is {node0} {relation0} {node1}?",
        10: "I'm not entirely certain, but {node0} appears to {relation0} {node1}.",
        11: "Could you clarify whether you mean {node0} or {node1}?",
        12: "In summary, {node0} {relation0} {node1}.",
        13: "{node0} {relation0} {node1}, and more specifically, {node2}.",
        14: "Now, regarding {node0}: {node1} {relation0} {node2}.",
        15: "Importantly, {node0} {relation0} {node1}.",
    },
    # Composite → multi-sentence
    "composite": {
        # Defined by recursion over single patterns
    }
}
```

---

## 7. Phase 5: T5-Small Fine-Tuning (Optional)

### 7.1 When to Fine-Tune T5

Consider fine-tuning ONLY if:
1. Template coverage < 90% of predicted intent sequences
2. Generated text is too rigid / robotic
3. User requires open-ended generation (not just templates)

### 7.2 Training Data Generation

Generate (prompt, target_text) pairs from templates + walk paths:

```python
def generate_finetune_data(walk, plan, template_decoder):
    """
    Input:
        walk: WalkResult(path=["cat", "mammal", "animal"],
                         path_labels=["cat", "mammal", "animal"],
                         path_relations=["is_a", "is_a"])
        plan: Plan(intent_sequence=[0, 1])

    Output:
        prompt: "intents: [0, 1] | nodes: [cat, mammal, animal] | relations: [is_a, is_a]"
        target: "A cat is a mammal. This means cat is a animal."

    Generate ~10K such pairs from training dataset.
    """
    prompt = f"intents: {plan.intent_sequence} | " \
             f"nodes: {walk.path_labels} | " \
             f"relations: {walk.path_relations}"
    target, ok = template_decoder.decode(
        walk.path_labels, walk.path_relations, plan.intent_sequence
    )
    if ok:
        return (prompt, target)
    return None
```

### 7.3 Fine-Tuning Execution

Already implemented in `decoder/t5_decoder.py:25-91` (`T5Decoder.finetune()`).

```python
# Default config from config_decoder.yaml:
finetune:
    learning_rate: 3e-5
    batch_size: 8
    epochs: 3
    warmup_steps: 500
    weight_decay: 0.01
```

```powershell
# Run fine-tuning
python -c "
from decoder.t5_decoder import T5Decoder
from decoder.config_loader import load_config

cfg = load_config('decoder/config_decoder.yaml')
t5 = T5Decoder(cfg['t5'], cfg['validation'])

# Generate 10K pairs
pairs = generate_finetune_data(training_data)
print(f'Training on {len(pairs)} pairs')

t5.finetune(pairs)
# Saves to t5-small (in-place) or a custom path
"
```

### 7.4 Resource Usage for T5 Fine-Tune

| Memory | CPU Time | RAM | 
|---|---|---|
| T5-small (60M params, fp32) | ~240 MB | 3-4 hrs (10K pairs, 3 epochs) |
| T5-small (60M params, int8) | ~60 MB | 5-6 hrs (slower due to quantize overhead) |
| T5-base (220M params) | ~880 MB | NOT recommended on 8GB |

**Verdict:** T5-small fine-tune fits on 8GB, but it's optional.

---

## 8. Phase 6: Full Pipeline Integration Testing

### 8.1 Existing Test Suite

The integration test suite (`integration_tests/test_integration.py`) already validates the full pipeline with 25 tests:

```
TestGLMXFullPipeline:
  ✓ test_dry_run_no_crash
  ✓ test_identity_short_circuit
  ✓ test_resonance_to_g2p_single_subgraph
  ✓ test_resonance_to_g2p_batch
  ✓ test_resonance_to_g2p_empty
  ✓ test_g2p_to_walker
  ✓ test_g2p_to_walker_empty_plan
  ✓ test_g2p_to_walker_batch
  ✓ test_walker_to_decoder
  ✓ test_walker_to_decoder_minimal_path
  ✓ test_walker_to_decoder_batch
  ✓ test_walker_to_decoder_empty_path
  ✓ test_resonance_to_answer
  ✓ test_resonance_to_answer_batch
  ✓ test_resonance_to_answer_empty
  ✓ test_learning_engine_activation
  ✓ test_learning_engine_activation_invalid
  ✓ test_full_pipeline_end_to_end
  ✓ test_full_pipeline_batch
  ✓ test_full_pipeline_with_empty_query
  ✓ test_full_pipeline_concurrent
  ✓ test_config_inheritance
  ✓ test_type_adapter_resonance_to_g2p
  ✓ test_type_adapter_g2p_to_walker
  ✓ test_type_adapter_walker_to_decoder
```

### 8.2 New Tests to Add

After training, add these tests:

| Test | Priority | Description |
|---|---|---|
| `test_trained_ffn_improves_over_synthetic` | High | Trained model beats synthetic-only on val set |
| `test_trained_ffn_plan_accuracy_threshold` | High | Accuracy > 80% on held-out test |
| `test_trained_ffn_rejects_unknown_graphs` | Medium | Low confidence on out-of-distribution |
| `test_template_expansion_coverage` | Medium | Templates cover > 90% of predicted sequences |
| `test_t5_finetune_improvement` | Low | Fine-tuned T5 scores higher on BLEU/Rouge |
| `test_plan_confidence_calibration` | High | Confidence scores match empirical accuracy |
| `test_model_quantization_no_degradation` | Medium | int8 model accuracy within 1% of fp32 |

### 8.3 Test Scripts

**File: `scripts/run_all_tests.ps1`**

```powershell
<#
.SYNOPSIS
    Run all GLM-X tests
.DESCRIPTION
    Runs unit tests, integration tests, and trained-model validation
#>

$Root = Split-Path -Parent $PSScriptRoot
$Results = @{}

Write-Host "=== G2P Unit Tests ===" -ForegroundColor Cyan
$Results.G2P = python -m pytest g2p\tests\test_g2p_all.py --json --tb=short

Write-Host "=== Decoder Unit Tests ===" -ForegroundColor Cyan
$Results.Decoder = python -m pytest decoder\tests\test_all.py --json --tb=short

Write-Host "=== Resonance Unit Tests ===" -ForegroundColor Cyan
$Results.Resonance = python -m pytest resonance\tests\test_unit\ --json --tb=short

Write-Host "=== Full Pipeline Integration ===" -ForegroundColor Cyan
$Results.Integration = python integration_tests\test_integration.py --json --tb=short

Write-Host "=== Walker Tests ===" -ForegroundColor Cyan
$Results.Walker = python -m pytest walker\tests\ --json --tb=short

Write-Host "=== Graph Tests ===" -ForegroundColor Cyan
$Results.Graph = python -m pytest graph\tests\ --json --tb=short

Write-Host "=== Trained Model Validation ===" -ForegroundColor Cyan
$Results.TrainedModel = python -m g2p.training.evaluate --model g2p\models\intent_ffn_best.pt

# Summary
$Passed = ($Results.Values | Where-Object { $_.success }).Count
$Total = $Results.Count
Write-Host "`n=== SUMMARY: $Passed/$Total test suites passed ===" -ForegroundColor Green
```

### 8.4 Test Criteria for Trained Model

```python
TRAINED_MODEL_THRESHOLDS = {
    "val_loss": 0.5,          # max acceptable
    "top1_accuracy": 0.85,    # min acceptable
    "sequence_accuracy": 0.70,
    "confidence_calibration_error": 0.10,  # ECE < 10%
    "inference_latency_ms": 5.0,           # per query
}
```

---

## 9. Phase 7: Evaluation, Benchmarking & Regression

### 9.1 Evaluation Suite

**File: `evaluation/plan_accuracy.py`**

```python
"""
Evaluate trained IntentFFN on held-out test set.

Metrics:
- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy
- Sequence accuracy (exact match)
- Sequence BLEU (partial match)
- Macro F1 per intent class
- Confusion matrix (16×16)
- Per-intent precision/recall/F1

Output: evaluation/reports/plan_accuracy_report.json
"""
```

**File: `evaluation/confidence_calibration.py`**

```python
"""
Confidence calibration analysis.

Metrics:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability diagram (confidence bins vs accuracy)
- Brier score

Output: evaluation/reports/calibration_report.json + .png
"""
```

**File: `evaluation/end_to_end_benchmark.py`**

```python
"""
End-to-end benchmark on full pipeline.

Metrics:
- End-to-end latency (ms per query)
- Throughput (queries/second)
- Memory peak (MB)
- Plan accuracy (ground truth available)
- Text quality (human eval sample, BLEU-4, ROUGE-L)

Scenarios:
- Single query (cold start)
- Single query (warm)
- Batch of 10
- Concurrent 5 queries
- 100-query stress test

Output: evaluation/reports/benchmark_report.json
"""
```

**File: `evaluation/report_generator.py`**

```python
"""
Generate HTML/PDF report combining all evaluations.

Sections:
1. Model architecture (table of layers, params)
2. Training curves (loss, accuracy vs epoch)
3. Hyperparameter sweep results
4. Confusion matrix
5. Calibration diagram
6. Benchmark results
7. Test results summary

Output: evaluation/reports/final_report.html
"""
```

### 9.2 Regression Test Benchmarks

Results stored in `evaluation/regression/`:

```json
{
    "benchmark_version": "1.0",
    "timestamp": "2026-05-15",
    "hardware": {
        "cpu": "AMD Ryzen 7 5700U",
        "ram_gb": 8,
        "os": "Windows 11"
    },
    "metrics": {
        "g2p_plan_accuracy_top1": 0.87,
        "g2p_plan_accuracy_top3": 0.96,
        "g2p_latency_ms": 0.8,
        "sentence_bert_latency_ms": 25.0,
        "walker_latency_ms": 5.0,
        "decoder_latency_ms_template": 0.3,
        "decoder_latency_ms_t5": 120.0,
        "end_to_end_latency_ms": 31.1,
        "end_to_end_throughput_qps": 32.0,
        "memory_peak_mb": 380,
        "confidence_ece": 0.04,
        "confidence_brier": 0.12
    }
}
```

### 9.3 Human Evaluation (Sample)

For text quality, sample 100 outputs and rate on a 1-5 scale:

| Criteria | Weight | Description |
|---|---|---|
| **Factual accuracy** | 0.4 | Does text correctly reflect the graph? |
| **Coherence** | 0.3 | Is it grammatically correct and readable? |
| **Intent alignment** | 0.2 | Does text fulfill the planned intent? |
| **Conciseness** | 0.1 | Appropriate length, not verbose |

**Target: average score ≥ 4.0 / 5.0**

---

## 10. Phase 8: Quantization & Production Deployment

### 10.1 Model Quantization

The 220K IntentFFN is already tiny, but quantize for consistency:

```python
def quantize_intent_ffn(model: IntentFFN) -> IntentFFN:
    """
    Quantize weights from fp32 to int8 using PyTorch dynamic quantization.

    fp32: 220K × 4 bytes = 880 KB
    int8: 220K × 1 byte = 220 KB

    Also applies to Sentence-BERT (22M params):
    fp32: 22M × 4 = 88 MB
    int8: 22M × 1 = 22 MB
    """
    import torch.quantization

    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # all Linear layers
        dtype=torch.qint8,
    )
    return quantized


def quantize_sentence_bert():
    """Sentence-BERT can be quantized or replaced with ONNX runtime."""
    # Option A: torch quantization (requires minimal changes)
    # Option B: ONNX Runtime (faster inference, larger setup)
    # Option C: keep fp32 (already 88 MB — fine for 8 GB)
```

### 10.2 Model Export

```python
# Export format: PyTorch JIT script + weights file

# IntentFFN
torch.jit.script(model).save("g2p/models/intent_ffn_jit.pt")
# or just save state_dict
torch.save(model.state_dict(), "g2p/models/intent_ffn_state.pt")

# Full pipeline config
import json
config_export = {
    "intent_ffn": "g2p/models/intent_ffn_state.pt",
    "sentence_bert": "sentence-transformers/all-MiniLM-L6-v2",
    "decoder_config": "decoder/config_decoder.yaml",
    "g2p_config": "configs/training/best_config.yaml",
    "quantized": True,
    "version": "1.0"
}
with open("g2p/models/model_manifest.json", "w") as f:
    json.dump(config_export, f, indent=2)
```

### 10.3 Deployment Checklist

```markdown
## Production Deployment Checklist

### Model Artifacts
- [ ] `g2p/models/intent_ffn_quantized.pt` (int8)
- [ ] `g2p/models/model_manifest.json`
- [ ] `decoder/config_decoder.yaml` (with expanded templates)
- [ ] `configs/training/best_config.yaml`

### Dependencies
- [ ] torch (CPU)
- [ ] numpy
- [ ] sentence-transformers
- [ ] pyyaml
- [ ] transformers (only if using T5)

### Files to Deploy
- [ ] `g2p/` — full G2P planner module
- [ ] `decoder/` — full decoder module
- [ ] `walker/` — walker module
- [ ] `resonance/` — resonance engine module
- [ ] `graph/` — graph store module
- [ ] `configs/` — all configs
- [ ] `config_core.py`
- [ ] `glmx_types.py`

### Before Go-Live
- [ ] Run all 25 integration tests
- [ ] Run plan accuracy eval on held-out test set
- [ ] Run 100-cycle end-to-end stress test
- [ ] Measure peak memory usage (should be <400 MB)
- [ ] Measure average latency (should be <50ms per query)
- [ ] Verify no regressions from synthetic baseline
```

---

## 11. Resource Budget & 8 GB Laptop Feasibility

### 11.1 Memory Budget (Peak)

| Component | Memory (fp32) | Memory (int8) |
|---|---|---|
| Python interpreter | ~50 MB | ~50 MB |
| Sentence-BERT (frozen) | ~200 MB | ~60 MB (quantized) |
| IntentFFN (trained) | ~1 MB | ~0.3 MB |
| Training batch (32 samples) | ~50 MB | ~50 MB |
| ConceptNet graph (processed) | ~1.5 GB | ~1.5 GB |
| Data loaders / caches | ~200 MB | ~200 MB |
| **Subtotal (training)** | **~2 GB** | **~1.86 GB** |
| T5-small (if used) | ~240 MB | ~60 MB |
| **Subtotal (with T5)** | **~2.24 GB** | **~1.92 GB** |
| OS overhead | ~1.5 GB | ~1.5 GB |
| **Total (training)** | **~3.74 GB** | **~3.42 GB** |
| **Headroom on 8 GB** | **~4.26 GB** | **~4.58 GB** |

### 11.2 CPU Time Estimates

| Task | Time | Parallelizable? |
|---|---|---|
| Download ConceptNet (~200 MB) | 5-10 min | No |
| Parse ConceptNet (8M edges) | 10-15 min | No |
| Generate 50K subgraph samples | 20-30 min | Yes (4 workers = 10 min) |
| Train IntentFFN (10K samples, 50 epochs) | 5-10 min | No |
| Hyperparameter sweep (200 runs) | 8-13 hrs | Yes (4 workers = 3 hrs) |
| Full pipeline integration tests | 2-5 min | No |
| End-to-end benchmark | 5-10 min | No |
| Export / quantize | 1 min | No |
| **Total (one full pass)** | **~10-15 hrs** | **~5-6 hrs with parallelism** |

### 11.3 What NOT to Do on 8 GB

```markdown
✗ Train/fine-tune T5-base (220M) — 880 MB is too large with other components
✗ Train/fine-tune GPT-2 (124M) — not part of architecture anyway
✗ Train Sentence-BERT from scratch — frozen, 22M params too costly
✗ Load full Wikipedia into memory — stream/process in chunks
✗ Full-batch gradient descent — use mini-batches (batch_size ≤ 64)
✗ Store 8M ConceptNet edges in Python objects — use numpy arrays or SQLite
```

---

## 12. Timeline & Milestones

### 12.1 Week-by-Week Schedule

```
Week 1: Environment + Data Pipeline
├── Mon: Verify all 25 integration tests pass
├── Tue: Download ConceptNet, build parser
├── Wed: Write ConceptNet → Subgraph converter
├── Thu: Generate 50K samples, validate, split
└── Fri: Augmentation script + dataset manifest

Week 2: Training + Hyperparameter Sweep
├── Mon: Enhance trainer (focal loss, scheduler, checkpoint)
├── Tue: Train baseline model (current synthetic data)
├── Wed: Train first ConceptNet model
├── Thu: Hyperparameter sweep (200 runs, automated)
└── Fri: Analyze results, select best config

Week 3: Evaluation + Template Expansion
├── Mon: Run full evaluation suite on best model
├── Tue: Expand templates from 15→50+, test coverage
├── Wed: Confidence calibration analysis
├── Thu: Add regression tests, update thresholds
└── Fri: Benchmark end-to-end latency, memory

Week 4: Integration + Documentation + Polish
├── Mon: Quantize model, verify no accuracy drop
├── Tue: Re-run all integration tests with trained model
├── Wed: Generate final report (HTML)
├── Thu: Update AGENTS.md, README, training docs
└── Fri: Final review, model freeze v1.0

Week 5 (Optional): T5 Fine-Tuning
├── Mon: Generate 10K (prompt, text) fine-tune pairs
├── Tue: Fine-tune T5-small
├── Wed: Evaluate fine-tuned T5 vs template decoder
├── Thu: Hybrid mode testing
└── Fri: Documentation + model export
```

### 12.2 Milestones & Exit Criteria

| Milestone | Date | Exit Criteria |
|---|---|---|
| **M0: Baseline** | Day 1 | All 25 integration tests pass on `main` |
| **M1: Data Ready** | Day 5 | 50K+ validated ConceptNet samples, train/val/test split done |
| **M2: Model Trained** | Day 12 | Best model selected (val_loss < 0.5, top1 > 85%) |
| **M3: Tested** | Day 19 | Evaluation report complete, regression tests pass |
| **M4: Release v1.0** | Day 26 | Quantized model deployed, all tests green, documentation done |
| **M5: T5 Release** | Day 33 | (Optional) T5 fine-tuned, hybrid mode validated |

---

## 13. Appendices

### 13.1 A: Intent Vocabulary Reference

```python
INTENT_VOCAB = {
    0:  "define",         # Define a concept by its class/type
    1:  "assert_fact",    # State a factual relationship
    2:  "explain_cause",  # Explain why something happens
    3:  "explain_effect", # Explain what results from something
    4:  "contrast",       # Highlight differences
    5:  "compare",        # Highlight similarities
    6:  "list",           # Enumerate items in a set
    7:  "example",        # Provide a concrete example
    8:  "conclude",       # Draw a conclusion
    9:  "question",       # Ask a question
    10: "uncertain",      # Express uncertainty
    11: "clarify",        # Ask for clarification
    12: "summarize",      # Summarize key points
    13: "elaborate",      # Provide more detail
    14: "transition",     # Shift topic
    15: "emphasize",      # Highlight importance
}
```

### 13.2 B: ConceptNet Relation Types Count

| Relation | Count (ConceptNet 5.7) | Maps to Intent |
|---|---|---|
| RelatedTo | 2,120,000 | 1 (assert_fact) |
| IsA | 1,180,000 | 0 (define) |
| PartOf | 710,000 | 6 (list) |
| HasA | 650,000 | 6 (list) |
| UsedFor | 540,000 | 1 (assert_fact) |
| CapableOf | 460,000 | 1 (assert_fact) |
| AtLocation | 390,000 | 1 (assert_fact) |
| Causes | 210,000 | 2 (explain_cause) |
| HasProperty | 190,000 | 1 (assert_fact) |
| Desires | 140,000 | 1 (assert_fact) |
| Synonym | 130,000 | 0 (define) |
| Antonym | 80,000 | 4 (contrast) |
| MannerOf | 50,000 | 0 (define) |
| CreatedBy | 40,000 | 1 (assert_fact) |
| InstanceOf | 30,000 | 7 (example) |
| SimilarTo | 20,000 | 5 (compare) |
| DerivedFrom | 15,000 | 1 (assert_fact) |
| *Total mapped* | *~6,965,000* | — |

### 13.3 C: Error Budget for Training

```python
# Allowable errors during training pipeline:

# Data pipeline: <1% invalid samples
assert invalid_ratio < 0.01, f"{invalid_ratio:.4f} samples failed validation"

# Training: no NaN gradients
assert not any(torch.isnan(p.grad).any() for p in model.parameters())

# Quantization: <1% accuracy drop
fp32_acc = evaluate(model_fp32)
int8_acc = evaluate(model_int8)
assert fp32_acc - int8_acc < 0.01, "Quantization degraded accuracy by >1%"

# Regression: no test regression
old_accuracy = load_regression_baseline()
new_accuracy = evaluate_current_model()
assert new_accuracy >= old_accuracy - 0.005, "Regression detected!"
```

### 13.4 D: Full Command Reference

```powershell
# ───────────────────────────────────────
# SETUP
# ───────────────────────────────────────
git clone https://github.com/project-genesis-ai-labs/glm_x_code.git
cd glm_x_code
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt   # (create if not exists)

# ───────────────────────────────────────
# DATA GENERATION
# ───────────────────────────────────────
python -m g2p.data_generation.conceptnet_loader --max-samples 50000 --max-hops 2
python -m g2p.data_generation.synthetic_augmentation --input g2p/data/conceptnet_50k.npy --multiplier 4
python -m g2p.data_generation.wikidata_loader --max-samples 10000

# ───────────────────────────────────────
# TRAINING
# ───────────────────────────────────────
python -m g2p.training.trainer --config configs/training/train_config.yaml --data g2p/data/train.npy
python -m g2p.training.hyper_sweep --study-name "intentffn_v1" --n-trials 200
python -m g2p.training.trainer --config g2p/models/sweep_results/best_config.yaml --data g2p/data/train.npy

# ───────────────────────────────────────
# EVALUATION
# ───────────────────────────────────────
python -m evaluation.plan_accuracy --model g2p/models/intent_ffn_best.pt --data g2p/data/test.npy
python -m evaluation.confidence_calibration --model g2p/models/intent_ffn_best.pt
python -m evaluation.end_to_end_benchmark --iterations 100
python -m evaluation.report_generator --output-dir evaluation/reports/

# ───────────────────────────────────────
# TESTING
# ───────────────────────────────────────
.\scripts\run_all_tests.ps1
python integration_tests\test_integration.py --json
python g2p\tests\test_g2p_all.py --json

# ───────────────────────────────────────
# EXPORT & DEPLOY
# ───────────────────────────────────────
python -c "from g2p.intent_ffn import IntentFFN; model=IntentFFN.load('g2p/models/intent_ffn_best.pt'); model.quantize().save('g2p/models/intent_ffn_quantized.pt')"

# ───────────────────────────────────────
# TEMPLATE EXPANSION
# ───────────────────────────────────────
python scripts/expand_templates.py --top-k 50
```

### 13.5 E: Key Architecture Parameters

```
Sentence-BERT:       all-MiniLM-L6-v2  →  384-dim embeddings (frozen)
IntentFFN:           384 → 128 → 128 → 16   (220K params, int8 = 220 KB)
Beam Search:         width=2, max_length=8, temp=1.0, rep_penalty=1.2
T5 Decoder:          t5-small (60M params, optional)
Resonance Engine:    Tier1 + Tier2 + ES meta-controller (θ ~20 params)
Walker:              Personalized PageRank (teleport=0.15, max_steps=100)
Hebbian Learning:    η=0.01, decay=0.001, eligibility trace τ=10
```

### 13.6 F: Related Files & Their Locations

| File | Purpose |
|---|---|
| `g2p/g2p_planner.py` | Main G2P planner (includes `train()`) |
| `g2p/intent_ffn.py` | The 220K MLP model to train |
| `g2p/beam_search.py` | Beam search decoder for intent sequences |
| `g2p/heuristic_planner.py` | Rule-based fallback planner |
| `g2p/graph_to_text.py` | Graph → text serialization (feature extraction) |
| `g2p/config.py` | All g2p config dataclasses |
| `g2p/types.py` | Subgraph, Plan datatypes |
| `g2p/train.py` | Current training function (baseline) |
| `g2p/tests/test_g2p_all.py` | 1287 lines of unit tests |
| `decoder/micro_decoder.py` | Main decoder orchestrator |
| `decoder/t5_decoder.py` | T5 decoder + fine-tune implementation |
| `decoder/template_decoder.py` | Template-based decoder |
| `decoder/config_decoder.yaml` | Decoder config + templates |
| `decoder/tests/test_all.py` | Decoder unit tests |
| `resonance/engine.py` | Resonance engine (ES controller) |
| `walker/graph_walker.py` | PPR-based walker |
| `learning/hebbian.py` | Hebbian learning engine |
| `integration_tests/test_integration.py` | Full pipeline integration tests (25 tests) |
| `config_core.py` | Core configuration loader |
| `glmx_types.py` | Shared type system |

---

> **Document version 1.0** — Last updated 2026-05-15  
> For questions, open an issue at https://github.com/project-genesis-ai-labs/glm_x_code/issues
