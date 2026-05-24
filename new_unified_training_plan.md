# GLM-X Unified Training Architecture Plan

## Architecture Overview (Target State)

```
Raw data (.parquet/.json/.csv)
        ↓
   DataLoader (format-agnostic)
        ↓
   Unified dict: {sentences: [...], triples: [(subj, rel, obj)]}
        ↓
   KGBuilderPipeline (spaCy + BGE + EntityResolver)
        ↓
   SQLiteGraphStore.write(dataset_name.db)
        ↓
   ─────────────────────────────────────
        ↓                          ↓
   GLM-X Pipeline           Unified Trainer
   (read-only from .db)     (reads .db, trains model)
        ↓                          ↓
   Answers + Walks          Checkpoints (IntentFFN + T5 + IntentBias)
                                    ↓
                            Next dataset → Load checkpoint → Train more
```

## Guiding Principles

| Principle | How |
|---|---|
| **Universal schema** | Fixed SQLite schema regardless of input. 4 tables: nodes, edges, embeddings, metadata |
| **Format-agnostic** | DataLoader abstraction handles .parquet / .json / .csv / .db uniformly |
| **Single model** | One IntentFFN + T5 + IntentBias checkpoint, updated by every dataset |
| **Continual learning** | Replay buffer (20% mix), lower LR on subsequent datasets, manifest tracking |
| **Decoupled graph ↔ model** | .db provides the graph, checkpoint provides the intelligence. Independent. |
| **Time-independent** | Train today, next week, next year — same script, same format, same model |
| **Zero heuristic** | No hardcoded rules in data loading, schema, or model. Everything adaptive |

---

## Phase 0: Unified Data Loaders

### New directory: `data_loader/`

```
data_loader/
  __init__.py
  base.py              # Abstract DataLoader
  parquet_loader.py    # .parquet → unified format
  json_loader.py       # .json → unified format
  csv_loader.py        # .csv → unified format
  sqlite_loader.py     # .db (already in schema) → skip KG build
  utils.py             # URI parsing, language filtering
```

### Unified intermediate format

Every loader produces:

```python
{
    "sentences": ["Paris is in France.", ...],      # for spaCy extraction
    "triples": [("Paris", "is in", "France"), ...],  # pre-extracted triples (optional)
    "metadata": {
        "source": "conceptnet",
        "language": "en",
        "num_entries": 5000,
    }
}
```

### Loader behavior by format

- **`.parquet`**: reads subject/predicate/object columns, extracts labels from URIs, filters by language. Already 80% done in `loader.py`.
- **`.json`**: expects `[{"head": "...", "relation": "...", "tail": "..."}]` or `[{"sentence": "..."}]` or `[{ "subject": "...", "predicate": "...", "object": "..." }]`.
- **`.csv`**: expects columns `head, relation, tail` or `sentence`.
- **`.db`**: loads SQLite with the unified schema directly → no KG extraction needed.

---

## Phase 1: SQLiteGraphStore

### New file: `graph/graph_component_implementation/sqlite_graph_store.py`

Implements the **same `GraphStore` protocol** as `DictGraphStore`:

```python
class SQLiteGraphStore(GraphStore):
    def __init__(self, db_path: str):
        # Opens SQLite connection
        # Loads all data into in-memory dicts for fast query
        # Periodically syncs to SQLite for persistence
    
    def get_node(self, node_id) -> Node
    def get_neighbors(self, node_id) -> List[Tuple[int, Edge]]
    def get_subgraph_by_embedding_similarity(self, query_emb, top_k) -> Subgraph
    def get_label(self, node_id) -> str
    def get_embedding(self, node_id) -> np.ndarray
    def save_state(self, path)  # path = .db file
    def load_state(cls, path)   # path = .db file
```

### Design choice

Wraps SQLite with in-memory dicts for O(1) reads (same as DictGraphStore). The SQLite file is the **canonical persistence format** — it's what you pass between machines and across training sessions. The in-memory dicts are loaded from SQLite at init time and synced back on `save_state()`.

- Query speed is identical to DictGraphStore (same dict lookups)
- Persistence is SQLite (not JSON files)
- Any process, any language, any time can read the .db

### Fixed SQLite Schema

```sql
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    node_type TEXT DEFAULT 'concept',
    activation REAL DEFAULT 0.5,
    use_count INTEGER DEFAULT 0,
    create_time REAL,
    sense_id INTEGER
);

CREATE TABLE edges (
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    strength REAL DEFAULT 0.9,
    confidence REAL DEFAULT 0.8,
    PRIMARY KEY (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);

CREATE TABLE embeddings (
    node_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

This schema is **immutable across all datasets**. Any dataset, regardless of source, maps to it.

### Migration path

DictGraphStore continues to work. Both implementations coexist. KG builder can output to either. DictGraphStore is deprecated after all paths are migrated.

---

## Phase 2: Refactor KGBuilder to output .db

### Modified: `kg_builder/pipeline.py`

```python
def build_graph_store(self, graph_data, db_path: str = None):
    store = SQLiteGraphStore()
    store.add_dataset(
        concepts=graph_data["concepts"],
        edges=graph_data["edges"],
        id_to_label=graph_data["id_to_label"],
        embeddings=graph_data["embeddings"],
    )
    if db_path:
        store.save_state(db_path)
    return store

def process_and_store(self, texts, db_path: str = None, max_docs=None):
    result = self.process_corpus(texts, max_docs=max_docs)
    store = self.build_graph_store(result["graph_data"], db_path=db_path)
    return store  # store._db_path points to the .db file
```

### New: `scripts/ingest.py` — universal data ingestion entry point

```bash
python scripts/ingest.py --input dataset.parquet --output kg/dataset.db
python scripts/ingest.py --input dataset.json --output kg/dataset.db
python scripts/ingest.py --input dataset.csv --output kg/dataset.db
python scripts/ingest.py --input dataset.db --output kg/dataset.db  # already .db, just copies
```

Flow:
1. Detect format from file extension
2. Load via appropriate DataLoader → unified format
3. If has triples directly, extract → resolve → build graph
4. If has sentences, run full spaCy pipeline
5. Save to .db

---

## Phase 3: Single Unified Training Pipeline

### New file: `scripts/universal_train.py`

```bash
# First dataset:
python scripts/universal_train.py \
    --db kg/dataset.db \
    --checkpoint checkpoints/unified/ \
    --lr 0.001 \
    --epochs 50

# Second dataset (continues training, detects existing checkpoint):
python scripts/universal_train.py \
    --db kg/dataset2.db \
    --checkpoint checkpoints/unified/ \
    --lr 0.0005 \
    --epochs 30
```

### Flow:
1. Load graph from .db via `SQLiteGraphStore.load_state()`
2. Generate training data from graph edges using `create_training_data()` (exists in `scripts/train_intent_ffn.py`)
3. If checkpoint exists:
   - Load IntentFFN, T5, IntentBiasTable from checkpoint
   - Set lower LR (e.g., 0.0005 vs 0.001 for first-time training)
   - Append to `trained_on_datasets` list
4. Train IntentFFN (with replay from previous datasets)
5. Fine-tune T5 decoder
6. Update IntentBiasTable from walker feedback
7. Save updated checkpoint

### Phase 3.1: T5 Fine-Tuning

**New file:** `model_training/training/t5_finetuner.py`

- Builds `(input_text, target_text)` pairs from graph edges
- Input: `"{intent_name}: {node0} {relation0} {node1}"`
- Target: `"{node0} {relation0} {node1}."`
- Uses HuggingFace `Seq2SeqTrainer` with configurable T5 model
- Early stopping, loss tracking
- For continual training: lower LR, append to dataset list

### Phase 3.2: Replay Buffer for Continual Learning

**New file:** `model_training/training/replay_buffer.py`

```python
class ReplayBuffer:
    def __init__(self, capacity: int = 1000):
        self.samples: List[Tuple[np.ndarray, int]] = []  # (embedding, intent)
    
    def add_dataset(self, X: np.ndarray, y: np.ndarray):
        # Reservoir sample up to capacity
        ...
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        # Mix replay samples with current batch
        ...
```

During training on dataset N, each training batch is:
- 80% from current dataset
- 20% from replay buffer (sampled from datasets 1..N-1)

This prevents catastrophic forgetting. The buffer stores embeddings (384 floats each), not raw text, so 1000 samples = ~1.5MB.

---

## Phase 4: Model Checkpoint Format

### Single checkpoint directory: `checkpoints/unified/`

```
checkpoints/unified/
  intent_ffn.pt        # IntentFFN state_dict + config
  t5_decoder/          # T5 model directory (config.json + model.safetensors)
  intent_bias.npy      # IntentBiasTable weights
  manifest.json        # Training history
```

### `manifest.json` format

```json
{
    "model_version": "2.0",
    "trained_on_datasets": [
        {
            "name": "conceptnet_v1",
            "source": "conceptnet.parquet",
            "trained_at": "2026-05-18T12:00:00Z",
            "num_edges": 139,
            "intent_ffn_val_loss": 0.2509,
            "t5_val_loss": 0.1234
        },
        {
            "name": "wikidata_geography",
            "source": "geography.parquet",
            "trained_at": "2026-05-25T12:00:00Z",
            "num_edges": 500,
            "intent_ffn_val_loss": 0.2013,
            "t5_val_loss": 0.0987
        }
    ],
    "config": {
        "ffn_input_dim": 384,
        "ffn_hidden_dim": 128,
        "ffn_output_dim": 16,
        "t5_model_name": "t5-small"
    }
}
```

### Modified: `model_training/models/glm_x_model.py`

Add a `continue_training()` method:
```python
def continue_training(self, new_db_path: str, config: Dict):
    # 1. Load checkpoint
    # 2. Load new graph from .db
    # 3. Generate training data
    # 4. Lower LR, append to manifest
    # 5. Train IntentFFN + T5 + Walker
    # 6. Save updated checkpoint
```

---

## Phase 5: End-to-End Entry Points

### `python glm_x_train.py`

```bash
# First dataset (initial training):
python glm_x_train.py \
    --data conceptnet.parquet \
    --output checkpoints/unified/

# Second dataset (continues training, auto-detects checkpoint):
python glm_x_train.py \
    --data wikidata.json \
    --output checkpoints/unified/

# Third dataset (a year later — same command, same checkpoint dir):
python glm_x_train.py \
    --data custom.csv \
    --output checkpoints/unified/
```

### `python glm_x_ask.py`

```bash
python glm_x_ask.py \
    --db kg/my_dataset.db \
    --checkpoint checkpoints/unified/ \
    --question "What is Paris in?"
```

The `.db` provides the graph. The checkpoint provides the trained IntentFFN + T5 + IntentBias. They are independent — you can mix and match. This is the key modularity: the graph is the data, the checkpoint is the trained intelligence.

---

## Phase 6: GLM-X Pipeline Integration

### Modified: `scripts/glmx_ask.py`

```python
class GLMXPipeline:
    def __init__(self, db_path: str = None, checkpoint_dir: str = None):
        if db_path:
            self.graph_store = SQLiteGraphStore.load_state(db_path)
        if checkpoint_dir:
            self.load_checkpoint(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint_dir: str):
        # Load IntentFFN, T5, IntentBias from checkpoint
        # No graph needed — the .db provides the graph
    
    def ask(self, question: str) -> Dict:
        # Same as current — resonance → planner → walker → decoder
        # Graph comes from .db, model comes from checkpoint
```

The `ask()` method stays **identical** — it only uses the GraphStore protocol interface. Whether the GraphStore is backed by dicts or SQLite makes no difference to the algorithm.

---

---

## Metrics & Evaluation Framework

### Architecture

```
universal_train.py / glm_x_train.py
        │
        ▼
   MetricsTracker (singleton, scoped per training run)
        │
        ├── DataLoaderMetrics      (from DataLoader)
        ├── KGBuilderMetrics       (from KGBuilderPipeline)
        ├── GraphStoreMetrics      (from SQLiteGraphStore)
        ├── IntentFFNMetrics       (from TrainingRun + evaluate)
        ├── T5DecoderMetrics       (from T5FineTuner)
        ├── WalkerMetrics          (from GraphWalker + IntentBiasTable)
        ├── ResonanceMetrics       (from Tier1Resonance)
        ├── DecoderMetrics         (from TemplateDecoder + T5Decoder)
        └── QAEvaluationMetrics    (from evaluate_full_pipeline)
                │
                ▼
   ┌─────────────────────────────────────────────┐
   │ 1. metrics/<dataset>/<timestamp>/           │
   │    ├── unified_metrics_summary.json          │ ← all metrics merged
   │    ├── {component}_metrics.json              │ ← per-component
   │    ├── training_curves.png                   │ ← loss + accuracy plots
   │    └── confusion_matrix.png                  │ ← IntentFFN confusion
   │                                              │
   │ 2. checkpoints/unified/metrics_history.jsonl │ ← append-only log
   │ 3. checkpoints/unified/manifest.json         │ ← dataset manifest
   └─────────────────────────────────────────────┘
```

### New file: `model_training/training/metrics_tracker.py`

```python
class MetricsTracker:
    def __init__(self, run_id: str, checkpoint_dir: str):
        self.run_id = run_id              # "2026-05-18T12-00-00_dataset_conceptnet"
        self.checkpoint_dir = checkpoint_dir
        self.dataset_name: str = ""
        self.source_file: str = ""
        self.is_continual: bool = False
        self.previous_checkpoint_path: str = ""

        # Per-component metric containers
        self.data_loader: "DataLoaderMetrics" = {}
        self.kg_builder: "KGBuilderMetrics" = {}
        self.graph_store: "GraphStoreMetrics" = {}
        self.intent_ffn: "IntentFFNMetrics" = {}
        self.t5_decoder: "T5DecoderMetrics" = {}
        self.walker: "WalkerMetrics" = {}
        self.resonance: "ResonanceMetrics" = {}
        self.decoder: "DecoderMetrics" = {}
        self.qa_eval: "QAEvaluationMetrics" = {}
        self.continual: "ContinualLearningMetrics" = {}

    def set_component(self, name: str, data: dict) -> None
    def get_component(self, name: str) -> dict

    def save_all(self) -> str  # saves to metrics/<dataset>/<timestamp>/
    def append_to_history(self) -> None  # appends to metrics_history.jsonl
    def generate_summary(self) -> dict  # merges all into one dict
    def add_epoch_metric(self, component: str, epoch_data: dict) -> None
    def plot_training_curves(self, output_path: str) -> None
    def plot_confusion_matrix(self, matrix: List[List[int]], output_path: str) -> None
```

---

### Per-Component Metric Schemas

Every component produces a dict matching its schema below. Fields with `?` are optional (may not be available in all runs).

#### 1. DataLoaderMetrics

```json
{
    "source_file": "conceptnet.parquet",
    "format": "parquet",
    "records_loaded": 5000,
    "records_after_filter": 3241,
    "sentences_extracted": 0,
    "triples_extracted": 3241,
    "load_time_seconds": 2.34,
    "parse_error_count": 12,
    "language_distribution": { "en": 3020, "unknown": 221 },
    "filter_reason_counts": {
        "non_english": 1689,
        "duplicate": 52,
        "invalid_uri": 18
    }
}
```

#### 2. KGBuilderMetrics

```json
{
    "sentences_processed": 140,
    "spans_extracted": 486,
    "triples_raw": 187,
    "triples_kept": 139,
    "unique_entities_before_resolution": 260,
    "unique_entities_after_resolution": 244,
    "entity_resolution_merge_rate": 0.0615,
    "entity_resolution_embedding_merges": 14,
    "embedding_dedup_merges": 2,
    "avg_triple_coherence": 0.74,
    "relation_type_count": 13,
    "triple_coherence_distribution": {
        "0.6-0.7": 23,
        "0.7-0.8": 67,
        "0.8-0.9": 41,
        "0.9-1.0": 8
    },
    "extraction_speed_sentences_per_second": 18.5,
    "build_time_seconds": 47.5
}
```

#### 3. GraphStoreMetrics

```json
{
    "node_count": 244,
    "edge_count": 139,
    "relation_types": ["causes", "developed", "discovered", ...],
    "avg_edges_per_node": 1.14,
    "density": 0.0047,
    "save_time_seconds": 0.12,
    "load_time_seconds": 0.08,
    "db_file_size_bytes": 245760,
    "embedding_dim": 384,
    "has_all_embeddings": true,
    "nodes_without_embeddings": 0,
    "dedup_merge_count": 2,
    "metadata_entries": {
        "dataset_name": "conceptnet_v1",
        "source": "conceptnet.parquet",
        "build_time": "2026-05-18T12:00:00Z"
    }
}
```

#### 4. IntentFFNMetrics

Training session overview:
```json
{
    "dataset_name": "conceptnet_v1",
    "is_continual_training": false,
    "total_samples": 139,
    "train_samples": 111,
    "val_samples": 14,
    "test_samples": 14,
    "num_classes": 6,
    "class_distribution": [28, 42, 13, 10, 15, 31],
    "class_names": {
        "0": "define", "1": "assert_fact", "2": "explain_cause",
        "4": "contrast", "6": "list", "13": "elaborate"
    },
    "epochs_trained": 37,
    "best_epoch": 32,
    "best_val_loss": 0.2509,
    "best_val_top1_accuracy": 0.8571,
    "final_train_loss": 0.1823,
    "final_train_top1_accuracy": 0.9189,
    "training_time_seconds": 45.2,
    "optimizer": "adamw",
    "loss_function": "focal",
    "focal_gamma": 2.0,
    "learning_rate": 0.001,
    "batch_size": 32,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 15,
    "scheduler": "cosine",
    "warmup_epochs": 5
}
```

Test evaluation:
```json
{
    "test_top1_accuracy": 0.8571,
    "test_top3_accuracy": 0.9286,
    "test_top5_accuracy": 1.0,
    "test_loss": 0.3274,
    "per_class_metrics": {
        "define": {
            "precision": 0.9, "recall": 0.85, "f1": 0.874,
            "support": 28, "top1_accuracy": 0.85, "top3_accuracy": 0.95
        },
        "assert_fact": {
            "precision": 0.92, "recall": 0.88, "f1": 0.899,
            "support": 42, "top1_accuracy": 0.88, "top3_accuracy": 0.97
        }
    },
    "confusion_matrix": [[28,0,0,0,0,0],[0,37,2,0,3,0],[0,1,12,0,0,0],[0,0,0,10,0,0],[1,2,0,0,12,0],[0,0,0,0,0,31]],
    "worst_class": { "name": "explain_cause", "f1": 0.75, "support": 13 },
    "best_class": { "name": "assert_fact", "f1": 0.899, "support": 42 }
}
```

Per-epoch history (stored in separate array, not in main metrics dict):
```json
{
    "epochs": [
        { "epoch": 1, "train_loss": 2.834, "val_loss": 2.761, "train_top1": 0.18, "val_top1": 0.21, "train_top3": 0.45, "val_top3": 0.50, "learning_rate": 1e-5 },
        { "epoch": 5, "train_loss": 1.023, "val_loss": 0.987, "train_top1": 0.57, "val_top1": 0.64, "train_top3": 0.82, "val_top3": 0.79, "learning_rate": 5e-4 },
        { "epoch": 32, "train_loss": 0.182, "val_loss": 0.251, "train_top1": 0.92, "val_top1": 0.86, "train_top3": 0.99, "val_top3": 0.93, "learning_rate": 1e-4 }
    ]
}
```

#### 5. T5DecoderMetrics

```json
{
    "t5_model_name": "t5-small",
    "is_continual_training": false,
    "train_samples": 120,
    "val_samples": 15,
    "test_samples": 15,
    "epochs_trained": 25,
    "best_epoch": 22,
    "best_val_loss": 0.187,
    "training_time_seconds": 180.5,
    "test_bleu_score": 0.891,
    "test_exact_match_rate": 0.854,
    "test_rouge_l_f1": 0.923,
    "val_perplexity": 1.204,
    "test_perplexity": 1.187,
    "output_length_mean": 8.3,
    "output_length_std": 2.1,
    "output_length_distribution": { "3-5": 2, "6-8": 8, "9-12": 4, "13+": 1 },
    "fallback_rate": 0.032,
    "validation_failure_rate": 0.021,
    "learning_rate": 0.0003,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "optimizer": "adamw",
    "max_input_length": 512,
    "max_output_length": 128,
    "num_beams": 4
}
```

Per-epoch history (stored separately):
```json
{
    "epochs": [
        { "epoch": 1, "train_loss": 3.124, "val_loss": 2.987, "val_perplexity": 19.82, "learning_rate": 3e-5 },
        { "epoch": 10, "train_loss": 0.452, "val_loss": 0.412, "val_perplexity": 1.51, "learning_rate": 1.5e-4 },
        { "epoch": 22, "train_loss": 0.201, "val_loss": 0.187, "val_perplexity": 1.20, "learning_rate": 3e-5 }
    ]
}
```

#### 6. WalkerMetrics

```json
{
    "total_walks_attempted": 240,
    "walks_completed": 220,
    "walks_successful": 210,
    "walk_completion_rate": 0.917,
    "walk_success_rate": 0.875,
    "avg_path_length": 2.3,
    "avg_path_confidence": 0.81,
    "avg_final_activation": 0.76,
    "dead_end_rate": 0.083,
    "intent_bias_entropy": 2.34,
    "intent_bias_convergence_steps": 120,
    "intent_bias_table_summary": {
        "intent_0": { "synonym": 1.2, "antonym": 0.8, "related_to": 1.0 },
        "intent_1": { "synonym": 1.0, "antonym": 0.9, "related_to": 1.3 }
    },
    "total_reward_accumulated": 187.5,
    "avg_reward_per_walk": 0.78,
    "bias_update_count": 240,
    "walker_config": {
        "max_steps": 5,
        "temperature": 0.1,
        "restart_on_dead_end": false,
        "scoring_formula": "strength * confidence * target_activation * intent_bias"
    }
}
```

#### 7. ResonanceMetrics

```json
{
    "total_resonance_calls": 240,
    "avg_activation_energy": 1.87,
    "activation_energy_std": 0.54,
    "avg_convergence_iterations": 4.2,
    "convergence_iterations_std": 0.8,
    "avg_propagation_steps": 3.8,
    "avg_seed_to_target_activation_spread": 0.42,
    "nodes_activated_above_threshold_avg": 18.5,
    "nodes_activated_above_threshold_std": 6.2,
    "avg_resonance_time_ms": 12.4,
    "resonance_time_ms_std": 3.1,
    "decay_lambda_value": 0.1,
    "propagation_threshold_value": 0.008,
    "edge_threshold_value": 0.02,
    "propagation_type": "wilson_cowan",
    "normalization": "budget_soft_cap",
    "tier_used_distribution": { "1": 240, "2": 0 }
}
```

#### 8. TemplateDecoderMetrics

```json
{
    "total_decode_calls": 240,
    "template_matched_count": 210,
    "template_match_rate": 0.875,
    "t5_decoder_success_count": 15,
    "t5_decoder_fallback_rate": 0.063,
    "concatenation_fallback_count": 15,
    "total_fallback_rate": 0.125,
    "output_validity_rate": 0.958,
    "avg_output_length_tokens": 8.7,
    "avg_output_length_chars": 42.3,
    "template_coverage": {
        "[1]": 85,
        "[1, 1]": 45,
        "[0]": 32,
        "[2]": 28,
        "[4]": 20
    },
    "validation_failures": {
        "too_short": 6,
        "no_node_mention": 4,
        "repetitive_ngrams": 0
    },
    "config": {
        "min_output_length": 3,
        "max_output_length": 200,
        "require_node_mention": true,
        "sentence_starters": [],
        "use_intent_prefix": false
    }
}
```

#### 9. QAEvaluationMetrics

```json
{
    "evaluation_name": "conceptnet_v1_qa",
    "qa_correct": 21,
    "qa_total": 24,
    "qa_accuracy": 0.875,
    "avg_latency_seconds": 1.87,
    "latency_std_seconds": 0.54,
    "latency_breakdown_ms": {
        "encode": { "mean": 12.3, "std": 2.1 },
        "subgraph": { "mean": 18.7, "std": 3.4 },
        "resonance": { "mean": 45.2, "std": 8.9 },
        "plan": { "mean": 8.1, "std": 1.5 },
        "walk": { "mean": 35.6, "std": 7.8 },
        "decode": { "mean": 5.3, "std": 1.2 }
    },
    "answer_confidence_mean": 0.81,
    "answer_confidence_std": 0.12,
    "walk_confidence_mean": 0.78,
    "walk_confidence_std": 0.09,
    "per_question_results": [
        {
            "question": "What did Albert Einstein develop?",
            "answer": "albert einstein developed the general theory of relativity.",
            "expected_contains": "theory of relativity",
            "correct": true,
            "confidence": 0.83,
            "walk_confidence": 0.81,
            "latency_seconds": 1.45,
            "latency_breakdown": { "encode": 0.012, "subgraph": 0.019, "resonance": 0.045, "plan": 0.008, "walk": 0.036, "decode": 0.005 },
            "walk_path": ["albert einstein", "the general theory of relativity"],
            "walk_edges": ["developed"],
            "template_matched": true
        }
    ],
    "failure_analysis": {
        "wrong_edge_selection": 2,
        "same_relation_multiple_targets": 1,
        "reverse_direction": 0,
        "template_fallback": 3
    },
    "query_types_evaluated": ["definition", "geography", "causation", "part_whole", "origin", "opposite"],
    "accuracy_by_type": {
        "definition": { "correct": 5, "total": 6, "accuracy": 0.833 },
        "geography": { "correct": 5, "total": 6, "accuracy": 0.833 },
        "causation": { "correct": 2, "total": 2, "accuracy": 1.0 },
        "part_whole": { "correct": 2, "total": 2, "accuracy": 1.0 },
        "origin": { "correct": 1, "total": 2, "accuracy": 0.5 },
        "opposite": { "correct": 2, "total": 2, "accuracy": 1.0 }
    }
}
```

#### 10. ContinualLearningMetrics

Computed by comparing the current training run's metrics against the previous run.

```json
{
    "previous_dataset_name": "conceptnet_v1",
    "previous_checkpoint_path": "checkpoints/unified/",
    "forgetting_evaluation": {
        "previous_dataset_test_top1_before": 0.8571,
        "previous_dataset_test_top1_after": 0.8432,
        "catastrophic_forgetting_score": 0.0139,
        "catastrophic_forgetting_threshold_warning": false,
        "per_class_forgetting": {
            "define": { "f1_before": 0.874, "f1_after": 0.862, "drop": 0.012 },
            "assert_fact": { "f1_before": 0.899, "f1_after": 0.891, "drop": 0.008 },
            "explain_cause": { "f1_before": 0.750, "f1_after": 0.720, "drop": 0.030 }
        },
        "forgotten_classes": ["explain_cause"]
    },
    "transfer_evaluation": {
        "new_dataset_random_init_top1": 0.7120,
        "new_dataset_pretrained_top1": 0.8912,
        "transfer_score": 0.1792,
        "transfer_improvement_percent": 25.2
    },
    "replay_buffer": {
        "total_samples": 27,
        "capacity": 1000,
        "utilization_percent": 2.7,
        "datasets_in_buffer": ["conceptnet_v1"],
        "sampling_ratio": 0.2
    },
    "learning_rate_adjustment": {
        "previous_lr": 0.001,
        "current_lr": 0.0005,
        "lr_decay_factor": 0.5,
        "warmup_epochs": 3
    }
}
```

---

### Metrics Storage Format & Directory Layout

Each training run produces two output paths:

```
# 1. Per-run detailed metrics (immutable snapshot)
checkpoints/unified/
  metrics/
    {dataset_name}_{timestamp}/
      manifest.json                    ← dataset metadata + run config
      data_loader_metrics.json
      kg_builder_metrics.json
      graph_store_metrics.json
      intent_ffn_metrics.json          ← includes epoch history
      t5_decoder_metrics.json          ← includes epoch history
      walker_metrics.json
      resonance_metrics.json
      decoder_metrics.json
      qa_evaluation_metrics.json
      continual_learning_metrics.json  ← only for runs after the first
      unified_metrics_summary.json     ← all above merged into one
      training_curves.png              ← loss + accuracy over epochs
      confusion_matrix.png             ← IntentFFN confusion matrix

# 2. Append-only history (single file, grows over time)
checkpoints/unified/
  metrics_history.jsonl                ← one JSON line per run (key metrics only)
```

### `manifest.json` (per-run)

```json
{
    "run_id": "2026-05-18T12-00-00_conceptnet_v1",
    "dataset_name": "conceptnet_v1",
    "source_file": "conceptnet.parquet",
    "trained_at": "2026-05-18T12:00:00Z",
    "duration_seconds": 295.4,
    "is_continual_training": false,
    "previous_run_id": null,
    "model_version": "2.0",
    "git_commit": "a1b2c3d4e5f6",
    "config_snapshot_path": "checkpoints/unified/config.yaml"
}
```

### `metrics_history.jsonl` (append-only)

Each line is a JSON object with only the top-level KPIs for quick scanning:

```json
{"run_id":"2026-05-18T12-00-00_conceptnet_v1","dataset":"conceptnet_v1","source":"conceptnet.parquet","timestamp":"2026-05-18T12:00:00Z","nodes":244,"edges":139,"intent_ffn_test_top1":0.8571,"intent_ffn_test_loss":0.3274,"t5_test_bleu":0.891,"t5_test_exact_match":0.854,"walk_success_rate":0.875,"qa_accuracy":0.875,"qa_correct":21,"qa_total":24,"training_time_min":4.92,"epochs":37,"catastrophic_forgetting":null,"transfer_score":null}
```

```json
{"run_id":"2026-06-01T10-00-00_wikidata_geography","dataset":"wikidata_geography","source":"geography.parquet","timestamp":"2026-06-01T10:00:00Z","nodes":500,"edges":320,"intent_ffn_test_top1":0.8912,"intent_ffn_test_loss":0.2512,"t5_test_bleu":0.912,"t5_test_exact_match":0.878,"walk_success_rate":0.912,"qa_accuracy":0.912,"qa_correct":21,"qa_total":23,"training_time_min":12.8,"epochs":42,"catastrophic_forgetting":0.0139,"transfer_score":0.1792}
```

### `metrics_history.md` (auto-generated human-readable table)

```markdown
# GLM-X Training History

| Run | Dataset | Date | Nodes | Edges | FFN Acc | T5 BLEU | Walk SR | QA Acc | Δ Forgetting | Training Time |
|-----|---------|------|-------|-------|---------|---------|---------|--------|-------------|--------------|
| 1 | conceptnet_v1 | 2026-05-18 | 244 | 139 | 0.857 | 0.891 | 0.875 | 0.875 | — | 5m |
| 2 | wikidata_geography | 2026-06-01 | 500 | 320 | 0.891 | 0.912 | 0.912 | 0.912 | 0.014 | 13m |
| 3 | custom_biology | 2027-01-15 | 1200 | 890 | 0.903 | 0.925 | 0.934 | 0.914 | 0.009 | 34m |
```

### `unified_metrics_summary.json` (all metrics in one file)

This is the most important file for programmatic consumption. It merges all component metrics into a single JSON with a defined key hierarchy:

```json
{
    "run_id": "2026-05-18T12-00-00_conceptnet_v1",
    "dataset": { "name": "conceptnet_v1", "source": "conceptnet.parquet", ... },
    "graph": { "nodes": 244, "edges": 139, ... },
    "intent_ffn": { "test_top1": 0.857, "test_top3": 0.929, "per_class": {...}, "epochs": [...] },
    "t5": { "test_bleu": 0.891, "test_exact_match": 0.854, "epochs": [...] },
    "walker": { "success_rate": 0.875, "avg_path_length": 2.3, ... },
    "resonance": { "avg_energy": 1.87, "avg_iterations": 4.2, ... },
    "decoder": { "template_match_rate": 0.875, "fallback_rate": 0.125, ... },
    "qa": { "accuracy": 0.875, "correct": 21, "total": 24, "by_type": {...}, "per_question": [...] },
    "continual": { "forgetting_score": null, "transfer_score": null, ... },
    "system": { "duration_seconds": 295.4, "model_version": "2.0", "git_commit": "a1b2c3d4", ... }
}
```

---

### New files for metrics infrastructure

| File | Purpose |
|---|---|
| `model_training/training/metrics_tracker.py` | Central MetricsTracker class |
| `model_training/training/metrics_schemas.py` | Schema validation + defaults for every component |
| `model_training/training/plotting.py` | Training curves + confusion matrix plots |
| `model_training/training/metrics_history.md` | Auto-generated human-readable history table |

### Modified files for metrics integration

| File | Change |
|---|---|
| `scripts/universal_train.py` | Instantiate MetricsTracker, pass to each component, save on completion |
| `scripts/glmx_ask.py` | Record latency breakdown per question, return in QA result dict |
| `scripts/evaluate_full_pipeline.py` | Generate QAEvaluationMetrics, pass to MetricsTracker |
| `model_training/training/trainer.py` | Return epoch-level metrics for MetricsTracker ingestion |
| `model_training/training/t5_finetuner.py` | Return epoch-level + test metrics for MetricsTracker |
| `g2p/g2p_planner.py` | Record plan confidence distribution, pass to MetricsTracker |
| `walker/graph_walker.py` | Expose walk statistics (completion, path length, dead-ends) |

---

## Complete File Summary

### New files

| File | Purpose |
|---|---|
| `data_loader/__init__.py` | Package init |
| `data_loader/base.py` | Abstract DataLoader |
| `data_loader/parquet_loader.py` | .parquet → unified format |
| `data_loader/json_loader.py` | .json → unified format |
| `data_loader/csv_loader.py` | .csv → unified format |
| `data_loader/sqlite_loader.py` | .db → skip KG build |
| `data_loader/utils.py` | URI parsing, language filtering |
| `graph/sqlite_graph_store.py` | SQLite-backed GraphStore (protocol-compatible) |
| `scripts/ingest.py` | Universal data ingestion entry point |
| `scripts/universal_train.py` | Unified training entry point (continual learning) |
| `glm_x_train.py` | End-to-end train CLI (root level) |
| `glm_x_ask.py` | End-to-end inference CLI (root level) |
| `training/t5_finetuner.py` | T5 fine-tuning loop |
| `training/replay_buffer.py` | Continual learning replay buffer |

### Existing files to modify

| File | Change |
|---|---|
| `kg_builder/pipeline.py` | Output to .db via SQLiteGraphStore |
| `kg_builder/graph_builder.py` | Support SQLite store output path |
| `scripts/glmx_ask.py` | Accept .db path + checkpoint dir params |
| `scripts/evaluate_full_pipeline.py` | Use .db path for graph store |
| `scripts/train_intent_ffn.py` | Refactor into universal_train.py |
| `model_training/models/glm_x_model.py` | Add `continue_training()` |
| `model_training/models/intent_ffn.py` | Add replay buffer support |
| `g2p/g2p_planner.py` | Support loading from checkpoint dict |
| `decoder/t5_decoder.py` | Enable fine-tuned model output path |

---

## Implementation Order

```
Phase 0: DataLoaders (.parquet → unified format)
    ↓
Phase 1: SQLiteGraphStore (GraphStore protocol + fixed schema)
    ↓
Phase 2: ingest.py (universal data → .db pipeline)
    ↓
Phase 3: universal_train.py (unified training + T5 + replay buffer)
    ↓
Phase 4: Checkpoint format + manifest.json + continue_training()
    ↓
Phase 5: glm_x_train.py + glm_x_ask.py (CLI entry points)
    ↓
Phase 6: Wire everything into GLMXPipeline (glmx_ask.py refactor)
```
