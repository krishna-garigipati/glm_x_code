# CNN/DailyMail Dataset Training Plan

## Overview
Train GLM-X on CNN/DailyMail news articles (287K articles) to answer factual questions about news events, entities, and relationships.

## Dataset
- **Source**: CNN/DailyMail via HuggingFace `datasets` (`cnn_dailymail`, 3.0.0)
- **Size**: 287,113 articles, ~1.2 GB JSONL
- **Structure**: `{"id", "article", "highlights"}` — full news article text with summary highlights
- **Location**: `model_training/dataset_cnn/cnn_dailymail_train.json`

## Pipeline

### Phase 1: Knowledge Graph Construction
1. **Sample** N articles from the JSONL file
2. Feed through `KGBuilderPipeline.process_corpus()`:
   - spaCy NER (PERSON, ORG, GPE, LOC, etc.)
   - Noun phrase extraction
   - Triple extraction (subject-relation-object)
   - Entity resolution via SBERT embedding similarity
   - Confidence scoring
3. Save as SQLite `.db` via `build_graph_store(store_type="sqlite")`

**Output**: `model_training/dataset_cnn/cnn_dailymail_dataset_data.db`

### Phase 2: IntentFFN Training
1. Load `.db` with `SQLiteGraphStore.load_state()`
2. Run `scripts/universal_train.py`:
   - Online gradient-based absorption via `OnlineLearner`
   - Adaptive convergence detection
   - QA evaluation against held-out queries
3. Save checkpoint: `checkpoints/cnn/intent_ffn.pt`

### Phase 3 (Optional): T5 Decoder Fine-tuning
1. Prepare article → highlights pairs
2. Fine-tune T5-small via `T5FineTuner`
3. Save to `checkpoints/cnn/t5_decoder/`

### Phase 4: Evaluation
- QA accuracy on sampled test queries
- Walk statistics, plan confidence, template match rate
- Confusion matrix for intent classification

## Sampling Strategy

| Sample Size | Articles | Est. Nodes | Est. Edges | Build Time (CPU) |
|-------------|----------|------------|------------|------------------|
| 1,000 | 1,000 | 5K-10K | 10K-30K | ~10-20 min |
| 5,000 | 5,000 | 25K-50K | 50K-150K | ~1-2 hrs |
| 10,000 | 10,000 | 50K-100K | 100K-300K | ~2-4 hrs |
| Full | 287,113 | 500K-2M | 1M-5M | ~24-48 hrs |

## Compute Requirements

| Component | Hardware | Time |
|-----------|----------|------|
| KG Builder (1K) | CPU, 8 GB RAM | 10-20 min |
| KG Builder (5K) | CPU, 16 GB RAM | 1-2 hrs |
| IntentFFN training | CPU or GPU (any) | 2-5 min |
| T5 fine-tuning (small) | 4-6 GB VRAM | 2-4 hrs |
| QA evaluation | CPU | 5-10 min |

## Directory Structure
```
model_training/dataset_cnn/
  cnn_dailymail_train.json       # Full dataset (287K articles)
  articles.jsonl                  # Articles with highlights
  corpus.txt                     # Raw article text only
  cnn_dailymail_dataset_data.db  # Built KG (SQLite)
  dataset_info.json              # Metadata

checkpoints/cnn/
  intent_ffn.pt                  # Trained IntentFFN
  manifest.json                  # Training run metadata
  t5_decoder/                    # Fine-tuned T5 (optional)
```

## Quality Checks
- [ ] Node count vs expected entities
- [ ] Edge count vs expected triples
- [ ] Relation type diversity
- [ ] Embedding coverage (% nodes with embeddings)
- [ ] Sample walk: pick a node, traverse neighbors
- [ ] Subgraph retrieval by embedding similarity
