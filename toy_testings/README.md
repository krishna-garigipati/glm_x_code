# GLM-X Toy Testings

This folder contains self-contained toy dataset and testing scripts for the GLM-X pipeline.
All tests run **offline** with **zero external dependencies** (no ConceptNet, no SentenceTransformer downloads).

## Structure

```
toy_testings/
├── toy_dataset.py          # Toy knowledge graph + graph store API tests
├── test_pipeline.py        # Full pipeline tests (resonance->extractor->walker->decoder)
├── run_all_tests.py        # Master test runner (runs all test suites)
├── README.md               # This file
├── pipeline_test_results.json  # Generated after pipeline tests
└── master_test_report.json     # Generated after master test run
```

## Toy Knowledge Graph

The toy dataset (`toy_dataset.py`) contains:
- **60+ concepts** across domains: temperature, size, emotions, animals, vehicles, elements, weather, seasons, logic, spaces
- **100+ edges** covering **ALL 16 canonical relations**:
  - `is_a`, `has_property`, `causes`, `caused_by`, `follows`, `precedes`
  - `contradicts`, `supports`, `associated_with`, `example_of`, `part_of`
  - `synonym`, `antonym`, `temporal_coincident`, `spatial_near`, `linguistic_maps`

Semantic embedding clusters are baked in (heat/cold, large/small, animal, vehicle, etc.)

## Test Queries

16 queries exercise each relation type + multi-hop:
- "What is the opposite of hot?" → antonym
- "What is a dog?" → is_a
- "What is fire?" → has_property
- "What causes lung cancer?" → caused_by
- "What does smoking cause?" → causes
- "What comes after summer?" → follows
- "What is part of a car?" → part_of
- "What is associated with water?" → associated_with
- "Give me an example of a bird" → example_of
- "What is the synonym of big?" → synonym
- "What happens at the same time as lightning?" → temporal_coincident
- "What is near the kitchen?" → spatial_near
- "What is another word for automobile?" → linguistic_maps
- "What contradicts a myth?" → contradicts
- "What supports a hypothesis?" → supports
- Multi-hop queries combining relations

## Running Tests

### Run all test suites (recommended)
```bash
cd updated_glmx/toy_testings
python run_all_tests.py
```

### Run individual test suites
```bash
# Graph store API tests (from graph/run_toy_dataset.py)
python -m graph.run_toy_dataset

# Graph unit tests
python -m graph.tests.test_unit

# Component unit tests (resonance, extractor, walker, decoder in isolation)
python test_pipeline.py --unit

# Full pipeline integration tests
python test_pipeline.py --pipeline

# Config validation
python ../scripts/validate_configs.py
```

### Using the toy dataset in your own tests
```python
from toy_testings.toy_dataset import build_dict_graph_store, TOY_QUERIES

graph = build_dict_graph_store()  # In-memory DictGraphStore
# ... use graph for testing ...
```

## Mock SentenceTransformer

`test_pipeline.py` includes a `MockSentenceTransformer` that:
- Produces deterministic 384-dim embeddings from text
- Encodes semantic signals in specific dimensions (heat, cold, animal, vehicle, cause, etc.)
- Caches embeddings for consistent results
- Requires **no downloads, no PyTorch, no sentence-transformers package**

## Expected Outputs

All tests should pass with:
- Graph store: 24 API operations verified
- Unit tests: 6 component tests passing
- Pipeline: 16/16 queries producing coherent answers
- Config validation: All 8 YAML configs valid

## Notes

- **No core files modified** - all tests use existing APIs
- **Fully offline** - no network access required
- **Deterministic** - fixed random seeds for reproducible results
- **Fast** - in-memory DictGraphStore, mock embeddings, ~30 seconds for full suite