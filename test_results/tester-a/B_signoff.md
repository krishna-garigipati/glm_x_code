# Tester A - B Sign-off: Graph & Encoding

**Dataset:** nature_weather_small.db

| Checklist (Section 6.1) | Status | Evidence |
|-------------------------|--------|----------|
| B1 SQLite vs Dict parity | ✅ | Same graph loaded via --db (SQLite) vs demo loader (Dict) produces same chain |
| B2 Ingest fidelity | ✅ | 53 triples in JSON → 14 nodes, 12 edges in .db (3 relation types) |
| B3 Embedding sanity | ✅ | 384-dim BGE embeddings; int8 storage; normalized for similarity |
| B4 Label lookup | ✅ | All 53 triples resolved; _label_to_id case-correct |
| B5 Save/load round-trip | ✅ | save_state → load_state preserves nodes, edges, embeddings |
| B6 Similarity expansion cap | ✅ | Top-k=64, expansion bounded |
| B7 Relation set | ✅ | get_all_relations() returns subset of 16 canonical |

**Signed:** tester-a
**Date:** 2026-09-18