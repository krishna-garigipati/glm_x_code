# Pre-registered golden set — tester-b-real-graph

Dataset  : test_results/tester-b-real-graph/datasets/food_biology_source.txt
Built by : kg_builder (KGBuilderPipeline: spaCy -> rule-based TripleExtractor
           -> zero-heuristic RelationMapper -> EntityResolver -> GraphBuilder
           -> SQLiteGraphStore), relations normalised to the 16 canonical keys.
Graph db : test_results/tester-b-real-graph/tester_b_real_graph.db
Status   : FROZEN — written before the evaluation runner executed. 18 goldens.

Pre-registration date: 2026-09-21

## Golden table

| id   | question                                | gold                | chain               | hops | relation        | note |
|------|-----------------------------------------|---------------------|---------------------|------|-----------------|------|
| trg01 | What is a salmon?                       | a fish              | is_a                | 1    | is_a            | direct |
| trg02 | What property does gold have?           | high density        | has_property        | 1    | has_property    | direct |
| trg03 | What does sugar cause?                  | tooth decay         | causes              | 1    | causes          | direct |
| trg04 | What is lung cancer caused by?          | smoking             | caused_by           | 1    | caused_by       | direct (verb-phrase edge) |
| trg05 | What follows night?                     | day                 | follows             | 1    | follows         | direct |
| trg06 | What precedes summer?                   | spring              | precedes            | 1    | precedes        | direct |
| trg07 | What does science contradict?           | superstition        | contradicts         | 1    | contradicts     | direct |
| trg08 | What does calcium support?              | bone health         | supports            | 1    | supports        | direct |
| trg09 | What is smoking associated with?        | lung cancer         | associated_with     | 1    | associated_with | direct |
| trg10 | What is an example of a fish?           | salmon              | example_of          | 1    | example_of      | direct (category->instance edge) |
| trg11 | What is part of the circulatory system? | the heart           | part_of             | 1    | part_of         | direct (consists-of edge) |
| trg12 | What is a synonym of insomnia?          | sleeplessness       | synonym             | 1    | synonym         | direct |
| trg13 | What is the opposite of day?            | night               | antonym             | 1    | antonym         | direct |
| trg14 | What does night coincide with?          | dusk                | temporal_coincident | 1    | temporal_coincident | direct |
| trg15 | What is near the heart?                 | the lungs           | spatial_near        | 1    | spatial_near    | direct |
| trg16 | What does casa translate to?            | house               | linguistic_maps     | 1    | linguistic_maps | direct |
| trg17 | What leads to tooth decay?              | sugar               | causes              | 1    | causes          | mirror: expected causes walked over caused_by (inverse-rel) |
| trg18 | What causes lung cancer?                | smoking             | causes              | 1    | causes          | mirror: expected causes walked over caused_by (inverse-rel) |

## Scoring (same gates as tester-c / tester-b)

- hops  : len(walk_path_edges) == expected
- object: gold node appears after the seed in the walked path, or in the answer text
- chain : extracted relation_chain == expected chain label (logged; non-blocking for pass/fail — PASS = hops AND object)