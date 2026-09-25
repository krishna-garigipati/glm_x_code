# Pre-registered golden set — tester-c-real-graph

Dataset  : test_results/tester-c-real-graph/datasets/health_science_source.txt
Built by : kg_builder (KGBuilderPipeline: spaCy -> rule-based TripleExtractor
           -> zero-heuristic RelationMapper -> EntityResolver -> GraphBuilder
           -> SQLiteGraphStore), relations normalised to the 16 canonical keys.
Graph db : test_results/tester-c-real-graph/tester_c_real_graph.db
Status   : FROZEN — written before the evaluation runner executed. 24 goldens.

Pre-registration date: 2026-09-25

> Label note (verified before final run): graph_data reports 72 nodes / 49 edges, but
> writing to SQLite runs the store's add_dataset embedding merge (cos(~1.0)), which
> collapses singular/plural near-duplicates -> stored graph has 70 nodes / 49 edges.
> Specifically "the lungs" merges into "the lung" (edges_dump.tsv, written from the
> pre-store graph_data, still shows the pre-merge label). Golden answers therefore
> use the labels of the STORED graph that glmx_ask actually walks.

The corpus deliberately contains the exact phrasings the IS-09 / IS-10 / IS-11 fixes
target, so this real-graph test doubles as their regression gate:
  - "Spring follows winter." / "Autumn precedes winter."  (IS-09 NER whole-sentence collapse)
  - "Bonjour translates to hello." / "Ciao translates to goodbye."  (IS-09 bare-pobj rescue)
  - "The heart is part of the circulatory system." (and stomach/lung)  (IS-10 partitive object)
  - "Insulin is a type of hormone." / "Vitamin D is an example of a nutrient."  (IS-10)
  - "Fish include salmon and tuna." / "Salmon and tuna are fish."  (IS-11 conjunction expansion:
    a coordinated object/subject becomes one edge PER atomic conjunct, no junk node)

## Golden table

| id   | question                                | gold                 | chain               | hops | relation        | note |
|------|-----------------------------------------|----------------------|---------------------|------|-----------------|------|
| trc01 | What is a virus?                        | a microorganism      | is_a                | 1    | is_a            | direct |
| trc02 | What property does vitamin C have?      | antioxidant properties | has_property      | 1    | has_property    | direct |
| trc03 | What does smoking cause?                | lung cancer          | causes              | 1    | causes          | direct |
| trc04 | What is lung cancer caused by?          | smoking              | caused_by           | 1    | caused_by       | direct (verb-phrase edge) |
| trc05 | What follows spring?                    | winter               | follows             | 1    | follows         | IS-09 probe: date-span collapse would DROP this sentence |
| trc06 | What precedes autumn?                   | winter               | precedes            | 1    | precedes        | IS-09 probe: date-span collapse would DROP this sentence |
| trc07 | What does science contradict?           | superstition         | contradicts         | 1    | contradicts     | direct |
| trc08 | What does calcium support?              | bone health          | supports            | 1    | supports        | direct |
| trc09 | What is smoking associated with?        | lung cancer          | associated_with     | 1    | associated_with | direct |
| trc10 | What is an example of a hormone?        | insulin              | example_of          | 1    | example_of      | direct (category->instance edge) |
| trc11 | What is the heart a part of?            | the circulatory system | part_of           | 1    | part_of         | IS-10 probe: "is part of" phrasing |
| trc12 | What is a synonym of insomnia?          | sleeplessness        | synonym             | 1    | synonym         | direct |
| trc13 | What is the opposite of day?            | night                | antonym             | 1    | antonym         | direct |
| trc14 | What does winter coincide with?         | flu season           | temporal_coincident | 1    | temporal_coincident | direct |
| trc15 | What is near the heart?                 | the lung             | spatial_near        | 1    | spatial_near    | direct (stored-node label; "the lungs" merged into "the lung") |
| trc16 | What does bonjour translate to?         | hello                | linguistic_maps     | 1    | linguistic_maps | IS-09 probe: bare-pobj rescue |
| trc17 | What leads to tooth decay?              | sugar                | causes              | 1    | causes          | mirror: expected causes walked over caused_by (inverse-rel) |
| trc18 | What causes insomnia?                   | stress               | causes              | 1    | causes          | mirror: expected causes walked over caused_by (inverse-rel) |
| trc19 | What does ciao translate to?            | goodbye              | linguistic_maps     | 1    | linguistic_maps | IS-09 probe #2: bare-pobj rescue |
| trc20 | What is the stomach a part of?          | the digestive system | part_of             | 1    | part_of         | IS-10 probe #2 |
| trc21 | What is the lung a part of?             | the respiratory system | part_of          | 1    | part_of         | IS-10 probe #3 |
| trc22 | What is a salmon?                        | fish                 | is_a                | 1    | is_a            | IS-11 probe: subject coordination split ("Salmon and tuna are fish.") |
| trc23 | What is a tuna?                          | fish                 | is_a                | 1    | is_a            | IS-11 probe: second conjunct not silently dropped |
| trc24 | What is an example of a fish?            | salmon               | example_of          | 1    | example_of      | IS-11 probe: object coordination split ("Fish include salmon and tuna.") |

## Bug-confirmation (edges that MUST exist in edges_dump.tsv)

Pre-registered, checked against the built edges_dump.tsv as a non-question gate:

| IS | expected edge (tab: source \t relation \t target)            |
|----|--------------------------------------------------------------|
| IS-09 | spring\tfollows\twinter                                    |
| IS-09 | autumn\tprecedes\twinter                                   |
| IS-09 | bonjour\tlinguistic_maps\thello                            |
| IS-09 | ciao\tlinguistic_maps\tgoodbye                             |
| IS-10 | the heart\tpart_of\tthe circulatory system                 |
| IS-10 | the stomach\tpart_of\tthe digestive system                 |
| IS-10 | the lung\tpart_of\tthe respiratory system                  |
| IS-10 | insulin\texample_of\thormone                               |
| IS-10 | vitamin d\texample_of\ta nutrient                          |
| IS-11 | fish\texample_of\tsalmon                                    |
| IS-11 | fish\texample_of\ttuna                                      |
| IS-11 | salmon\tis_a\tfish                                          |
| IS-11 | tuna\tis_a\tfish                                            |

Whole-pipeline integrity gate: 47 corpus sentences must yield exactly 49 edges.
Four of those come from the two coordinated-entity sentences, which the IS-11 fix
splits into one edge per atomic conjunct ("Fish include salmon and tuna." -> two
edges; "Salmon and tuna are fish." -> two edges). No sentence is silently dropped
and unmapped raw relations must be empty.

## Scoring (same gates as tester-b-real-graph / tester-c)

- hops  : len(walk_path_edges) == expected
- object: gold node appears after the seed in the walked path, or in the answer text
- chain : extracted relation_chain == expected chain label (logged; non-blocking for pass/fail — PASS = hops AND object)