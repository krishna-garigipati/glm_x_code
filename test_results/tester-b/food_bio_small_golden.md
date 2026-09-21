# food_bio_small — Pre-registered Golden Questions (Tester-B)

Pre-registered **before** any own-domain run, per EXPERIMENTATION.md Section 3.1 / 13.5.
Anti-bias: this list is fixed once committed; it may not be edited after seeing results.

Dataset: `food_bio_small.db` (Food & Biology, Small 10–100 nodes)
Assigned relations (Section 13.2): `is_a`, `part_of`, `has_property`, `example_of`, `synonym`, `antonym`
Extra relations added on purpose (Section 5.4 / 13.2): `causes`
Edge cases (Section 13.3, tester-B): missing relation (honest fallback), duplicate labels

Golden rule: min 8 goldens/dataset, ≥2 per assigned relation.

## In-graph goldens (12)

| id | question | expected answer node | expected chain | expected hops |
|----|----------|----------------------|----------------|---------------|
| fbs01 | What is a salmon? | fish | [is_a] | 1 |
| fbs02 | What is a robin? | bird | [is_a] | 1 |
| fbs03 | What is the yolk a part of? | egg | [part_of] | 1 |
| fbs04 | What is the petal a part of? | flower | [part_of] | 1 |
| fbs05 | What property does honey have? | sweet | [has_property] | 1 |
| fbs06 | What property does lemon have? | sour | [has_property] | 1 |
| fbs07 | Give me an example of a bird | robin | [example_of] | 1 |
| fbs08 | Give me an example of a fish | salmon | [example_of] | 1 |
| fbs09 | What is another word for happy? | glad | [synonym] | 1 |
| fbs10 | What is another word for small? | little | [synonym] | 1 |
| fbs11 | What is the opposite of hot? | cold | [antonym] | 1 |
| fbs12 | What is the opposite of sweet? | sour | [antonym] | 1 |

## Out-of-list causal golden (1) — tests the intentionally-added `causes` relation (Section 5.4)

| id | question | expected answer node | expected chain | expected hops |
|----|----------|----------------------|----------------|---------------|
| fbs13 | What leads to tooth decay? | sugar | [causes] | 1 |

## Edge-case goldens (2) — tester-B assigned edge cases (Section 13.3)

| id | question | expected answer node | expected chain | expected hops | edge case |
|----|----------|----------------------|----------------|---------------|-----------|
| fbs14 | What follows photosynthesis? | (honest fallback sentence, Section 6.4) | [has_property] fallback | 0 | missing relation — must emit the exact configured fallback, never a fabricated chain |
| fbs15 | What is rice? | grain | [is_a] | 1 | duplicate labels — graph holds case-colliding `rice`/`Rice` as distinct nodes; correct `_label_to_id`, no silent merge corruption |

## B4 label-lookup accompaniment (not a walk golden)
- `_label_to_id` must resolve every label used by edges (case-correct, incl. `rice` and `Rice`) with no KeyError (B-cell matrix, B4).

Expected hops follow P2 chain-aware stop: single-relation chains walk exactly 1 hop.
Expected chain for fbs13: the walker bias table (config_walker.yaml causes: causes 1.8) prefers `causes`.