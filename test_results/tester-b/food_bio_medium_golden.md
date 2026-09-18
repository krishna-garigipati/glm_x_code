# food_bio_medium — tester-b pre-registered golden questions

Domain: Food & Biology (tester-b, Section 13.2)
Size: Medium (100-1,000 nodes) — designed at 194 nodes
Relations in graph (9): is_a, part_of, has_property, example_of, synonym, antonym,
causes, caused_by, associated_with
Absent relations (7): follows, precedes, contradicts, supports, spatial_near,
temporal_coincident, linguistic_maps
Edge cases (Section 13.3, tester-b): fbm19 = missing relation (honest fallback),
fbm20 = duplicate labels (`corn` / `Corn`, no silent merge).

PRE-REGISTRATION: this file was written BEFORE the first run of food_bio_medium.db.
Per Section 3.1 / 13.5, the list is FROZEN after the run begins (≥8 goldens, ≥2 per
assigned relation).

## Assigned relations (is_a, part_of, has_property, example_of, synonym, antonym)

| id    | question | expected answer node | expected chain | expected hops |
|-------|----------|----------------------|----------------|---------------|
| fbm01 | What is a mammal? | animal | [is_a] | 1 |
| fbm02 | What is a penguin? | bird | [is_a] | 1 |
| fbm03 | What is a salmon? | fish | [is_a] | 1 |
| fbm04 | What is the yolk a part of? | egg | [part_of] | 1 |
| fbm05 | What is the petal a part of? | flower | [part_of] | 1 |
| fbm06 | What is the wing a part of? | bird | [part_of] | 1 |
| fbm07 | What property does honey have? | sweet | [has_property] | 1 |
| fbm08 | What property does chili have? | spicy | [has_property] | 1 |
| fbm09 | What property does snow have? | cold | [has_property] | 1 |
| fbm10 | Give me an example of a bird | robin | [example_of] | 1 |
| fbm11 | Give me an example of a fish | salmon | [example_of] | 1 |
| fbm12 | What is another word for big? | large | [synonym] | 1 |
| fbm13 | What is another word for quick? | fast | [synonym] | 1 |
| fbm14 | What is the opposite of dark? | light | [antonym] | 1 |
| fbm15 | What is the opposite of wet? | dry | [antonym] | 1 |

## Additional in-graph relations (Section 5.4 bands + coverage)

| id    | question | expected answer node | expected chain | expected hops |
|-------|----------|----------------------|----------------|---------------|
| fbm16 | What leads to tooth decay? | sugar | [causes] | 1 |
| fbm17 | What is tooth decay caused by? | sugar | [caused_by] | 1 |
| fbm18 | Tell me something associated with water | rain | [associated_with] | 1 |

## Edge cases (Section 13.3 tester-b)

| id    | question | expected result | expected chain | expected hops |
|-------|----------|-----------------|----------------|---------------|
| fbm19 | What organism lives close to plankton? | EXACT Section 6.4 honest-fallback sentence; never a fabricated chain | [has_property] (default fallback) | 0 (isolated node `plankton`, no edges) |
| fbm20 | What is corn? | grain | [is_a] | 1 |

Notes:
- fbm19 design-time verification (measured with BAAI/bge-small-en-v1.5 before this run):
  clause "organism lives close to plankton" best-matches `spatial_near` @ 0.670 (absent
  from graph) over `part_of` @ 0.628 (present) -> extractor graph-filter drops it ->
  default-chain fallback. `plankton` is an isolated node, so the walk is empty and the
  honest no-relation sentence is the ONLY valid output. This avoids the fbs14 failure
  mode (a confident match to a PRESENT relation such as `causes` at 0.83).
- fbm20: `corn` and `Corn` are distinct nodes, both `is_a -> grain`; case-insensitive
  collision must NOT merge them (B4 label lookup case-correct, no silent merge).
- fbm03/fbm11 intentionally mirror the proven small-dataset probes (salmon->fish).
- fbm16 mirrors fbs13 (sugar->tooth decay, causes) for cross-dataset stability.
- See food_bio_small_golden.md for the small-dataset pre-registration (fbs01-fbs15).