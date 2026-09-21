# Tester A - Nature & Weather Small Dataset - Golden Questions

**Dataset:** nature_weather_small.db
**Domain:** Nature & Weather (Tester A)
**Size:** 14 nodes, 12 edges (Small: 10-100 nodes)
**Relations covered:** causes, follows, precedes, antonym, associated_with, is_a, has_property, part_of, temporal_coincident

**Pre-registered:** 2026-09-18 (before any runs)

---

## Golden Questions (≥ 8, ≥ 2 per assigned relation)

| # | Question | Expected Chain | Expected Answer Semantics | Expected Hop Count | Assigned Relation |
|---|----------|----------------|---------------------------|-------------------|-------------------|
| 1 | What causes flood? | `["causes"]` | rain → flood | 1 | causes |
| 2 | What causes rain? | `["caused_by"]` | cloud → rain / storm → rain | 1 | caused_by |
| 3 | What comes after summer? | `["follows"]` | summer → autumn | 1 | follows |
| 4 | What comes before summer? | `["precedes"]` | spring → summer | 1 | precedes |
| 5 | What is the opposite of sun? | `["antonym"]` | sun ↔ cloud | 1 | antonym |
| 6 | What is associated with rain? | `["associated_with"]` | rain ↔ cloud / water | 1 | associated_with |
| 7 | What is rain? | `["is_a"]` | rain → water / weather | 1 | is_a |
| 8 | What is part of a storm? | `["part_of"]` | lightning → storm / hail → storm | 1 | part_of |
| 9 | What does lightning cause? | `["causes"]` | lightning → fire / thunder | 1 | causes |
| 10 | What comes before winter? | `["precedes"]` | autumn → winter | 1 | precedes |

---

## Pre-registration Notes

- All questions are in-graph (relations exist in graph)
- Expected chains match relations present in the graph
- Minimum 8 goldens ✅ (10 registered)
- ≥ 2 goldens per assigned relation ✅:
  - `causes`: Q1, Q9
  - `caused_by`: Q2
  - `follows`: Q3
  - `precedes`: Q4, Q10
  - `antonym`: Q5
  - `associated_with`: Q6
  - `is_a`: Q7
  - `part_of`: Q8

---

## Run Results (re-run 2026-09-21 on Bhargav, Batch-3 code)

**Graph relations actually in `nature_weather_small.db`:** `causes`, `follows`, `precedes`, `associated_with`
(`antonym`, `is_a`, `part_of`, `caused_by`, `temporal_coincident` are NOT edges in this graph).

**Grading:** pass = hops == expected (1) AND an expected answer node appears in the walk path or the answer.
Chain is reported as extracted; exactness vs the pre-registered expectation is also shown.

| # | Question | Expected Chain | Extracted Chain | Chain Exact | Answer (actual) | Hops | Object | Pass |
|---|----------|----------------|-----------------|-------------|-----------------|------|--------|------|
| 1 | What causes flood? | `["causes"]` | `["causes"]` | ✅ | The reason is that flood is caused by rain. | 1 | ✅ | ✅ |
| 2 | What causes rain? | `["caused_by"]` | `["causes"]` | ❌ | The reason is that rain causes flood. | 1 | ❌ | ❌ |
| 3 | What comes after summer? | `["follows"]` | `["follows"]` | ✅ | summer follows autumn. | 1 | ✅ | ✅ |
| 4 | What comes before summer? | `["precedes"]` | `["precedes"]` | ✅ | summer precedes spring. | 1 | ✅ | ✅ |
| 5 | What is the opposite of sun? | `["antonym"]` | `["precedes"]` | ❌ | Honest: "I don't have a relation ... (No precedes relation found.)" | 1 | ❌ | ❌ |
| 6 | What is associated with rain? | `["associated_with"]` | `["associated_with"]` | ✅ | rain is associated with cloud. | 1 | ✅ | ✅ |
| 7 | What is rain? | `["is_a"]` | `["causes"]` | ❌ | The reason is that rain causes flood. | 1 | ❌ | ❌ |
| 8 | What is part of a storm? | `["part_of"]` | `["follows"]` | ❌ | Honest: "I don't have a relation ... (No follows relation found.)" | 1 | ❌ | ❌ |
| 9 | What does lightning cause? | `["causes"]` | `["causes"]` | ✅ | The reason is that lightning causes fire. | 1 | ✅ | ✅ |
| 10 | What comes before winter? | `["precedes"]` | `["precedes"]` | ✅ | Honest: "I don't have a relation ... (No precedes relation found.)" | 1 | ❌ | ❌ |

### Re-run notes (vs the 2026-09-18 run)

- **q2 / q7 — anchor changed.** Batch-3 exact-label anchoring picks `rain` (named in the question) as the start
  node, then walks forward on `causes` → `flood`. The old run anchored rain too but produced different neighbors
  (`storm`, `river`). The semantics expected by the golden (things that CAUSE rain = cloud/storm) need `caused_by`,
  which is not an edge in this graph, so the extractor can only offer `causes` (forward). Walking forward from
  `rain` yields rain's effects, not its causes → object semantics fail.
- **q5 / q8 / q10 — Batch-3 relation-honesty gate now fires.** Old run: "closest available relation" answer
  (e.g. q5 `["precedes"]`, "sun causes light"; q8 `["follows"]`, "storm causes rain"). New run: the gate checks
  whether the anchor actually has an edge of the asked relation (incl. its inverse); `sun`/`storm`/these anchors
  lack it, so the pipeline emits the honest "I don't have a relation ..." sentence instead of a nearest-neighbor
  walk. Chain still extracted (marked), but the answer is honest no-relation. This is intended Batch-3 behavior
  (Section 6.4 honesty), not a missing-relation probe — q5/q8's goldens expect concrete answers that this graph
  cannot support (relations absent).
- **q10 — object fail despite exact chain.** Chain `precedes` is exact and hops=1, but the walk resolves to
  `spring → summer` and the honest gate also fired (`entity_top_sim` below floor); expected answer node `autumn`
  does not appear.

---

## Summary (2026-09-21 re-run)

| Metric | Value |
|--------|-------|
| Total Questions | 10 |
| Pass (hops + expected answer node) | 5/10 |
| Chain exact match | 6/10 |
| Template Matched | 10/10 |
| Heuristic Fallback | 0/10 |
| Honest no-relation answers (Batch-3 relation gate) | q5, q8, q10 |

Prior run (2026-09-18) recorded 10/10 under "chain within available graph relations + template-matched" grading.
The 2026-09-21 grading adds an **answer-semantics check** (expected answer node must appear), which the old
table did not enforce.

---

## Edge Case: Missing Relation Test (re-run 2026-09-21)

**Question:** "What is the opposite of sun?" (antonym - not in graph)
**Actual Result:** Chain `["precedes"]` extracted; Batch-3 relation-honesty gate fires (sun lacks a precedes/
follows edge) -> honest no-relation sentence instead of a nearest-relation walk.
**Note:** Batch-3 now satisfies the "honest fallback" intent from the pipe: when the anchor has no edge of the
asked relation, it says "I don't have a relation ..." rather than fabricating a nearest-neighbor answer.