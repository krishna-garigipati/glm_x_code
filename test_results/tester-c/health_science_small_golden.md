# tester-c: Health & Science Small - Pre-registered Goldens

Dataset: `test_results/tester-c/datasets/health_science_small.db`
- 40 nodes, 23 edges, relations: causes, contradicts, linguistic_maps, part_of, spatial_near, supports

Assigned relations (Section 13.2, tester-C): `supports`, `contradicts`, `spatial_near`,
`linguistic_maps` + 2 more from tester list A/B -> `causes` (A), `part_of` (B).

Pre-registered: **2026-09-21** (before any runs). No golden edited after running.

## Golden Questions

| id   | question                          | relation        | hops_expected | answer (gold node) | expected chain    |
|------|-----------------------------------|-----------------|---------------|--------------------|-------------------|
| hsc01 | What supports health?             | supports        | 1             | exercise           | [supports]        |
| hsc02 | What is supported by sunlight?    | supports        | 1             | vitamin_d          | [supports]        |
| hsc03 | What contradicts health?          | contradicts     | 1             | smoking            | [contradicts]     |
| hsc04 | What does sugar contradict?       | contradicts     | 1             | healthy_teeth      | [contradicts]     |
| hsc05 | What is near the heart?           | spatial_near    | 1             | lungs              | [spatial_near]    |
| hsc06 | What is located near the liver?   | spatial_near    | 1             | stomach            | [spatial_near]    |
| hsc07 | What is the term for cure?        | linguistic_maps | 1             | treat              | [linguistic_maps] |
| hsc08 | What do you call a doctor?        | linguistic_maps | 1             | physician          | [linguistic_maps] |
| hsc09 | What leads to fever?              | causes          | 1             | virus              | [causes]          |
| hsc10 | What causes insomnia?             | causes          | 1             | stress             | [causes]          |
| hsc11 | What is the heart a part of?      | part_of         | 1             | circulatory_system | [part_of]         |
| hsc12 | What is the stomach a part of?    | part_of         | 1             | digestive_system   | [part_of]         |

## Coverage

- 12 goldens: 2 per assigned relation (6 relations -> Section 5.5 "min 2 per assigned relation").
- All chains are hops=1 forward walks (or inverted-mirror causal walks for hsc09/hsc10,
  e.g. virus --causes--> fever; question anchors fever and walks the caused_by mirror).

## Grading rule (same as tester-b/tester-a)

- PASS = hops == hops_expected AND expected gold node in walk path (object_ok).
- For "cause-source" goldens (hsc09/hsc10) expected answer = the source node.