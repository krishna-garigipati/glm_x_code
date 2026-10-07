## GLM-X OFFICIAL ARCHITECTURE & IMPLEMENTATION CONTRACT
## Version: 3.3.2 – COMPLETE PoC BINDING VERSION
## Status: MANDATORY FOR ALL PARALLEL DEVELOPMENT
<!--  -->
## This is the single source of truth.
## Every team and every developer MUST strictly follow this file.
## No deviations are allowed without explicit written agreement.
<!--  -->

system:
  name: "GLM-X"
  goal: "Proof of Concept of an offline, deterministic, non-LLM knowledge-graph QA system"
  version: "3.3.2-PoC"
  embedding_model: "BAAI/bge-small-en-v1.5"
  embedding_dim: 384
  generative_llm_at_inference: false
  learning_at_inference: false
  offline: true
  deterministic: true

## ------------------------------------------------------------
## 1. CENTRAL INVARIANT (HIGHEST LAW)
## ------------------------------------------------------------
central_invariant: |
  The relation_chain produced by the Planner is the contract.
  The graph is the evidence.
  The final answer must be true to both.

  relation_chain ≈ walked path ≈ generated sentence

  If these three do not agree, the system has failed.

## ------------------------------------------------------------
## 2. PROOF OF CONCEPT SUCCESS CRITERIA
## ------------------------------------------------------------
poc_success_criteria:
  mandatory:
    - Full pipeline (Encode → Spread → Plan → Walk → Decode) works without crashing
    - Answers are driven by relation_chain (NOT intent_sequence)
    - Every correct answer is traceable to real nodes and edges in the graph
    - Accuracy ≥ 90% on the frozen PoC evaluation set
    - Fully deterministic (same question → identical answer when learning is off)
    - System can refuse honestly when it does not know
    - No generative LLM is used at inference time
    - Direction of main relations (causes/caused_by, part_of, follows/precedes) is mostly correct

  strong_recommended:
    - Heuristic fallback triggers correctly on nonsense / unanswerable questions
    - Short multi-hop questions work at a basic level
    - Answers are reasonably natural and directionally correct
    - Median latency < 100 ms per question (after model load) on CPU

  one_sentence_claim: |
    GLM-X can answer questions over a knowledge graph by extracting a relation chain,
    walking real edges, and generating a correct template-based answer — fully offline,
    deterministically, and without any generative LLM at inference time — reaching ≥ 90%
    accuracy on a controlled test set.

## ------------------------------------------------------------
## 3. PIPELINE (FIXED ORDER)
## ------------------------------------------------------------
pipeline:
  - Encode
  - Spread          # Resonance
  - Plan            # Query Relation Extractor
  - Walk
  - Decode
  - Learn           # Offline / training only

## ------------------------------------------------------------
## 4. 16 CANONICAL RELATIONS (CLOSED SET)
## ------------------------------------------------------------
relations:
  closed_set: true
  count: 16
  list:
    - is_a
    - has_property
    - causes
    - caused_by
    - follows
    - precedes
    - contradicts
    - supports
    - associated_with
    - example_of
    - part_of
    - synonym
    - antonym
    - temporal_coincident
    - spatial_near
    - linguistic_maps

  inverse_pairs:
    causes: caused_by
    caused_by: causes
    precedes: follows
    follows: precedes
    part_of: has_part
    is_a: is_a

  rule: "No team may add, remove, or rename any relation without explicit agreement."

## ------------------------------------------------------------
## 5. GRAPH BUILDING RULES (DETAILED)
## ------------------------------------------------------------
graph_building:
  principle: |
    The graph is the memory of the system.
    For PoC we use small, clean, hand-crafted graphs only.
    Graph quality is part of system correctness.

  process:
    1: Choose a simple, clear domain
    2: Create unambiguous nodes
    3: Add edges using only the 16 canonical relations
    4: Set high strength and confidence (0.8 – 1.0) for PoC facts
    5: Explicitly support direction pairs and short multi-hop paths
    6: Version and freeze the graph

  node_rules:
    - Use clear and unambiguous labels
    - Avoid vague or overloaded names
    - Include both specific entities and broader categories when multi-hop is required

  edge_rules:
    - Only the 16 allowed relations may be used
    - Every important fact must exist as an explicit edge
    - Direction must be intentional
    - Support common multi-hop patterns (e.g. part_of → is_a)
    - Do not expect the system to invent missing knowledge

  size_strategy:
    phase_1: "Start with 40–60 nodes and reach high accuracy"
    phase_2: "Expand to cover all 16 relations"
    phase_3: "Add controlled direction and multi-hop cases"
    forbidden: "Do not evaluate PoC on large noisy graphs before the clean graph works"

## ------------------------------------------------------------
## 6. STAGE: ENCODE
## ------------------------------------------------------------
encode:
  input: "raw question text"
  model: "BAAI/bge-small-en-v1.5"
  output: "384-dimensional normalized embedding"
  allowed_usage:
    - Seed node selection
    - Optional light steering
  forbidden:
    - Driving topological spreading
    - Being the primary signal for relation decisions

## ------------------------------------------------------------
## 7. STAGE: SPREAD (RESONANCE)
## ------------------------------------------------------------
spread:
  name: "Resonance"
  purpose: "Select the relevant part of the graph for the current question"
  algorithm: "Wilson-Cowan style activation spreading"
  activation_range: [0.01, 1.0]
  seed_activation: 1.0
  update_style: "synchronous"

  tiers:
    tier1:
      top_k: 64
      max_iterations: 4
    tier2:
      top_k: 1024
      max_iterations: 8
      analogy_enabled: true
      condition: "Only if Tier 1 energy is low"

  important_rules:
    - Spreading is effectively undirected
    - Direction of the answer is NOT decided in this stage
    - Output is a Subgraph

## ------------------------------------------------------------
## 8. STAGE: PLAN (QUERY RELATION EXTRACTOR)
## ------------------------------------------------------------
plan:
  name: "Query Relation Extractor"
  purpose: "Decide what relationship the question is asking about"
  operative_signal: "relation_chain"
  dormant_signal: "intent_sequence"

  method:
    - Split question into clauses
    - Match against relation descriptor banks and cue phrases
    - Use literal matching + embedding similarity
    - Build ordered relation_chain

  parameters:
    similarity_threshold: 0.35
    max_chain_length: 3
    collapse_consecutive_repeats: true

  descriptor_banks:
    requirement: |
      Every relation must have a sufficiently rich set of cue phrases and descriptors.
      Minimum expectation: at least 5–8 strong descriptors per relation for PoC.
      Direction-sensitive relations (causes/caused_by, precedes/follows, part_of) 
      must have clearly distinct descriptors.

  fallback:
    enabled: true
    flag: "heuristic_fallback_used"
    default_chain: ["has_property"]
    preferred_trigger: |
      No strong relation cue words or phrases from any descriptor bank 
      are present in the question.
      (Do not rely only on similarity score threshold — scores of real and 
      nonsense questions overlap.)
    behaviour: |
      Set heuristic_fallback_used = True
      Use default_chain
      Disclose that the answer is a heuristic guess
      Still attempt a useful answer when possible

## ------------------------------------------------------------
## 9. STAGE: WALK (DETAILED BEHAVIOUR)
## ------------------------------------------------------------
walk:
  purpose: "Follow the relation_chain through the real graph and collect evidence"
  max_steps: 6
  allow_cycles: false
  allow_backtrack: false

  direction_model: |
    Walker is forward-only.
    Inverse relations are supported by pre-mirroring edges into the subgraph.

  mirroring:
    enabled: true
    pairs:
      - causes ↔ caused_by
      - precedes ↔ follows
      - part_of ↔ has_part
      - is_a is treated as reversible
    action: "Inject mirrored edges with inverse relation labels before walking begins"
    exclusive: true
    exclusive_note: |
      Mirroring is PERMITTED ONLY for the pairs listed above, and every mirrored
      edge MUST carry the inverse relation label. A relation that appears in no
      pair MUST NOT be mirrored, and its forward label MUST NOT be reused on a
      reversed edge. Reusing the forward label asserts a directed triple the graph
      never stored, which is false for directional relations (supports,
      example_of, has_property) and merely unstored for relations that happen to
      be symmetric (contradicts, spatial_near, temporal_coincident,
      linguistic_maps). Both are contract violations; the second is not excusable
      merely because the sentence happens to be true.
      Where a symmetric relation must be read from either endpoint, BOTH
      directions are stored as explicit edges in the graph, so the symmetry is
      auditable data rather than invisible inference.

  deviations:
    - id: is_a-forward-only
      clause: "mirroring.pairs -> is_a is treated as reversible"
      status: AGREED_NARROWER_INTERPRETATION
      agreed: 2026-10-02
      direction: narrower
      note: |
        is_a is implemented FORWARD-ONLY, which is narrower than the clause above
        and therefore cannot over-answer. ACCEPTED as a deliberate narrower
        interpretation (sign-off recorded 2026-10-02); it is not a pending item
        and does not block the staged evaluation.

        Reason: is_a's declared inverse label is is_a itself, so a mirrored copy
        is indistinguishable by label from a real forward edge. Measured on
        test_results/stage_a/toy_graph.db with mirroring enabled, answering
        "What is a tree?" places BOTH `tree is_a plant` (stored) and
        `tree is_a oak` (mirror of the stored `oak is_a tree`) in the walker
        subgraph. The stored edge won on score in that measurement, so the failure
        is latent and score-dependent rather than guaranteed -- but direction
        cannot be recovered from a relation label alone, so the collision cannot
        be closed without a schema change (an orientation marker on edges).

        Cost: zero measurable recall on Stage A or Stage B. Both suites reach
        100% with is_a forward-only.

  anchor_selection:
    rules:
      - Prefer the main entity mentioned in the question
      - Fall back to the highest-similarity seed node if necessary
      - If no reliable anchor can be found → trigger honesty gate

  candidate_selection:
    hard_filter: |
      Prefer (or keep only) edges whose relation matches the expected 
      relation from the chain or its mirror.
    primary_steering: "relation_bias table"

  scoring:
    final_formula: |
      score = strength × confidence × target_activation × relation_bias(expected, actual)
    dominant_signal: "relation_bias"
    cosine_similarity: "REMOVED from the main scoring formula"
    note: |
      Cosine similarity may be used only as a very weak optional tie-breaker 
      if explicitly justified. It must never overpower relation_bias.

  required_behaviour:
    - Must stay faithful to the relation_chain
    - Must prefer correct direction
    - Must not jump to semantically similar but relationally wrong nodes
    - Must return a real path that exists in the graph

  output_must_include:
    - path
    - path_edges
    - walk_confidence

## ------------------------------------------------------------
## 10. STAGE: DECODE
## ------------------------------------------------------------
decode:
  mode: "template"
  template_selection: "Based on relations actually walked (path_edges)"

  strict_rules:
    - Never assert a relation that was not walked
    - Direction must be correct
    - Prefer specific nodes over generic category nodes
    - Language should be clear and reasonably natural

  honesty_gates:
    - weak_anchor
    - wrong_relation_from_anchor
    - no_valid_path

  honesty_behaviour: |
    Produce a concise and clear refusal.
    Prefer "I don't know" or "I don't have that information" over inventing answers.

## ------------------------------------------------------------
## 11. STAGE: LEARN
## ------------------------------------------------------------
learn:
  active_only_in: "training / offline mode"
  inference_behaviour: "Completely frozen and deterministic"

## ------------------------------------------------------------
## 12. DETERMINISM REQUIREMENTS
## ------------------------------------------------------------
determinism:
  mandatory:
    - No random sampling at inference
    - Fixed temperature (or greedy decoding)
    - No dropout or stochastic behaviour
    - Same graph + same question → identical answer
    - Learning must be fully disabled during evaluation runs

## ------------------------------------------------------------
## 13. TRACE & EXPLAINABILITY REQUIREMENTS
## ------------------------------------------------------------
trace_requirements:
  every_answer_must_be_able_to_produce:
    - Original question
    - relation_chain
    - Selected anchor
    - Full walked path
    - path_edges
    - Final generated sentence
    - Whether heuristic_fallback was used
    - Whether an honesty gate was triggered

  purpose: "Required for debugging and for proving the central invariant during PoC"

## ------------------------------------------------------------
## 14. STANDARD FAILURE BEHAVIOUR
## ------------------------------------------------------------
failure_handling:
  when_plan_fails: "Use heuristic_fallback"
  when_no_valid_walk: "Trigger honesty gate → clear refusal"
  when_anchor_not_found: "Trigger honesty gate → clear refusal"
  when_template_fails: "Safe fallback or honest refusal"
  never: "Invent unsupported relations or nodes"

## ------------------------------------------------------------
## 15. MECHANISM PERFORMANCE TARGETS (MANDATORY FOR PoC)
## ------------------------------------------------------------
mechanism_performance_targets:

  overall:
    accuracy: "≥ 90%"
    determinism: "100%"
    crash_free: "100%"

  plan_query_relation_extractor:
    correct_relation_chain: "≥ 92–95%"
    fallback_trigger_accuracy: "≥ 90%"

  walk:
    direction_accuracy: "≥ 90%"
    chain_following_accuracy: "≥ 90%"          # path_edges match relation_chain
    hard_filter_effectiveness: "≥ 95%"
    correct_edge_ranked_first: "≥ 90%"         # on clean one-hop cases

  decode:
    template_correctness: "≥ 95%"              # sentence matches walked relations
    no_invented_relations: "≥ 98%"

  honesty_gates:
    correct_refusal_rate: "≥ 90%"

  central_invariant:
    hold_rate: "≥ 90%"                         # relation_chain ≈ path ≈ sentence

  question_category_targets:
    simple_one_hop: "≥ 95%"
    direction_pairs: "≥ 90%"
    short_multi_hop: "≥ 70–80%"
    honesty_out_of_graph: "≥ 90%"
    nonsense_fallback: "Fallback must trigger on ≥ 90% of cases"

## ------------------------------------------------------------
## 16. GRAPH DATA CREATION & TESTING STRATEGY (MANDATORY FOR PoC)
## ------------------------------------------------------------
graph_data_and_testing:

  principle: |
    For the Proof of Concept, use small, clean, hand-crafted graphs.
    Do not rely on large noisy data. Quality of graph data is part of system correctness.

  recommended_test_layers:
    - name: "Toy Graph"
      purpose: "Basic pipeline correctness"
      size: "30–80 nodes"
      goal: "Prove the full pipeline works end-to-end"

    - name: "Relation Coverage Graph"
      purpose: "Test all 16 relations"
      size: "~100–150 nodes"
      goal: "Prove every canonical relation can be used"

    - name: "Direction & Multi-hop Graph"
      purpose: "Test direction and short chains"
      size: "~100 nodes"
      goal: "Prove causes/caused_by, part_of, follows/precedes and simple multi-hop"

    - name: "Honesty & Fallback Graph"
      purpose: "Negative and edge cases"
      size: "Small"
      goal: "Prove honest refusal and heuristic_fallback"

    - name: "Frozen PoC Evaluation Set"
      purpose: "Final scoring"
      goal: "Reach ≥ 90% accuracy"

  graph_building_rules:
    - Use only the 16 canonical relations
    - Prefer clear, unambiguous facts
    - Keep strength and confidence high (0.8 – 1.0) for PoC edges
    - Start with a very small clean graph and make it work perfectly before growing
    - Version and freeze both the graph and the question set

  node_creation_guidelines:
    - Use clear entity and concept names
    - Avoid ambiguous labels
    - Include both specific entities and broader categories when needed for multi-hop

  edge_creation_guidelines:
    - Every important fact must be an explicit edge
    - Add direction pairs deliberately (e.g. causes and caused_by)
    - Support short multi-hop paths (e.g. fin → fish → animal)
    - Do not expect the system to invent missing edges

  question_set_rules:
    - For every important edge, write at least one clear question
    - Include direction pair questions
    - Include a few short multi-hop questions
    - Include out-of-graph and nonsense questions for honesty/fallback
    - Freeze the question set before final evaluation (pre-registered)

  recommended_poc_test_matrix:
    simple_one_hop:
      minimum_questions: 30
      target_pass_rate: "≥ 95%"
    direction_pairs:
      minimum_questions: 10
      target_pass_rate: "≥ 90%"
    short_multi_hop:
      minimum_questions: 8
      target_pass_rate: "≥ 70–80%"
    honesty_out_of_graph:
      minimum_questions: 8
      target_pass_rate: "≥ 90%"
    nonsense_fallback:
      minimum_questions: 5
      target: "Fallback must trigger"
    overall:
      target_pass_rate: "≥ 90%"

  evaluation_requirements:
    - Record relation_chain, walked path, path_edges, and final answer for every question
    - Verify the central invariant on every example
    - Measure determinism by running the same questions multiple times
    - Separate failures caused by missing graph data from failures caused by code

  practical_workflow:
    1: "Build a small clean graph (40–60 nodes)"
    2: "Write matching questions and freeze them"
    3: "Run full pipeline and debug until accuracy is high"
    4: "Gradually add more relations and multi-hop cases"
    5: "Only after the small graph works well, move to larger controlled graphs"
    6: "Final PoC score must be reported on a frozen evaluation set"

## ------------------------------------------------------------
## 17. FORBIDDEN ACTIONS
## ------------------------------------------------------------
forbidden:
  - Using IntentFFN or making intent_sequence the main driving signal
  - Adding generative LLM at inference time
  - Making inference non-deterministic
  - Putting strong cosine similarity back into Walker scoring
  - Allowing Decode to claim relations that were not walked
  - Ignoring or weakening the central invariant
  - Adding new relations without agreement
  - Running learning during normal inference
  - Testing PoC only on large noisy graphs without a clean controlled set

## ------------------------------------------------------------
## 18. PARALLEL DEVELOPMENT RULES
## ------------------------------------------------------------
parallel_development_rules:
  - This file is the single source of truth
  - All teams must implement against this contract
  - Any change that affects the central invariant or PoC criteria requires explicit approval
  - Each team is responsible for keeping its component compliant
  - Determinism and offline constraints are non-negotiable
  - Graph data quality is part of system correctness
  - All evaluation graphs and question sets used for PoC scoring must be frozen and versioned

## ------------------------------------------------------------
## 19. DEFINITION OF DONE (PoC)
## ------------------------------------------------------------
definition_of_done:
  - All mandatory PoC success criteria are met
  - Central invariant holds on the evaluation set
  - Heuristic fallback works with the specified trigger
  - Direction is mostly correct
  - System is fully deterministic
  - Clean controlled graphs + frozen question sets exist
  - Full traces can be produced for every answer
  - Mechanism performance targets are met
  - Implementation matches this contract exactly

## END OF MANDATORY CONTRACT – VERSION 3.3.2
