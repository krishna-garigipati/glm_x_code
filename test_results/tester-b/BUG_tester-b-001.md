tag:             P-blocker
component:       resonance/planner (integration: Tier1 top_k gate + glmx_ask walker embedding hand-off)
issue_id:        tester-b-001
date:            2026-09-18
dataset:         food_bio_medium (196 nodes / 181 edges / 9 relations)

command:         python scripts/glmx_ask.py --db test_results/tester-b/datasets/food_bio_medium.db -q "What property does snow have?"
input:           What property does snow have?

expected:        golden fbm09 (pre-registered): answer node "cold", relation_chain ["has_property"],
                 hops 1. snow -> cold is a real graph edge (has_property, strength 0.95).

actual:          Pipeline crash during the walk:
                 walker.exceptions.EmbeddingLookupError: No embedding provider configured

traceback:
  File "scripts/glmx_ask.py", line 867, in main
      result = pipeline.ask(args.question)
  File "scripts/glmx_ask.py", line 609, in ask
      walk = self.walker.walk(walker_sub, walker_plan)
  File "walker/graph_walker.py", line 169, in walk
      path_embeddings = [self._resolve_embedding(start_node, subgraph)]
  File "walker/graph_walker.py", line 437, in _resolve_embedding
      raise EmbeddingLookupError("No embedding provider configured")
  walker.exceptions.EmbeddingLookupError: No embedding provider configured

repro:           1) python scripts/glmx_ask.py --db test_results/tester-b/datasets/food_bio_medium.db
                    -q "What property does snow have?"   -> crash (above)
                 2) food_bio_medium_runner.py loop: fbm01-fbm08 OK, fbm09 crashes -> crash (above)
                 3) debug_embedding_crash.py shows the mechanism (see root cause).
                 Reproduced 2/2. Deterministic for this question.

root cause:
  Tier1Resonance._gate uses gate_type="top_k" with tier.top_k = 64 (resonance/tier1.py:310-312).
  Low-propagated boundary seeds are pruned out of the activation map. resonated.node_embeddings
  (resonance/tier1.py:162-170) is built ONLY from that pruned activation map -> partial set
  (64 of 71 nodes). glmx_ask step 5 (scripts/glmx_ask.py:551-558) trusts
  resonated.node_embeddings as-is whenever it is not None, so the walker's subgraph lacks the
  embedding for the start node. GraphWalker._resolve_embedding (walker/graph_walker.py:433-441)
  has no fallback provider (self._embedding_provider is None), so it raises instead of reading
  the graph store.
  Trigger for fbm09: "snow" is the target seed; resonance pruned it below top-64; glmx_ask boosts
  its activation to 1.0 only AFTER resonance (glmx_ask.py:541-542), so the walker selects it as
  start node -> embedding lookup fails.
  Verifiable data: target snow(195), ice(194), water(193), plankton(196), kind(166), wet(172),
  orchid(69) are seeds absent from the 64-entry node_embeddings even though present in
  resonated.nodes AND in the graph store (get_embedding/get_node both return valid embeddings).

proposed fix:    (for lead decision)
  A. glmx_ask.py: when building walker_sub.node_embeddings, merge the resolved embeddings for
     ALL resonated.nodes (fallback loop already exists but is skipped because
     resonated.node_embeddings is not None even when partial).,
  B. or provide an embedding provider to the walker for on-demand lookups, or
  C. resonance: never prune seed/target nodes' embeddings from node_embeddings (keep top_k
     for propagation but carry embeddings for all seeds).

impact on tester-b: Medium golden run (fbm01-fbm20) is BLOCKED at fbm09. 8/20 questions
  executed without reaching the reporter; results incomplete. Per Section 8.1: experiment
  frozen until lead acts.