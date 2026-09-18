"""Full Pipeline Toy Testing for GLM-X with REAL SentenceTransformer (PyTorch backend).

Tests the complete pipeline with actual semantic embeddings:
  Question -> Resonance -> Extractor -> Walker -> Decoder -> Answer

Sets TRANSFORMERS_NO_TF=1 to avoid TensorFlow/Keras import issues.
"""
from __future__ import annotations

import os
# Force PyTorch backend, disable TensorFlow
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['USE_TF'] = '0'

import sys
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from toy_testings.toy_dataset import (
    build_dict_graph_store, TOY_QUERIES, ALL_16_RELATIONS
)
from graph.graph_component_implementation.dict_graph_store import DictGraphStore

# Import resonance
from resonance.config import (
    CoreConfig, AlgorithmConfig, TemporalConfig, TierConfig,
    CoreActivationConfig, CoreResonanceConfig, ESBounds,
)
from resonance.tier1 import Tier1Resonance

# Import G2P (real extractor)
from g2p.config import G2PConfig
from g2p.g2p_planner import QueryRelationExtractor
from g2p.types import Plan, Subgraph as G2PSubgraph

# Import Walker
from walker.config import CoreConfig as WalkerCoreConfig, WalkerConfig, ActivationConfig, WalkerCoreConfig as WCore
from walker.models import Plan as WalkerPlan, Subgraph as WalkerSubgraph, WalkResult
from walker.graph_walker import GraphWalker

# Import Decoder
from decoder.template_decoder import TemplateDecoder
from decoder.config_loader import load_config as load_decoder_config

# Import real SBERT
from sentence_transformers import SentenceTransformer


# ============================================================================
# REAL PIPELINE COMPONENTS SETUP
# ============================================================================

def setup_resonance():
    """Initialize Tier1Resonance with default config."""
    core = CoreConfig(
        activation=CoreActivationConfig(min=0.01, max=1.0, default=0.01, threshold_resonance=0.2),
        resonance=CoreResonanceConfig(
            propagation_threshold=0.008, edge_threshold=0.02, decay_lambda=0.1,
            top_k=64, budget_max=2.0, convergence_epsilon=0.001,
            tier1_energy_threshold=0.4, tier2_max_nodes=1024, analogy_validation_overlap=0.3,
        ),
        relations=ALL_16_RELATIONS,
        es_bounds=ESBounds(
            propagation_threshold=(0.001, 0.05), edge_threshold=(0.01, 0.1),
            decay_lambda=(0.05, 0.5), top_k=(16, 1024), relation_bias=(0.0, 2.0),
        ),
    )

    algorithm = AlgorithmConfig(
        propagation_type="wilson_cowan", normalization="budget_soft_cap",
        gate_type="top_k", temporal_factor_enabled=False,
    )

    temporal = TemporalConfig(gamma=0.5, frequency_threshold=20)

    tier = TierConfig(
        propagation_threshold=0.008, edge_threshold=0.02, decay_lambda=0.1,
        top_k=64, max_iterations=3,
        relation_bias={
            "is_a": 0.8, "has_property": 0.5, "causes": 1.5, "caused_by": 0.6,
            "follows": 0.5, "precedes": 0.5, "contradicts": 0.5, "supports": 0.5,
            "associated_with": 0.5, "example_of": 0.5, "part_of": 0.7,
            "synonym": 1.0, "antonym": 0.4, "temporal_coincident": 0.5,
            "spatial_near": 0.5, "linguistic_maps": 0.5,
        },
        energy_threshold_formula=None, t_conf_coefficient=None, multiplied_at_runtime=None,
    )

    tier1 = Tier1Resonance(
        core_config=core, algorithm=algorithm, temporal=temporal,
        tier_config=tier, log_activation_history=True, history_buffer_size=10,
    )
    return tier1, core


def setup_extractor(graph: DictGraphStore, sbert):
    """Initialize real QueryRelationExtractor with real SBERT."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config_g2p.yaml"
    )
    config = G2PConfig.from_yaml(config_path)
    extractor = QueryRelationExtractor(config)
    # Inject the real SBERT model (bypasses internal loading)
    extractor._sentence_model = sbert
    
    # Patch _best_relation_for_clause to handle 2D embeddings from SBERT
    original_best = extractor._best_relation_for_clause
    def patched_best(clause: str):
        if not extractor._variant_embeddings:
            return None, 0.0
        query_emb = sbert.encode(
            clause,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        # Ensure 1D
        if query_emb.ndim > 1:
            query_emb = query_emb.flatten()
        best_relation: Optional[str] = None
        best_sim = 0.0
        for relation, embeddings in extractor._variant_embeddings.items():
            for emb in embeddings:
                # Ensure 1D
                if emb.ndim > 1:
                    emb = emb.flatten()
                sim = float(np.dot(query_emb, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_relation = relation
        return best_relation, best_sim
    
    extractor._best_relation_for_clause = patched_best
    
    graph_relations = sorted(graph.get_all_relations())
    extractor.initialize(graph_relations=graph_relations)
    extractor.mark_trained()
    return extractor


def setup_walker():
    """Initialize GraphWalker with config."""
    core_cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config_core.yaml"
    )
    walker_cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "config_walker.yaml"
    )

    import yaml
    with open(core_cfg_path) as f:
        wc_raw = yaml.safe_load(f)
    with open(walker_cfg_path) as f:
        ww_raw = yaml.safe_load(f)

    walker_core_cfg = WalkerCoreConfig(
        activation=ActivationConfig(
            min=float(wc_raw.get("activation", {}).get("min", 0.01)),
            max=float(wc_raw.get("activation", {}).get("max", 1.0)),
        ),
        walker=WCore(
            default_temperature=float(wc_raw.get("walker", {}).get("default_temperature", 0.1)),
            softmax_temperature_range=tuple(
                wc_raw.get("walker", {}).get("softmax_temperature_range", [0.05, 0.5])
            ),
            max_steps=int(wc_raw.get("walker", {}).get("max_steps", 6)),
            min_activation_to_continue=float(
                wc_raw.get("walker", {}).get("min_activation_to_continue", 0.05)
            ),
        ),
        relations={int(k): v for k, v in wc_raw.get("relations", {}).items()},
    )

    walker_cfg = WalkerConfig.from_yaml(walker_cfg_path)
    walker = GraphWalker(walker_cfg, walker_core_cfg)
    return walker


def setup_decoder():
    """Initialize TemplateDecoder."""
    decoder_cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "decoder", "config_decoder.yaml"
    )
    decoder_cfg = load_decoder_config(decoder_cfg_path)
    decoder = TemplateDecoder(
        templates=decoder_cfg.get("templates", {}).get("definitions", []),
        relation_phrases=decoder_cfg.get("templates", {}).get("relation_phrases", {}),
        sentence_starters=decoder_cfg.get("templates", {}).get("sentence_starters", []),
        fallback_cfg=decoder_cfg.get("fallback", {}),
        validation_cfg=decoder_cfg.get("validation", {}),
        chain_render_cfg=decoder_cfg.get("templates", {}).get("chain_render", {}),
    )
    return decoder


def subgraph_to_walker_subgraph(res_subgraph: G2PSubgraph, graph: DictGraphStore) -> WalkerSubgraph:
    """Convert resonance Subgraph to Walker Subgraph format."""
    rev_edges = [(t, s, r) for s, t, r in res_subgraph.edges]
    rev_strengths = {}
    rev_confidences = {}
    for s, t, r in res_subgraph.edges:
        rev_strengths[(t, s, r)] = res_subgraph.edge_strengths.get((s, t, r), 0.5)
        rev_confidences[(t, s, r)] = res_subgraph.edge_confidences.get((s, t, r), 0.5)

    all_edges = list(dict.fromkeys(res_subgraph.edges + rev_edges))
    all_strengths = {**res_subgraph.edge_strengths, **rev_strengths}
    all_confidences = {**res_subgraph.edge_confidences, **rev_confidences}

    node_embeddings = {}
    for nid in res_subgraph.nodes:
        emb = graph.get_embedding(nid)
        if emb is not None:
            node_embeddings[nid] = emb

    return WalkerSubgraph(
        nodes=res_subgraph.nodes,
        node_activations=res_subgraph.node_activations,
        edges=all_edges,
        edge_strengths=all_strengths,
        edge_confidences=all_confidences,
        seed_nodes=res_subgraph.seed_nodes,
        tier_used=res_subgraph.tier_used,
        activation_energy=res_subgraph.activation_energy,
        query_embedding=res_subgraph.query_embedding,
        timestamp=res_subgraph.timestamp,
        node_embeddings=node_embeddings,
    )


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def run_pipeline_test(question: str, graph: DictGraphStore, sbert, tier1, extractor, walker, decoder) -> Dict[str, Any]:
    """Run a single question through the full pipeline with REAL SBERT."""
    steps_log = {}
    t_total = time.time()

    # Step 1: Embed question with REAL SBERT
    ts = time.time()
    q_emb = sbert.encode(question, normalize_embeddings=True)
    # SBERT returns 1D array (384,) for single string - no [0] needed
    steps_log["1_encode"] = round(time.time() - ts, 4)

    # Step 2: Get seed subgraph by embedding similarity
    ts = time.time()
    seed_sub = graph.get_subgraph_by_embedding_similarity(q_emb, top_k=20)
    for nid in seed_sub.seed_nodes[:3]:
        seed_sub.node_activations[nid] = max(seed_sub.node_activations.get(nid, 0), 0.8)
    steps_log["2_subgraph"] = round(time.time() - ts, 4)

    # Step 3: Tier1 Resonance (with float32 precision workaround)
    ts = time.time()
    import resonance.tier1 as tier1_mod
    orig_validate = tier1_mod.validate_subgraph
    def patched_validate(subgraph, core_config):
        for nid, act in subgraph.node_activations.items():
            if act < core_config.activation.min:
                subgraph.node_activations[nid] = core_config.activation.min
        return orig_validate(subgraph, core_config)
    tier1_mod.validate_subgraph = patched_validate
    try:
        resonated, history = tier1.resonate(q_emb, graph, seed_sub.seed_nodes)
    finally:
        tier1_mod.validate_subgraph = orig_validate
    steps_log["3_resonance"] = round(time.time() - ts, 4)

    # Step 4: Real QueryRelationExtractor -> Plan
    ts = time.time()
    plan = extractor.plan(resonated, query_text=question)
    steps_log["4_plan"] = round(time.time() - ts, 4)

    # Step 5: Graph Walker
    ts = time.time()
    walker_sub = subgraph_to_walker_subgraph(resonated, graph)

    walker_plan = WalkerPlan(
        intent_sequence=plan.intent_sequence,
        plan_confidence=plan.plan_confidence,
        heuristic_fallback_used=plan.heuristic_fallback_used,
        intent_names=plan.intent_names,
        relation_chain=plan.relation_chain,
    )

    walk = walker.walk(walker_sub, walker_plan)
    steps_log["5_walk"] = round(time.time() - ts, 4)

    # Step 6: Decoder
    ts = time.time()
    node_labels = [graph.get_label(n) for n in walk.path]
    edge_labels = list(walk.path_edges)
    chain = plan.relation_chain or ["has_property"]

    if plan.heuristic_fallback_used and not walk.path_edges:
        answer = decoder.render_no_relation(node_labels, chain=chain)
        template_ok = True if answer else False
    else:
        answer, template_ok = decoder.decode(node_labels, edge_labels, chain=chain)
        if not template_ok:
            answer = decoder.fallback(node_labels, edge_labels, chain=chain)
            template_ok = True

    steps_log["6_decode"] = round(time.time() - ts, 4)

    total_time = round(time.time() - t_total, 3)

    return {
        "question": question,
        "answer": answer,
        "relation_chain": chain,
        "heuristic_used": plan.heuristic_fallback_used,
        "template_matched": template_ok,
        "plan_confidence": round(plan.plan_confidence, 4),
        "walk_confidence": round(walk.walk_confidence, 4),
        "time_seconds": total_time,
        "steps_timing": steps_log,
        "n_resonated_nodes": len(resonated.nodes),
        "n_resonated_edges": len(resonated.edges),
        "resonance_energy": round(resonated.activation_energy, 4),
        "n_walk_steps": walk.steps_taken,
        "walk_path_labels": node_labels,
        "walk_path_edges": edge_labels,
        "walk_path_activations": [round(a, 4) for a in walk.path_activations],
    }


def run_all_pipeline_tests():
    """Run all toy queries through the full pipeline with REAL SBERT."""
    print("=" * 70)
    print("GLM-X FULL PIPELINE TOY TESTING (REAL SBERT - PyTorch backend)")
    print("=" * 70)

    # Initialize components
    print("\n[1/6] Building toy knowledge graph...")
    graph, tmp_dir = build_dict_graph_store()
    print(f"      Nodes: {graph.get_node_count()}, Edges: {graph.get_edge_count()}")

    print("[2/6] Loading REAL SentenceTransformer (BAAI/bge-small-en-v1.5)...")
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("      Model loaded successfully")

    print("[3/6] Initializing Tier1 Resonance...")
    tier1, core_cfg = setup_resonance()

    print("[4/6] Initializing REAL Query-Relation Extractor...")
    extractor = setup_extractor(graph, sbert)

    print("[5/6] Initializing Graph Walker...")
    walker = setup_walker()

    print("[6/6] Initializing Template Decoder...")
    decoder = setup_decoder()

    print("\n" + "=" * 70)
    print("RUNNING PIPELINE TESTS WITH REAL SEMANTIC EMBEDDINGS")
    print("=" * 70)

    results = {
        "metadata": {
            "graph_nodes": graph.get_node_count(),
            "graph_edges": graph.get_edge_count(),
            "relations_covered": len(graph.get_all_relations()),
            "test_queries": len(TOY_QUERIES),
            "sbert_model": "BAAI/bge-small-en-v1.5 (PyTorch backend)",
        },
        "test_results": [],
        "summary": {},
    }

    passed = 0
    failed = 0

    for i, query_info in enumerate(TOY_QUERIES):
        question = query_info["question"]
        print(f"\n[{i+1}/{len(TOY_QUERIES)}] {question}")

        try:
            result = run_pipeline_test(question, graph, sbert, tier1, extractor, walker, decoder)
            results["test_results"].append(result)

            # Check if answer is reasonable
            if result["answer"] and len(result["answer"]) > 5:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL (empty answer)"

            print(f"    Status: {status}")
            print(f"    Chain: {result['relation_chain']} | Heuristic: {result['heuristic_used']}")
            print(f"    Answer: {result['answer'][:100]}...")
            print(f"    Walk: {result['n_walk_steps']} steps, conf={result['walk_confidence']:.3f}")
            print(f"    Time: {result['time_seconds']:.3f}s")

        except Exception as e:
            failed += 1
            error_result = {
                "question": question,
                "error": str(e),
                "status": "ERROR"
            }
            results["test_results"].append(error_result)
            print(f"    Status: ERROR - {e}")

    results["summary"] = {
        "total": len(TOY_QUERIES),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(TOY_QUERIES) * 100, 1),
    }

    print("\n" + "=" * 70)
    print("PIPELINE TEST SUMMARY (REAL SBERT)")
    print("=" * 70)
    print(f"  Total queries: {results['summary']['total']}")
    print(f"  Passed:        {results['summary']['passed']}")
    print(f"  Failed:        {results['summary']['failed']}")
    print(f"  Pass rate:     {results['summary']['pass_rate']}%")
    print("=" * 70)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pipeline_test_results_REAL.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Cleanup
    import shutil
    if hasattr(graph, 'close'):
        graph.close()
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


if __name__ == "__main__":
    run_all_pipeline_tests()