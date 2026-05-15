from __future__ import annotations

import logging
import time
import textwrap
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")

from resonance.config import AlgorithmConfig, CoreConfig, TemporalConfig, TierConfig
from resonance.energy import compute_activation_energy
from resonance.tier1 import Tier1Resonance
from resonance.tests.fixtures.config_provider import build_minimal_core_config, build_minimal_resonance_config
from resonance.tests.fixtures.toy_data import build_animal_kingdom_graph
from resonance.tests.fixtures.toy_graph_store import ToyGraphStore

np.set_printoptions(precision=4, suppress=True)

ALGORITHM = AlgorithmConfig(
    propagation_type="wilson_cowan",
    normalization="budget_soft_cap",
    gate_type="top_k",
    temporal_factor_enabled=False,
)

TEMPORAL = TemporalConfig(gamma=0.5, frequency_threshold=20)

def _build_tier1_config() -> TierConfig:
    rc = build_minimal_resonance_config()
    return rc.tier1

def _build_core() -> CoreConfig:
    return build_minimal_core_config()

def _make_tier1(log_history: bool = True) -> Tier1Resonance:
    return Tier1Resonance(
        core_config=_build_core(),
        algorithm=ALGORITHM,
        temporal=TEMPORAL,
        tier_config=_build_tier1_config(),
        log_activation_history=log_history,
        history_buffer_size=1000,
    )


def _map_labels(graph: ToyGraphStore) -> Tuple[Dict[str, int], Dict[int, str]]:
    label_to_id = {}
    id_to_label = {}
    for node in graph.get_all_nodes():
        label_to_id[node.label] = node.id
        id_to_label[node.id] = node.label
    return label_to_id, id_to_label


def _describe_node_type(graph: ToyGraphStore, node_id: int) -> str:
    node = graph.get_node(node_id)
    if node is None:
        return "Unknown"
    return node.node_type


def _neighbor_details(graph: ToyGraphStore, node_id: int) -> List[Tuple[int, str, str, float, float]]:
    details = []
    for nid, edge in graph.get_neighbors(node_id):
        n = graph.get_node(nid)
        label = n.label if n else f"Node-{nid}"
        details.append((nid, label, edge.relation_type, edge.strength, edge.confidence))
    return details


def _simulate_propagation(
    tier1: Tier1Resonance,
    graph: ToyGraphStore,
    seed_id: int,
    seed_label: str,
    max_iter: int = 4,
) -> Dict:
    query = np.zeros(384, dtype=np.float32)
    subgraph, history = tier1.resonate(query, graph, [seed_id], max_iterations=max_iter)

    _, id_to_label = _map_labels(graph)

    steps: List[Dict] = []
    prev_activations: Dict[int, float] = {seed_id: 1.0}

    tier1_cfg = _build_tier1_config()
    core_cfg = _build_core()

    temp_activations: Dict[int, float] = {seed_id: 1.0}
    temp_edges: Dict[Tuple[int, int, str], Tuple[float, float]] = {}

    for step in range(max_iter):
        temp_activations = tier1._propagate(graph, temp_activations, temp_edges)

        sorted_acts = sorted(temp_activations.items(), key=lambda x: x[1], reverse=True)
        step_info = {
            "step": step + 1,
            "activations": dict(sorted_acts[:30]),
            "top5": [(id_to_label.get(nid, f"N{nid}"), round(val, 4)) for nid, val in sorted_acts[:5]],
            "num_active": len(temp_activations),
        }
        steps.append(step_info)

    all_activations = dict(temp_activations)

    return {
        "seed_label": seed_label,
        "seed_id": seed_id,
        "iterations": max_iter,
        "steps": steps,
        "final_subgraph": subgraph,
        "final_activations": all_activations,
        "activation_history": history,
    }


def analyze_scenario(
    graph: ToyGraphStore,
    seed_label: str,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    expected_high: List[str],
    expected_low: List[str],
) -> Dict:
    tier1 = _make_tier1(log_history=True)
    seed_id = label_to_id[seed_label]

    result = _simulate_propagation(tier1, graph, seed_id, seed_label, max_iter=4)
    result["expected_high"] = expected_high
    result["expected_low"] = expected_low
    result["label_to_id"] = label_to_id
    result["id_to_label"] = id_to_label

    final = result["final_activations"]
    sorted_final = sorted(final.items(), key=lambda x: x[1], reverse=True)

    result["sorted_final"] = [(id_to_label[nid], val) for nid, val in sorted_final]

    seed_nid = label_to_id.get(result["seed_label"])
    high_checks = []
    for label in expected_high:
        nid = label_to_id.get(label)
        if nid is None:
            high_checks.append((label, "NOT_IN_GRAPH", False))
        else:
            activated = nid in final
            val = final.get(nid, 0.0)
            is_direct = seed_nid is not None and any(nid == e[0] for e in graph.get_neighbors(seed_nid))
            threshold = 0.08 if is_direct else 0.04
            high_checks.append((label, val, activated and val >= threshold))
    result["high_checks"] = high_checks

    low_checks = []
    for label in expected_low:
        nid = label_to_id.get(label)
        if nid is None:
            low_checks.append((label, "NOT_IN_GRAPH", True))
        else:
            activated = nid in final
            val = final.get(nid, 0.0)
            low_checks.append((label, val, not activated or val < 0.06))
    result["low_checks"] = low_checks

    activation_energy = compute_activation_energy(result["final_subgraph"])
    result["activation_energy"] = activation_energy

    result["converged"] = len(result["activation_history"]) < 4 if result["activation_history"] else False

    step_counts = [s["num_active"] for s in result["steps"]]
    result["active_node_counts"] = step_counts
    result["stable"] = len(step_counts) < 2 or abs(step_counts[-1] - step_counts[-2]) <= 2

    del tier1
    return result


def print_scenario_report(scenario_name: str, result: Dict, graph: ToyGraphStore):
    sep = "=" * 72
    sub_sep = "-" * 72

    print(f"\n{sep}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"  Seed: {result['seed_label']} (ID={result['seed_id']})")
    print(f"{sep}")

    print(f"\n  Activation Energy: {result['activation_energy']:.4f}")
    print(f"  Converged: {result['converged']}")
    print(f"  Stable final step: {result['stable']}")
    print(f"  Active node counts per step: {result['active_node_counts']}")

    print(f"\n{sub_sep}")
    print(f"  ITERATION-BY-ITERATION ACTIVATION EVOLUTION")
    print(f"{sub_sep}")

    for step_info in result["steps"]:
        print(f"\n  --- Step {step_info['step']} ({step_info['num_active']} active nodes) ---")
        print(f"  Top-5: {step_info['top5']}")
        top3_nodes = list(step_info["activations"].items())[:3]
        for nid, val in top3_nodes:
            label = result["id_to_label"].get(nid, f"N{nid}")
            ntype = _describe_node_type(graph, nid)
            neighbors = _neighbor_details(graph, nid)
            relevant = [(lbl, rel, s, c) for (nnid, lbl, rel, s, c) in neighbors if nnid in step_info["activations"] or nnid == result["seed_id"]]
            print(f"    {label} ({ntype}): act={val:.4f}")
            for nlbl, rel, s, c in relevant[:4]:
                print(f"      <- {rel} (s={s:.2f}, c={c:.2f})")

    print(f"\n{sub_sep}")
    print(f"  RANKED FINAL ACTIVATIONS (Top 15)")
    print(f"{sub_sep}")
    for label, val in result["sorted_final"][:15]:
        marker = ""
        nid = result["label_to_id"].get(label)
        if nid is not None and nid in result["final_subgraph"].seed_nodes:
            marker = " [SEED]"
        elif label in result["expected_high"]:
            marker = " [EXPECTED HIGH]"
        print(f"    {label:25s} {val:.4f}{marker}")

    print(f"\n{sub_sep}")
    print(f"  SEMANTIC CORRECTNESS CHECKS")
    print(f"{sub_sep}")

    high_fail = 0
    for label, val, ok in result["high_checks"]:
        status = "PASS" if ok else "FAIL"
        if not ok:
            high_fail += 1
        print(f"    [{status}] {label}: act={val} (expected HIGH)")

    low_fail = 0
    for label, val, ok in result["low_checks"]:
        status = "PASS" if ok else "FAIL"
        if not ok:
            low_fail += 1
        print(f"    [{status}] {label}: act={val} (expected LOW)")

    result["high_fail"] = high_fail
    result["low_fail"] = low_fail

    print(f"\n{sub_sep}")
    print(f"  PROPAGATION PATHS")
    print(f"{sub_sep}")
    seed_nid = result["seed_id"]
    direct = _neighbor_details(graph, seed_nid)
    print(f"  Direct neighbors of {result['seed_label']} (seed):")
    for nid, lbl, rel, s, c in direct:
        act = result["final_activations"].get(nid, 0.0)
        print(f"    -> {lbl:20s} via {rel:15s} (s={s:.2f}, c={c:.2f}) act={act:.4f}")

    secondary_nodes = set()
    for nid, lbl, rel, s, c in direct:
        sub = _neighbor_details(graph, nid)
        for nnid, nlbl, nrel, ns, nc in sub:
            if nnid != seed_nid:
                secondary_nodes.add((nnid, nlbl, lbl, nrel, ns, nc))

    if secondary_nodes:
        print(f"\n  Secondary (2-hop) connections with activations:")
        for nnid, nlbl, via, rel, s, c in sorted(secondary_nodes, key=lambda x: result["final_activations"].get(x[0], 0), reverse=True)[:8]:
            act = result["final_activations"].get(nnid, 0.0)
            if act > 0:
                print(f"    {nlbl:20s} <- {rel:15s} (via {via:15s}, s={s:.2f}, c={c:.2f}) act={act:.4f}")

    return result


def check_numerical_stability(graph: ToyGraphStore, label_to_id: Dict[str, int], id_to_label: Dict[int, str], seed_labels: List[str]):
    print("\n" + "=" * 72)
    print("  NUMERICAL STABILITY & DETERMINISM ANALYSIS")
    print("=" * 72)

    seed_label = seed_labels[0]
    seed_id = label_to_id[seed_label]

    all_runs = []
    for run in range(5):
        tier1 = _make_tier1(log_history=False)
        query = np.zeros(384, dtype=np.float32)
        subgraph, _ = tier1.resonate(query, graph, [seed_id], max_iterations=4)
        activations = subgraph.node_activations
        all_runs.append(activations)
        del tier1

    all_keys = set()
    for a in all_runs:
        all_keys.update(a.keys())
    all_keys = sorted(all_keys)

    print(f"\n  Determinism check: 5 runs with seed={seed_label}")
    print(f"  Total unique nodes across runs: {len(all_keys)}")

    deviations = 0
    for key in all_keys:
        vals = [a.get(key, 0.0) for a in all_runs]
        if max(vals) - min(vals) > 1e-6:
            deviations += 1
            if deviations <= 5:
                lbl = id_to_label.get(key, f"N{key}")
                print(f"    Non-deterministic: {lbl} values={[round(v, 6) for v in vals]}")

    print(f"  Nodes with any deviation: {deviations}/{len(all_keys)}")
    print(f"  Deterministic: {'PASS (all identical)' if deviations == 0 else f'FAIL ({deviations} deviations)'}")

    print(f"\n  Float precision check (NaN/Inf):")
    nan_inf_count = 0
    for run_idx, a in enumerate(all_runs):
        for nid, val in a.items():
            if not np.isfinite(val):
                nan_inf_count += 1
                lbl = id_to_label.get(nid, f"N{nid}")
                print(f"    Non-finite in run {run_idx}: {lbl}={val}")
    if nan_inf_count == 0:
        print("    No NaN/Inf values in any run [PASS]")

    return all_runs


def check_contradiction_handling(graph: ToyGraphStore, label_to_id: Dict[str, int], id_to_label: Dict[int, str]):
    print("\n" + "=" * 72)
    print("  CONTRADICTION HANDLING ANALYSIS")
    print("=" * 72)

    pairs = [
        ("whale", "fish", "contradicts"),
        ("cat", "bird", "contradicts"),
    ]

    for src_label, tgt_label, rel in pairs:
        src_id = label_to_id[src_label]
        tgt_id = label_to_id[tgt_label]

        tier1 = _make_tier1(log_history=True)
        query = np.zeros(384, dtype=np.float32)
        subgraph1, hist1 = tier1.resonate(query, graph, [src_id], max_iterations=4)
        acts1 = subgraph1.node_activations
        tgt_act = acts1.get(tgt_id, 0.0)
        tgt_label_actual = id_to_label[tgt_id]

        print(f"\n  Contradiction: {src_label} --[{rel}]--> {tgt_label_actual}")
        print(f"    Activation of {tgt_label_actual} from seed={src_label}: {tgt_act:.4f}")
        print(f"    Relation bias for 'contradicts': 0.3")

        edge_found = False
        for nid, edge in graph.get_neighbors(src_id):
            if nid == tgt_id and edge.relation_type == rel:
                edge_found = True
                edge_weight = edge.strength * edge.confidence
                print(f"    Edge: strength={edge.strength}, confidence={edge.confidence}, weight={edge_weight:.4f}")
                print(f"    Effective input = act(src) * weight * relation_bias = 1.0 * {edge_weight:.4f} * 0.3 = {1.0 * edge_weight * 0.3:.4f}")
                break

        if not edge_found:
            rev_edge_found = False
            for nid, edge in graph.get_neighbors(tgt_id):
                if nid == src_id and edge.relation_type == rel:
                    rev_edge_found = True
                    edge_weight = edge.strength * edge.confidence
                    print(f"    Reverse edge found: strength={edge.strength}, confidence={edge.confidence}, weight={edge_weight:.4f}")
                    print(f"    Effective input = 1.0 * {edge_weight:.4f} * 0.3 = {1.0 * edge_weight * 0.3:.4f}")
                    break
            if not rev_edge_found:
                print(f"    No contradict edge found between {src_label} and {tgt_label_actual}")

        print(f"    Assessment: ", end="")
        if tgt_act < 0.05:
            print(f"Contradiction successfully suppresses activation [{tgt_act:.4f} < 0.05]")
        elif tgt_act < 0.15:
            print(f"Contradiction partially suppresses [{tgt_act:.4f}]")
        else:
            print(f"Contradiction may be weak [{tgt_act:.4f}]")

        del tier1


def check_analogy_quality(graph: ToyGraphStore, label_to_id: Dict[str, int], id_to_label: Dict[int, str]):
    print("\n" + "=" * 72)
    print("  ANALOGY / SIMILARITY RETRIEVAL ANALYSIS")
    print("=" * 72)

    from resonance.analogy import AnalogyFinder
    from resonance.config import AnalogyParameters
    core = build_minimal_core_config()
    rc = build_minimal_resonance_config()
    ap = rc.tier2.analogy_parameters

    af = AnalogyFinder(core, ap, embedding_dim=32)

    test_nodes = ["dog", "shark", "whale", "bat", "penguin"]
    print(f"\n  LSH Analogy Retrieval (top_k=3, overlap_validation=True):")
    for label in test_nodes:
        nid = label_to_id[label]
        try:
            analogs = af.get_analogy_leaps(nid, graph, top_k=3)
            print(f"\n    Seed={label} (ID={nid}):")
            if analogs:
                for aid, sim in analogs:
                    alabel = id_to_label.get(aid, f"N{aid}")
                    atype = _describe_node_type(graph, aid)
                    print(f"      -> {alabel:20s} ({atype:10s}) similarity={sim:.4f}")
            else:
                print(f"      (no analogies found)")
        except Exception as e:
            print(f"      ERROR: {e}")

    print(f"\n  Brute-force (embedding) similarity check for query vs all nodes:")
    query = np.zeros(384, dtype=np.float32)
    sub = graph.get_subgraph_by_embedding_similarity(query, top_k=5)
    if sub.nodes:
        for nid in sub.nodes:
            lbl = id_to_label.get(nid, f"N{nid}")
            print(f"    {lbl:20s} base_act={sub.node_activations[nid]:.4f}")


def check_overpropagation(graph: ToyGraphStore, label_to_id: Dict[str, int], id_to_label: Dict[int, str], all_results: Dict[str, Dict]):
    print("\n" + "=" * 72)
    print("  OVER-PROPAGATION / UNDER-PROPAGATION ANALYSIS")
    print("=" * 72)

    all_activated = {}
    for scenario_name, result in all_results.items():
        for nid, val in result["final_activations"].items():
            if nid not in all_activated:
                all_activated[nid] = []
            all_activated[nid].append((scenario_name, val))

    exploding = 0
    for nid, entries in all_activated.items():
        max_val = max(v for _, v in entries)
        if max_val > 0.95:
            lbl = id_to_label.get(nid, f"N{nid}")
            print(f"    Near-max activation: {lbl}={max_val:.4f} (from {[s for s, _ in entries]})")
            exploding += 1

    if exploding == 0:
        print("  No exploding activations (>0.95) [PASS]")

    print(f"\n  Cross-scenario activation counts:")
    for nid, entries in sorted(all_activated.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        lbl = id_to_label.get(nid, f"N{nid}")
        scenarios = ", ".join([f"{s}({v:.3f})" for s, v in entries])
        print(f"    {lbl:20s}: activated in {len(entries)}/5 scenarios [{scenarios}]")

    for scenario_name, result in all_results.items():
        seed_id = result["seed_id"]
        final = result["final_activations"]
        unrelated = [(nid, val) for nid, val in final.items() if nid != seed_id and _describe_node_type(graph, nid) not in ("Property", "Concept")]
        print(f"\n  Scenario '{scenario_name}': Entity activations beyond seed:")
        if unrelated:
            for nid, val in sorted(unrelated, key=lambda x: x[1], reverse=True)[:5]:
                lbl = id_to_label.get(nid, f"N{nid}")
                path_info = _get_propagation_path_summary(graph, nid, seed_id, label_to_id, id_to_label)
                print(f"    {lbl:20s} act={val:.4f} path={path_info}")
        else:
            print(f"    (none)")


def _get_propagation_path_summary(
    graph: ToyGraphStore,
    target_id: int,
    seed_id: int,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
) -> str:
    if target_id == seed_id:
        return "self"

    for nid, edge in graph.get_neighbors(seed_id):
        if nid == target_id:
            return f"direct({edge.relation_type})"

    for mid, edge1 in graph.get_neighbors(seed_id):
        for nid, edge2 in graph.get_neighbors(mid):
            if nid == target_id:
                m_label = id_to_label.get(mid, f"N{mid}")
                return f"via {m_label}({edge1.relation_type}->{edge2.relation_type})"

    return "3+ hops"


def check_normalization_stability(all_results: Dict[str, Dict]):
    print("\n" + "=" * 72)
    print("  NORMALIZATION & BUDGET ANALYSIS")
    print("=" * 72)

    core = _build_core()
    budget_max = core.resonance.budget_max

    print(f"  Budget soft-cap max: {budget_max}")

    for scenario_name, result in all_results.items():
        final = result["final_activations"]
        total_activation = sum(final.values())
        num_active = len(final)
        print(f"  {scenario_name:20s}: total_act={total_activation:.4f}, num_active={num_active}, budget_capped={total_activation > budget_max if total_activation > budget_max + 0.01 else False}")


def check_activation_ranking_quality(all_results: Dict[str, Dict], id_to_label: Dict[int, str]):
    print("\n" + "=" * 72)
    print("  ACTIVATION RANKING QUALITY ANALYSIS")
    print("=" * 72)

    for scenario_name, result in all_results.items():
        sorted_acts = result["sorted_final"]
        print(f"\n  Scenario '{scenario_name}' (seed={result['seed_label']}):")
        print(f"    {'Rank':<5} {'Node':<20} {'Activation':<10} {'Type':<12} {'Plausible':<10}")
        print(f"    {'-'*57}")
        for rank, (label, val) in enumerate(sorted_acts[:10], 1):
            nid = result["label_to_id"].get(label)
            ntype = _describe_node_type(graph, nid) if nid is not None else "?"
            plausible = _judge_plausible(result["seed_label"], label, rank)
            print(f"    {rank:<5} {label:<20} {val:<10.4f} {ntype:<12} {plausible:<10}")


def _judge_plausible(seed: str, activated: str, rank: int) -> str:
    immediate = {
        "dog": ["mammal", "animal", "has_fur", "carnivore", "lives_on_land"],
        "penguin": ["bird", "animal", "has_wings", "lives_in_water"],
        "shark": ["fish", "animal", "carnivore", "predator", "lives_in_water"],
        "whale": ["mammal", "animal", "lives_in_water", "gives_birth"],
        "bat": ["mammal", "animal", "has_wings", "can_fly", "has_fur"],
    }
    related = {
        "dog": ["cat", "has_backbone", "is_warm_blooded", "gives_birth"],
        "penguin": ["has_backbone", "is_warm_blooded", "lays_eggs"],
        "shark": ["has_backbone", "is_cold_blooded", "gives_birth", "lays_eggs"],
        "whale": ["has_backbone", "is_warm_blooded", "has_fur"],
        "bat": ["has_backbone", "is_warm_blooded", "gives_birth", "bird", "eagle", "sparrow"],
    }
    if rank == 1 and activated == seed:
        return "YES(seed)"
    if activated in immediate.get(seed, []):
        return "YES"
    if activated in related.get(seed, []):
        return "OK"
    return "?" if rank <= 10 else "LOW"


def check_propagation_decay(all_results: Dict[str, Dict], id_to_label: Dict[int, str]):
    print("\n" + "=" * 72)
    print("  PROPAGATION DECAY ANALYSIS")
    print("=" * 72)

    for scenario_name, result in all_results.items():
        print(f"\n  Scenario '{scenario_name}' (seed={result['seed_label']}):")
        step_traces = result["steps"]
        for si, step in enumerate(step_traces):
            top_vals = [v for _, v in step["top5"]]
            print(f"    Step {si+1}: top5 activations = {top_vals} | active nodes = {step['num_active']}")

        sorted_final = result["sorted_final"]
        if len(sorted_final) >= 3:
            first, second, third = sorted_final[:3]
            ratio_12 = second[1] / max(first[1], 1e-8)
            ratio_13 = third[1] / max(first[1], 1e-8)
            print(f"    Seed activation: {sorted_final[0][1]:.4f}")
            print(f"    1st/2nd ratio: {ratio_12:.3f} ({first[0]}:{first[1]:.4f} / {second[0]}:{second[1]:.4f})")
            print(f"    1st/3rd ratio: {ratio_13:.3f}")


def main():
    print("=" * 72)
    print("  GLM-X RESONANCE COMPONENT — SEMANTIC VALIDATION REPORT")
    print("  Generation Time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 72)

    global graph
    graph = build_animal_kingdom_graph()
    label_to_id, id_to_label = _map_labels(graph)

    print(f"\n  Animal Kingdom Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

    scenarios = [
        ("dog",   "Hierarchical: dog -> mammal -> animal + properties",
         ["mammal", "animal", "has_fur", "lives_on_land", "has_backbone", "is_warm_blooded", "gives_birth"],
         ["shark", "fish", "lives_in_water", "has_wings", "can_fly", "lays_eggs", "is_cold_blooded"]),
        ("penguin", "Contradiction: penguin -> bird + can_fly conflict",
         ["bird", "animal", "has_wings", "has_backbone", "is_warm_blooded", "lays_eggs"],
         ["can_fly", "has_fur", "is_cold_blooded", "gives_birth"]),
        ("shark",  "Marine predator: shark -> fish + predator + contradictions",
         ["fish", "animal", "carnivore", "predator", "lives_in_water", "has_backbone", "gives_birth"],
         ["has_fur", "mammal", "lives_on_land", "is_warm_blooded"]),
        ("whale",  "Mammal in water: whale -> mammal + fish contradiction",
         ["mammal", "animal", "lives_in_water", "is_warm_blooded", "gives_birth", "has_backbone"],
         ["fish", "has_fur", "lays_eggs", "is_cold_blooded", "lives_on_land"]),
        ("bat",    "Mixed: bat -> mammal + wings + fly",
         ["mammal", "animal", "can_fly", "has_wings", "has_fur", "is_warm_blooded", "gives_birth", "has_backbone"],
         ["bird", "fish", "lives_in_water", "lays_eggs", "is_cold_blooded"]),
    ]

    all_results = {}

    for seed_label, desc, exp_high, exp_low in scenarios:
        scenario_name = f"{seed_label.upper()} ({desc})"
        result = analyze_scenario(graph, seed_label, label_to_id, id_to_label, exp_high, exp_low)
        all_results[seed_label] = result
        print_scenario_report(scenario_name, result, graph)

    print("\n" + "=" * 72)
    print("  CROSS-CUTTING ANALYSES")
    print("=" * 72)

    check_contradiction_handling(graph, label_to_id, id_to_label)
    check_analogy_quality(graph, label_to_id, id_to_label)
    check_numerical_stability(graph, label_to_id, id_to_label, ["dog"])
    check_overpropagation(graph, label_to_id, id_to_label, all_results)
    check_normalization_stability(all_results)
    check_activation_ranking_quality(all_results, id_to_label)
    check_propagation_decay(all_results, id_to_label)

    print("\n" + "=" * 72)
    print("  SEMANTIC ANOMALY DETECTION")
    print("=" * 72)

    total_high_fail = 0
    total_low_fail = 0
    for scenario_name, result in all_results.items():
        total_high_fail += result["high_fail"]
        total_low_fail += result["low_fail"]

    print(f"\n  Total expected-high failures: {total_high_fail}")
    print(f"  Total expected-low failures:  {total_low_fail}")
    print(f"  Overall semantic correctness: ", end="")
    if total_high_fail == 0 and total_low_fail == 0:
        print("ALL CHECKS PASS")
    elif total_high_fail <= 3 and total_low_fail <= 3:
        print(f"MINOR ISSUES ({total_high_fail} high, {total_low_fail} low)")
    else:
        print(f"SIGNIFICANT ISSUES ({total_high_fail} high, {total_low_fail} low)")

    unstable = [s for s, r in all_results.items() if not r["stable"]]
    if unstable:
        print(f"  Unstable scenarios (oscillating): {unstable}")

    print(f"\n{'='*72}")
    print(f"  END OF SEMANTIC VALIDATION REPORT")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
