"""Level-2 evaluation: isolate P1 (walker semantic-similarity scoring).

Feeds the SAME subgraph + plan to two GraphWalker instances that differ ONLY
in weight_target_similarity (0.0 = old behavior, 1.0 = P1). The smoke node has
two identical out-edges (associated_with -> tobacco / -> fire), same strength,
confidence and activation, so the ONLY discriminating signal is the query
embedding similarity of the target node.

Expected:
    weight=0.0 -> tobacco and fire picked ~50/50 (coin flip at score parity)
    weight=1.0 -> consistently the more similar target (tobacco)

Usage:
    python test_results/lead/toy_eval_walker_ab.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataclasses import dataclass, replace

import numpy as np
from sentence_transformers import SentenceTransformer

from walker.config import CoreConfig, WalkerConfig
from walker.graph_walker import GraphWalker
from walker.models import Plan as WalkerPlan
from walker.models import Subgraph as WalkerSubgraph

DB_PATH = ROOT / "configs" / "config_core.yaml"
WALKER_PATH = ROOT / "configs" / "config_walker.yaml"

SMOKE, TOBACCO, FIRE = 1, 2, 3


@dataclass
class TrialRun:
    variant: str
    counts: dict
    trials: int


def build_subgraph(sbert: SentenceTransformer, query: str) -> WalkerSubgraph:
    labels = {SMOKE: "smoke", TOBACCO: "tobacco", FIRE: "fire"}
    raw = sbert.encode(list(labels.values()), normalize_embeddings=True, show_progress_bar=False)
    embeddings = {nid: np.asarray(raw[i], dtype=np.float32) for i, nid in enumerate(labels)}

    edges = [
        (SMOKE, TOBACCO, "associated_with"),
        (SMOKE, FIRE, "associated_with"),
    ]
    strengths = {e: 0.9 for e in edges}
    confidences = {e: 0.9 for e in edges}
    activations = {SMOKE: 1.0, TOBACCO: 0.5, FIRE: 0.5}

    return WalkerSubgraph(
        nodes=[SMOKE, TOBACCO, FIRE],
        node_activations=activations,
        edges=edges,
        edge_strengths=strengths,
        edge_confidences=confidences,
        seed_nodes=[SMOKE],
        tier_used=1,
        activation_energy=1.0,
        query_embedding=np.asarray(
            sbert.encode(query, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        ),
        timestamp=0.0,
        node_embeddings=embeddings,
    )


def make_walker(core_cfg: CoreConfig, walker_cfg: WalkerConfig, similarity_weight: float) -> GraphWalker:
    scoring = replace(walker_cfg.scoring, weight_target_similarity=float(similarity_weight))
    cfg = replace(walker_cfg, scoring=scoring)
    return GraphWalker(cfg, core_cfg, random_seed=1234)


def main() -> None:
    core_cfg = CoreConfig.from_yaml(str(DB_PATH))
    walker_cfg = WalkerConfig.from_yaml(str(WALKER_PATH))

    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    query = "what is smoking related to?"
    subgraph = build_subgraph(sbert, query)
    plan = WalkerPlan(
        intent_sequence=None,
        plan_confidence=1.0,
        heuristic_fallback_used=False,
        intent_names=None,
        relation_chain=["associated_with"],
    )

    q_emb = subgraph.query_embedding
    for nid, name in ((TOBACCO, "tobacco"), (FIRE, "fire")):
        sim = float(np.dot(q_emb, subgraph.node_embeddings[nid]))
        print(f"cos(query, {name:8s}) = {sim:.3f}")

    trials = 21
    print("\nRunning walker A/B (" + str(trials) + " trials each, seed=1234)...\n")

    results = []
    for variant, weight in (("weight=0.0 (old)", 0.0), ("weight=1.0 (P1)", 1.0)):
        walker = make_walker(core_cfg, walker_cfg, weight)
        counts = {TOBACCO: 0, FIRE: 0}
        for _ in range(trials):
            result = walker.walk(subgraph, plan)
            chosen = result.path[1]
            counts[chosen] += 1
        results.append(TrialRun(variant, counts, trials))
        print(f"{variant:20s} -> tobacco: {counts[TOBACCO]}/{trials}   fire: {counts[FIRE]}/{trials}")

    print("\nNote: at weight=0.0 both candidates are score-identical (coin flip). "
          "Stable tobacco picks at weight=1.0 demonstrate P1.")


if __name__ == "__main__":
    main()