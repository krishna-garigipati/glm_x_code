"""Diagnostic: dump walker/candidate decisions for the failing food_bio_small goldens.

Level-1 helper for Section 8 failure classification. Not part of the golden set;
only inspects WHY fbs07/12/14 failed as observed.
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("glmx.walker")
log.setLevel(logging.DEBUG)

QS = ["Give me an example of a bird", "What is the opposite of sweet?", "What follows photosynthesis?"]


def make_pipeline(db):
    p = GLMXPipeline()
    p.graph_store = SQLiteGraphStore.load_state(db)
    p.load_models()
    w = p.walker
    object.__setattr__(w._walker_config.debug, "log_decision_scores", True)
    object.__setattr__(w._walker_config.debug, "log_path_taken", True)
    return p


def main():
    db = str(ROOT / "test_results" / "tester-b" / "datasets" / "food_bio_small.db")
    p = make_pipeline(db)
    for q in QS:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("=" * 70)
        r = p.ask(q)
        print(f"target_entity_ids/seed selection:")
        print(f"  path: {r['walk_path_labels']}")
        print(f"  edges: {r['walk_path_edges']}")
        print(f"  chain: {r['relation_chain']}  heuristic={r['heuristic_used']} plan_conf={r['confidence']:.4f}")
        rd = r.get("relation_details")
        if rd:
            for det in rd:
                if isinstance(det, dict):
                    print(f"  rel detail: {det}")
        embs = p.sbert.encode([q] + [c for c in [
            "is_a", "example_of", "has_property", "part_of", "synonym", "antonym", "causes",
            "Give me an example of a bird", "What is the opposite of sweet?",
            "What follows photosynthesis?",
        ] if c != q], normalize_embeddings=True)
        print(f"  q emb shape ok: {embs.shape}")


if __name__ == "__main__":
    main()