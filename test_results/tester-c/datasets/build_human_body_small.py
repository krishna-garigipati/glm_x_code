"""Build human_body_small.db: tester-c Human Body / Medicine graph (Phase G).

Training-domain disjoint vocabulary. Targets >= 12 canonical relations,
~78 edges, heavy part_of + supports/contradicts.

Usage:
    python test_results/tester-c/datasets/build_human_body_small.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "human_body_small.db"

CONCEPTS = [
    "body", "skeleton", "rib cage", "spine", "rib", "vertebra", "arm", "hand",
    "finger", "elbow", "shoulder", "leg", "foot", "toe", "knee", "head", "neck",
    "skin", "muscle", "bone", "tendon", "ligament", "cartilage", "femur",
    "heart", "lungs", "liver", "kidney", "stomach", "intestines", "pancreas",
    "brain",
    "trachea", "esophagus", "diaphragm", "bladder", "eyes", "ears",
    "blood vessel", "artery", "vein", "capillary", "neuron", "nerve cell",
    "red blood cell", "white blood cell", "blood cell", "blood", "cell", "tissue",
    "organ", "body part",
    "skeletal system", "muscular system", "nervous system", "circulatory system",
    "respiratory system", "digestive system",
    "breathing", "respiration", "heartbeat", "pulse", "digestion", "hunger",
    "headache", "insomnia", "recovery", "fitness", "lung damage",
    "exercise", "dehydration", "stress", "sleep", "smoking", "junk food",
    "calcium", "vitamin d", "oxygen", "bone health", "heart health", "health",
    "cardiology", "neurology",
    "warm", "hard", "soft", "strong", "flexible", "healthy", "sick",
    "cold", "belly", "abdomen", "cuore",
]

EDGES = [
    # --- is_a taxonomy ---
    {"source": "heart", "target": "organ", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "lungs", "target": "organ", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "liver", "target": "organ", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "kidney", "target": "organ", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "stomach", "target": "organ", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "skin", "target": "organ", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "organ", "target": "body part", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "hand", "target": "body part", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "arm", "target": "body part", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "foot", "target": "body part", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "head", "target": "body part", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "leg", "target": "body part", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "finger", "target": "body part", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "muscle", "target": "tissue", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "bone", "target": "tissue", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "neuron", "target": "nerve cell", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "red blood cell", "target": "blood cell", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "white blood cell", "target": "blood cell", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "blood cell", "target": "cell", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "artery", "target": "blood vessel", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "vein", "target": "blood vessel", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "capillary", "target": "blood vessel", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    # --- part_of (source=part, target=whole) ---
    {"source": "hand", "target": "arm", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "finger", "target": "hand", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "elbow", "target": "arm", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "shoulder", "target": "arm", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "arm", "target": "body", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "foot", "target": "leg", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "toe", "target": "foot", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "knee", "target": "leg", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "leg", "target": "body", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "head", "target": "body", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "neck", "target": "body", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "skin", "target": "body", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "rib", "target": "rib cage", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "rib cage", "target": "skeleton", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "vertebra", "target": "spine", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "spine", "target": "skeleton", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "bone", "target": "skeleton", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "skeleton", "target": "body", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "heart", "target": "circulatory system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "blood vessel", "target": "circulatory system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "lungs", "target": "respiratory system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "trachea", "target": "respiratory system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "brain", "target": "nervous system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "neuron", "target": "nervous system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "stomach", "target": "digestive system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    # --- spatial_near ---
    {"source": "shoulder", "target": "neck", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    # --- linguistic_maps ---
    {"source": "heart", "target": "cuore", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    # --- has_property ---
    {"source": "blood", "target": "warm", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "bone", "target": "hard", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "muscle", "target": "strong", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "skin", "target": "flexible", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "brain", "target": "soft", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    # --- causes / caused_by ---
    {"source": "smoking", "target": "lung damage", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "dehydration", "target": "headache", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "stress", "target": "insomnia", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "sleep", "target": "recovery", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "exercise", "target": "fitness", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "lung damage", "target": "smoking", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    {"source": "headache", "target": "dehydration", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    {"source": "insomnia", "target": "stress", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    # --- antonym ---
    {"source": "healthy", "target": "sick", "relation": "antonym", "strength": 0.98, "confidence": 0.98},
    {"source": "warm", "target": "cold", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "hard", "target": "soft", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    # --- synonym ---
    {"source": "breathing", "target": "respiration", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    {"source": "heartbeat", "target": "pulse", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    {"source": "abdomen", "target": "belly", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    # --- example_of ---
    {"source": "femur", "target": "bone", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "heart", "target": "organ", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "red blood cell", "target": "blood cell", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    # --- supports ---
    {"source": "calcium", "target": "bone health", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    {"source": "vitamin d", "target": "bone health", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    {"source": "exercise", "target": "heart health", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    # --- contradicts ---
    {"source": "smoking", "target": "health", "relation": "contradicts", "strength": 0.9, "confidence": 0.9},
    {"source": "junk food", "target": "fitness", "relation": "contradicts", "strength": 0.9, "confidence": 0.9},
    # --- associated_with ---
    {"source": "heart", "target": "cardiology", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "brain", "target": "neurology", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "red blood cell", "target": "oxygen", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "stomach", "target": "hunger", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
]


def main() -> None:
    from _build_common import build_db
    build_db(OUT, CONCEPTS, EDGES, dataset_name="human_body_small",
             min_relations=12, min_edges=60)


if __name__ == "__main__":
    main()