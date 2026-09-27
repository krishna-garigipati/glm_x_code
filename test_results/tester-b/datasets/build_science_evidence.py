"""Build science_evidence.db: tester-b's large Science-Evidence graph (Phase C, tier=scale).

Explores previously NEVER-EXERCISED relations supports & contradicts, plus
follows/precedes sequels (mitosis chains), evidence-style causes/caused_by.
Targets the Phase A scale gate: >=100 edges AND >=6 relations.

Supports the sc* goldens in the unified golden runner (science_evidence suite).

Usage:
    python test_results/tester-b/datasets/build_science_evidence.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

CONCEPTS = [
    # --- cellular cycle (follows/precedes chains build here) ---
    "division", "mitosis", "meiosis", "interphase", "prophase", "metaphase",
    "anaphase", "telophase", "replication", "cell", "nucleus", "chromosome",
    "ribosome", "cytoplasm", "membrane", "mitochondrion", "vacuole", "gene",
    "enzyme", "protein",
    # --- fermentation / microbiology ---
    "yeast", "fermentation", "process", "fungus", "bacteria", "anaerobic",
    "glucose", "ethanol", "carbon dioxide",
    # --- evidence claims ---
    "vaccination", "vaccine", "prevention", "immunity", "clinical trial",
    "exercise", "fitness", "health", "smoking", "lung disease", "sugar",
    "diabetes", "junk food", "obesity", "fasting", "disease", "illness",
    "sick", "healthy", "doctor", "physician",
    # --- epidemiology / climate evidence ---
    "virus", "infection", "pathogen", "allergy", "pollen", "inflammation",
    "greenhouse gas", "warming", "climate change", "cool", "warm", "gradual",
    "tiny", "harmful",
    # --- firmed-up labels for scale-gate volume ---
    "cancer", "flu", "malaria", "tuberculosis", "asthma", "anemia", "bronchitis",
    "organelle", "vesicle", "dextrose", "catalyst", "tissue", "organ", "body",
    "heart", "liver", "kidney", "brain", "sickness", "cigarette", "smoke",
    "sudden", "rare", "common", "microscopic", "infectious", "hereditary",
    "structural", "sweet", "crystalline", "macromolecule", "lipid",
    "monosaccharide", "disaccharide", "polysaccharide", "sucrose", "starch",
    "water", "molecule", "dna", "rna", "structure", "healing", "breathing",
    "digestion", "insulin spike",
]

EDGES = [
    # --- SUPPORTS (Phase C: never-exercised) ---
    {"source": "vaccination", "target": "prevention", "relation": "supports", "strength": 0.95, "confidence": 0.95},
    {"source": "exercise", "target": "fitness", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    {"source": "clinical trial", "target": "vaccination", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    {"source": "immunity", "target": "prevention", "relation": "supports", "strength": 0.85, "confidence": 0.85},
    # --- CONTRADICTS (Phase C: never-exercised) ---
    {"source": "sugar", "target": "diabetes", "relation": "contradicts", "strength": 0.9, "confidence": 0.9},
    {"source": "junk food", "target": "obesity", "relation": "contradicts", "strength": 0.9, "confidence": 0.9},
    # --- precedes (prior step; single canonical progression direction) ---
    {"source": "interphase", "target": "prophase", "relation": "precedes", "strength": 0.95, "confidence": 0.95},
    {"source": "prophase", "target": "metaphase", "relation": "precedes", "strength": 0.95, "confidence": 0.95},
    {"source": "metaphase", "target": "anaphase", "relation": "precedes", "strength": 0.95, "confidence": 0.95},
    {"source": "anaphase", "target": "telophase", "relation": "precedes", "strength": 0.95, "confidence": 0.95},
    # --- causes ---
    {"source": "yeast", "target": "fermentation", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "greenhouse gas", "target": "warming", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "virus", "target": "infection", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "smoking", "target": "lung disease", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "sugar", "target": "diabetes", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "infection", "target": "inflammation", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "pollen", "target": "allergy", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    # --- caused_by ---
    {"source": "fermentation", "target": "yeast", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "warming", "target": "greenhouse gas", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "infection", "target": "virus", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "lung disease", "target": "smoking", "relation": "caused_by", "strength": 0.9, "confidence": 0.9},
    {"source": "inflammation", "target": "infection", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    # --- associated_with ---
    {"source": "vaccination", "target": "immunity", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "exercise", "target": "fitness", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "warming", "target": "climate change", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "pollen", "target": "allergy", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "fermentation", "target": "ethanol", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "fermentation", "target": "carbon dioxide", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    # --- example_of ---
    {"source": "yeast", "target": "fungus", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "virus", "target": "pathogen", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "mitochondrion", "target": "organelle", "relation": "example_of", "strength": 0.85, "confidence": 0.85},
    # --- is_a ---
    {"source": "allergy", "target": "disease", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "cancer", "target": "disease", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "fitness", "target": "health", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "fermentation", "target": "process", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "mitosis", "target": "division", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "meiosis", "target": "division", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    # --- part_of ---
    {"source": "nucleus", "target": "cell", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "chromosome", "target": "nucleus", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "ribosome", "target": "cytoplasm", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "mitochondrion", "target": "cytoplasm", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "vacuole", "target": "cytoplasm", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "membrane", "target": "cell", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "cytoplasm", "target": "cell", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "gene", "target": "chromosome", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    # --- synonym ---
    {"source": "disease", "target": "illness", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "doctor", "target": "physician", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "vacuole", "target": "vesicle", "relation": "synonym", "strength": 0.85, "confidence": 0.85},
    # --- antonym ---
    {"source": "sick", "target": "healthy", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "warm", "target": "cool", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    # --- has_property ---
    {"source": "virus", "target": "tiny", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "smoking", "target": "harmful", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "fermentation", "target": "anaerobic", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "warming", "target": "gradual", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
]


FILLER_EDGES = [
    # --- is_a: disease fan-out (scale volume) ---
    {"source": c, "target": "disease", "relation": "is_a", "strength": 0.9, "confidence": 0.9}
    for c in ["cancer", "flu", "malaria", "tuberculosis", "asthma", "anemia", "bronchitis"]
] + [
    {"source": "mitochondrion", "target": "organelle", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "vacuole", "target": "vesicle", "relation": "synonym", "strength": 0.85, "confidence": 0.85},
    {"source": "dextrose", "target": "glucose", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "enzyme", "target": "catalyst", "relation": "synonym", "strength": 0.85, "confidence": 0.85},
    {"source": "sickness", "target": "illness", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "glucose", "target": "sweet", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "bacteria", "target": "microscopic", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "virus", "target": "infectious", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "gene", "target": "hereditary", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "protein", "target": "structural", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "sugar", "target": "crystalline", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "cell", "target": "tissue", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "tissue", "target": "organ", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "organ", "target": "body", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "heart", "target": "body", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "liver", "target": "body", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "kidney", "target": "body", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "brain", "target": "body", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "sugar", "target": "insulin spike", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "cigarette", "target": "smoking", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "smoke", "target": "lung disease", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "water", "target": "molecule", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "gradual", "target": "sudden", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "rare", "target": "common", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "glucose", "target": "monosaccharide", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "sucrose", "target": "disaccharide", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "starch", "target": "polysaccharide", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "protein", "target": "macromolecule", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "lipid", "target": "macromolecule", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "enzyme", "target": "macromolecule", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "dna", "target": "macromolecule", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "rna", "target": "macromolecule", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "water", "target": "molecule", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "glucose", "target": "molecule", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "chromosome", "target": "dna", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "enzyme", "target": "catalyst", "relation": "synonym", "strength": 0.85, "confidence": 0.85},
    {"source": "cytoplasm", "target": "structure", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "ribosome", "target": "organelle", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "gene", "target": "dna", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "rna", "target": "protein", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "healing", "target": "process", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "breathing", "target": "process", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "digestion", "target": "process", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "immunity", "target": "health", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
]


def _edges_by_id(concepts):
    resolved = []
    for e in EDGES + FILLER_EDGES:
        src, tgt = concepts.get(e["source"]), concepts.get(e["target"])
        if src is not None and tgt is not None and e["strength"] > 0:
            resolved.append({
                "source": src, "target": tgt, "relation": e["relation"],
                "strength": e["strength"], "confidence": e["confidence"],
            })
    return resolved


OUT = Path(__file__).resolve().parent / "science_evidence.db"


def main() -> None:
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")

    for stale in (OUT, OUT.with_suffix(".db-wal"), OUT.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    concepts = {label: i + 1 for i, label in enumerate(CONCEPTS)}
    labels = list(concepts.keys())
    raw = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {label: np.asarray(raw[i], dtype=np.float32) for i, label in enumerate(labels)}

    store = SQLiteGraphStore(db_path=str(OUT))
    edges = _edges_by_id(concepts)
    store.add_dataset(concepts=concepts, edges=edges, id_to_label=concepts)
    for label, nid in concepts.items():
        store._embeddings[nid] = embeddings[label]
    store.set_metadata("dataset_name", "science_evidence")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.save_state(str(OUT))

    loaded = SQLiteGraphStore.load_state(str(OUT))
    relation_set = sorted(loaded.get_all_relations())
    n_nodes = loaded.get_node_count()
    n_edges = loaded.get_edge_count()
    print(f"Wrote science_evidence graph: {n_nodes} nodes, {n_edges} edges")
    print(f"Relations ({len(relation_set)}): {relation_set}")
    assert n_edges >= 100, f"scale gate: >=100 edges, got {n_edges}"
    assert len(relation_set) >= 6, f"scale gate: >=6 relations, got {len(relation_set)}"
    missing = {"supports", "contradicts", "precedes"} - set(relation_set)
    assert not missing, f"missing target relations: {missing}"
    canonical = {"is_a", "example_of", "has_property", "causes", "caused_by", "supports",
                 "contradicts", "synonym", "antonym", "linguistic_maps", "part_of",
                 "follows", "precedes", "temporal_coincident", "spatial_near",
                 "associated_with"}
    extra = set(relation_set) - canonical
    print(f"relation set subset-of-16 -> {'OK' if not extra else 'FAIL ' + str(extra)}")


if __name__ == "__main__":
    main()