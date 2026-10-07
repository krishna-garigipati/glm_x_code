"""Build history_civilizations.db: tester-c History & Civilizations graph (Phase G).

Training-domain disjoint vocabulary. Targets >= 13 canonical relations,
~81 edges, incl. follows / precedes / temporal_coincident.

Usage:
    python test_results/tester-c/datasets/build_history_civilizations.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "history_civilizations.db"

CONCEPTS = [
    "ancient rome", "roman empire", "roman republic", "ancient greece", "athens",
    "sparta", "ancient egypt", "mesopotamia", "babylon", "persian empire",
    "byzantine empire", "aztec empire", "inca empire", "mongol empire",
    "ottoman empire", "han dynasty", "carthage", "phoenicia", "rome",
    "stone age", "bronze age", "iron age", "middle ages", "renaissance",
    "industrial revolution", "classical antiquity", "historical era",
    "empire", "republic", "city-state", "kingdom", "monarchy", "democracy",
    "autocracy", "government", "ruler", "emperor", "pharaoh", "king", "senator",
    "gladiator", "warrior", "scribe", "philosopher", "politician", "soldier", "democratic",
    "pyramid", "colosseum", "acropolis", "parthenon", "great wall", "temple",
    "monument", "aqueduct", "arena", "senate",
    "invention of writing", "recorded history", "invention of the wheel",
    "transportation", "silk road", "trade routes", "spread of disease",
    "food surplus", "cities", "myth", "historical record", "hieroglyphs",
    "understanding of egypt", "artifacts", "ancient world",
    "hammurabi's code", "law code", "julius caesar", "alexander the great",
    "cleopatra", "ramses", "socrates", "first olympic games", "founding of rome",
    "printing press", "camel",
    "ancient", "modern", "powerful", "warlike", "peace", "war", "long",
    "polis", "kaiser", "tsar", "rex",
]

EDGES = [
    # --- is_a taxonomy ---
    {"source": "roman empire", "target": "empire", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "persian empire", "target": "empire", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "byzantine empire", "target": "empire", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "mongol empire", "target": "empire", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "ottoman empire", "target": "empire", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "empire", "target": "government", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "roman republic", "target": "republic", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "republic", "target": "government", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "monarchy", "target": "government", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "democracy", "target": "government", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "autocracy", "target": "government", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "athens", "target": "city-state", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "sparta", "target": "city-state", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "carthage", "target": "city-state", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "city-state", "target": "government", "relation": "is_a", "strength": 0.85, "confidence": 0.85},
    {"source": "pharaoh", "target": "ruler", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "emperor", "target": "ruler", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "king", "target": "ruler", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "senator", "target": "politician", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "gladiator", "target": "warrior", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "soldier", "target": "warrior", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "pyramid", "target": "monument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "colosseum", "target": "monument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "great wall", "target": "monument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "parthenon", "target": "temple", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "temple", "target": "monument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "stone age", "target": "historical era", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "bronze age", "target": "historical era", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "iron age", "target": "historical era", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "middle ages", "target": "historical era", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "renaissance", "target": "historical era", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    # --- part_of (source=part, target=whole) ---
    {"source": "athens", "target": "ancient greece", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "sparta", "target": "ancient greece", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "rome", "target": "roman empire", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "acropolis", "target": "athens", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "parthenon", "target": "acropolis", "relation": "part_of", "strength": 0.95, "confidence": 0.95},
    {"source": "colosseum", "target": "rome", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "senate", "target": "roman republic", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    # --- has_property ---
    {"source": "ancient rome", "target": "powerful", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "mongol empire", "target": "powerful", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "sparta", "target": "warlike", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "athens", "target": "democratic", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "ancient egypt", "target": "ancient", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "pyramid", "target": "ancient", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "silk road", "target": "long", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    # --- causes / caused_by ---
    {"source": "invention of writing", "target": "recorded history", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "invention of the wheel", "target": "transportation", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "trade routes", "target": "spread of disease", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "food surplus", "target": "cities", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "recorded history", "target": "invention of writing", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    {"source": "spread of disease", "target": "trade routes", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    # --- follows (source BEFORE target; "what comes after X" -> follows, seed X) ---
    {"source": "stone age", "target": "bronze age", "relation": "follows", "strength": 0.95, "confidence": 0.95},
    {"source": "bronze age", "target": "iron age", "relation": "follows", "strength": 0.95, "confidence": 0.95},
    {"source": "iron age", "target": "middle ages", "relation": "follows", "strength": 0.9, "confidence": 0.9},
    {"source": "middle ages", "target": "renaissance", "relation": "follows", "strength": 0.9, "confidence": 0.9},
    {"source": "renaissance", "target": "industrial revolution", "relation": "follows", "strength": 0.9, "confidence": 0.9},
    # --- precedes (source AFTER target; "what comes before X" -> precedes, seed X) ---
    {"source": "bronze age", "target": "stone age", "relation": "precedes", "strength": 0.95, "confidence": 0.95},
    {"source": "iron age", "target": "bronze age", "relation": "precedes", "strength": 0.95, "confidence": 0.95},
    {"source": "middle ages", "target": "iron age", "relation": "precedes", "strength": 0.9, "confidence": 0.9},
    {"source": "renaissance", "target": "middle ages", "relation": "precedes", "strength": 0.9, "confidence": 0.9},
    {"source": "industrial revolution", "target": "renaissance", "relation": "precedes", "strength": 0.9, "confidence": 0.9},
    # --- antonym ---
    {"source": "ancient", "target": "modern", "relation": "antonym", "strength": 0.98, "confidence": 0.98},
    {"source": "war", "target": "peace", "relation": "antonym", "strength": 0.98, "confidence": 0.98},
    {"source": "democracy", "target": "autocracy", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    # --- synonym ---
    {"source": "monarchy", "target": "kingdom", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "city-state", "target": "polis", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    # --- example_of ---
    {"source": "pyramid", "target": "monument", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "gladiator", "target": "warrior", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "socrates", "target": "philosopher", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "hammurabi's code", "target": "law code", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    # --- linguistic_maps ---
    {"source": "emperor", "target": "kaiser", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    {"source": "emperor", "target": "tsar", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    {"source": "king", "target": "rex", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    # --- supports ---
    {"source": "hieroglyphs", "target": "understanding of egypt", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    {"source": "artifacts", "target": "ancient world", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    # --- contradicts ---
    {"source": "myth", "target": "historical record", "relation": "contradicts", "strength": 0.85, "confidence": 0.85},
    # --- temporal_coincident ---
    {"source": "printing press", "target": "renaissance", "relation": "temporal_coincident", "strength": 0.9, "confidence": 0.9},
    # --- associated_with ---
    {"source": "camel", "target": "silk road", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "pharaoh", "target": "pyramid", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "gladiator", "target": "colosseum", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "senator", "target": "senate", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "socrates", "target": "athens", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
]


def main() -> None:
    from _build_common import build_db
    build_db(OUT, CONCEPTS, EDGES, dataset_name="history_civilizations",
             min_relations=13, min_edges=60)


if __name__ == "__main__":
    main()