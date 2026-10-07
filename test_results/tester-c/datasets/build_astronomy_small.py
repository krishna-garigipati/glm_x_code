"""Build astronomy_small.db: tester-c Astronomy/Space graph (Phase G, cross-domain).

Zero vocabulary overlap with the food/weather/geo/zoology/bio test tiers. Targets
>= 10 distinct canonical relations and ~78 edges (scale-shaped probe graph).

Usage:
    python test_results/tester-c/datasets/build_astronomy_small.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "astronomy_small.db"

CONCEPTS = [
    "celestial body", "cosmic object", "planet", "terrestrial planet", "gas giant",
    "ice giant", "dwarf planet", "star", "main sequence star", "red dwarf",
    "sun", "moon", "natural satellite", "solar system", "asteroid", "asteroid belt",
    "comet", "halley's comet", "meteor", "milky way", "spiral galaxy", "galaxy cluster",
    "galaxy", "universe", "nebula", "supernova", "black hole",
    "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune",
    "orbit", "gravity", "tides", "seasons", "heat", "light", "earth tilt",
    "big bang", "universe expansion", "contraction", "redshift", "dark matter",
    "eclipse", "solar eclipse", "lunar eclipse",
    "telescope", "observatory", "space station", "astronaut", "rocket",
    "hubble telescope", "mars rover", "space probe", "evidence of water",
    "astronomy", "space exploration", "stella", "lune",
    "hot", "cold", "red", "bright", "dark", "gaseous", "rocky", "luminous",
    "craters", "rings", "aurora", "daystar",
]

EDGES = [
    # --- is_a taxonomy ---
    {"source": "mercury", "target": "terrestrial planet", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "venus", "target": "terrestrial planet", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "earth", "target": "terrestrial planet", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "mars", "target": "terrestrial planet", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "jupiter", "target": "gas giant", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "saturn", "target": "gas giant", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "uranus", "target": "ice giant", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "neptune", "target": "ice giant", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "terrestrial planet", "target": "planet", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "gas giant", "target": "planet", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "ice giant", "target": "planet", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "dwarf planet", "target": "planet", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "planet", "target": "celestial body", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "star", "target": "celestial body", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "comet", "target": "celestial body", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "asteroid", "target": "celestial body", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "meteor", "target": "celestial body", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "nebula", "target": "celestial body", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "sun", "target": "main sequence star", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "main sequence star", "target": "star", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "red dwarf", "target": "star", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "milky way", "target": "spiral galaxy", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "spiral galaxy", "target": "galaxy", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "halley's comet", "target": "comet", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "hubble telescope", "target": "telescope", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "mars rover", "target": "space probe", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    # --- part_of (source=part, target=whole) ---
    {"source": "sun", "target": "solar system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "moon", "target": "solar system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "earth", "target": "solar system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "asteroid belt", "target": "solar system", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "solar system", "target": "milky way", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "milky way", "target": "galaxy cluster", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "galaxy cluster", "target": "universe", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "rings", "target": "saturn", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "craters", "target": "moon", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    {"source": "aurora", "target": "earth", "relation": "part_of", "strength": 0.8, "confidence": 0.8},
    # --- has_property ---
    {"source": "mars", "target": "red", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "sun", "target": "hot", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "jupiter", "target": "gaseous", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "earth", "target": "rocky", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "black hole", "target": "dark", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "sun", "target": "bright", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "star", "target": "luminous", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "moon", "target": "cold", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    # --- causes / caused_by ---
    {"source": "gravity", "target": "orbit", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "moon", "target": "tides", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "sun", "target": "heat", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "sun", "target": "light", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "earth tilt", "target": "seasons", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "big bang", "target": "universe expansion", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "orbit", "target": "gravity", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    {"source": "tides", "target": "moon", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    {"source": "seasons", "target": "earth tilt", "relation": "caused_by", "strength": 0.8, "confidence": 0.8},
    {"source": "universe expansion", "target": "big bang", "relation": "caused_by", "strength": 0.8, "confidence": 0.8},
    # --- spatial_near ---
    {"source": "sun", "target": "mercury", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    {"source": "asteroid belt", "target": "jupiter", "relation": "spatial_near", "strength": 0.8, "confidence": 0.8},
    {"source": "earth", "target": "moon", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    {"source": "space station", "target": "earth", "relation": "spatial_near", "strength": 0.85, "confidence": 0.85},
    # --- supports ---
    {"source": "redshift", "target": "big bang", "relation": "supports", "strength": 0.85, "confidence": 0.85},
    {"source": "hubble telescope", "target": "universe expansion", "relation": "supports", "strength": 0.85, "confidence": 0.85},
    {"source": "mars rover", "target": "evidence of water", "relation": "supports", "strength": 0.85, "confidence": 0.85},
    # --- example_of ---
    {"source": "jupiter", "target": "gas giant", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "halley's comet", "target": "comet", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "hubble telescope", "target": "telescope", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    # --- antonym ---
    {"source": "hot", "target": "cold", "relation": "antonym", "strength": 0.98, "confidence": 0.98},
    {"source": "bright", "target": "dark", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "universe expansion", "target": "contraction", "relation": "antonym", "strength": 0.85, "confidence": 0.85},
    # --- synonym ---
    {"source": "daystar", "target": "sun", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "celestial body", "target": "cosmic object", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    # --- linguistic_maps ---
    {"source": "star", "target": "stella", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    {"source": "moon", "target": "lune", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    # --- associated_with ---
    {"source": "astronaut", "target": "space station", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "telescope", "target": "astronomy", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "astronaut", "target": "space exploration", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "mars rover", "target": "mars", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "black hole", "target": "dark matter", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
]


def main() -> None:
    from _build_common import build_db
    build_db(OUT, CONCEPTS, EDGES, dataset_name="astronomy_small",
             min_relations=10, min_edges=60)


if __name__ == "__main__":
    main()