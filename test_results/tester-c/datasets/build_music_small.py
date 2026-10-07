"""Build music_small.db: tester-c Music & Instruments graph (Phase G).

Training-domain disjoint vocabulary. Targets >= 11 canonical relations,
~73 edges, incl. linguistic_maps (Italian) + temporal_coincident.

Usage:
    python test_results/tester-c/datasets/build_music_small.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "music_small.db"

CONCEPTS = [
    "guitar", "electric guitar", "acoustic guitar", "piano", "keyboard",
    "violin", "cello", "double bass", "harp", "trumpet", "trombone", "saxophone",
    "clarinet", "flute", "oboe", "bassoon", "drum", "cymbal", "marimba",
    "string instrument", "wind instrument", "brass instrument", "woodwind instrument",
    "percussion instrument", "keyboard instrument", "musical instrument",
    "strings", "fretboard", "bridge", "neck", "bow", "reed", "mouthpiece",
    "valve", "drumhead", "keys", "pedal", "bell",
    "rhythm", "melody", "harmony", "tempo", "chord", "note", "scale", "octave",
    "pitch", "beat", "verse", "chorus", "tune",
    "classical", "jazz", "rock", "blues", "folk", "pop", "electronic", "opera",
    "hip hop", "country", "music genre",
    "band", "orchestra", "symphony", "choir", "quartet",
    "musician", "composer", "conductor", "singer", "drummer", "guitarist",
    "lyricist", "violinist", "sheet music", "concert", "album", "studio", "stage",
    "loud", "soft", "fast", "slow", "high", "low", "sharp", "flat",
    "hearing loss", "practice", "improvement", "excessive volume", "song", "track",
    "violino", "chitarra", "tamburo", "crescendo", "climax", "versatile",
]

EDGES = [
    # --- is_a taxonomy ---
    {"source": "electric guitar", "target": "guitar", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "acoustic guitar", "target": "guitar", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "guitar", "target": "string instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "violin", "target": "string instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "cello", "target": "string instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "double bass", "target": "string instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "harp", "target": "string instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "clarinet", "target": "woodwind instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "flute", "target": "woodwind instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "oboe", "target": "woodwind instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "bassoon", "target": "woodwind instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "saxophone", "target": "woodwind instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "trumpet", "target": "brass instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "trombone", "target": "brass instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "drum", "target": "percussion instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "cymbal", "target": "percussion instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "marimba", "target": "percussion instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "piano", "target": "keyboard instrument", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "keyboard", "target": "keyboard instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "string instrument", "target": "musical instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "woodwind instrument", "target": "musical instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "brass instrument", "target": "musical instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "percussion instrument", "target": "musical instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "keyboard instrument", "target": "musical instrument", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "classical", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "jazz", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "rock", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "blues", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "folk", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "pop", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "electronic", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "opera", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "hip hop", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "country", "target": "music genre", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    # --- part_of (source=part, target=whole) ---
    {"source": "bridge", "target": "guitar", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "neck", "target": "guitar", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "strings", "target": "violin", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "bow", "target": "violin", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "reed", "target": "clarinet", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "mouthpiece", "target": "trumpet", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "valve", "target": "trumpet", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "drumhead", "target": "drum", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "keys", "target": "piano", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "pedal", "target": "piano", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "bell", "target": "trumpet", "relation": "part_of", "strength": 0.85, "confidence": 0.85},
    # --- has_property ---
    {"source": "drum", "target": "loud", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "cello", "target": "low", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "flute", "target": "high", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "violin", "target": "soft", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "piano", "target": "versatile", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "guitar", "target": "versatile", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    # --- causes / caused_by ---
    {"source": "excessive volume", "target": "hearing loss", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "practice", "target": "improvement", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "hearing loss", "target": "excessive volume", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    # --- antonym ---
    {"source": "loud", "target": "soft", "relation": "antonym", "strength": 0.98, "confidence": 0.98},
    {"source": "fast", "target": "slow", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "high", "target": "low", "relation": "antonym", "strength": 0.98, "confidence": 0.98},
    {"source": "sharp", "target": "flat", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    # --- synonym ---
    {"source": "melody", "target": "tune", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    {"source": "song", "target": "track", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    # --- example_of ---
    {"source": "trumpet", "target": "brass instrument", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "drum", "target": "percussion instrument", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "jazz", "target": "music genre", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    # --- linguistic_maps ---
    {"source": "guitar", "target": "chitarra", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    {"source": "violin", "target": "violino", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    {"source": "drum", "target": "tamburo", "relation": "linguistic_maps", "strength": 0.9, "confidence": 0.9},
    # --- temporal_coincident ---
    {"source": "crescendo", "target": "climax", "relation": "temporal_coincident", "strength": 0.85, "confidence": 0.85},
    # --- associated_with ---
    {"source": "guitar", "target": "rock", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "saxophone", "target": "jazz", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "conductor", "target": "orchestra", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "singer", "target": "opera", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "composer", "target": "classical", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "drummer", "target": "band", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
]


def main() -> None:
    from _build_common import build_db
    build_db(OUT, CONCEPTS, EDGES, dataset_name="music_small",
             min_relations=11, min_edges=60)


if __name__ == "__main__":
    main()