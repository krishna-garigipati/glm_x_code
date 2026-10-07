"""Build computing_small.db: tester-c Computers & Internet graph (Phase G).

Training-domain disjoint vocabulary. Targets >= 10 canonical relations,
~81 edges.

Usage:
    python test_results/tester-c/datasets/build_computing_small.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "computing_small.db"

CONCEPTS = [
    "computer", "personal computer", "laptop", "desktop", "tablet", "smartphone",
    "notebook", "workstation", "server", "mainframe", "supercomputer",
    "hardware", "software", "processor", "cpu", "gpu", "graphics card", "chip",
    "motherboard", "ram", "memory", "hard drive", "storage", "solid state drive",
    "keyboard", "mouse", "monitor", "screen", "printer", "webcam", "battery",
    "operating system", "windows", "linux", "android", "macos", "browser",
    "chrome", "firefox", "safari", "application", "app", "program", "source code",
    "code", "compiler", "debugger", "bug", "glitch",
    "network", "internet", "router", "wifi", "web server", "cloud",
    "cloud storage", "malware", "virus", "ransomware", "phishing", "firewall",
    "encryption", "encrypted", "plain text", "privacy", "security", "malicious",
    "password", "data", "database", "programmer", "software engineer", "developer",
    "hacker", "user", "malware infection", "data loss", "software crash",
    "system failure", "overheating", "shutdown", "cracked software",
    "outdated software", "security breach", "video game",
    "fast", "slow", "powerful", "reliable", "portable", "secure", "dangerous",
    "vulnerable", "online", "offline",
]

EDGES = [
    # --- is_a taxonomy ---
    {"source": "laptop", "target": "personal computer", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "desktop", "target": "personal computer", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "tablet", "target": "personal computer", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "smartphone", "target": "personal computer", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "personal computer", "target": "computer", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "notebook", "target": "laptop", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "workstation", "target": "computer", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "server", "target": "computer", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "mainframe", "target": "computer", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "supercomputer", "target": "computer", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "cpu", "target": "processor", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "gpu", "target": "processor", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "processor", "target": "hardware", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "chip", "target": "hardware", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "motherboard", "target": "hardware", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "ram", "target": "hardware", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "hard drive", "target": "hardware", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "keyboard", "target": "hardware", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "monitor", "target": "hardware", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "printer", "target": "hardware", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "operating system", "target": "software", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "windows", "target": "operating system", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "linux", "target": "operating system", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "android", "target": "operating system", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "macos", "target": "operating system", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "browser", "target": "software", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "chrome", "target": "browser", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "firefox", "target": "browser", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "safari", "target": "browser", "relation": "is_a", "strength": 0.95, "confidence": 0.95},
    {"source": "application", "target": "software", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "program", "target": "software", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "compiler", "target": "software", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    {"source": "debugger", "target": "software", "relation": "is_a", "strength": 0.9, "confidence": 0.9},
    # --- part_of (source=part, target=whole) ---
    {"source": "keyboard", "target": "laptop", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "screen", "target": "smartphone", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "battery", "target": "smartphone", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "ram", "target": "computer", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "motherboard", "target": "computer", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "cpu", "target": "computer", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "hard drive", "target": "computer", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "gpu", "target": "graphics card", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    {"source": "graphics card", "target": "computer", "relation": "part_of", "strength": 0.9, "confidence": 0.9},
    # --- has_property ---
    {"source": "smartphone", "target": "portable", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "laptop", "target": "portable", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "supercomputer", "target": "fast", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "mainframe", "target": "reliable", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "server", "target": "reliable", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "computer", "target": "powerful", "relation": "has_property", "strength": 0.85, "confidence": 0.85},
    {"source": "firewall", "target": "secure", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "encryption", "target": "secure", "relation": "has_property", "strength": 0.9, "confidence": 0.9},
    {"source": "virus", "target": "malicious", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    {"source": "ransomware", "target": "dangerous", "relation": "has_property", "strength": 0.95, "confidence": 0.95},
    # --- causes / caused_by ---
    {"source": "virus", "target": "malware infection", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "phishing", "target": "data loss", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "bug", "target": "software crash", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "overheating", "target": "shutdown", "relation": "causes", "strength": 0.9, "confidence": 0.9},
    {"source": "malware", "target": "system failure", "relation": "causes", "strength": 0.85, "confidence": 0.85},
    {"source": "malware infection", "target": "virus", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    {"source": "data loss", "target": "phishing", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    {"source": "software crash", "target": "bug", "relation": "caused_by", "strength": 0.85, "confidence": 0.85},
    {"source": "system failure", "target": "malware", "relation": "caused_by", "strength": 0.8, "confidence": 0.8},
    # --- antonym ---
    {"source": "online", "target": "offline", "relation": "antonym", "strength": 0.98, "confidence": 0.98},
    {"source": "encrypted", "target": "plain text", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    {"source": "fast", "target": "slow", "relation": "antonym", "strength": 0.95, "confidence": 0.95},
    {"source": "secure", "target": "vulnerable", "relation": "antonym", "strength": 0.9, "confidence": 0.9},
    # --- synonym ---
    {"source": "program", "target": "application", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    {"source": "bug", "target": "glitch", "relation": "synonym", "strength": 0.95, "confidence": 0.95},
    {"source": "laptop", "target": "notebook", "relation": "synonym", "strength": 0.9, "confidence": 0.9},
    # --- example_of ---
    {"source": "chrome", "target": "browser", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "ram", "target": "hardware", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    {"source": "windows", "target": "operating system", "relation": "example_of", "strength": 0.9, "confidence": 0.9},
    # --- supports ---
    {"source": "firewall", "target": "security", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    {"source": "encryption", "target": "privacy", "relation": "supports", "strength": 0.9, "confidence": 0.9},
    # --- contradicts ---
    {"source": "cracked software", "target": "security breach", "relation": "contradicts", "strength": 0.9, "confidence": 0.9},
    {"source": "outdated software", "target": "security breach", "relation": "contradicts", "strength": 0.85, "confidence": 0.85},
    # --- associated_with ---
    {"source": "router", "target": "wifi", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "hacker", "target": "malware", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "programmer", "target": "source code", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "cloud", "target": "cloud storage", "relation": "associated_with", "strength": 0.9, "confidence": 0.9},
    {"source": "gpu", "target": "video game", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
    {"source": "user", "target": "password", "relation": "associated_with", "strength": 0.85, "confidence": 0.85},
]


def main() -> None:
    from _build_common import build_db
    build_db(OUT, CONCEPTS, EDGES, dataset_name="computing_small",
             min_relations=10, min_edges=60)


if __name__ == "__main__":
    main()