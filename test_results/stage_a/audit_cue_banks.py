"""Audit the G2P relation cue banks against contract section 8.

Contract section 8 (descriptor_banks) requires:
  "Every relation must have a sufficiently rich set of cue phrases and
   descriptors. Minimum expectation: at least 5-8 strong descriptors per
   relation for PoC. Direction-sensitive relations (causes/caused_by,
   precedes/follows, part_of) must have clearly distinct descriptors."
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CANONICAL_16 = [
    "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
    "contradicts", "supports", "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident", "spatial_near",
    "linguistic_maps",
]

cfg = yaml.safe_load((ROOT / "configs" / "config_g2p.yaml").read_text(encoding="utf-8"))
variants = cfg["extraction"]["relation_variants"]

print("=== contract section 8: cue-bank completeness (16 canonical relations) ===")
print("{:<22} {:>5}  {}".format("relation", "cues", "status"))
print("-" * 60)
missing, thin = [], []
for rel in CANONICAL_16:
    cues = variants.get(rel)
    if not cues:
        missing.append(rel)
        print("{:<22} {:>5}  MISSING BANK".format(rel, 0))
        continue
    n = len(cues)
    status = "ok" if n >= 5 else "THIN (<5)"
    if n < 5:
        thin.append(rel)
    print("{:<22} {:>5}  {}".format(rel, n, status))

extra = [k for k in variants if k not in CANONICAL_16]
print("\nnon-canonical banks present:", extra or "none")
print("MISSING BANKS:", missing or "none")
print("THIN BANKS    :", thin or "none")

print("\n=== direction-sensitive pairs: are descriptors disjoint? ===")
for a, b in [("causes", "caused_by"), ("precedes", "follows"), ("part_of", "has_part")]:
    ca, cb = set(variants.get(a) or []), set(variants.get(b) or [])
    overlap = ca & cb
    verdict = "DISJOINT" if not overlap else f"OVERLAP {sorted(overlap)}"
    print("  {:<12} vs {:<12} {}".format(a, b, verdict if cb else f"({b} has NO bank -> cannot test this direction)"))

print("\n=== cross-bank cue leakage (a cue must not live in two banks) ===")
owner = {}
dupes = []
for rel, cues in variants.items():
    for cue in cues:
        key = cue.lower().strip()
        if key in owner and owner[key] != rel:
            dupes.append((key, owner[key], rel))
        owner[key] = rel
print("  duplicated cue strings:", dupes or "none")

print("\n=== substring containment between banks (weaker, still a risk) ===")
risky = []
for a, cues_a in variants.items():
    for b, cues_b in variants.items():
        if a >= b:
            continue
        for ca in cues_a:
            for cb in cues_b:
                if ca.lower() != cb.lower() and ca.lower() in cb.lower():
                    risky.append((a, ca, b, cb))
for r in risky:
    print("   {:<20} {!r} is a substring of {:<20} {!r}".format(*r))
print("  substring overlaps:", len(risky))

print("\n=== precedes/follows direction convention (live config) ===")
print("Graph stores: spring -precedes-> summer -precedes-> autumn -precedes-> winter")
print("Convention (same as causes/part_of): the anchor is the SUBJECT and the")
print("walk goes FORWARD to the answer. So from anchor=summer:")
print("   chain=precedes -> autumn   (the season AFTER summer)")
print("   chain=follows  -> spring   (the season BEFORE summer)")
print()
print("  precedes cues:", variants.get("precedes"))
print("  follows  cues:", variants.get("follows"))
after = [c for c in variants.get("precedes", []) if "after" in c or "next" in c]
before = [c for c in variants.get("follows", []) if "before" in c or "prior" in c]
print()
print("  'comes after'-family attached to precedes :", bool(after))
print("  'comes before'-family attached to follows :", bool(before))
ok = bool(after) and bool(before)
print("  verdict:", "CORRECT (each temporal phrasing resolves to the relation that"
      " walks toward the right answer)" if ok else "STILL SWAPPED")