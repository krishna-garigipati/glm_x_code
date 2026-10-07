"""Probe candidate Stage A questions for the chain the planner actually extracts.

Run BEFORE freezing the question set so the frozen set is not built on
phrasings that were never checked. Read-only.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from g2p.config import G2PConfig
from g2p.g2p_planner import QueryRelationExtractor

cfg = G2PConfig.from_yaml(str(ROOT / "configs" / "config_g2p.yaml"))
cfg.validate()
p = QueryRelationExtractor(cfg)
p._initialized = True
p._variant_embeddings = {}
p._best_relation_for_clause = lambda c: (None, 0.0)  # force literal-only, fast

MULTIHOP = [
    "What is the fin part of, and what is a fish?",
    "What is the gill part of, and what is a fish?",
    "What is the tail part of, and what is a dog?",
    "What is the wing part of, and what is a bird?",
    "What is the trunk part of, and what is a tree?",
    "What is ice part of, and what is water?",
    "What causes sunlight, and what is associated with photosynthesis?",
    "What is the fin part of, and what is associated with fish?",
    "What is the wing part of, and what is associated with bird?",
    "What is the trunk part of, and what is a tree?",
]

# pre-fix these mapped to the relation whose name reads naturally in English
# but walks the opposite way from the anchor. After the swap the "comes after"
# family must resolve to `precedes` and the "comes before" family to `follows`.
TEMPORAL = [
    ("Which season comes after summer?", ["precedes"]),
    ("Which season comes after spring?", ["precedes"]),
    ("Which season comes after autumn?", ["precedes"]),
    ("Which season happens next after spring?", ["precedes"]),
    ("Which season comes before summer?", ["follows"]),
    ("Which season comes before winter?", ["follows"]),
    ("Which season comes before autumn?", ["follows"]),
    ("Which season happened before summer?", ["follows"]),
]

ONEHOP = [
    ("What is a dog?", ["is_a"]),
    ("What is a salmon?", ["is_a"]),
    ("What is the fin a part of?", ["part_of"]),
    ("What causes lightning?", ["causes"]),
    ("What is thunder caused by?", ["caused_by"]),
    ("What comes after summer?", ["precedes"]),
    ("What comes before winter?", ["follows"]),
    ("What is the opposite of cold?", ["antonym"]),
    ("What is another word for doctor?", ["synonym"]),
    ("What evidence supports fossils?", ["supports"]),
    ("What contradicts robin?", ["contradicts"]),
    ("What is associated with rain?", ["associated_with"]),
    ("What property does honey have?", ["has_property"]),
    ("Give me an example of ice?", ["example_of"]),
    ("What do you call a farm in Spanish?", ["linguistic_maps"]),
    ("What is near river?", ["spatial_near"]),
    ("What occurs during autumn?", ["temporal_coincident"]),
]

print("=== MULTI-HOP CANDIDATES (want len(chain)==2) ===")
ok = 0
for q in MULTIHOP:
    pl = p.extract(q)
    good = len(pl.relation_chain or []) == 2
    ok += good
    print(f"  [{'OK ' if good else 'BAD'}] {pl.relation_chain}  heur={pl.heuristic_fallback_used}  {q}")
print(f"multi-hop usable: {ok}/{len(MULTIHOP)}")

print("\n=== ONE-HOP CANDIDATES (want chain==expected) ===")
bad = 0
for q, want in ONEHOP:
    pl = p.extract(q)
    got = list(pl.relation_chain or [])
    good = got == want
    bad += (not good)
    print(f"  [{'OK ' if good else 'BAD'}] got={got} want={want} heur={pl.heuristic_fallback_used}  {q}")
print(f"one-hop mismatches: {bad}/{len(ONEHOP)}")

print("\n=== TEMPORAL DIRECTION (post-swap) ===")
tbad = 0
for q, want in TEMPORAL:
    pl = p.extract(q)
    got = list(pl.relation_chain or [])
    good = got == want
    tbad += (not good)
    print(f"  [{'OK ' if good else 'BAD'}] got={got} want={want}  {q}")
print(f"temporal mismatches: {tbad}/{len(TEMPORAL)}")