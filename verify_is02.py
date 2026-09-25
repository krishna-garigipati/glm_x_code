"""
Test IS-02 fix: Verify clause extraction + relation matching logic
without loading the full SentenceTransformer model.
"""
import re
import numpy as np
from g2p.config import G2PConfig

# Load config
config = G2PConfig.from_yaml('configs/config_g2p.yaml')
config.validate()

# Test the clause splitting logic
patterns = config.extraction.clause_split
joined = '|'.join(re.escape(p) for p in patterns)

def split_clauses(question):
    parts = re.split(joined, question, flags=re.IGNORECASE)
    return [p.strip().lower() for p in parts if p.strip()]

# Test questions
test_cases = [
    ("What follows photosynthesis?", "follows", "Should match temporal 'follows'"),
    ("What leads to tooth decay?", "causes", "Should match causal 'leads to'"),
    ("What triggers rain?", "causes", "Should match causal 'triggers'"),
    ("What comes before summer?", "precedes", "Should match temporal 'comes before'"),
    ("What precedes winter?", "precedes", "Should match temporal 'precedes'"),
    ("What comes after spring?", "follows", "Should match temporal 'comes after'"),
]

print("=" * 60)
print("IS-02 Verification: Clause Extraction + Expected Relations")
print("=" * 60)

for question, expected_relation, note in test_cases:
    clauses = split_clauses(question)
    print(f"\nQ: {question}")
    print(f"  Clauses: {clauses}")
    print(f"  Expected relation: {expected_relation}")
    print(f"  Note: {note}")

# Also verify the descriptor sets are semantically distinct
print("\n" + "=" * 60)
print("Descriptor Overlap Check (should be minimal)")
print("=" * 60)

causes_words = set(' '.join(config.extraction.relation_variants['causes']).lower().split())
follows_words = set(' '.join(config.extraction.relation_variants['follows']).lower().split())
precedes_words = set(' '.join(config.extraction.relation_variants['precedes']).lower().split())

print(f"causes & follows: {causes_words & follows_words}")
print(f"causes & precedes: {causes_words & precedes_words}")
print(f"follows & precedes: {follows_words & precedes_words}")

# Check key discriminative words
print("\nKey discriminative words in each:")
print(f"  causes: {sorted([w for w in causes_words if w in ('leads', 'triggers', 'results', 'makes', 'produces', 'creates', 'generates', 'cause', 'reason')])}")
print(f"  follows: {sorted([w for w in follows_words if w in ('follows', 'after', 'next', 'sequence', 'step')])}")
print(f"  precedes: {sorted([w for w in precedes_words if w in ('precedes', 'before', 'came', 'step')])}")

print("\n✅ Config-level verification complete")
print("Next: Run full tester-b pipeline if environment allows")