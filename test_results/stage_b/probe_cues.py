"""Probe G2P chain extraction for the Stage B phrasings that failed.

Stage B's first run produced 13 failures, most of them because a cue that reads
correctly in English is not literally present once the clause splitter has
consumed "what". This prints the extracted chain for each candidate rewording
so the frozen question set can be fixed from evidence instead of guesswork.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

DB = Path(__file__).resolve().parent / "relation_coverage.db"

CANDIDATES = [
    # --- has_property: "is known for" must stay ADJACENT after the split ---
    ("What is honey known for?", "honey"),
    ("What is known for ice?", "ice"),
    ("What is known for rain?", "rain"),
    ("What is known for the ant?", "ant"),
    ("What is known for the whale?", "whale"),
    # --- is_a with an article-less subject ---
    ("What is water?", "water"),
    ("What type of thing is water?", "water"),
    ("What type of liquid is water?", "water"),
    # --- caused_by without the colliding "is a" fragment ---
    ("What is a flood caused by?", "flood"),
    ("What is the flood caused by?", "flood"),
    ("What is the crop failure caused by?", "crop failure"),
    # --- 2-hop chains ---
    ("What is the tail part of, and what is known for the whale?", "tail"),
    ("What is a dog, and what is known for a dog?", "dog"),
    ("What is a dog, and what is known for the dog?", "dog"),
    ("What is the fin part of, and what is a shark?", "fin"),
    # --- example_of forward ---
    ("What is the hailstone an example of?", "hailstone"),
    ("What is the robin an example of?", "robin"),
    # --- supports / contradicts ---
    # The cue bank lists "supports" (plural), so "What does X support?" does not
    # match it and the extractor falls back to has_property. "substantiates" is
    # a stored-direction cue, and the reverse cues ("is supported by", "backed
    # by", "evidence for") would all walk the mirror and render a false claim.
    ("What does the fossil record support?", "fossil record"),
    ("What does the fossil record substantiate?", "fossil record"),
    ("What claim does the fossil record substantiate?", "fossil record"),
    ("What is supported by the fossil record?", "fossil record"),
    ("What is backed by the fossil record?", "fossil record"),
    ("What is backed by the dna record?", "dna record"),
    ("What is the evidence for the deep time claim?", "deep time claim"),
    ("What contradicts the deep time claim?", "deep time claim"),
    # --- temporal_coincident reverse ---
    ("What occurs during autumn?", "autumn"),
    # --- linguistic_maps reverse ---
    ("What do you call a dog in Spanish?", "dog"),
    # --- spatial_near / part_of ---
    ("What is near river?", "river"),
    ("What is the cloud part of?", "cloud"),
]

pipeline = GLMXPipeline()
pipeline._seed = 0
pipeline._no_learning = True
pipeline._measure = True
pipeline.graph_store = SQLiteGraphStore.load_state(str(DB))
pipeline.load_models()

print(f"{'question':<54} {'anchor':<22} {'chain':<26} answer")
print("-" * 150)
for q, want_anchor in CANDIDATES:
    res = pipeline.ask(q)
    chain = res.get("relation_chain")
    anchor = (res.get("selected_anchor") or {}).get("label")
    flag = "" if anchor == want_anchor else f"  <-- anchor {want_anchor!r}"
    print(f"{q:<54} {str(anchor):<22} {str(chain):<26} "
          f"{(res.get('answer') or '')[:44]}{flag}")
    print(f"{'':<54} edges={res.get('walk_path_edges')} labels={res.get('walk_path_labels')}")