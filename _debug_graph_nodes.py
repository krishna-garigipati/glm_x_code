import sys, os
sys.path.insert(0, '.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.disable(logging.WARNING)

from scripts.kb_loader import KGBuilderPipeline, KGBuilderConfig
from scripts.evaluate_full_pipeline import TRUE_FACTS

config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=False)
pipeline = KGBuilderPipeline(config)
corpus = "\n".join(TRUE_FACTS)
result = pipeline.process_corpus(corpus.split("\n"), show_progress=False)
gd = result["graph_data"]

all_labels = list(gd['id_to_label'].values())
print("=== CHECKING WHICH ENTITIES EXIST AS NODES ===")

checks = [
    'alexander fleming', 'penicillin', 'fleming',
    'smoking', 'lung cancer', 'causes', 'cause',
    'cpu', 'the cpu', 'computer', 'a computer', 'part of a computer',
    'hot', 'cold', 'light', 'darkness',
    'paper', 'china',
    'a diamond', 'diamond', 'diamonds', 'a gemstone', 'gemstone',
]

for check in checks:
    found = [l for l in all_labels if check in l.lower()]
    if found:
        print(f"  {check:25s} -> FOUND: {found}")
    else:
        print(f"  {check:25s} -> NOT FOUND in graph nodes")

print("\n=== ALL EDGES ===")
for e in gd['edges']:
    src = gd['id_to_label'][e['source']]
    tgt = gd['id_to_label'][e['target']]
    print(f"  {src:35s} --[{e['relation']:20s}]--> {tgt}")
