import json
import sqlite3

import numpy as np

c = sqlite3.connect("test_results/stage_a/toy_graph.db")
meta = dict(c.execute("select key,value from metadata").fetchall())
journal = json.loads(meta.get("merge_journal", "[]"))
print("merge_journal entries:", len(journal))
for e in journal:
    print("   keep={keep!r} drop={drop!r} sim={similarity}".format(**e))

print()
print("cosine similarity of the merged pair (bge-small):")
try:
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    enc = sbert.encode(["dog", "animal", "cat", "mammal"], normalize_embeddings=True)
    enc = np.asarray(enc, dtype=np.float32)
    for i, a in enumerate(["dog", "animal", "cat", "mammal"]):
        for j, b in enumerate(["dog", "animal", "cat", "mammal"]):
            if j <= i:
                continue
            print("   {:>8} vs {:<8} {:.4f} {}".format(
                a, b, float(np.dot(enc[i], enc[j])),
                "MERGED" if float(np.dot(enc[i], enc[j])) >= 0.92 else ""))
except Exception as exc:  # pragma: no cover - offline guard
    print("   (skipped:", exc, ")")