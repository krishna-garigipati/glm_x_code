#!/usr/bin/env python
"""Build KG from a corpus file and evaluate GLM-X accuracy.

Usage:
    python scripts/evaluate_full_pipeline.py --corpus path/to/corpus.txt [--db path/to/graph.db]
"""
import sys, time, json, re, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
from sentence_transformers import SentenceTransformer
from kg_builder import KGBuilderPipeline, KGBuilderConfig


def load_corpus(path: str) -> list:
    path = Path(path)
    if path.suffix == ".jsonl":
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get("article", obj.get("text", line.strip())))
        return texts
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


TEST_QUERIES = [
    ("What did Albert Einstein develop?", "theory of relativity"),
    ("What did Marie Curie discover?", "radium"),
    ("What did Isaac Newton formulate?", "laws of motion"),
    ("Who developed the polio vaccine?", "Jonas Salk"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What is Paris in?", "France"),
    ("Where is Tokyo?", "Japan"),
    ("What is London in?", "England"),
    ("What causes lung cancer?", "Smoking"),
    ("What does exercise cause?", "good health"),
    ("What is the CPU part of?", "computer"),
    ("What is the heart part of?", "circulatory system"),
    ("What is a dog?", "animal"),
    ("What is a rose?", "flower"),
    ("What originated in Ethiopia?", "Coffee"),
    ("What originated in China?", "Paper"),
    ("What is the opposite of hot?", "cold"),
    ("What is the opposite of light?", "darkness"),
    ("What did Isaac Newton discover?", "laws of motion"),
    ("Where is the Nile river?", "Egypt"),
    ("What does smoking cause?", "lung cancer"),
    ("What is a diamond?", "gemstone"),
    ("What is in France?", "Paris"),
    ("What is in Japan?", "Tokyo"),
]


def build_ground_truth(corpus: list) -> dict:
    ground_truth = {}
    for line in corpus:
        parts = line.rstrip(".").split(" is part of ")
        if len(parts) == 2:
            e1, e2 = parts[0].strip(), parts[1].strip()
            ground_truth[(e1.lower(), "is part of", e2.lower())] = (e1, "is part of", e2)
            continue
        parts = line.rstrip(".").split(" is the opposite of ")
        if len(parts) == 2:
            e1, e2 = parts[0].strip(), parts[1].strip()
            ground_truth[(e1.lower(), "is the opposite of", e2.lower())] = (e1, "is the opposite of", e2)
            continue
        parts = line.rstrip(".").split(" is in ")
        if len(parts) == 2:
            e1, e2 = parts[0].strip(), parts[1].strip()
            ground_truth[(e1.lower(), "is in", e2.lower())] = (e1, "is in", e2)
            continue
        parts = line.rstrip(".").split(" is an ")
        if len(parts) == 2:
            e1 = parts[0].strip()
            if e1.startswith("A "): e1 = e1[2:]
            elif e1.startswith("An "): e1 = e1[3:]
            ground_truth[(e1.lower(), "is", e2.lower() if len(parts) > 1 else "")] = (e1, "is", e2)
            continue
        parts = line.rstrip(".").split(" is a ")
        if len(parts) == 2:
            e1 = parts[0].strip()
            if e1.startswith("A "): e1 = e1[2:]
            ground_truth[(e1.lower(), "is", e2.lower() if len(parts) > 1 else "")] = (e1, "is", e2)
            continue
        parts = line.rstrip(".").split(" is ")
        if len(parts) == 2:
            e1, e2 = parts[0].strip(), parts[1].strip()
            if e2 and not e2.startswith("the "):
                ground_truth[(e1.lower(), "is", e2.lower())] = (e1, "is", e2)
                continue
        for verb in ["developed", "discovered", "formulated", "invented",
                      "introduced", "established", "built", "originated in",
                      "has", "causes", "caused"]:
            vparts = line.rstrip(".").split(f" {verb} ", 1)
            if len(vparts) == 2:
                e1, e2 = vparts[0].strip(), vparts[1].strip()
                if e1 and e2:
                    ground_truth[(e1.lower(), verb, e2.lower())] = (e1, verb, e2)
                    break
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Evaluate GLM-X KG building and QA")
    parser.add_argument("--corpus", "-c", required=True, help="Path to corpus file (.txt or .jsonl)")
    parser.add_argument("--db", "-d", default=None, help="Output .db path (optional)")
    parser.add_argument("--spacy", default="en_core_web_sm", help="spaCy model")
    parser.add_argument("--sbert", default="BAAI/bge-small-en-v1.5", help="SBERT model")
    args = parser.parse_args()

    # Load corpus
    corpus = load_corpus(args.corpus)
    print(f"\nLoaded {len(corpus)} sentences from {args.corpus}")

    # Build ground truth
    ground_truth = build_ground_truth(corpus)
    print(f"Built {len(ground_truth)} ground-truth triples")

    # Build KG
    print(f"\n=== Building KG ===")
    config = KGBuilderConfig(spaCy_model=args.spacy, verbose=False, embed_merge_threshold=0.92)
    pipeline = KGBuilderPipeline(config)
    t0 = time.time()
    result = pipeline.process_corpus(corpus, show_progress=True)
    build_time = time.time() - t0
    gd = result["graph_data"]
    stats = result["stats"]
    print(f"Built in {build_time:.1f}s: {stats['nodes']} nodes, {stats['edges']} edges, {stats['relation_types']} types")

    # Save to SQLite
    if args.db:
        db_path = Path(args.db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = pipeline.build_graph_store(gd, store_type="sqlite", db_path=str(db_path))
        print(f"Saved to {db_path}")
    else:
        db_path = Path("checkpoints/unified/graphs/eval_graph.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = pipeline.build_graph_store(gd, store_type="sqlite", db_path=str(db_path))
        print(f"Saved to {db_path}")

    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore
    sqlite_store = SQLiteGraphStore.load_state(str(db_path))

    # Compute embeddings
    sbert = SentenceTransformer(args.sbert)
    nids = sorted(store._nodes.keys())
    labels = [store._nodes[nid].label for nid in nids]
    embs = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=True)
    for nid, emb in zip(nids, embs):
        sqlite_store._embeddings[nid] = emb.astype(np.float32)
    sqlite_store.save_state(str(db_path))

    edge_set = set()
    for e in gd["edges"]:
        src = gd["id_to_label"][e["source"]]
        tgt = gd["id_to_label"][e["target"]]
        edge_set.add((src, e["relation"], tgt))

    # Evaluate extraction accuracy
    print(f"\n=== EXTRACTION ACCURACY ===")
    correct = 0
    total_expected = len(ground_truth)
    false_positives = []
    false_negatives = []
    matched_expected = set()

    for src, rel, tgt in sorted(edge_set):
        key = (src.lower(), rel.lower(), tgt.lower())
        if key in ground_truth:
            correct += 1
            matched_expected.add(key)
        else:
            found_any = False
            for ek, ev in ground_truth.items():
                ek_src, ek_rel, ek_tgt = ek
                if rel == ek_rel:
                    if (ek_src in src.lower() or src.lower() in ek_src) and \
                       (ek_tgt in tgt.lower() or tgt.lower() in ek_tgt):
                        correct += 1
                        matched_expected.add(ek)
                        found_any = True
                        break
            if not found_any:
                false_positives.append((src, rel, tgt))

    for ek, ev in ground_truth.items():
        if ek not in matched_expected:
            false_negatives.append(ev)

    precision = correct / (correct + len(false_positives)) * 100 if (correct + len(false_positives)) > 0 else 0
    recall = correct / total_expected * 100 if total_expected > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Ground truth triples: {total_expected}")
    print(f"Correctly extracted:  {correct}")
    print(f"False positives:      {len(false_positives)}")
    print(f"False negatives:      {len(false_negatives)}")
    print(f"Precision:            {precision:.1f}%")
    print(f"Recall:               {recall:.1f}%")
    print(f"F1 Score:             {f1:.1f}%")

    # Run GLM-X QA
    print(f"\n=== GLM-X QA EVALUATION ===")
    from scripts.glmx_ask import GLMXPipeline
    from scripts.train_intent_ffn import train_intent_ffn_from_graph

    print("\n=== AUTO-TRAINING IntentFFN ===")
    train_result = train_intent_ffn_from_graph(gd)
    if train_result is not None:
        print("IntentFFN trained and saved")
    else:
        print("IntentFFN training skipped")

    glmx = GLMXPipeline()
    glmx.graph_store = sqlite_store
    checkpoint_path = Path("checkpoints/unified/graphs")
    model_path = checkpoint_path.parent / "intent_ffn.pt"
    glmx.load_models(model_path=str(model_path) if model_path.exists() else None)
    print("GLM-X pipeline loaded!")

    qa_correct = 0
    qa_total = 0
    qa_results = []

    for question, expected in TEST_QUERIES:
        qa_total += 1
        try:
            t0 = time.time()
            result = glmx.ask(question)
            elapsed = time.time() - t0
            answer = result.get("answer", "").lower()
            walk_labels = " ".join(result.get("walk_path_labels", [])).lower()
            full_response = answer + " " + walk_labels
            is_correct = expected.lower() in full_response
            if is_correct:
                qa_correct += 1
            qa_results.append({
                "question": question,
                "answer": result.get("answer", ""),
                "expected": expected,
                "correct": is_correct,
                "confidence": result.get("confidence", 0),
                "walk_confidence": result.get("walk_confidence", 0),
                "time": round(elapsed, 2),
            })
            status = chr(10003) if is_correct else chr(10007)
            print(f"  {status} Q: {question}")
            print(f"     A: {result.get('answer', '')[:80]}")
            if not is_correct:
                print(f"     Expected: {expected}")
        except Exception as e:
            print(f"  ! Q: {question} -> ERROR: {e}")

    qa_accuracy = qa_correct / qa_total * 100 if qa_total > 0 else 0
    print(f"\n=== QA RESULTS ===")
    print(f"Correct: {qa_correct}/{qa_total} ({qa_accuracy:.1f}%)")

    print(f"\n{'='*60}")
    print(f"KG BUILD: {stats['nodes']} nodes, {stats['edges']} edges in {build_time:.1f}s")
    print(f"EXTRACTION: P={precision:.1f}% R={recall:.1f}% F1={f1:.1f}%")
    print(f"QA ACCURACY: {qa_accuracy:.1f}% ({qa_correct}/{qa_total})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
