#!/usr/bin/env python
"""Test dynamic KG builder + embedding-based question answering."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from kg_builder import KGBuilderPipeline, KGBuilderConfig

BASE = Path(__file__).parent.parent
CORPUS_PATH = BASE / "kg_builder" / "tests" / "corpus_1000.txt"

nlp = spacy.load("en_core_web_sm")
encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")


def load_corpus(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def extract_question_entities(question: str) -> list:
    doc = nlp(question)
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    for chunk in doc.noun_chunks:
        t = chunk.text.strip()
        if t.lower() not in ("what", "why", "how", "who", "which", "tell me", "something"):
            entities.append(chunk.text)
    tokens = [t.text for t in doc if t.pos_ in ("PROPN", "NOUN") and len(t.text) > 2]
    entities.extend(tokens)
    return list(set(e.strip() for e in entities if e.strip()))


def extract_question_relation(question: str) -> str:
    doc = nlp(question)
    content_words = [t.text for t in doc if t.pos_ in ("VERB", "ADP", "ADV", "SCONJ")]
    if not content_words:
        content_words = [t.text for t in doc if t.pos_ in ("VERB", "AUX", "ADP")]
    return " ".join(content_words).lower().strip()


def main():
    texts = load_corpus(str(CORPUS_PATH))
    print(f"Corpus: {len(texts)} sentences\n")

    print("Building KG...")
    config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=False)
    pipeline = KGBuilderPipeline(config)
    t0 = time.time()
    result = pipeline.process_corpus(texts, show_progress=True)
    build_time = time.time() - t0
    stats = result["stats"]
    graph_data = result["graph_data"]
    print(f"Built in {build_time:.1f}s: {stats['nodes']} nodes, {stats['edges']} edges, {stats['relation_types']} types\n")

    relation_embeddings = {}
    for edge in graph_data["edges"]:
        src = graph_data["id_to_label"][edge["source"]]
        tgt = graph_data["id_to_label"][edge["target"]]
        rel = edge["relation"]
        rel_text = f"{src} {rel} {tgt}"
        emb = encoder.encode(rel_text, normalize_embeddings=True)
        relation_embeddings[(src, rel, tgt)] = emb

    questions = [
        "What did Albert Einstein discover?",
        "What did Marie Curie discover?",
        "Where is Paris?",
        "What originated in India?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")

        entities = extract_question_entities(question)
        q_rel = extract_question_relation(question)
        q_emb = encoder.encode(question, normalize_embeddings=True)

        print(f"  Entities: {entities}")
        print(f"  Relation: {q_rel}")

        if not entities:
            print(f"  A: (no entities found)")
            continue

        entity_matches = {}
        for ent in entities:
            ent_lower = ent.lower()
            exact_matches = [l for l in graph_data["concepts"] if ent_lower == l.lower()]
            if exact_matches:
                entity_matches[ent] = (exact_matches[0], 1.0)
                continue
            substring_matches = [l for l in graph_data["concepts"] if ent_lower in l.lower() or l.lower() in ent_lower]
            if substring_matches:
                best = max(substring_matches, key=len)
                entity_matches[ent] = (best, 0.85)
                continue
            ent_emb = encoder.encode(ent, normalize_embeddings=True)
            best_label, best_sim = "", -1.0
            for label in graph_data["concepts"]:
                node_emb = graph_data["embeddings"].get(label)
                if node_emb is not None:
                    sim = float(np.dot(ent_emb, node_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_label = label
            entity_matches[ent] = (best_label, best_sim)

        matched_entities = {label for ent, (label, sim) in entity_matches.items() if sim > 0.45 and label}

        print(f"  Matched graph nodes: {matched_entities}")

        if not matched_entities:
            print(f"  A: (no matching entities in graph)")
            continue

        doc = nlp(question)
        subj_entities = {t.text.lower() for t in doc if t.dep_ in ("nsubj", "nsubjpass")}
        answers = []
        for edge in graph_data["edges"]:
            src = graph_data["id_to_label"][edge["source"]]
            tgt = graph_data["id_to_label"][edge["target"]]
            rel = edge["relation"]

            if src in matched_entities or tgt in matched_entities:
                rel_emb = relation_embeddings.get((src, rel, tgt))
                if rel_emb is not None:
                    rel_sim = float(np.dot(q_emb, rel_emb))
                else:
                    rel_sim = 0.0

                direction_bonus = 0.0
                if src in matched_entities and any(e in src.lower() for e in subj_entities):
                    direction_bonus = 0.1
                elif tgt in matched_entities:
                    direction_bonus = 0.05

                answers.append({
                    "source": src,
                    "relation": rel,
                    "target": tgt,
                    "confidence": edge["confidence"],
                    "relation_similarity": round(rel_sim + direction_bonus, 4),
                    "direction": "outgoing" if src in matched_entities else "incoming",
                })

        answers.sort(key=lambda x: (x["relation_similarity"], x["confidence"]), reverse=True)

        print(f"  Found {len(answers)} related edges")
        top = [a for a in answers if a["relation_similarity"] > 0.2][:5]
        if not top:
            top = answers[:3]

        if top:
            print(f"  A: {top[0]['source']} {top[0]['relation']} {top[0]['target']} (rel_sim={top[0]['relation_similarity']})")
            for a in top[1:]:
                print(f"     also: {a['source']} {a['relation']} {a['target']} (rel_sim={a['relation_similarity']})")
        else:
            print(f"  A: (no relevant edges found)")


if __name__ == "__main__":
    main()
