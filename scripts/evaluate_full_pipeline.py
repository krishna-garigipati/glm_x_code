#!/usr/bin/env python
"""Build KG from clean, verified-correct facts and evaluate GLM-X accuracy."""
import sys, time, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
from sentence_transformers import SentenceTransformer
from kg_builder import KGBuilderPipeline, KGBuilderConfig

# ── 1. Define ~200 verified-true facts ──────────────────────────────────────
# Format: (source, raw_sentence) — raw_sentence is what the pipeline sees
# After extraction, we expect the triple extraction to produce (source, relation, target)

TRUE_FACTS = [
    # === Scientific discoveries & theories (lines 1-25) ===
    "Albert Einstein developed the theory of relativity.",
    "Marie Curie discovered radium.",
    "Isaac Newton formulated the laws of motion.",
    "Alexander Graham Bell invented the telephone.",
    "Thomas Edison invented the light bulb.",
    "Charles Darwin developed the theory of evolution.",
    "Gregor Mendel discovered the laws of inheritance.",
    "Louis Pasteur developed the germ theory of disease.",
    "Jonas Salk developed the polio vaccine.",
    "Alexander Fleming discovered penicillin.",
    "James Watson discovered the structure of DNA.",
    "Nikola Tesla invented the alternating current motor.",
    "Alan Turing developed the Turing machine.",
    "Grace Hopper invented the first compiler.",
    "Stephen Hawking developed the theory of Hawking radiation.",
    "Galileo Galilei discovered the moons of Jupiter.",
    "Michael Faraday discovered electromagnetic induction.",
    "Max Planck introduced quantum theory.",
    "Niels Bohr developed the Bohr model of the atom.",
    "Enrico Fermi built the first nuclear reactor.",
    "Rosalind Franklin discovered the structure of DNA.",
    "Dmitri Mendeleev developed the periodic table.",
    "Werner Heisenberg formulated the uncertainty principle.",
    "Edward Jenner developed the smallpox vaccine.",
    "Robert Koch discovered the tuberculosis bacterium.",

    # === Geographic facts (lines 26-55) ===
    "Paris is in France.",
    "London is in England.",
    "Berlin is in Germany.",
    "Rome is in Italy.",
    "Madrid is in Spain.",
    "Tokyo is in Japan.",
    "Beijing is in China.",
    "Moscow is in Russia.",
    "Sydney is in Australia.",
    "Cairo is in Egypt.",
    "New York is in the United States.",
    "Toronto is in Canada.",
    "Amsterdam is in the Netherlands.",
    "Vienna is in Austria.",
    "Dublin is in Ireland.",
    "Oslo is in Norway.",
    "Stockholm is in Sweden.",
    "Athens is in Greece.",
    "Bangkok is in Thailand.",
    "The Amazon river is in Brazil.",
    "The Nile river is in Egypt.",
    "The Seine river is in France.",
    "The Thames river is in England.",
    "The Danube river is in Germany.",
    "The Yangtze river is in China.",
    "The Great Wall is in China.",
    "The Eiffel Tower is in France.",
    "The Colosseum is in Italy.",
    "The Taj Mahal is in India.",
    "The Acropolis is in Greece.",

    # === Causation (lines 56-68) ===
    "Smoking causes lung cancer.",
    "Exercise causes good health.",
    "Pollution causes global warming.",
    "Vaccination causes immunity.",
    "Education causes economic growth.",
    "Stress causes heart disease.",
    "Solar energy causes clean electricity.",
    "Earthquakes cause tsunamis.",
    "Reading causes knowledge.",
    "Gravity causes orbital motion.",
    "Friction causes heat.",
    "Photosynthesis causes oxygen production.",
    "Natural selection causes evolution.",

    # === Part-whole relationships (lines 69-79) ===
    "The CPU is part of a computer.",
    "The heart is part of the circulatory system.",
    "The liver is part of the digestive system.",
    "An engine is part of a car.",
    "A chapter is part of a book.",
    "A petal is part of a flower.",
    "The nucleus is part of an atom.",
    "A cell is part of an organism.",
    "A wheel is part of a bicycle.",
    "A wing is part of an airplane.",
    "A processor is part of a smartphone.",

    # === Definitions (lines 80-94) ===
    "A dog is an animal.",
    "A cat is an animal.",
    "An elephant is an animal.",
    "A tiger is an animal.",
    "A lion is an animal.",
    "A whale is an animal.",
    "A dolphin is an animal.",
    "A penguin is an animal.",
    "A kangaroo is an animal.",
    "A panda is an animal.",
    "A horse is an animal.",
    "A rose is a flower.",
    "A tulip is a flower.",
    "A diamond is a gemstone.",
    "A ruby is a gemstone.",

    # === Properties (lines 95-104) ===
    "Water has high heat capacity.",
    "Metals have electrical conductivity.",
    "Diamonds have extreme hardness.",
    "Birds have feathers.",
    "Fish have gills.",
    "Mammals have hair.",
    "Trees have roots.",
    "Stars have nuclear fusion.",
    "Computers have memory.",
    "Plants have chlorophyll.",

    # === Origins (lines 105-114) ===
    "Coffee originated in Ethiopia.",
    "Chocolate originated in Mesoamerica.",
    "Paper originated in China.",
    "Democracy originated in Ancient Greece.",
    "The compass originated in China.",
    "Algebra originated in the Middle East.",
    "Jazz music originated in New Orleans.",
    "Buddhism originated in India.",
    "Chess originated in India.",
    "Opera originated in Italy.",

    # === Opposites (lines 115-129) ===
    "Hot is the opposite of cold.",
    "Light is the opposite of darkness.",
    "Love is the opposite of hate.",
    "Peace is the opposite of war.",
    "Wealth is the opposite of poverty.",
    "Knowledge is the opposite of ignorance.",
    "Order is the opposite of chaos.",
    "Truth is the opposite of lies.",
    "Courage is the opposite of fear.",
    "Freedom is the opposite of captivity.",
    "Strength is the opposite of weakness.",
    "Growth is the opposite of decay.",
    "Silence is the opposite of noise.",
    "Summer is the opposite of winter.",
    "Day is the opposite of night.",

    # === Additional multi-word entity tests (lines 130-140) ===
    "Albert Einstein developed the general theory of relativity.",
    "Isaac Newton formulated the universal law of gravitation.",
    "Charles Darwin developed the theory of evolution by natural selection.",
    "Max Planck introduced the quantum theory of radiation.",
    "James Clerk Maxwell formulated the theory of electromagnetism.",
    "Marie Curie discovered the element of radium.",
    "Ernest Rutherford discovered the nucleus of the atom.",
    "John Dalton developed the atomic theory of matter.",
    "Niels Bohr developed the quantum model of the atom.",
    "Werner Heisenberg formulated the matrix mechanics of quantum theory.",
    "Paul Dirac formulated the Dirac equation of quantum mechanics.",
]

np.random.seed(42)
np.random.shuffle(TRUE_FACTS)
corpus = list(TRUE_FACTS)

# Save
corpus_path = Path("kg_builder/tests/corpus_clean.txt")
corpus_path.write_text("\n".join(corpus), encoding="utf-8")
print(f"Clean corpus: {len(corpus)} verified-true sentences → {corpus_path}")

# ── 2. Build ground-truth map from corpus ───────────────────────────────────
# For each sentence, extract the expected triple manually
GROUND_TRUTH = {}
for line in TRUE_FACTS:
    parts = line.rstrip(".").split(" is part of ")
    if len(parts) == 2:
        e1, e2 = parts[0].strip(), parts[1].strip()
        GROUND_TRUTH[(e1.lower(), "is part of", e2.lower())] = (e1, "is part of", e2)
        continue
    parts = line.rstrip(".").split(" is the opposite of ")
    if len(parts) == 2:
        e1, e2 = parts[0].strip(), parts[1].strip()
        GROUND_TRUTH[(e1.lower(), "is the opposite of", e2.lower())] = (e1, "is the opposite of", e2)
        continue
    parts = line.rstrip(".").split(" is in ")
    if len(parts) == 2:
        e1, e2 = parts[0].strip(), parts[1].strip()
        GROUND_TRUTH[(e1.lower(), "is in", e2.lower())] = (e1, "is in", e2)
        continue
    parts = line.rstrip(".").split(" is an ")
    if len(parts) == 2:
        e1 = parts[0].strip()
        if e1.startswith("A "): e1 = e1[2:]
        elif e1.startswith("An "): e1 = e1[3:]
        GROUND_TRUTH[(e1.lower(), "is", e2.lower() if len(parts) > 1 else "")] = (e1, "is", e2)
        continue
    parts = line.rstrip(".").split(" is a ")
    if len(parts) == 2:
        e1 = parts[0].strip()
        if e1.startswith("A "): e1 = e1[2:]
        GROUND_TRUTH[(e1.lower(), "is", e2.lower() if len(parts) > 1 else "")] = (e1, "is", e2)
        continue
    parts = line.rstrip(".").split(" is ")
    if len(parts) == 2:
        e1, e2 = parts[0].strip(), parts[1].strip()
        if e2 and not e2.startswith("the "):
            GROUND_TRUTH[(e1.lower(), "is", e2.lower())] = (e1, "is", e2)
            continue
    # Verb-based: subject VERB object
    for verb in ["developed", "discovered", "formulated", "invented", "created",
                 "introduced", "established", "built", "originated in", "has",
                 "causes", "caused"]:
        vparts = line.rstrip(".").split(f" {verb} ", 1)
        if len(vparts) == 2:
            e1, e2 = vparts[0].strip(), vparts[1].strip()
            if e1 and e2:
                GROUND_TRUTH[(e1.lower(), verb, e2.lower())] = (e1, verb, e2)
                break

# ── 3. Build KG using the pipeline ──────────────────────────────────────────
print("\n=== Building KG from clean corpus ===")
config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=False)
pipeline = KGBuilderPipeline(config)
t0 = time.time()
result = pipeline.process_corpus(corpus, show_progress=True)
build_time = time.time() - t0
gd = result["graph_data"]
stats = result["stats"]
print(f"Built in {build_time:.1f}s: {stats['nodes']} nodes, {stats['edges']} edges, {stats['relation_types']} types")

# List all nodes
print("\nAll graph nodes:")
for nid, lbl in sorted(gd['id_to_label'].items(), key=lambda x: x[0]):
    print(f"  [{nid}] {lbl}")

# List all edges
print("\nAll edges:")
edge_set = set()
for e in gd['edges']:
    src = gd['id_to_label'][e['source']]
    tgt = gd['id_to_label'][e['target']]
    edge_set.add((src, e['relation'], tgt))
    print(f"  {src} --[{e['relation']}]--> {tgt}")

# ── 4. Evaluate extraction accuracy ──────────────────────────────────────────
print(f"\n=== EXTRACTION ACCURACY ===")
correct = 0
total_expected = len(GROUND_TRUTH)
false_positives = []
false_negatives = []
matched_expected = set()

for src, rel, tgt in sorted(edge_set):
    key = (src.lower(), rel, tgt.lower())
    if key in GROUND_TRUTH:
        correct += 1
        matched_expected.add(key)
    else:
        # Check if it's a reasonable variant (e.g., expanded entity name)
        found_any = False
        for ek, ev in GROUND_TRUTH.items():
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

for ek, ev in GROUND_TRUTH.items():
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

if false_positives:
    print(f"\nFalse positives ({len(false_positives)}):")
    for s, r, t in false_positives[:10]:
        print(f"  {s} --[{r}]--> {t}")
if false_negatives:
    print(f"\nFalse negatives (first 10 of {len(false_negatives)}):")
    for s, r, t in false_negatives[:10]:
        print(f"  {s} --[{r}]--> {t}")

# ── 5. Build graph store and run GLM-X queries ──────────────────────────────
print(f"\n=== GLM-X QA EVALUATION ===")
store = pipeline.build_graph_store(gd)
print(f"GraphStore: {store.get_node_count()} nodes")

from scripts.glmx_ask import GLMXPipeline
from scripts.train_intent_ffn import train_intent_ffn_from_graph

print("\n=== AUTO-TRAINING IntentFFN ===")
train_result = train_intent_ffn_from_graph(gd)
if train_result is not None:
    print("IntentFFN trained and saved — pipeline will load the KG model automatically")
else:
    print("IntentFFN training skipped — pipeline will use ConceptNet model")

glmx = GLMXPipeline()
glmx.graph_store = store
glmx.load_models()
print("GLM-X pipeline loaded!")

test_queries = [
    # (question, expected_answer_contains)
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

qa_correct = 0
qa_total = 0
qa_results = []

for question, expected in test_queries:
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
        status = "✓" if is_correct else "✗"
        print(f"  {status} Q: {question}")
        print(f"     A: {result.get('answer', '')[:80]}")
        if not is_correct:
            print(f"     Expected: {expected}")
    except Exception as e:
        print(f"  ! Q: {question} → ERROR: {e}")

qa_accuracy = qa_correct / qa_total * 100 if qa_total > 0 else 0
print(f"\n=== QA RESULTS ===")
print(f"Correct: {qa_correct}/{qa_total} ({qa_accuracy:.1f}%)")
for r in qa_results:
    status = "✓" if r["correct"] else "✗"
    print(f"  {status} {r['question'][:50]:50s} → {r['answer'][:40]:40s} [{r['expected']}] conf={r['confidence']:.2f} walk={r['walk_confidence']:.2f}")

print(f"\n{'='*60}")
print(f"KG BUILD: {stats['nodes']} nodes, {stats['edges']} edges in {build_time:.1f}s")
print(f"EXTRACTION: P={precision:.1f}% R={recall:.1f}% F1={f1:.1f}%")
print(f"QA ACCURACY: {qa_accuracy:.1f}% ({qa_correct}/{qa_total})")
print(f"{'='*60}")
