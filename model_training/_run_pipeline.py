#!/usr/bin/env python
"""End-to-end pipeline test on true-facts corpus.
    python model_training/_run_pipeline.py
"""

import sys, json, time, logging
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

from kg_builder import KGBuilderPipeline, KGBuilderConfig
from sentence_transformers import SentenceTransformer
from model_training.pipeline import Pipeline

TRUE_FACTS = [
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


def main():
    t_start = time.time()
    base = Path(__file__).resolve().parent
    db_path = str(base / "graphs" / "true_facts.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 1: Build KG from facts ===")
    config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=False, embed_merge_threshold=0.92)
    pipeline = KGBuilderPipeline(config)
    result = pipeline.process_corpus(TRUE_FACTS, show_progress=True)
    gd = result["graph_data"]
    stats = result["stats"]
    logger.info("KG built: %d nodes, %d edges, %d types in %.1fs",
                 stats["nodes"], stats["edges"], len(stats["relation_types"]), stats["time_seconds"])

    logger.info("=== Step 2: Save to SQLite .db ===")
    store = pipeline.build_graph_store(gd, store_type="sqlite", db_path=db_path)
    logger.info("Saved: %d nodes, %d edges", store.get_node_count(), store.get_edge_count())

    logger.info("=== Step 3: Compute BGE embeddings ===")
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    nids = sorted(store._nodes.keys())
    labels = [store._nodes[nid].label for nid in nids]
    embs = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=True)
    for nid, emb in zip(nids, embs):
        store._embeddings[nid] = emb.astype(np.float32)
        node = store._nodes[nid]
        store._nodes[nid] = type(node)(
            id=node.id, label=node.label, node_type=node.node_type,
            embedding=emb.astype(np.float32), activation=node.activation,
            use_count=node.use_count, create_time=node.create_time,
        )
    store.save_state(db_path)
    logger.info("Embeddings: %d nodes", len(store._embeddings))

    logger.info("=== Step 4: Run pipeline ===")
    pipe = Pipeline(config_dir="configs")
    pipe.initialize()
    pipe.load_graph(db_path)
    eval_results = pipe.evaluate()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {elapsed:.1f}s total")
    print(f"  QA Accuracy: {eval_results['qa_accuracy']:.1f}% ({eval_results['qa_correct']}/{eval_results['qa_total']})")
    print(f"{'='*60}")

    if eval_results["per_question"]:
        for r in eval_results["per_question"]:
            mark = "✓" if r["correct"] else "✗"
            print(f"  {mark} {r['question'][:50]:50s} → {r['answer'][:40]:40s}")


if __name__ == "__main__":
    main()
