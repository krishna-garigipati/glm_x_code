#!/usr/bin/env python
"""Train IntentFFN on our KG relations and save the model."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np

from kg_builder import KGBuilderPipeline, KGBuilderConfig
from g2p.g2p_planner import G2PPlanner
from g2p.types import Subgraph as G2PSubgraph
from g2p.config import G2PConfig

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "model_training" / "config.yaml"
SAVED_MODELS_DIR = BASE_DIR / "model_training" / "saved_models"

KG_RELATION_TO_INTENT = {
    "developed": 0, "discovered": 0, "formulated": 0,
    "invented": 0, "built": 0, "introduced": 0,
    "established": 0, "created": 0,
    "is": 1, "is in": 1, "have": 1, "has": 1,
    "causes": 2, "cause": 2, "caused": 2, "leads to": 2,
    "originated in": 13,
    "is part of": 6,
    "is the opposite of": 4,
}

DEFAULT_INTENT = 1

# ── 200 verified-true facts (same as evaluate_full_pipeline.py) ──
TRAIN_FACTS = [
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


def build_kg(texts):
    config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=False)
    pipeline = KGBuilderPipeline(config)
    result = pipeline.process_corpus(texts, show_progress=True)
    gd = result["graph_data"]
    store = pipeline.build_graph_store(gd)
    return gd, store


def create_training_data(gd):
    id_to_label = gd["id_to_label"]
    edges = gd["edges"]
    training_data = []
    intent_counts = {}

    for e in edges:
        rel = e["relation"]
        intent_id = KG_RELATION_TO_INTENT.get(rel, DEFAULT_INTENT)
        intent_counts[intent_id] = intent_counts.get(intent_id, 0) + 1

        src_id = e["source"]
        tgt_id = e["target"]

        subgraph = G2PSubgraph(
            nodes=[src_id, tgt_id],
            node_activations={src_id: 0.8, tgt_id: 0.6},
            edges=[(src_id, tgt_id, rel)],
            edge_strengths={(src_id, tgt_id, rel): 1.0},
            edge_confidences={(src_id, tgt_id, rel): e.get("confidence", 0.8)},
            seed_nodes=[src_id],
            tier_used=1,
            activation_energy=1.0,
            query_embedding=np.zeros(384, dtype=np.float32),
            timestamp=time.time(),
        )

        training_data.append((subgraph, [intent_id]))

    logging.info(f"Training data: {len(training_data)} samples")
    for intent_id, count in sorted(intent_counts.items()):
        name = G2PPlanner.INTENT_NAMES.get(intent_id, f"unknown_{intent_id}")
        logging.info(f"  intent {intent_id} ({name}): {count} samples")

    return training_data


def train_intent_ffn_from_graph(gd):
    import torch
    training_data = create_training_data(gd)

    if len(training_data) == 0:
        print("ERROR: No training data generated!")
        return None

    planner = G2PPlanner(G2PConfig.from_yaml(str(CONFIG_PATH)))
    planner.set_label_map(gd["id_to_label"])
    planner.initialize()

    print(f"Training IntentFFN on {len(training_data)} samples...")
    metrics = planner.train(training_data)
    print(f"Training complete: {metrics['epochs_trained']} epochs, best_val_loss={metrics['best_val_loss']:.6f}")

    kg_dir = SAVED_MODELS_DIR / "kg"
    kg_dir.mkdir(parents=True, exist_ok=True)
    model_path = kg_dir / "intent_ffn_best.pt"

    torch.save({
        "model_state_dict": planner.intent_ffn.state_dict(),
        "config": {
            "input_dim": planner.config.ffn.input_dim,
            "hidden_dim": planner.config.ffn.hidden_dim,
            "output_dim": planner.config.ffn.output_dim,
            "num_layers": planner.config.ffn.num_layers,
            "classifier_input_dim": planner.config.ffn.classifier_input_dim,
            "classifier_hidden_dim": planner.config.ffn.classifier_hidden_dim,
            "classifier_num_layers": planner.config.ffn.classifier_num_layers,
        },
        "training_metrics": metrics,
        "relation_to_intent": KG_RELATION_TO_INTENT,
        "intent_names": G2PPlanner.INTENT_NAMES,
    }, model_path)
    print(f"Model saved to {model_path}")

    planner.mark_trained()
    return planner, metrics, model_path


def main():
    print("=== Phase 1: Train IntentFFN on KG Relations ===\n")

    print("Building KG from clean corpus...")
    gd, store = build_kg(TRAIN_FACTS)
    print(f"KG: {len(gd['id_to_label'])} nodes, {len(gd['edges'])} edges, {gd['relation_types']} types")

    result = train_intent_ffn_from_graph(gd)
    if result is None:
        return
    planner, metrics, model_path = result

    print("\nVerification — testing planner on first 10 edges:")
    training_data = create_training_data(gd)
    correct = 0
    for i, (subgraph, _) in enumerate(training_data[:10]):
        plan = planner.plan(subgraph)
        expected_intent = training_data[i][1][0]
        predicted = plan.intent_sequence[0]
        is_ok = predicted == expected_intent
        if is_ok:
            correct += 1
        n0 = gd["id_to_label"][subgraph.nodes[0]]
        n1 = gd["id_to_label"][subgraph.nodes[1]]
        rel = subgraph.edges[0][2]
        mark = "✓" if is_ok else "✗"
        print(f"  {mark} {n0} --[{rel}]--> {n1}: "
              f"pred={predicted}({plan.intent_names[0]}) "
              f"exp={expected_intent}({G2PPlanner.INTENT_NAMES.get(expected_intent, '?')}) "
              f"conf={plan.plan_confidence:.4f}")
    print(f"\nFirst-10 accuracy: {correct}/10 ({correct/10*100:.0f}%)")


if __name__ == "__main__":
    main()
