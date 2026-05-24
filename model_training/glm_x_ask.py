#!/usr/bin/env python
"""GLM-X Inference CLI. Loads graph + checkpoint, answers a question.
Usage:
    python -m model_training.glm_x_ask --db path/to/graph.db --question "What is Paris in?"
"""

import sys, json, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("glm_x_ask")

from model_training.pipeline import Pipeline


def main():
    import argparse
    p = argparse.ArgumentParser(description="GLM-X Inference")
    p.add_argument("--db", required=True, help="Path to .db graph file")
    p.add_argument("--config", default="configs", help="Config directory")
    p.add_argument("--question", "-q", required=True, help="Question to answer")
    args = p.parse_args()

    pipeline = Pipeline(config_dir=args.config)
    pipeline.initialize()
    pipeline.load_graph(args.db)

    result = pipeline.answer(args.question)
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"  Confidence: {result['confidence']:.4f} | Matched: {result['template_matched']}")
    print(f"  Path: {' → '.join(result['path_labels'])}")
    print(f"  Time: {result['time_seconds']}s")


if __name__ == "__main__":
    main()
