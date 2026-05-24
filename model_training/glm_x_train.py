#!/usr/bin/env python
"""GLM-X Evaluation CLI. Initializes pipeline, loads graph, runs QA evaluation.

Usage:
    python -m model_training.glm_x_train --db path/to/graph.db
"""
import sys, json, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from model_training.pipeline import Pipeline


def main():
    import argparse
    p = argparse.ArgumentParser(description="GLM-X Evaluation")
    p.add_argument("--db", required=True, help="Path to SQLite .db graph file")
    p.add_argument("--config", default="configs", help="Config directory")
    args = p.parse_args()

    pipeline = Pipeline(config_dir=args.config)
    pipeline.initialize()
    pipeline.load_graph(args.db)

    results = pipeline.evaluate()
    print(json.dumps({"qa_accuracy": results["qa_accuracy"], "qa_correct": results["qa_correct"],
                       "qa_total": results["qa_total"]}, indent=2))


if __name__ == "__main__":
    main()
