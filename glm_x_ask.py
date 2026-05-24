#!/usr/bin/env python
"""GLM-X Inference CLI. Ask questions against a trained model."""
import sys, argparse, json, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("glm_x_ask")


def main():
    parser = argparse.ArgumentParser(description="GLM-X Question Answering")
    parser.add_argument("--db", default=None, help="Path to .db graph file")
    parser.add_argument("--checkpoint", default=str(Path(__file__).parent / "checkpoints" / "unified"),
                        help="Checkpoint directory with trained models")
    parser.add_argument("--question", "-q", required=True, help="Question to answer")
    args = parser.parse_args()

    from scripts.glmx_ask import GLMXPipeline

    pipeline = GLMXPipeline()

    if args.db:
        db_path = Path(args.db)
        if not db_path.exists():
            print(f"Error: DB file not found: {db_path}", file=sys.stderr)
            sys.exit(1)
        from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore
        pipeline.graph_store = SQLiteGraphStore.load_state(str(db_path))
    else:
        pipeline.load_graph(max_edges_per_rel=2000)

    checkpoint_path = Path(args.checkpoint)
    model_path = checkpoint_path / "intent_ffn.pt"
    pipeline.load_models(model_path=str(model_path) if model_path.exists() else None)

    result = pipeline.ask(args.question)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
