#!/usr/bin/env python
"""GLM-X Training CLI. Delegates to the unified training pipeline."""
import sys, argparse, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("glm_x_train")


def main():
    parser = argparse.ArgumentParser(description="GLM-X Training Pipeline")
    parser.add_argument("--data", required=True, help="Path to input .db graph file or corpus text")
    parser.add_argument("--output", default=str(Path(__file__).parent / "checkpoints" / "unified"),
                        help="Checkpoint / output directory")
    parser.add_argument("--config", default=None, help="Path to training config YAML")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--sbert", default="BAAI/bge-small-en-v1.5", help="Sentence transformer model")
    args = parser.parse_args()

    from scripts.universal_train import train_on_db

    result = train_on_db(
        db_path=args.data,
        checkpoint_dir=args.output,
        sbert_model=args.sbert,
        metrics=None,
    )
    print(f"\nTraining complete: {result.get('qa_accuracy', 0):.1f}% accuracy")
    return result


if __name__ == "__main__":
    main()
