#!/usr/bin/env python
"""Download CNN/DailyMail and store as JSONL for KG ingestion."""
import sys, json, logging, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("download_cnn")

OUT_DIR = Path(__file__).resolve().parent.parent / "model_training" / "dataset_cnn"

def main():
    import tensorflow_datasets as tfds

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading cnn_dailymail (3.0.0)...")
    ds = tfds.load("cnn_dailymail", split="train", shuffle_files=False)
    logger.info("Dataset loaded")

    articles_path = OUT_DIR / "articles.jsonl"
    highlights_path = OUT_DIR / "highlights.jsonl"
    text_path = OUT_DIR / "corpus.txt"

    t0 = time.time()
    count = 0
    with open(articles_path, "w", encoding="utf-8") as fa, \
         open(highlights_path, "w", encoding="utf-8") as fh, \
         open(text_path, "w", encoding="utf-8") as ft:
        for example in ds:
            article = example["article"].numpy().decode("utf-8")
            highlights = example["highlights"].numpy().decode("utf-8")
            fa.write(json.dumps({"id": count, "article": article}) + "\n")
            fh.write(json.dumps({"id": count, "highlights": highlights}) + "\n")
            ft.write(article + "\n")
            count += 1
            if count % 10000 == 0:
                elapsed = time.time() - t0
                logger.info(f"  {count} articles ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    logger.info(f"Downloaded {count} articles in {elapsed:.1f}s")
    logger.info(f"Articles: {articles_path}")
    logger.info(f"Highlights: {highlights_path}")
    logger.info(f"Corpus: {text_path}")

    with open(OUT_DIR / "dataset_info.json", "w") as f:
        json.dump({
            "name": "cnn_dailymail",
            "version": "3.0.0",
            "total_articles": count,
            "download_time_seconds": round(elapsed, 1),
            "date": time.strftime("%Y-%m-%d"),
        }, f, indent=2)
    logger.info("Done")

if __name__ == "__main__":
    main()
