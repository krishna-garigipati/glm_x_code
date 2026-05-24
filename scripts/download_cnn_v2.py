import sys, json, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
out_dir = Path("D:/Machine Learning/project_genesis/GLM_X/glm_x_code/model_training/dataset_cnn")
out_dir.mkdir(parents=True, exist_ok=True)

# Use TF record approach - stream from TFDS
import tensorflow_datasets as tfds

# First download & prepare (will extract archives)
builder = tfds.builder("cnn_dailymail/3.0.0")
print("Downloading and preparing cnn_dailymail/3.0.0...")
dl_config = tfds.download.DownloadConfig(
    extract_dir=str(out_dir / "extracted"),
    manual_dir=None,
)
builder.download_and_prepare(download_dir=str(out_dir / "tfds_cache"))
print("Dataset ready!")

# Now stream to JSONL
ds = builder.as_dataset(split="train")
articles_path = out_dir / "articles.jsonl"
corpus_path = out_dir / "corpus.txt"

t0 = time.time()
count = 0
with open(articles_path, "w", encoding="utf-8") as fa, \
     open(corpus_path, "w", encoding="utf-8") as ft:
    for example in ds:
        article = example["article"].numpy().decode("utf-8")
        highlights = example["highlights"].numpy().decode("utf-8")
        fa.write(json.dumps({"id": count, "article": article, "highlights": highlights}) + "\n")
        ft.write(article + "\n")
        count += 1
        if count % 10000 == 0:
            print(f"  {count} articles ({time.time()-t0:.1f}s)")

elapsed = time.time() - t0
print(f"Downloaded {count} articles in {elapsed:.1f}s")

with open(out_dir / "dataset_info.json", "w") as f:
    json.dump({
        "name": "cnn_dailymail",
        "version": "3.0.0",
        "total_articles": count,
        "download_time_seconds": round(elapsed, 1),
        "date": time.strftime("%Y-%m-%d"),
    }, f, indent=2)
print(f"Saved to {out_dir}")
