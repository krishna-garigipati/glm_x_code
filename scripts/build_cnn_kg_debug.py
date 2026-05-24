#!/usr/bin/env python
"""Build KG from CNN articles — 1000 articles with checkpoint/resume."""
import sys, json, time, os, logging, pickle, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.alignment import EntityRegistry, RelationRegistry

for noisy in ["sentence_transformers", "transformers", "httpx", "urllib3",
               "huggingface_hub", "filelock", "PIL"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

log_file = Path(__file__).resolve().parent.parent / "model_training" / "dataset_cnn" / "kg_build.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
if log_file.exists():
    log_file.unlink()

file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("%(message)s"))
root = logging.getLogger()
root.setLevel(logging.DEBUG)
root.addHandler(file_handler)
root.addHandler(stream_handler)
logger = logging.getLogger("build_kg")

DATA_FILE = Path(__file__).resolve().parent.parent / "model_training" / "dataset_cnn" / "cnn_dailymail_train.json"
OUT_DB = Path(__file__).resolve().parent.parent / "model_training" / "dataset_cnn" / "cnn_dailymail_dataset_data.db"
CHECKPOINT = Path(__file__).resolve().parent.parent / "model_training" / "dataset_cnn" / "build_checkpoint.pkl"
ENTITY_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "model_training" / "dataset_cnn" / "entity_registry.pkl"
RELATION_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "model_training" / "dataset_cnn" / "relation_registry.pkl"
NUM_ARTICLES = 1000
SAVE_EVERY = 200

# Delete DB at start (not checkpoint)
if OUT_DB.exists():
    OUT_DB.unlink()

print("=" * 70)
print(f"CNN/DAILYMAIL KG BUILD — 1000 articles with checkpoint/resume")
print(f"Output: {OUT_DB}")
print("=" * 70)

# ── 1. Read ──────────────────────────────────────────────────────────────
print("\n[1/6] Reading articles...")
articles = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= NUM_ARTICLES:
            break
        articles.append(json.loads(line).get("article", ""))
print(f"  {len(articles)} articles, {sum(len(a) for a in articles):,} chars total")

# ── 2. Init ──────────────────────────────────────────────────────────────
print("\n[2/6] Initializing KGBuilder pipeline...")
from kg_builder import KGBuilderPipeline, KGBuilderConfig
from sentence_transformers import SentenceTransformer

config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=False, embed_merge_threshold=0.92)
pipeline = KGBuilderPipeline(config)
pipeline.initialize()

dp = pipeline.doc_processor
te = pipeline.triple_extractor
er = pipeline.entity_resolver
gb = pipeline.graph_builder

te._sbert = None  # skip coherence check

# Shared SBERT for both registries (load once)
shared_sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
# Use same SBERT for EntityResolver too
er._sbert = shared_sbert

entity_registry = EntityRegistry(sbert_model=shared_sbert, registry_path=str(ENTITY_REGISTRY_PATH))
relation_registry = RelationRegistry(sbert_model=shared_sbert, registry_path=str(RELATION_REGISTRY_PATH))

# ── 3. spaCy batch ──────────────────────────────────────────────────────
print("\n[3/6] spaCy batch processing...")
from kg_builder.cross_sentence_linker import CrossSentenceLinker
from tqdm import tqdm

T0 = time.time()
all_sentences_list = dp.process_batch(articles, batch_size=32)
t_spacy = time.time() - T0
print(f"  {len(articles)} docs in {t_spacy:.1f}s ({t_spacy/len(articles):.2f}s/doc)")

# ── 4. Extract triples (resumable) ────────────────────────────────────────
print("\n[4/6] Extracting triples...")

# Load checkpoint if exists
start_from = 0
all_triple_results = []
if CHECKPOINT.exists():
    with open(CHECKPOINT, "rb") as f:
        saved = pickle.load(f)
    all_triple_results = saved["triples"]
    start_from = saved["doc_index"] + 1
    # Restore entity resolver state
    er._canonical_to_id = saved["canonical_to_id"]
    er._id_to_canonical = {int(k): v for k, v in saved["id_to_canonical"].items()}
    er._embeddings = saved["embeddings"]
    er._surface_forms = saved["surface_forms"]
    er._entity_freq = saved["entity_freq"]
    er._next_id = saved["next_id"]
    print(f"  Resumed from doc {start_from-1} ({len(all_triple_results)} triples saved)")

t_start = time.time()
for doc_idx in tqdm(range(start_from, len(articles)), desc="Articles", unit="doc", ncols=80,
                     initial=start_from, total=len(articles)):
    t0 = time.time()
    sentences = all_sentences_list[doc_idx]
    linker = CrossSentenceLinker()

    raw_triple_data = []
    for sent in sentences:
        if len(sent.get("entities", [])) == 0 and len(sent.get("text", "")) < 10:
            continue
        try:
            extracted = te.extract_or_escalate(sent)
        except Exception:
            continue
        for item in extracted:
            t = item["triple"]
            raw_conn = item.get("raw_connector", t[1])
            raw_triple_data.append((
                t[0], t[1], t[2],
                item.get("rel_confidence", 0.5),
                raw_conn,
                sent.get("text", ""),
            ))

    if raw_triple_data:
        linked = linker.link([(d[0], d[1], d[2], d[3], d[4]) for d in raw_triple_data])
        original_keys = {(d[0].lower().strip(), d[1], d[2].lower().strip()) for d in raw_triple_data}

        all_entities = []
        for item in linked:
            all_entities.append(item[0])
            all_entities.append(item[2])
        resolved = er.resolve_batch(all_entities)

        # Cross-build entity alignment via EntityRegistry
        unique_resolved = list(set(resolved))
        unique_canonical, _ = entity_registry.align_batch(unique_resolved)
        resolved_to_canonical = dict(zip(unique_resolved, unique_canonical))

        for idx, item in enumerate(linked):
            e1, rel, e2, rel_conf, raw_conn = item
            re1 = resolved_to_canonical.get(resolved[idx * 2], resolved[idx * 2])
            re2 = resolved_to_canonical.get(resolved[idx * 2 + 1], resolved[idx * 2 + 1])
            if re1 == re2:
                continue
            key = (e1.lower().strip(), rel, e2.lower().strip())
            level = 1 if key in original_keys else 2
            all_triple_results.append({
                "triple": (re1, raw_conn, re2),
                "level": level,
                "relation_confidence": rel_conf,
                "resolution_method": "embedding",
                "doc_index": doc_idx,
                "raw_connector": raw_conn,
            })

        if (doc_idx + 1) % SAVE_EVERY == 0 or doc_idx == len(articles) - 1:
            with open(CHECKPOINT, "wb") as f:
                pickle.dump({
                    "triples": all_triple_results,
                    "doc_index": doc_idx,
                    "canonical_to_id": er._canonical_to_id,
                    "id_to_canonical": {str(k): v for k, v in er._id_to_canonical.items()},
                    "embeddings": er._embeddings,
                    "surface_forms": er._surface_forms,
                    "entity_freq": er._entity_freq,
                    "next_id": er._next_id,
                }, f)
            entity_registry.save(str(ENTITY_REGISTRY_PATH))
            relation_registry.save(str(RELATION_REGISTRY_PATH))
        num_new = len(all_triple_results)
        tqdm.write(f"  CHECKPOINT at doc {doc_idx} ({num_new} triples)")

t_extract = time.time() - t_start

# Clean up checkpoint
CHECKPOINT.unlink(missing_ok=True)
print(f"\n  Extraction: {len(all_triple_results)} triples in {t_extract:.1f}s "
      f"({t_extract/len(articles):.2f}s/doc)")

# ── 5. Canonicalize relations (embedding-based, zero heuristics) ────────
print("\n[5/6] Canonicalizing relations (embedding-based)...")

unique_raw = len(set(t["triple"][1] for t in all_triple_results))
all_rels = [t["triple"][1] for t in all_triple_results]
canonical_rels, _ = relation_registry.canonicalize_batch(all_rels)
for t, canonical_rel in zip(all_triple_results, canonical_rels):
    t["triple"] = (t["triple"][0], canonical_rel, t["triple"][2])
unique_final = len(set(t["triple"][1] for t in all_triple_results))
print(f"  Relation types: {unique_raw} → {unique_final} ({unique_raw - unique_final} merged)")

# ── 6. Build graph + save DB + BGE ──────────────────────────────────────
print(f"\n[6/6] Building graph, saving DB, computing BGE embeddings...")
t0 = time.time()
gd = gb.build(all_triple_results)
print(f"  Graph: {gd['node_count']} nodes, {gd['edge_count']} edges, {len(gd['relation_types'])} types")

store = pipeline.build_graph_store(gd, store_type="sqlite", db_path=str(OUT_DB))
print(f"  SQLite: {store.get_node_count()} nodes, {store.get_edge_count()} edges")

nids = sorted(store._nodes.keys())
labels = [store._nodes[nid].label for nid in nids]
print(f"  Encoding {len(labels)} node labels for BGE...")
embs = shared_sbert.encode(labels, normalize_embeddings=True)
for nid, emb in zip(nids, embs):
    store._embeddings[nid] = emb.astype(np.float32)

store.set_metadata("dataset_name", "cnn_dailymail_1k")
store.set_metadata("sample_size", str(NUM_ARTICLES))
store.save_state(str(OUT_DB))
db_size = os.path.getsize(str(OUT_DB)) / (1024 * 1024)

# Save registries for cross-build persistence
entity_registry.save(str(ENTITY_REGISTRY_PATH))
relation_registry.save(str(RELATION_REGISTRY_PATH))
t_total = time.time() - T0

print(f"\n{'=' * 70}")
print(f"BUILD COMPLETE — 1000 articles")
print(f"{'=' * 70}")
print(f"  Articles:      {NUM_ARTICLES}")
print(f"  Triples:       {len(all_triple_results)}")
print(f"  Nodes:         {gd['node_count']}")
print(f"  Edges:         {gd['edge_count']}")
print(f"  Relation types: {len(gd['relation_types'])} (was {unique_raw})")
print(f"  Embeddings:    {len(gd['embeddings'])}")
print(f"  DB:            {OUT_DB.name} ({db_size:.2f} MB)")
print(f"  spaCy:         {t_spacy:.0f}s")
print(f"  Extraction:    {t_extract:.0f}s")
print(f"  Total:         {t_total:.0f}s")
print()
