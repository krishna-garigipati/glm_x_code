"""Build the 'real graph' for tester-b using the KGBuilder pipeline.

Pipeline (full): corpus TXT -> DocumentProcessor (spaCy) -> TripleExtractor
(rule-based, ALL relations discovered from text) -> RelationMapper (zero-heuristic
raw-connector labels) -> EntityResolver (merges near-duplicate entities) ->
GraphBuilder (confidence-gated) -> SQLiteGraphStore.

The extractor labels every edge with the raw connective text it finds between two
entities (e.g. "is a", "is part of", "causes"). Those raw labels are then normalised
to the system's 16 canonical relation names via REL_MAP below so the GLM-X
extractor/walker can consume the graph; anything not covered by the map is kept as-is
and reported. No relation is dropped purely because it was not curated in advance.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kg_builder import KGBuilderPipeline, KGBuilderConfig

CORPUS = OUT / "datasets" / "food_biology_source.txt"
DB_PATH = OUT / "tester_b_real_graph.db"
STATS_PATH = OUT / "build_stats.json"
EDGES_PATH = OUT / "edges_dump.tsv"

REL_MAP = {
    "is a": "is_a",
    "is an": "is_a",
    "is": "is_a",
    "are": "is_a",
    "has": "has_property",
    "have": "has_property",
    "causes": "causes",
    "cause": "causes",
    "is caused by": "caused_by",
    "leads to": "causes",
    "follows": "follows",
    "precedes": "precedes",
    "contradicts": "contradicts",
    "supports": "supports",
    "is associated with": "associated_with",
    "is a type of": "example_of",
    "are a type of": "example_of",
    "include": "example_of",
    "includes": "example_of",
    "is part of": "part_of",
    "are part of": "part_of",
    "consists of": "part_of",
    "is synonymous with": "synonym",
    "contrasts": "antonym",
    "coincides with": "temporal_coincident",
    "is near": "spatial_near",
    "are near": "spatial_near",
    "is located near": "spatial_near",
    "translates to": "linguistic_maps",
}

# Raw connectors whose SURFACE subject is the semantic target of the canonical
# relation. Writing them subject->object would store the canonical edge
# backwards: "Fish include salmon." must store salmon --[example_of]--> fish
# (instance example_of category) and "The circulatory system consists of the
# heart." must store the heart --[part_of]--> the circulatory system
# (part part_of whole). Swap source/target so stored edges match the canonical
# orientation (fixes trg10/trg11; rendering stays canonical, traversal-only).
CANONICAL_SWAP_RELATIONS = frozenset({"include", "includes", "consists of"})


def main():
    for suffix in ("", "-wal", "-shm"):
        stale = Path(str(DB_PATH) + suffix)
        if stale.exists():
            stale.unlink()

    texts = [ln.strip() for ln in CORPUS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[build] corpus: {CORPUS} ({len(texts)} sentences)")

    config = KGBuilderConfig(spaCy_model="en_core_web_sm", verbose=True)
    pipeline = KGBuilderPipeline(config)
    result = pipeline.process_corpus(texts, show_progress=True)
    graph_data = result["graph_data"]
    stats = result["stats"]

    raw_rels = Counter(e["relation"] for e in graph_data["edges"])

    unknown = sorted(r for r in raw_rels if r not in REL_MAP)
    mapped = {r: REL_MAP.get(r, r) for r in raw_rels}

    rel_map = {raw: REL_MAP[raw] for raw in raw_rels if raw in REL_MAP}
    for e in graph_data["edges"]:
        raw_rel = e["relation"]
        mapped_rel = rel_map.get(raw_rel, raw_rel)
        if mapped_rel in ("part_of", "example_of") and raw_rel in CANONICAL_SWAP_RELATIONS:
            e["source"], e["target"] = e["target"], e["source"]
        e["relation"] = mapped_rel
    mapped_rels = Counter(e["relation"] for e in graph_data["edges"])
    graph_data["relation_types"] = list(mapped_rels)

    store = pipeline.build_graph_store(
        graph_data, store_type="sqlite", db_path=str(DB_PATH)
    )

    with EDGES_PATH.open("w", encoding="utf-8") as f:
        for e in sorted(graph_data["edges"], key=lambda x: (x["relation"], x["source"])):
            f.write(f"{graph_data['id_to_label'][e['source']]}\t{e['relation']}\t"
                    f"{graph_data['id_to_label'][e['target']]}\n")

    out = {
        "corpus": {"file": str(CORPUS.name), "sentences": len(texts)},
        "extraction_stats": stats,
        "raw_relation_counts": dict(sorted(raw_rels.items(), key=lambda kv: -kv[1])),
        "unmapped_raw_relations": unknown,
        "mapped_relation_counts": dict(sorted(mapped_rels.items(), key=lambda kv: -kv[1])),
        "nodes": graph_data["node_count"],
        "edges": graph_data["edge_count"],
        "db": str(DB_PATH.name),
    }
    STATS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== RAW relations found by KGBuilder ===")
    for r, c in sorted(raw_rels.items(), key=lambda kv: -kv[1]):
        print(f"  {r!r}: {c}")
    print("  [[UNMAPPED]] ->", unknown)
    print("\n=== MAPPED canonical relations ===")
    for r, c in sorted(mapped_rels.items(), key=lambda kv: -kv[1]):
        print(f"  {r}: {c}")
    print(f"\nGraph: {graph_data['node_count']} nodes, {graph_data['edge_count']} edges")
    print(f"DB: {DB_PATH}; stats: {STATS_PATH}; edges: {EDGES_PATH}")


if __name__ == "__main__":
    main()