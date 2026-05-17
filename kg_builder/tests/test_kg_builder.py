"""Test KGBuilder pipeline on 500-line corpus and report metrics."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from kg_builder import KGBuilderPipeline, KGBuilderConfig


def load_corpus(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    corpus_path = Path(__file__).parent / "corpus_500.txt"
    results_path = Path(__file__).parent / "kg_build_results.json"

    texts = load_corpus(str(corpus_path))
    print(f"Loaded {len(texts)} sentences from corpus")

    config = KGBuilderConfig(
        spaCy_model="en_core_web_sm",
        verbose=True,
    )

    pipeline = KGBuilderPipeline(config)
    t0 = time.time()

    result = pipeline.process_corpus(texts, show_progress=True)

    build_time = time.time() - t0
    stats = result["stats"]
    graph_data = result["graph_data"]

    store = pipeline.build_graph_store(graph_data)

    entity_count = pipeline.entity_resolver._next_id - 1
    surface_form_count = len(pipeline.entity_resolver._surface_forms)
    embed_cache_count = len(pipeline.entity_resolver._embeddings)

    metrics = {
        "pipeline": "KGBuilderPipeline",
        "corpus": {
            "file": "corpus_500.txt",
            "sentences": len(texts),
            "total_chars": sum(len(t) for t in texts),
        },
        "extraction": {
            "total_time_seconds": round(build_time, 2),
            "sentences_per_second": round(len(texts) / build_time, 1) if build_time > 0 else 0,
            "raw_triples": stats["raw_triples"],
            "kept_triples": stats["kept_triples"],
            "triples_per_second": stats.get("triples_per_second", 0),
            "filter_drop_pct": round(
                (1 - stats["kept_triples"] / max(stats["raw_triples"], 1)) * 100, 1
            ),
        },
        "graph": {
            "nodes": stats["nodes"],
            "edges": stats["edges"],
            "relation_types": stats["relation_types"],
            "avg_edges_per_node": round(stats["edges"] / max(stats["nodes"], 1), 2),
            "graph_density": round(
                stats["edges"] / (stats["nodes"] * (stats["nodes"] - 1) / 2), 6
            ) if stats["nodes"] > 1 else 0,
        },
        "entity_resolution": {
            "unique_canonical_entities": entity_count,
            "surface_form_aliases": surface_form_count,
            "entities_with_embeddings": embed_cache_count,
            "merge_ratio": round(
                (entity_count + surface_form_count) / max(entity_count, 1), 2
            ),
        },
        "relation_type_breakdown": {},
    }

    for rel_type in set(graph_data["relation_types"]):
        count = sum(1 for e in graph_data["edges"] if e["relation"] == rel_type)
        metrics["relation_type_breakdown"][rel_type] = {
            "count": count,
            "pct": round(count / max(len(graph_data["edges"]), 1) * 100, 1),
        }

    repeated = {}
    for e in graph_data["edges"]:
        key = (e["source"], e["relation"], e["target"])
        repeated[key] = repeated.get(key, 0) + 1
    metrics["graph"]["duplicate_edge_groups"] = sum(1 for v in repeated.values() if v > 1)

    sample_triples = []
    for item in result.get("graph_data", {}).get("edges", [])[:10]:
        src = result["graph_data"]["id_to_label"][item["source"]]
        tgt = result["graph_data"]["id_to_label"][item["target"]]
        sample_triples.append({
            "source": src,
            "relation": item["relation"],
            "target": tgt,
            "confidence": item["confidence"],
        })
    metrics["sample_triples"] = sample_triples

    connector_nodes = {}
    for e in graph_data["edges"]:
        for nid in (e["source"], e["target"]):
            connector_nodes[nid] = connector_nodes.get(nid, 0) + 1
    top_connectors = sorted(connector_nodes.items(), key=lambda x: -x[1])[:5]
    metrics["graph"]["top_connector_nodes"] = [
        {"label": graph_data["id_to_label"][nid], "connections": count}
        for nid, count in top_connectors
    ]

    print("\n" + "=" * 65)
    print("KG BUILDER TEST RESULTS")
    print("=" * 65)
    print(f"Corpus: {len(texts)} sentences processed in {build_time:.1f}s")
    print(f"  Extraction: {stats['raw_triples']} raw triples, {stats['kept_triples']} kept")
    print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    print(f"  Relation types: {stats['relation_types']}")
    print(f"  Entity resolution: {entity_count} canonical, {surface_form_count} aliases")
    print(f"\nTop-5 connector nodes:")
    for item in metrics["graph"]["top_connector_nodes"]:
        print(f"  {item['label']}: {item['connections']} connections")
    print(f"\nRelation breakdown:")
    for rtype, info in sorted(metrics["relation_type_breakdown"].items(),
                              key=lambda x: -x[1]["count"]):
        print(f"  {rtype}: {info['count']} ({info['pct']}%)")
    print(f"\nSample triples:")
    for t in sample_triples[:5]:
        print(f"  ({t['source']}, {t['relation']}, {t['target']}) conf={t['confidence']}")

    with open(str(results_path), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
