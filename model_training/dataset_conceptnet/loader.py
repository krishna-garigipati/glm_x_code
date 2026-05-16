import logging
import numpy as np
import pyarrow.parquet as pq
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from collections import Counter

from .relation_map import CONCEPTNET_RELATION_MAP

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "conceptnet" / "data"

RELATION_SHARD_MAP = {
    "Antonym": [0],
    "RelatedTo": [5, 6, 7],
    "Synonym": [8],
}


def _concept_label(uri: str) -> str:
    parts = uri.strip("/").split("/")
    name_idx = 4 if parts[0] == "http:" else 3
    if len(parts) > name_idx:
        return parts[name_idx].replace("_", " ")
    return uri


def download_conceptnet(max_samples: Optional[int] = None, english_only: bool = True) -> List[dict]:
    n = max_samples or 5000
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")

    index_to_file = {int(f.stem.split("-")[1]): f for f in parquet_files}

    all_triples = []
    total_needed = n * 2  # buffer 2x for filtering
    per_rel = max(1, total_needed // len(RELATION_SHARD_MAP))

    for rel_name, shard_indices in RELATION_SHARD_MAP.items():
        needed = per_rel
        for sidx in shard_indices:
            if needed <= 0:
                break
            fpath = index_to_file.get(sidx)
            if fpath is None:
                continue

            pf = pq.ParquetFile(fpath)
            n_groups = pf.metadata.num_row_groups
            for gi in range(n_groups):
                if needed <= 0:
                    break
                tbl = pf.read_row_groups([gi], columns=["subject", "predicate", "object"])
                for row in tbl.to_pylist():
                    head = _concept_label(row["subject"])
                    tail = _concept_label(row["object"])
                    all_triples.append({
                        "head": head,
                        "relation": rel_name,
                        "tail": tail,
                        "weight": 1.0,
                    })
                    needed -= 1
                    if needed <= 0:
                        break
            if len(all_triples) >= n:
                break
        if len(all_triples) >= n:
            break

    rng = np.random.RandomState(42)
    rng.shuffle(all_triples)
    all_triples = all_triples[:n]

    logger.info(f"Loaded {len(all_triples)} ConceptNet triples from local parquet")
    rel_counts = Counter(t["relation"] for t in all_triples)
    logger.info(f"Relation distribution: {dict(rel_counts.most_common())}")
    return all_triples


def filter_known_relations(triples: List[dict]) -> List[dict]:
    known = CONCEPTNET_RELATION_MAP.keys()
    filtered = [t for t in triples if t["relation"] in known]
    unknown_rels = set(t["relation"] for t in triples) - known
    if unknown_rels:
        logger.debug(f"Unknown relation types: {sorted(unknown_rels)}")
    logger.info(f"Relation filter: {len(triples)} -> {len(filtered)} (kept)")
    return filtered


def build_concept_graph(triples: List[dict]) -> Tuple[Dict[str, int], List[dict]]:
    concept_to_id: Dict[str, int] = {}
    edges: List[dict] = []
    next_id = 0

    for t in triples:
        for c in (t["head"], t["tail"]):
            if c not in concept_to_id:
                concept_to_id[c] = next_id
                next_id += 1
        edges.append({
            "head_id": concept_to_id[t["head"]],
            "head": t["head"],
            "relation": t["relation"],
            "tail_id": concept_to_id[t["tail"]],
            "tail": t["tail"],
            "weight": t.get("weight", 1.0),
        })

    logger.info(f"Built graph: {len(concept_to_id)} concepts, {len(edges)} edges")
    return concept_to_id, edges
