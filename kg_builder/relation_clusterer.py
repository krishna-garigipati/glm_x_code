import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RelationClusterer:
    """
    Online clustering of relation embeddings with adaptive threshold.

    Groups semantically similar connector texts (e.g., "developed" and "created")
    into the same cluster while keeping distinct meanings separate
    (e.g., "in" as location vs "in" as involvement).

    The threshold adapts based on:
    - Number of clusters seen (more clusters → lower threshold)
    - Internal coherence of each cluster
    - Merge/split success rate
    """

    def __init__(self, initial_threshold: float = 0.55, min_threshold: float = 0.35, max_threshold: float = 0.85):
        self._initial_threshold = initial_threshold
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._centroids: Dict[str, np.ndarray] = {}
        self._member_texts: Dict[str, List[str]] = {}
        self._member_embeddings: Dict[str, List[np.ndarray]] = {}
        self._member_counts: Dict[str, int] = {}
        self._internal_sims: Dict[str, List[float]] = {}
        self._next_id = 0
        self._total_assigned = 0
        self._merge_successes = 0
        self._merge_attempts = 0

    @property
    def num_clusters(self) -> int:
        return len(self._centroids)

    @property
    def threshold(self) -> float:
        t = self._initial_threshold - 0.05 * np.log1p(self.num_clusters)
        return float(np.clip(t, self._min_threshold, self._max_threshold))

    def _ema(self, old: np.ndarray, new: np.ndarray, count: int) -> np.ndarray:
        alpha = 1.0 / max(count + 1, 1)
        centroid = (1 - alpha) * old + alpha * new
        return centroid / (np.linalg.norm(centroid) + 1e-8)

    def assign(self, raw_text: str, embedding: np.ndarray) -> str:
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        if self.num_clusters == 0:
            return self._create_cluster(raw_text, emb)

        best_id = None
        best_sim = -1.0
        for cid, centroid in self._centroids.items():
            sim = float(np.dot(centroid, emb))
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        adaptive_threshold = self._cluster_adaptive_threshold(best_id, best_sim)

        if best_sim > adaptive_threshold and best_id is not None:
            return self._add_to_cluster(best_id, raw_text, emb, best_sim)
        else:
            return self._create_cluster(raw_text, emb)

    def _cluster_adaptive_threshold(self, cluster_id: Optional[str], sim: float) -> float:
        base = self.threshold
        if cluster_id is None or cluster_id not in self._internal_sims:
            return base
        sims = self._internal_sims[cluster_id]
        if len(sims) < 3:
            return base
        mean_sim = float(np.mean(sims[-20:]))
        std_sim = float(np.std(sims[-20:])) + 1e-6
        return max(base, mean_sim - 1.5 * std_sim)

    def _create_cluster(self, raw_text: str, emb: np.ndarray) -> str:
        cid = f"rel_{self._next_id}"
        self._next_id += 1
        self._centroids[cid] = emb
        self._member_texts[cid] = [raw_text]
        self._member_embeddings[cid] = [emb]
        self._member_counts[cid] = 1
        self._internal_sims[cid] = [1.0]
        self._total_assigned += 1
        return cid

    def _add_to_cluster(self, cid: str, raw_text: str, emb: np.ndarray, sim: float) -> str:
        count = self._member_counts[cid]
        self._centroids[cid] = self._ema(self._centroids[cid], emb, count)
        self._member_texts[cid].append(raw_text)
        self._member_embeddings[cid].append(emb)
        self._member_counts[cid] = count + 1
        self._internal_sims[cid].append(sim)
        self._total_assigned += 1

        if self._member_counts[cid] % 20 == 0:
            self._check_split(cid)

        return cid

    def _check_split(self, cid: str):
        embs = self._member_embeddings[cid]
        if len(embs) < 10:
            return
        recent = np.stack(embs[-10:])
        centroid = self._centroids[cid]
        sims = np.dot(recent, centroid)
        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))
        if mean_sim < 0.4 and std_sim < 0.15:
            logger.info("Splitting cluster %s (mean_sim=%.3f, std=%.3f)", cid, mean_sim, std_sim)
            low_sim_mask = sims < mean_sim - std_sim
            if np.any(low_sim_mask):
                outlier_emb = recent[low_sim_mask][0]
                outlier_text = self._member_texts[cid][-10:][low_sim_mask.tolist().index(True) if isinstance(low_sim_mask, np.ndarray) else 0]
                self._create_cluster(outlier_text, outlier_emb)
                self._member_texts[cid] = self._member_texts[cid][:-10] + [self._member_texts[cid][-10:][i] for i in range(10) if not low_sim_mask[i]]
                self._member_embeddings[cid] = self._member_embeddings[cid][:-10] + [self._member_embeddings[cid][-10:][i] for i in range(10) if not low_sim_mask[i]]
                self._member_counts[cid] = len(self._member_texts[cid])

    def merge_similar_clusters(self):
        self._merge_attempts += 1
        merged = 0
        cids = list(self._centroids.keys())
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                ci, cj = cids[i], cids[j]
                if ci not in self._centroids or cj not in self._centroids:
                    continue
                sim = float(np.dot(self._centroids[ci], self._centroids[cj]))
                if sim > self.threshold + 0.1:
                    smaller, larger = (ci, cj) if self._member_counts[ci] < self._member_counts[cj] else (cj, ci)
                    count = self._member_counts[larger]
                    self._centroids[larger] = self._ema(self._centroids[larger], self._centroids[smaller], count)
                    self._member_texts[larger].extend(self._member_texts[smaller])
                    self._member_counts[larger] += self._member_counts[smaller]
                    del self._centroids[smaller]
                    del self._member_texts[smaller]
                    del self._member_embeddings[smaller]
                    del self._member_counts[smaller]
                    del self._internal_sims[smaller]
                    merged += 1
                    self._merge_successes += 1
        if merged:
            logger.info("Merged %d cluster pairs, total clusters: %d", merged, self.num_clusters)

    def get_cluster_info(self) -> List[Dict]:
        info = []
        for cid in sorted(self._centroids.keys(), key=lambda x: self._member_counts.get(x, 0), reverse=True):
            texts = self._member_texts.get(cid, [])
            unique_texts = list(set(texts))
            most_common = max(set(texts), key=texts.count) if texts else ""
            info.append({
                "cluster_id": cid,
                "count": self._member_counts.get(cid, 0),
                "unique_connectors": len(unique_texts),
                "top_connector": most_common,
                "all_connectors": sorted(unique_texts)[:10],
            })
        return info
