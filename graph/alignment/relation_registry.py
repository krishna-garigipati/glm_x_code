"""
Zero-heuristic relation canonicalization registry.

Persists relation cluster centroids across pipeline runs so that
semantically similar relations from different corpora (e.g.,
"related to" and "is related to") map to the same canonical ID.

No string patterns, no word lists, no lemma rules.
All decisions are cosine similarity in BGE embedding space.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RelationRegistry:
    """
    Persistent dynamic relation canonicalization.

    Maintains cluster centroids for relation embeddings.
    New relations are assigned to the nearest cluster if similarity
    exceeds an adaptive threshold; otherwise a new cluster is created.
    """

    def __init__(
        self,
        sbert_model: Optional[SentenceTransformer] = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        registry_path: Optional[str] = None,
        initial_threshold: float = 0.78,
    ):
        self._sbert = sbert_model
        self._model_name = model_name
        self._centroids: Dict[str, np.ndarray] = {}
        self._members: Dict[str, List[str]] = {}
        self._member_counts: Dict[str, int] = {}
        self._canonical_to_cluster: Dict[str, str] = {}
        self._next_cluster_id: int = 0
        self._threshold: float = initial_threshold
        self._similarity_history: List[float] = []
        self._total_canonicalizations: int = 0
        self._total_merges: int = 0
        self._total_new_clusters: int = 0

        if registry_path and os.path.exists(registry_path):
            self.load(registry_path)

    def _get_sbert(self):
        if self._sbert is not None:
            return self._sbert
        self._sbert = SentenceTransformer(self._model_name)
        return self._sbert

    def canonicalize_batch(
        self, texts: List[str], embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Canonicalize a batch of relation texts.

        Args:
            texts: Raw relation strings.
            embeddings: Pre-computed embeddings (optional).

        Returns:
            (canonical_labels, embeddings) — one per input text.
        """
        if not texts:
            return [], np.array([])

        sbert = self._get_sbert()
        if embeddings is None:
            embeddings = sbert.encode(texts, normalize_embeddings=True)

        results: List[str] = [""] * len(texts)

        if not self._centroids:
            for text, emb in zip(texts, embeddings):
                cid = self._create_cluster(text, emb)
                results.append(cid)
            results = results[0:0]  # reset
            for text, emb in zip(texts, embeddings):
                cid = self._create_cluster(text, emb)
                results.append(cid)
            self._total_canonicalizations += len(texts)
            return results, embeddings

        cids = list(self._centroids.keys())
        centroid_arr = np.array([self._centroids[c] for c in cids])

        sims = embeddings @ centroid_arr.T
        best_sims = np.max(sims, axis=1)
        best_indices = np.argmax(sims, axis=1)

        self._similarity_history.extend(best_sims.tolist())
        self._adapt_threshold()

        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            if text in self._canonical_to_cluster:
                results[i] = self._canonical_to_cluster[text]
                continue

            if best_sims[i] >= self._threshold:
                cid = cids[best_indices[i]]
                self._add_to_cluster(cid, text, emb)
                results[i] = cid
                self._total_merges += 1
            else:
                cid = self._create_cluster(text, emb)
                results[i] = cid
                self._total_new_clusters += 1

        self._total_canonicalizations += len(texts)
        return results, embeddings

    def canonicalize(
        self, text: str, embedding: Optional[np.ndarray] = None
    ) -> str:
        """Single-text canonicalization."""
        texts, _ = self.canonicalize_batch(
            [text], None if embedding is None else embedding[None]
        )
        return texts[0]

    def _create_cluster(self, text: str, emb: np.ndarray) -> str:
        cid = f"rel_{self._next_cluster_id}"
        self._next_cluster_id += 1
        self._centroids[cid] = emb.copy()
        self._members[cid] = [text]
        self._member_counts[cid] = 1
        self._canonical_to_cluster[text] = cid
        return cid

    def _add_to_cluster(self, cid: str, text: str, emb: np.ndarray):
        count = self._member_counts.get(cid, 0)
        self._centroids[cid] = (
            self._centroids[cid] * 0.85 + emb * 0.15
        )
        self._centroids[cid] /= np.linalg.norm(self._centroids[cid]) + 1e-8
        self._members.setdefault(cid, []).append(text)
        self._member_counts[cid] = count + 1
        self._canonical_to_cluster[text] = cid

    def _adapt_threshold(self):
        if len(self._similarity_history) < 50:
            return
        recent = self._similarity_history[-200:]
        mean_sim = float(np.mean(recent))
        std_sim = float(np.std(recent))
        adaptive = mean_sim - 1.5 * std_sim
        self._threshold = max(0.65, min(0.92, adaptive))

    def get_top_text(self, cid: str) -> str:
        members = self._members.get(cid, [])
        if not members:
            return cid
        from collections import Counter
        return Counter(members).most_common(1)[0][0]

    def get_cluster_info(self) -> List[dict]:
        info = []
        for cid in sorted(self._centroids.keys(),
                          key=lambda x: self._member_counts.get(x, 0),
                          reverse=True):
            texts = self._members.get(cid, [])
            unique = list(set(texts))
            top = max(set(texts), key=texts.count) if texts else cid
            info.append({
                "cluster_id": cid,
                "count": self._member_counts.get(cid, 0),
                "unique": len(unique),
                "top_text": top,
            })
        return info

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def num_clusters(self) -> int:
        return len(self._centroids)

    def state_dict(self) -> dict:
        return {
            "centroids": {k: v for k, v in self._centroids.items()},
            "members": self._members,
            "member_counts": self._member_counts,
            "canonical_to_cluster": self._canonical_to_cluster,
            "next_cluster_id": self._next_cluster_id,
            "threshold": self._threshold,
            "total_canonicalizations": self._total_canonicalizations,
            "total_merges": self._total_merges,
            "total_new_clusters": self._total_new_clusters,
        }

    def save(self, path: str):
        path = str(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)
        logger.info(
            "RelationRegistry saved: %d clusters, %d canonicals -> %s",
            self.num_clusters, len(self._canonical_to_cluster), path,
        )

    def load(self, path: str):
        path = str(path)
        if not os.path.exists(path):
            logger.warning("RelationRegistry not found: %s", path)
            return
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._centroids = state["centroids"]
        self._members = state["members"]
        self._member_counts = state["member_counts"]
        self._canonical_to_cluster = state["canonical_to_cluster"]
        self._next_cluster_id = state["next_cluster_id"]
        self._threshold = state.get("threshold", 0.78)
        self._total_canonicalizations = state.get("total_canonicalizations", 0)
        self._total_merges = state.get("total_merges", 0)
        self._total_new_clusters = state.get("total_new_clusters", 0)
        logger.info(
            "RelationRegistry loaded: %d clusters, threshold=%.3f",
            self.num_clusters, self._threshold,
        )

    def absorb_clusterer(self, clusterer) -> int:
        """Absorb clusters from a kg_builder RelationClusterer."""
        from kg_builder.relation_clusterer import RelationClusterer
        if not isinstance(clusterer, RelationClusterer):
            return 0
        new_ids = []
        for info in clusterer.get_cluster_info():
            cid = info["cluster_id"]
            if cid in self._centroids:
                continue
            members = clusterer._member_texts.get(cid, [])
            centroid = clusterer._centroids.get(cid)
            if centroid is None or not members:
                continue
            self._centroids[cid] = centroid
            self._members[cid] = members
            self._member_counts[cid] = len(members)
            for text in set(members):
                self._canonical_to_cluster[text] = cid
            new_ids.append(cid)
        if new_ids:
            next_num = max(int(cid.split("_")[1]) for cid in self._centroids) + 1
            self._next_cluster_id = max(self._next_cluster_id, next_num)
        logger.info("Absorbed %d clusters from RelationClusterer", len(new_ids))
        return len(new_ids)
