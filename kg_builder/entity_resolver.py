import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import deque

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EntityResolver:
    def __init__(
        self,
        embed_similarity_threshold: float = 0.72,
        graph_neighbor_overlap_threshold: float = 0.4,
        sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self._base_embed_threshold = embed_similarity_threshold
        self._base_graph_threshold = graph_neighbor_overlap_threshold
        self._embed_threshold = embed_similarity_threshold
        self._graph_threshold = graph_neighbor_overlap_threshold
        self._sbert_model_name = sbert_model_name
        self._sbert: Optional[SentenceTransformer] = None
        self._canonical_to_id: Dict[str, int] = {}
        self._id_to_canonical: Dict[int, str] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._neighbors: Dict[str, Set[str]] = {}
        self._surface_forms: Dict[str, str] = {}
        self._next_id: int = 1
        self._resolve_count: int = 0
        self._merge_window: deque = deque(maxlen=20)
        self._entity_freq: Dict[str, int] = {}
        self._merge_success_by_type: Dict[str, int] = {"name": 0, "embed": 0, "graph": 0}

    def _lazy_init(self):
        if self._sbert is not None:
            return
        self._sbert = SentenceTransformer(self._sbert_model_name)

    @staticmethod
    def _normalize(label: str) -> str:
        label = label.lower().strip()
        if label.endswith("'s"):
            label = label[:-2]
        elif label.endswith("s'") and len(label) > 2:
            label = label[:-1]
        return label.strip()

    def _update_thresholds(self):
        self._resolve_count += 1
        merge_rate = 0.5
        if self._merge_window:
            merge_rate = sum(self._merge_window) / len(self._merge_window)
        entity_progress = min(1.0, len(self._canonical_to_id) / 200)
        adapt = 0.12 * (0.5 - merge_rate)
        decay = 0.15 * (1.0 - entity_progress)
        self._embed_threshold = max(0.50, self._base_embed_threshold + adapt - decay)
        self._graph_threshold = max(0.25, self._base_graph_threshold + adapt * 0.5 - decay * 0.5)
        self._embed_threshold = round(self._embed_threshold, 3)
        self._graph_threshold = round(self._graph_threshold, 3)

    def resolve(self, label: str, neighbors: Optional[Set[str]] = None) -> str:
        label_norm = self._normalize(label)
        if not label_norm:
            return label_norm
        if label_norm in self._canonical_to_id:
            self._entity_freq[label_norm] = self._entity_freq.get(label_norm, 0) + 1
            self._extend_neighbors(label_norm, neighbors)
            self._merge_window.append(True)
            self._merge_success_by_type["name"] += 1
            return label_norm
        if label_norm in self._surface_forms:
            canonical = self._surface_forms[label_norm]
            self._entity_freq[canonical] = self._entity_freq.get(canonical, 0) + 1
            self._extend_neighbors(canonical, neighbors)
            self._merge_window.append(True)
            self._merge_success_by_type["name"] += 1
            return canonical
        if len(self._canonical_to_id) > 0:
            self._update_thresholds()
            merged = self._try_name_merge(label_norm)
            if merged:
                self._surface_forms[label_norm] = merged
                self._entity_freq[merged] = self._entity_freq.get(merged, 0) + 1
                self._extend_neighbors(merged, neighbors)
                self._merge_window.append(True)
                self._merge_success_by_type["name"] += 1
                return merged
            merged = self._try_embedding_merge(label_norm, neighbors)
            if merged:
                self._surface_forms[label_norm] = merged
                self._entity_freq[merged] = self._entity_freq.get(merged, 0) + 1
                self._merge_window.append(True)
                self._merge_success_by_type["embed"] += 1
                return merged
            merged = self._try_graph_merge(label_norm, neighbors)
            if merged:
                self._surface_forms[label_norm] = merged
                self._entity_freq[merged] = self._entity_freq.get(merged, 0) + 1
                self._merge_window.append(True)
                self._merge_success_by_type["graph"] += 1
                return merged
        self._merge_window.append(False)
        return self._register_new(label_norm, neighbors)

    def _try_name_merge(self, label: str) -> Optional[str]:
        label_words = set(label.split())
        if len(label_words) < 1:
            return None
        best_match = None
        best_score = 0.0
        threshold = 0.5
        cnt = len(self._canonical_to_id)
        if cnt < 50:
            threshold = 0.4
        elif cnt > 200:
            threshold = 0.6
        for canonical in list(self._canonical_to_id.keys()):
            canonical_words = set(canonical.split())
            overlap = label_words & canonical_words
            if len(overlap) == 0:
                continue
            score = len(overlap) / min(len(label_words), len(canonical_words))
            if score >= threshold and score > best_score:
                best_score = score
                best_match = canonical
        if best_match is not None:
            return best_match
        for canonical in list(self._canonical_to_id.keys()):
            if len(label) >= 4 and len(canonical) >= 4:
                if label in canonical or canonical in label:
                    freq_factor = 1.0
                    f = self._entity_freq.get(canonical, 0)
                    if f > 2:
                        freq_factor = 0.85
                    if freq_factor >= 0.85:
                        return canonical
        return None

    def _extend_neighbors(self, canonical: str, neighbors: Optional[Set[str]]):
        if neighbors and canonical in self._neighbors:
            self._neighbors[canonical] |= neighbors

    def _try_embedding_merge(self, label: str, neighbors: Optional[Set[str]]) -> Optional[str]:
        self._lazy_init()
        emb = self._sbert.encode(label)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        best_canonical = None
        best_sim = -1.0
        for canonical, existing_emb in self._embeddings.items():
            sim = float(np.dot(emb, existing_emb))
            word_bonus = 0.0
            label_w = set(label.split())
            canon_w = set(canonical.split())
            if label_w & canon_w:
                word_bonus = 0.04
            freq_penalty = 0.0
            f = self._entity_freq.get(canonical, 0)
            if f > 5:
                freq_penalty = 0.02
            effective_sim = sim + word_bonus - freq_penalty
            if effective_sim > best_sim:
                best_sim = effective_sim
                best_canonical = canonical
        if best_sim >= self._embed_threshold and best_canonical is not None:
            self._embeddings[best_canonical] = (
                self._embeddings[best_canonical] * 0.7 + emb * 0.3
            )
            self._embeddings[best_canonical] /= (
                np.linalg.norm(self._embeddings[best_canonical]) + 1e-8
            )
            self._extend_neighbors(best_canonical, neighbors)
            return best_canonical
        return None

    def _try_graph_merge(self, label: str, neighbors: Optional[Set[str]]) -> Optional[str]:
        if not neighbors or len(neighbors) == 0:
            return None
        best_canonical = None
        best_overlap = -1.0
        for canonical, existing_neighbors in self._neighbors.items():
            if len(existing_neighbors) == 0:
                continue
            intersection = len(neighbors & existing_neighbors)
            union = len(neighbors | existing_neighbors)
            if union == 0:
                continue
            overlap = intersection / union
            freq_penalty = 0.0
            f = self._entity_freq.get(canonical, 0)
            if f > 5:
                freq_penalty = 0.05
            adjusted = overlap - freq_penalty
            if adjusted > best_overlap:
                best_overlap = adjusted
                best_canonical = canonical
        if best_overlap >= self._graph_threshold and best_canonical is not None:
            self._neighbors[best_canonical] |= neighbors
            return best_canonical
        return None

    def _register_new(self, label: str, neighbors: Optional[Set[str]]) -> str:
        nid = self._next_id
        self._next_id += 1
        self._canonical_to_id[label] = nid
        self._id_to_canonical[nid] = label
        self._neighbors[label] = neighbors or set()
        self._entity_freq[label] = 1
        return label

    def get_canonical_id(self, label: str) -> Optional[int]:
        label_norm = label.lower().strip()
        if label_norm in self._canonical_to_id:
            return self._canonical_to_id[label_norm]
        if label_norm in self._surface_forms:
            canonical = self._surface_forms[label_norm]
            if canonical in self._canonical_to_id:
                return self._canonical_to_id[canonical]
        return None

    def get_id_to_label(self) -> Dict[int, str]:
        return dict(self._id_to_canonical)

    def get_labels(self) -> List[str]:
        return list(self._canonical_to_id.keys())

    def get_embedding(self, label: str) -> Optional[np.ndarray]:
        label_norm = label.lower().strip()
        canonical = self._surface_forms.get(label_norm, label_norm)
        return self._embeddings.get(canonical)

    def compute_all_embeddings(self) -> Dict[str, np.ndarray]:
        self._lazy_init()
        missing = [lbl for lbl in self._canonical_to_id if lbl not in self._embeddings]
        if missing:
            embs = self._sbert.encode(missing)
            for lbl, emb in zip(missing, embs):
                self._embeddings[lbl] = emb / (np.linalg.norm(emb) + 1e-8)
        return dict(self._embeddings)
