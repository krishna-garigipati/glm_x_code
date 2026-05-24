import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import deque

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EntityResolver:
    def __init__(
        self,
        embed_merge_threshold: float = 0.92,
        sbert_model_name: str = "BAAI/bge-small-en-v1.5",
        model: Optional[SentenceTransformer] = None,
    ):
        self._merge_threshold = embed_merge_threshold
        self._sbert_model_name = sbert_model_name
        self._sbert: Optional[SentenceTransformer] = model
        self._canonical_to_id: Dict[str, int] = {}
        self._id_to_canonical: Dict[int, str] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._surface_forms: Dict[str, str] = {}
        self._next_id: int = 1
        self._entity_freq: Dict[str, int] = {}
        self._resolve_count: int = 0

    def _lazy_init(self):
        if self._sbert is not None:
            return
        self._sbert = SentenceTransformer(self._sbert_model_name)
        logger.info("EntityResolver SBERT model '%s' loaded", self._sbert_model_name)

    @staticmethod
    def _normalize(label: str) -> str:
        label = label.lower().strip()
        if label.endswith("'s"):
            label = label[:-2]
        elif label.endswith("s'") and len(label) > 2:
            label = label[:-1]
        return label.strip()

    def resolve_batch(self, labels: List[str]) -> List[str]:
        self._lazy_init()
        results = [None] * len(labels)
        unresolved_idx: List[int] = []
        unresolved_norm: List[str] = []

        for i, label in enumerate(labels):
            n = self._normalize(label)
            if not n:
                results[i] = n
            elif n in self._canonical_to_id:
                self._entity_freq[n] += 1
                self._resolve_count += 1
                results[i] = n
            elif n in self._surface_forms:
                c = self._surface_forms[n]
                self._entity_freq[c] += 1
                self._resolve_count += 1
                results[i] = c
            elif len(self._canonical_to_id) > 0:
                unresolved_idx.append(i)
                unresolved_norm.append(n)
            else:
                results[i] = self._register_new(n)

        if unresolved_norm:
            embs = self._sbert.encode(unresolved_norm, normalize_embeddings=True)
            if not self._embeddings:
                for idx, norm, emb in zip(unresolved_idx, unresolved_norm, embs):
                    canon = self._register_new(norm)
                    self._embeddings[norm] = emb
                    results[idx] = canon
                return results

            # Batch compare: stack all existing embeddings, broadcast dot product
            existing_labels = list(self._embeddings.keys())
            existing_arr = np.stack([self._embeddings[k] for k in existing_labels])
            # embs: (M, D)  existing_arr: (N, D)  → sims: (M, N)
            sims = np.dot(embs, existing_arr.T)
            best_indices = np.argmax(sims, axis=1)
            best_sims = np.max(sims, axis=1)

            for idx, norm, emb, best_n, best_sim in zip(
                unresolved_idx, unresolved_norm, embs, best_indices, best_sims
            ):
                if best_sim >= self._merge_threshold:
                    best = existing_labels[best_n]
                    self._surface_forms[norm] = best
                    self._entity_freq[best] += 1
                    self._resolve_count += 1
                    self._embeddings[best] = (
                        self._embeddings[best] * 0.85 + emb * 0.15
                    )
                    self._embeddings[best] /= (
                        np.linalg.norm(self._embeddings[best]) + 1e-8
                    )
                    results[idx] = best
                else:
                    canon = self._register_new(norm)
                    self._embeddings[norm] = emb
                    results[idx] = canon

        return results

    def start_batch(self):
        pass

    def resolve(self, label: str, neighbors: Optional[Set[str]] = None) -> str:
        label_norm = self._normalize(label)
        if not label_norm:
            return label_norm
        if label_norm in self._canonical_to_id:
            self._entity_freq[label_norm] = self._entity_freq.get(label_norm, 0) + 1
            self._resolve_count += 1
            return label_norm
        if label_norm in self._surface_forms:
            canonical = self._surface_forms[label_norm]
            self._entity_freq[canonical] = self._entity_freq.get(canonical, 0) + 1
            self._resolve_count += 1
            return canonical
        if len(self._canonical_to_id) > 0:
            self._resolve_count += 1
            merged = self._try_embedding_merge(label_norm)
            if merged:
                self._surface_forms[label_norm] = merged
                self._entity_freq[merged] = self._entity_freq.get(merged, 0) + 1
                return merged
        return self._register_new(label_norm)

    def _try_embedding_merge(self, label: str) -> Optional[str]:
        self._lazy_init()
        emb = self._sbert.encode(label, normalize_embeddings=True)
        best_canonical = None
        best_sim = -1.0
        for canonical, existing_emb in self._embeddings.items():
            sim = float(np.dot(emb, existing_emb))
            if sim > best_sim and sim >= self._merge_threshold:
                best_sim = sim
                best_canonical = canonical
        if best_canonical is not None:
            self._embeddings[best_canonical] = (
                self._embeddings[best_canonical] * 0.85 + emb * 0.15
            )
            self._embeddings[best_canonical] /= (
                np.linalg.norm(self._embeddings[best_canonical]) + 1e-8
            )
            return best_canonical
        return None

    def _register_new(self, label: str) -> str:
        nid = self._next_id
        self._next_id += 1
        self._canonical_to_id[label] = nid
        self._id_to_canonical[nid] = label
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
        return self._embeddings.get(label)

    def compute_all_embeddings(self) -> Dict[str, np.ndarray]:
        self._lazy_init()
        missing = [lbl for lbl in self._canonical_to_id if lbl not in self._embeddings]
        if missing:
            embs = self._sbert.encode(missing)
            for lbl, emb in zip(missing, embs):
                self._embeddings[lbl] = emb / (np.linalg.norm(emb) + 1e-8)
        return dict(self._embeddings)
