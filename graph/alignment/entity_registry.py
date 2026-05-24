"""
Zero-heuristic entity alignment registry.

Persists entity embedding space across pipeline runs so that
entities from different corpora are mapped to the same canonical
identity when they are semantically similar.

No string patterns, no word lists, no rules.
All decisions are cosine similarity in BGE embedding space.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EntityRegistry:
    """
    Persistent entity alignment using embedding similarity only.

    Maintains a growing set of canonical entity embeddings.
    New entities are aligned by: encode → cosine vs all canonicals → merge or register.
    Threshold adapts dynamically based on the similarity distribution observed.
    """

    def __init__(
        self,
        sbert_model: Optional[SentenceTransformer] = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        registry_path: Optional[str] = None,
        initial_threshold: float = 0.85,
    ):
        self._sbert = sbert_model
        self._model_name = model_name
        self._canonical_label_to_emb: Dict[str, np.ndarray] = {}
        self._canonical_label_to_id: Dict[str, int] = {}
        self._id_to_canonical_label: Dict[int, str] = {}
        self._surface_to_canonical: Dict[str, str] = {}
        self._next_id: int = 1

        # Threshold is not hardcoded — adapts from data
        self._threshold: float = initial_threshold
        self._similarity_history: List[float] = []
        self._total_alignments: int = 0
        self._total_merges: int = 0
        self._total_registrations: int = 0

        if registry_path and os.path.exists(registry_path):
            self.load(registry_path)

    def _get_sbert(self):
        if self._sbert is not None:
            return self._sbert
        self._sbert = SentenceTransformer(self._model_name)
        logger.info("EntityRegistry loaded SBERT model: %s", self._model_name)
        return self._sbert

    def align_batch(
        self, labels: List[str], embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Align a batch of entity labels against the existing registry.

        Args:
            labels: Raw entity labels.
            embeddings: Pre-computed embeddings (optional, saves one encode call).

        Returns:
            (canonical_labels, node_ids) — one per input label.
        """
        if not labels:
            return [], []

        sbert = self._get_sbert()
        if embeddings is None:
            embeddings = sbert.encode(labels, normalize_embeddings=True)

        results_labels: List[str] = [""] * len(labels)
        results_ids: List[int] = [0] * len(labels)

        if not self._canonical_label_to_emb:
            for i, (label, emb) in enumerate(zip(labels, embeddings)):
                canonical = label.lower().strip()
                self._canonical_label_to_emb[canonical] = emb
                nid = self._next_id
                self._canonical_label_to_id[canonical] = nid
                self._id_to_canonical_label[nid] = canonical
                self._next_id += 1
                results_labels[i] = canonical
                results_ids[i] = nid
                self._total_registrations += 1
            self._total_alignments += len(labels)
            return results_labels, results_ids

        existing_labels = list(self._canonical_label_to_emb.keys())
        existing_arr = np.array([self._canonical_label_to_emb[k] for k in existing_labels])

        sims = embeddings @ existing_arr.T
        best_sims: np.ndarray = np.max(sims, axis=1)
        best_indices: np.ndarray = np.argmax(sims, axis=1)

        self._similarity_history.extend(best_sims.tolist())
        self._adapt_threshold()

        for i, (label, emb) in enumerate(zip(labels, embeddings)):
            label_key = label.lower().strip()
            if label_key in self._surface_to_canonical:
                canonical = self._surface_to_canonical[label_key]
                nid = self._canonical_label_to_id[canonical]
                results_labels[i] = canonical
                results_ids[i] = nid
                continue

            if label_key in self._canonical_label_to_emb:
                nid = self._canonical_label_to_id[label_key]
                results_labels[i] = label_key
                results_ids[i] = nid
                continue

            if best_sims[i] >= self._threshold:
                canonical = existing_labels[best_indices[i]]
                nid = self._canonical_label_to_id[canonical]
                self._surface_to_canonical[label_key] = canonical
                self._canonical_label_to_emb[canonical] = (
                    self._canonical_label_to_emb[canonical] * 0.85 + emb * 0.15
                )
                self._canonical_label_to_emb[canonical] /= (
                    np.linalg.norm(self._canonical_label_to_emb[canonical]) + 1e-8
                )
                results_labels[i] = canonical
                results_ids[i] = nid
                self._total_merges += 1
            else:
                canonical = label_key
                self._canonical_label_to_emb[canonical] = emb
                nid = self._next_id
                self._canonical_label_to_id[canonical] = nid
                self._id_to_canonical_label[nid] = canonical
                self._next_id += 1
                results_labels[i] = canonical
                results_ids[i] = nid
                self._total_registrations += 1

        self._total_alignments += len(labels)
        return results_labels, results_ids

    def align(self, label: str, embedding: Optional[np.ndarray] = None) -> Tuple[str, int]:
        """Single-label alignment. Wraps align_batch."""
        labels, ids = self.align_batch([label], None if embedding is None else embedding[None])
        return labels[0], ids[0]

    def _adapt_threshold(self):
        if len(self._similarity_history) < 50:
            return
        recent = self._similarity_history[-200:]
        mean_sim = float(np.mean(recent))
        std_sim = float(np.std(recent))
        adaptive = mean_sim - 1.5 * std_sim
        self._threshold = max(0.70, min(0.95, adaptive))

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def size(self) -> int:
        return len(self._canonical_label_to_emb)

    def state_dict(self) -> dict:
        return {
            "canonical_label_to_emb": self._canonical_label_to_emb,
            "canonical_label_to_id": self._canonical_label_to_id,
            "id_to_canonical_label": {str(k): v for k, v in self._id_to_canonical_label.items()},
            "surface_to_canonical": self._surface_to_canonical,
            "next_id": self._next_id,
            "threshold": self._threshold,
            "total_alignments": self._total_alignments,
            "total_merges": self._total_merges,
            "total_registrations": self._total_registrations,
        }

    def save(self, path: str):
        path = str(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)
        logger.info("EntityRegistry saved: %d canonicals -> %s", self.size, path)

    def load(self, path: str):
        path = str(path)
        if not os.path.exists(path):
            logger.warning("EntityRegistry not found: %s", path)
            return
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._canonical_label_to_emb = state["canonical_label_to_emb"]
        self._canonical_label_to_id = state["canonical_label_to_id"]
        self._id_to_canonical_label = {int(k): v for k, v in state["id_to_canonical_label"].items()}
        self._surface_to_canonical = state["surface_to_canonical"]
        self._next_id = state["next_id"]
        self._threshold = state.get("threshold", 0.85)
        self._total_alignments = state.get("total_alignments", 0)
        self._total_merges = state.get("total_merges", 0)
        self._total_registrations = state.get("total_registrations", 0)
        logger.info(
            "EntityRegistry loaded: %d canonicals, %d surfaces, threshold=%.3f",
            self.size, len(self._surface_to_canonical), self._threshold,
        )

    def absorb_entity_resolver(self, er) -> int:
        """Absorb state from a kg_builder EntityResolver after a pipeline run."""
        er = er
        canonicals = er._canonical_to_id
        surfaces = er._surface_forms
        new = 0
        for label, nid in canonicals.items():
            if label not in self._canonical_label_to_id:
                self._canonical_label_to_id[label] = nid
                self._id_to_canonical_label[nid] = label
                emb = er._embeddings.get(label)
                if emb is not None:
                    self._canonical_label_to_emb[label] = emb
                new += 1
        for variant, canonical in surfaces.items():
            if variant not in self._surface_to_canonical:
                self._surface_to_canonical[variant] = canonical
        if self._next_id <= max(self._canonical_label_to_id.values(), default=0):
            self._next_id = max(self._canonical_label_to_id.values()) + 1
        logger.info("Absorbed %d new canonicals from EntityResolver", new)
        return new

    def get_canonical_id(self, label: str) -> Optional[int]:
        label_key = label.lower().strip()
        if label_key in self._canonical_label_to_id:
            return self._canonical_label_to_id[label_key]
        if label_key in self._surface_to_canonical:
            return self._canonical_label_to_id.get(self._surface_to_canonical[label_key])
        return None

    def get_all_canonical_labels(self) -> List[str]:
        return list(self._canonical_label_to_emb.keys())

    def get_embedding(self, label: str) -> Optional[np.ndarray]:
        return self._canonical_label_to_emb.get(label.lower().strip())
