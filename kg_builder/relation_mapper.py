import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RelationMapper:
    """
    Zero-heuristic relation mapper.

    - No static word lists, no lemmatization rules, no if/else branches.
    - The raw connector text (lowercased) IS the relation label.
    - The relation embedding is computed from the full context:
        f"{subj} {connector} {obj}" + optional sentence context
    - Embedding is used for semantic relation matching during graph queries.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 model: Optional[SentenceTransformer] = None):
        self._model_name = model_name
        self._model = model
        self._cache: Dict[str, np.ndarray] = {}

    def _lazy_init(self):
        if self._model is not None:
            return
        self._model = SentenceTransformer(self._model_name)

    def classify(
        self, subj: str, connector: str, obj: str, context: str = ""
    ) -> Tuple[str, float]:
        rel = connector.strip().lower()
        if not rel:
            rel = connector.strip().lower()
        return rel, 1.0

    def classify_relation_text(self, text: str) -> Tuple[str, float]:
        rel = text.strip().lower()
        if not rel:
            rel = text.strip().lower()
        return rel, 1.0

    def encode_relation(
        self, subj: str, connector: str, obj: str, context: str = ""
    ) -> np.ndarray:
        self._lazy_init()
        relation_text = f"{subj} {connector} {obj}"
        if context:
            relation_text = f"{relation_text} {context}"
        cached = self._cache.get(relation_text)
        if cached is not None:
            return cached
        emb = self._model.encode(relation_text, normalize_embeddings=True)
        self._cache[relation_text] = emb
        return emb

    def encode_relation_text(self, text: str) -> np.ndarray:
        self._lazy_init()
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        emb = self._model.encode(text, normalize_embeddings=True)
        self._cache[text] = emb
        return emb

    @property
    def known_relations(self) -> List[str]:
        return []
