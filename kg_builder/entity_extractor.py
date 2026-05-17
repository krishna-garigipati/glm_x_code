import logging
from typing import Dict, List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EntityExtractor:
    def __init__(
        self,
        ner_labels: Optional[List[str]] = None,
        sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_np_length: int = 1,
    ):
        self._ner_labels = set(ner_labels or [
            "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP",
        ])
        self._sbert: Optional[SentenceTransformer] = None
        self._sbert_model_name = sbert_model_name
        self._min_np_length = min_np_length

    def _lazy_load_sbert(self):
        if self._sbert is not None:
            return
        self._sbert = SentenceTransformer(self._sbert_model_name)
        logger.info("SBERT model '%s' loaded for entity extraction", self._sbert_model_name)

    def extract(self, sentence: Dict) -> List[str]:
        entities: Set[str] = set()
        doc = sentence.get("doc")
        if doc is None:
            return []
        for ent in sentence.get("entities", []):
            label = ent.label_
            if label in self._ner_labels:
                text = ent.text.strip().lower()
                if text:
                    entities.add(text)
        for chunk in doc.noun_chunks:
            text = chunk.text.strip().lower()
            words = text.split()
            if len(words) >= self._min_np_length and not text.isspace():
                entities.add(text)
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "attr", "iobj"):
                full_text = self._expand_np(token).strip().lower()
                if len(full_text) > 1 and not full_text.isspace():
                    entities.add(full_text)
        sorted_entities = sorted(entities, key=len, reverse=True)
        return sorted_entities

    def extract_key_concepts(self, text: str, top_k: int = 5) -> List[str]:
        self._lazy_load_sbert()
        doc = self._nlp(text) if hasattr(self, '_nlp') and self._nlp else None
        candidates = []
        if doc:
            for chunk in doc.noun_chunks:
                candidates.append(chunk.text.strip().lower())
        if len(candidates) <= top_k:
            return candidates
        emb = self._sbert.encode(text)
        chunk_embs = self._sbert.encode(candidates)
        sims = np.dot(chunk_embs, emb) / (
            np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(emb) + 1e-8
        )
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [candidates[i] for i in top_indices]

    @staticmethod
    def _expand_np(token) -> str:
        tokens = []
        for child in token.subtree:
            if child.dep_ in ("compound", "amod", "det", "nmod", "poss"):
                tokens.append(child.text)
        tokens.append(token.text)
        for child in token.children:
            if child.dep_ == "conj":
                tokens.append(child.text)
        return " ".join(tokens)
