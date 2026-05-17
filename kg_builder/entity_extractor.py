import logging
from typing import Dict, List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_STOP_ENTITIES = frozenset({
    "someone", "something", "everyone", "everything", "nobody", "nothing",
    "anyone", "anything", "somebody", "anybody", "one", "ones",
    "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "there", "here", "a", "an", "the",
})


class EntityExtractor:
    def __init__(
        self,
        ner_labels: List[str] = None,
        sbert_model_name: str = "BAAI/bge-small-en-v1.5",
        min_np_length: int = 2,
        extract_common_nouns: bool = True,
        model: Optional[SentenceTransformer] = None,
    ):
        self._ner_labels = ner_labels or []
        self._min_np_length = min_np_length
        self._extract_common_nouns = extract_common_nouns
        self._sbert = model
        self._sbert_model_name = sbert_model_name
        self._min_np_length = min_np_length
        self._extract_common_nouns = extract_common_nouns

    def _lazy_load_sbert(self):
        if self._sbert is not None:
            return
        self._sbert = SentenceTransformer(self._sbert_model_name)
        logger.info("SBERT model '%s' loaded for entity extraction", self._sbert_model_name)

    def _is_stop(self, text: str) -> bool:
        return text.strip().lower() in _STOP_ENTITIES

    def extract(self, sentence: Dict) -> List[str]:
        entities: Set[str] = set()
        doc = sentence.get("doc")
        if doc is None:
            return []

        for ent in sentence.get("entities", []):
            label = ent.label_
            if label in self._ner_labels:
                text = ent.text.strip()
                if text and not self._is_stop(text):
                    entities.add(text)

        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            words = text.split()
            if len(words) >= self._min_np_length and not text.isspace() and not self._is_stop(text):
                if not self._is_stop(chunk.root.text):
                    entities.add(text)

        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "attr", "iobj"):
                if self._is_stop(token.text):
                    continue
                full_text = self._expand_np(token).strip()
                if len(full_text) > 1 and not full_text.isspace():
                    if not self._is_stop(full_text):
                        entities.add(full_text)

        if self._extract_common_nouns:
            for token in doc:
                if token.pos_ == "NOUN" and not token.is_stop and not self._is_stop(token.text):
                    if token.dep_ not in ("det", "prep", "punct", "case"):
                        if len(token.text) > 2:
                            entities.add(token.text)
                if token.pos_ == "PROPN" and not self._is_stop(token.text):
                    if len(token.text) > 1:
                        entities.add(token.text)

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
            if child.dep_ in ("compound", "amod", "det", "nmod", "poss", "nummod"):
                if not child.is_stop or child.dep_ == "compound":
                    tokens.append(child.text)
        tokens.append(token.text)
        for child in token.children:
            if child.dep_ == "conj":
                if not child.is_stop:
                    tokens.append(child.text)
        result = " ".join(tokens)
        return result
