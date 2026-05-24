import logging
from typing import Dict, List, Optional

import spacy
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self._nlp: Optional[spacy.Language] = None
        self._model_name = model_name

    def _lazy_load(self):
        if self._nlp is not None:
            return
        try:
            self._nlp = spacy.load(self._model_name)
            logger.info("spaCy model '%s' loaded", self._model_name)
        except OSError:
            logger.warning("spaCy model '%s' not found, downloading...", self._model_name)
            spacy.cli.download(self._model_name)
            self._nlp = spacy.load(self._model_name)

    def process(self, text: str) -> List[Dict]:
        self._lazy_load()
        doc = self._nlp(text)
        self._merge_entities(doc)
        sentences = []
        for sent in doc.sents:
            entities = list(sent.ents)
            sentences.append({
                "text": sent.text,
                "doc": sent.as_doc(),
                "entities": entities,
                "entity_labels": list(set(e.label_ for e in entities)),
            })
        return sentences

    def process_batch(self, texts: List[str], batch_size: int = 32) -> List[List[Dict]]:
        self._lazy_load()
        all_sentences = []
        for doc in self._nlp.pipe(texts, batch_size=batch_size):
            self._merge_entities(doc)
            sentences = []
            for sent in doc.sents:
                entities = list(sent.ents)
                sentences.append({
                    "text": sent.text,
                    "doc": sent.as_doc(),
                    "entities": entities,
                    "entity_labels": list(set(e.label_ for e in entities)),
                })
            all_sentences.append(sentences)
        return all_sentences

    @staticmethod
    def _merge_entities(doc: Doc):
        spans = []
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"):
                spans.append(ent)
        if spans:
            with doc.retokenize() as retokenizer:
                for span in spans:
                    retokenizer.merge(span)
