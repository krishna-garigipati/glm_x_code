import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EntitySpan:
    __slots__ = ("text", "tok_start", "tok_end", "dep", "pos", "label")
    def __init__(self, text: str, tok_start: int, tok_end: int, dep: str = "", pos: str = "", label: str = ""):
        self.text = text
        self.tok_start = tok_start
        self.tok_end = tok_end
        self.dep = dep
        self.pos = pos
        self.label = label

    def __repr__(self):
        return f"Span({self.text!r}[{self.tok_start}:{self.tok_end}]/{self.dep})"


class TripleExtractor:
    _COPULAR_CONNECTORS = frozenset({"is", "are", "was", "were", "be", "am"})
    _PARTITIVE_PHRASES = {
        "part": "part of",
        "component": "part of",
        "member": "part of",
        "constituent": "part of",
        "element": "part of",
        "segment": "part of",
        "type": "a type of",
        "kind": "a type of",
        "sort": "a type of",
        "form": "a type of",
        "variety": "a type of",
        "instance": "a type of",
        "example": "a type of",
    }
    _PARTITIVE_RE = re.compile(
        r"^(?:the |a |an )?(?P<word>part|component|member|constituent|element|segment|"
        r"type|kind|sort|form|variety|instance|example)s? of (?P<obj>.+?)\s*$",
        re.IGNORECASE,
    )

    def __init__(self, enable_llm: bool = False, llm_model: str = "phi-3-mini"):
        self._enable_llm = enable_llm
        self._llm_model = llm_model
        self._nlp: Optional[Any] = None
        self._relation_mapper: Any = None
        self._sbert: Optional[SentenceTransformer] = None
        self._coherence_threshold: float = 0.65

    def set_nlp(self, nlp):
        self._nlp = nlp

    def set_relation_mapper(self, mapper):
        self._relation_mapper = mapper

    def set_sbert(self, model: SentenceTransformer, coherence_threshold: float = 0.65):
        self._sbert = model
        self._coherence_threshold = coherence_threshold

    def _extract_spans(self, doc) -> List[EntitySpan]:
        spans: List[EntitySpan] = []
        seen_texts: set = set()

        non_punct_idx = [t.i for t in doc if not t.is_punct]
        for ent in doc.ents:
            if non_punct_idx and ent.start == 0 and non_punct_idx[-1] < ent.end:
                continue
            text = ent.text.strip()
            if text and len(text) > 1:
                key = (text.lower(), ent.start)
                if key not in seen_texts:
                    seen_texts.add(key)
                    spans.append(EntitySpan(
                        text=ent.text, tok_start=ent.start, tok_end=ent.end,
                        dep="ent", pos=ent.label_, label=ent.label_
                    ))

        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if not text or len(text) <= 1:
                continue
            key = (text.lower(), chunk.start)
            if key not in seen_texts:
                new_end = self._expand_chunk_end(doc, chunk.root.i, chunk.end)
                full_text = doc[chunk.start:new_end].text.strip()
                seen_texts.add(key)
                seen_texts.add((full_text.lower(), chunk.start))
                spans.append(EntitySpan(
                    text=full_text, tok_start=chunk.start, tok_end=new_end,
                    dep="np", pos=chunk.root.pos_
                ))

        # Add standalone ADJ/ADV tokens that act as subjects (e.g. "Hot" in "Hot is the opposite of cold.")
        for token in doc:
            if token.pos_ in ("ADJ", "ADV") and token.dep_ in ("nsubj", "nsubjpass", "attr"):
                if len(token.text) > 1:
                    key = (token.text.lower(), token.i)
                    if key not in seen_texts:
                        seen_texts.add(key)
                        spans.append(EntitySpan(
                            text=token.text, tok_start=token.i, tok_end=token.i + 1,
                            dep=token.dep_, pos=token.pos_, label=""
                        ))

        # Add bare NOUN/PROPN tokens not in any span (e.g. "penicillin" in "Alexander Fleming discovered penicillin.")
        existing_ranges = [(s.tok_start, s.tok_end) for s in spans]
        for token in doc:
            if (token.pos_ in ("NOUN", "PROPN") or token.dep_ == "pobj") and len(token.text) > 1:
                covered = any(tok_start <= token.i < tok_end for tok_start, tok_end in existing_ranges)
                if not covered:
                    key = (token.text.lower(), token.i)
                    if key not in seen_texts:
                        seen_texts.add(key)
                        spans.append(EntitySpan(
                            text=token.text, tok_start=token.i, tok_end=token.i + 1,
                            dep=token.dep_, pos=token.pos_, label=""
                        ))

        spans.sort(key=lambda s: s.tok_start)
        return spans

    @staticmethod
    def _expand_chunk_end(doc, root_i: int, chunk_end: int) -> int:
        end = chunk_end
        for tok in doc[root_i:]:
            if tok.dep_ in ("prep", "agent") and tok.head.i == root_i:
                for child in tok.subtree:
                    end = max(end, child.i + 1)
            elif tok.dep_ == "conj" and tok.head.i == root_i and tok.pos_ == "NOUN":
                for child in tok.subtree:
                    end = max(end, child.i + 1)
        return end

    def _connector_text(self, doc, span1: EntitySpan, span2: EntitySpan) -> str:
        if span2.tok_start <= span1.tok_end:
            return ""
        tokens_between = [t for t in doc if span1.tok_end <= t.i < span2.tok_start]
        connector = " ".join(t.text for t in tokens_between).strip()
        if not connector:
            connector = span1.text
        return connector

    def _split_partitive_object(self, connector: str, obj_text: str) -> Tuple[str, str]:
        conn = connector.strip().lower()
        if conn not in self._COPULAR_CONNECTORS:
            return connector, obj_text
        m = self._PARTITIVE_RE.match(obj_text.strip())
        if not m:
            return connector, obj_text
        word = m.group("word").lower()
        phrase = self._PARTITIVE_PHRASES.get(word)
        if phrase is None:
            return connector, obj_text
        inner = m.group("obj").strip()
        if not inner:
            return connector, obj_text
        return f"{connector.strip()} {phrase}", inner

    def extract(self, sentence: Dict) -> List[Tuple[str, str, str, float, str]]:
        doc = sentence.get("doc")
        if doc is None:
            return []
        spans = self._extract_spans(doc)
        spans = self._deduplicate_spans(spans)
        if len(spans) < 2:
            return []
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        sent_text = sentence.get("text", "")
        sent_emb = None
        candidates: List[Tuple[EntitySpan, EntitySpan, str, str]] = []
        for i in range(len(spans) - 1):
            s1, s2 = spans[i], spans[i + 1]
            connector = self._connector_text(doc, s1, s2)
            if not connector or len(connector.split()) > 8:
                continue
            connector, obj_text = self._split_partitive_object(connector, s2.text)
            if obj_text != s2.text:
                s2 = EntitySpan(obj_text, s2.tok_start, s2.tok_end, s2.dep, s2.pos, s2.label)
            if root is not None:
                if not (s1.tok_end <= root.i <= s2.tok_start):
                    continue
            triple_text = f"{s1.text} {connector} {s2.text}"
            candidates.append((s1, s2, connector, triple_text))

        triples = []
        seen: set = set()
        if self._sbert is not None and sent_text and candidates:
            sent_emb = self._sbert.encode(sent_text, normalize_embeddings=True)
            trip_embs = self._sbert.encode(
                [c[3] for c in candidates], normalize_embeddings=True
            )
            for (s1, s2, connector, _), trip_emb in zip(candidates, trip_embs):
                if float(np.dot(sent_emb, trip_emb)) < self._coherence_threshold:
                    continue
                rel, conf = self._classify_relation(s1.text, connector, s2.text)
                key = (s1.text.lower().strip(), rel, s2.text.lower().strip())
                if key not in seen and s1.text.lower() != s2.text.lower():
                    seen.add(key)
                    triples.append((s1.text, rel, s2.text, conf, connector))
        else:
            for s1, s2, connector, _ in candidates:
                rel, conf = self._classify_relation(s1.text, connector, s2.text)
                key = (s1.text.lower().strip(), rel, s2.text.lower().strip())
                if key not in seen and s1.text.lower() != s2.text.lower():
                    seen.add(key)
                    triples.append((s1.text, rel, s2.text, conf, connector))
        triples = self._expand_conj(doc, triples, seen)
        return triples

    def _deduplicate_spans(self, spans: List[EntitySpan]) -> List[EntitySpan]:
        if len(spans) <= 1:
            return spans
        keep = []
        for i, s in enumerate(spans):
            subsumed = False
            for j, t in enumerate(spans):
                if i != j and t.tok_start <= s.tok_start and s.tok_end <= t.tok_end:
                    if t.tok_end - t.tok_start > s.tok_end - s.tok_start:
                        subsumed = True
                        break
            if not subsumed:
                keep.append(s)
        return keep

    def _classify_relation(self, subj: str, connector: str, obj: str) -> Tuple[str, float]:
        if self._relation_mapper is not None:
            return self._relation_mapper.classify(subj, connector, obj)
        return connector.lower().strip(), 1.0

    def _expand_conj(self, doc, triples, seen):
        expanded = list(triples)
        subj_lower = ""
        obj_lower = ""
        for subj, rel, obj, conf, raw_conn in triples:
            for token in doc:
                if token.dep_ != "conj":
                    continue
                head = token.head
                head_lower = head.text.lower().strip()
                token_lower = token.text.lower().strip()
                subj_lower = subj.lower().strip()
                obj_lower = obj.lower().strip()
                if not token_lower:
                    continue
                if head_lower == subj_lower and token_lower != obj_lower:
                    key = (token_lower, rel, obj_lower)
                    if key not in seen:
                        seen.add(key)
                        expanded.append((token.text, rel, obj, conf * 0.9, raw_conn))
                elif head_lower == obj_lower and token_lower != subj_lower:
                    key = (subj_lower, rel, token_lower)
                    if key not in seen:
                        seen.add(key)
                        expanded.append((subj, rel, token.text, conf * 0.9, raw_conn))
        return expanded

    def extract_or_escalate(self, sentence: Dict) -> List[Dict]:
        result = self.extract(sentence)
        if result:
            return [{"triple": (t[0], t[1], t[2]), "level": 1, "rel_confidence": t[3], "raw_connector": t[4]} for t in result]
        return []
