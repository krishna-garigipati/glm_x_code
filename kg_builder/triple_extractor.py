import logging
from typing import Dict, List, Optional, Tuple

import spacy
from spacy.matcher import DependencyMatcher

logger = logging.getLogger(__name__)


class TripleExtractor:
    def __init__(self, enable_llm: bool = False, llm_model: str = "phi-3-mini"):
        self._matcher: Optional[DependencyMatcher] = None
        self._nlp: Optional[spacy.Language] = None
        self._enable_llm = enable_llm
        self._llm_model = llm_model
        self._llm_extractor = None

    def _lazy_init(self, nlp_vocab):
        if self._matcher is not None:
            return
        self._matcher = DependencyMatcher(nlp_vocab)
        self._compile_patterns()
        if self._enable_llm:
            self._init_llm()

    def set_nlp(self, nlp):
        self._nlp = nlp

    def _compile_patterns(self):
        patterns = {
            "svo": [
                {"RIGHT_ID": "root", "RIGHT_ATTRS": {"POS": {"IN": ["VERB", "AUX"]}}},
                {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "subj",
                 "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
                {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "obj",
                 "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "obj", "iobj", "attr", "nmod"]}}},
            ],
            "copula": [
                {"RIGHT_ID": "root", "RIGHT_ATTRS": {"POS": "AUX", "LEMMA": {"IN": ["be", "become", "seem"]}}},
                {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "subj",
                 "RIGHT_ATTRS": {"DEP": "nsubj"}},
                {"LEFT_ID": "root", "REL_OP": ">", "RIGHT_ID": "attr",
                 "RIGHT_ATTRS": {"DEP": {"IN": ["attr", "acomp"]}}},
            ],
            "prep_obj": [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
                 "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prep",
                 "RIGHT_ATTRS": {"DEP": "prep"}},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj",
                 "RIGHT_ATTRS": {"DEP": "pobj"}},
            ],
            "passive": [
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB", "TAG": "VBN"}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subj",
                 "RIGHT_ATTRS": {"DEP": "nsubjpass"}},
                {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "agent",
                 "RIGHT_ATTRS": {"DEP": {"IN": ["agent", "pobj"]}}},
            ],
            "possessive": [
                {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}},
                {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "poss",
                 "RIGHT_ATTRS": {"DEP": "poss"}},
                {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "nmod",
                 "RIGHT_ATTRS": {"DEP": "nmod", "POS": "NOUN"}},
            ],
            "apposition": [
                {"RIGHT_ID": "head", "RIGHT_ATTRS": {"POS": {"IN": ["PROPN", "NOUN"]}}},
                {"LEFT_ID": "head", "REL_OP": ">", "RIGHT_ID": "appos",
                 "RIGHT_ATTRS": {"DEP": "appos"}},
            ],
            "noun_prep": [
                {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}},
                {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "prep",
                 "RIGHT_ATTRS": {"DEP": "prep"}},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj",
                 "RIGHT_ATTRS": {"DEP": "pobj"}},
            ],
            "be_prep": [
                {"RIGHT_ID": "be", "RIGHT_ATTRS": {"POS": "AUX", "LEMMA": "be"}},
                {"LEFT_ID": "be", "REL_OP": ">", "RIGHT_ID": "subj",
                 "RIGHT_ATTRS": {"DEP": "nsubj"}},
                {"LEFT_ID": "be", "REL_OP": ">", "RIGHT_ID": "prep",
                 "RIGHT_ATTRS": {"DEP": "prep"}},
                {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "pobj",
                 "RIGHT_ATTRS": {"DEP": "pobj"}},
            ],
        }
        for name, pattern in patterns.items():
            self._matcher.add(name, [pattern])

    def _init_llm(self):
        try:
            from spacy_llm import pipeline_component
            self._llm_extractor = True
            logger.info("spacy-llm available")
        except ImportError:
            logger.warning("spacy-llm not installed; LLM cascade disabled")
            self._enable_llm = False

    def extract(self, sentence: Dict) -> List[Tuple[str, str, str]]:
        doc = sentence.get("doc")
        if doc is None:
            return []
        self._lazy_init(doc.vocab)
        triples = []
        seen = set()
        matches = self._matcher(doc)
        for match_id, token_ids in matches:
            pattern_name = doc.vocab.strings[match_id]
            tokens = [doc[i] for i in token_ids]
            triple = self._tokens_to_triple(tokens, pattern_name)
            if triple:
                e1, verb, e2 = triple
                key = (e1.lower().strip(), verb.lower().strip(), e2.lower().strip())
                if key not in seen:
                    seen.add(key)
                    triples.append(triple)
        conj_triples = []
        for triple in triples:
            e1, verb, e2 = triple
            for token in doc:
                if token.dep_ != "conj":
                    continue
                head = token.head
                head_lower = head.text.lower().strip()
                if head_lower == e1.lower().strip():
                    new_t = (token.text, verb, e2)
                    key = (new_t[0].lower(), new_t[1].lower(), new_t[2].lower())
                    if key not in seen:
                        seen.add(key)
                        conj_triples.append(new_t)
                elif head_lower == e2.lower().strip():
                    new_t = (e1, verb, token.text)
                    key = (new_t[0].lower(), new_t[1].lower(), new_t[2].lower())
                    if key not in seen:
                        seen.add(key)
                        conj_triples.append(new_t)
        triples.extend(conj_triples)
        nmod_triples = []
        for triple in triples:
            e1, verb, e2 = triple
            for token in doc:
                if token.text.lower().strip() != e2.lower().strip():
                    continue
                for child in token.children:
                    if child.dep_ == "nmod":
                        parts = [t.text for t in child.subtree]
                        nmod_text = " ".join(parts)
                        key = (e1.lower(), verb.lower(), nmod_text.lower())
                        if key not in seen and nmod_text.lower() != e2.lower():
                            seen.add(key)
                            nmod_triples.append((e1, verb, nmod_text))
        triples.extend(nmod_triples)
        return triples

    def extract_or_escalate(self, sentence: Dict) -> List[Dict]:
        result = self.extract(sentence)
        if result:
            return [{"triple": t, "level": 1} for t in result]
        if self._enable_llm and self._llm_extractor:
            try:
                llm_triples = self._llm_extract(sentence["text"])
                if llm_triples:
                    return [{"triple": t, "level": 2} for t in llm_triples]
            except Exception as e:
                logger.debug("LLM extraction failed: %s", e)
        return []

    def _llm_extract(self, text: str) -> List[Tuple[str, str, str]]:
        return []

    @staticmethod
    def _tokens_to_triple(tokens: list, pattern: str) -> Optional[Tuple[str, str, str]]:
        if pattern == "svo":
            subj, verb, obj = None, None, None
            for tok in tokens:
                if tok.dep_ in ("nsubj", "nsubjpass"):
                    subj = tok.text
                elif tok.pos_ in ("VERB", "AUX"):
                    verb = tok.lemma_
                elif tok.dep_ in ("dobj", "attr", "acomp"):
                    obj = tok.text
            if subj and verb and obj:
                return (subj, verb, obj)
        elif pattern == "copula":
            subj, obj, verb = None, None, "is"
            for tok in tokens:
                if tok.dep_ == "nsubj":
                    subj = tok.text
                elif tok.dep_ == "attr":
                    obj = tok.text
                    verb = "is"
                elif tok.dep_ == "acomp":
                    obj = tok.text
                    verb = "has"
            if subj and obj:
                return (subj, verb, obj)
        elif pattern in ("prep_obj", "passive"):
            subj, verb, obj = None, None, None
            for tok in tokens:
                if tok.dep_ in ("nsubj", "nsubjpass"):
                    subj = tok.text
                elif tok.pos_ == "VERB":
                    verb = tok.lemma_
                elif tok.dep_ in ("pobj", "agent"):
                    obj = tok.text
            if subj and verb and obj:
                prep_text = ""
                for tok in tokens:
                    if tok.dep_ == "prep":
                        prep_text = tok.text
                        break
                rel = f"{verb}_{prep_text}" if prep_text else verb
                return (subj, rel, obj)
        elif pattern == "possessive":
            poss, head, nmod = None, None, None
            for tok in tokens:
                if tok.dep_ == "poss":
                    poss = tok.text
                elif tok.dep_ in ("ROOT", "nsubj"):
                    head = tok.text
                elif tok.dep_ == "nmod":
                    nmod = tok.text
            if poss and head:
                return (poss, "has", head)
            if head and nmod:
                return (head, "has", nmod)
        elif pattern == "apposition":
            if len(tokens) >= 2:
                return (tokens[0].text, "is", tokens[1].text)
        elif pattern == "noun_prep":
            if len(tokens) >= 3:
                noun, prep, pobj = tokens[0], tokens[1], tokens[2]
                if prep.text == "of":
                    return (noun.text, "part_of", pobj.text)
                if prep.text in ("in", "on", "at", "near", "inside"):
                    return (noun.text, "located_in", pobj.text)
                if prep.text in ("from", "out_of"):
                    return (noun.text, "originated_in", pobj.text)
        elif pattern == "be_prep":
            if len(tokens) >= 4:
                be, subj, prep, pobj = tokens[0], tokens[1], tokens[2], tokens[3]
                if prep.text in ("in", "on", "at", "near", "inside", "across", "along"):
                    return (subj.text, "located_in", pobj.text)
                if prep.text in ("from", "out_of"):
                    return (subj.text, "originated_in", pobj.text)
                if prep.text == "of":
                    return (subj.text, "part_of", pobj.text)
                if prep.text in ("for", "with"):
                    return (subj.text, "has_property", pobj.text)
        return None
