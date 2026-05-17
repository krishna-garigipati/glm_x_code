import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RelationMapper:
    def __init__(
        self,
        relation_map: Optional[Dict[str, str]] = None,
        antonym_triggers: Optional[List[str]] = None,
        synonym_triggers: Optional[List[str]] = None,
        sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        default_map = {
            "is": "is_a", "are": "is_a", "was": "is_a", "were": "is_a",
            "be": "is_a", "become": "is_a", "became": "is_a",
            "remain": "is_a", "stay": "is_a",
            # has_property
            "has": "has_property", "have": "has_property", "had": "has_property",
            "contains": "has_property", "contain": "has_property",
            "includes": "has_property", "include": "has_property",
            "features": "has_property", "feature": "has_property",
            "possess": "has_property", "possesses": "has_property",
            "compose": "has_property", "comprises": "has_property", "comprise": "has_property",
            # part_of
            "consists_of": "part_of", "consist_of": "part_of",
            "belongs_to": "part_of", "belong_to": "part_of",
            "member_of": "part_of", "part_of": "part_of",
            # causes
            "causes": "causes", "caused": "causes", "cause": "causes",
            "produce": "causes", "produces": "causes", "produced": "causes",
            "generate": "causes", "generates": "causes", "generated": "causes",
            "strengthen": "causes", "strengthens": "causes",
            "enable": "causes", "enables": "causes", "enabled": "causes",
            "allow": "causes", "allows": "causes",
            "lead_to": "causes", "leads_to": "causes", "led_to": "causes",
            "result_in": "causes", "results_in": "causes", "resulted_in": "causes",
            "trigger": "causes", "triggers": "causes", "triggered": "causes",
            "reduce": "causes", "reduces": "causes", "reduced": "causes",
            "increase": "causes", "increases": "causes", "increased": "causes",
            "improve": "causes", "improves": "causes", "improved": "causes",
            "provide": "causes", "provides": "causes", "provided": "causes",
            "power": "causes", "powers": "causes", "powered": "causes",
            "support": "causes", "supports": "causes", "supported": "causes",
            "protect": "causes", "protects": "causes", "protected": "causes",
            "transform": "causes", "transformed": "causes",
            "convert": "causes", "converted": "causes",
            "spread": "causes", "spreads": "causes",
            "arise_from": "causes",
            # located_in (specific spatial)
            "located_in": "located_in", "locate": "located_in",
            "located": "located_in", "situated_in": "located_in",
            "situated": "located_in", "flow_through": "located_in",
            "flow": "located_in", "lives_in": "located_in",
            "live_in": "located_in", "reside_in": "located_in",
            "exists_in": "located_in",
            # originated_in (specific origin)
            "originate_in": "originated_in", "originate": "originated_in",
            "originated": "originated_in", "come_from": "originated_in",
            "come": "originated_in", "comes": "originated_in",
            "arise": "originated_in", "arose": "originated_in",
            "emerge": "originated_in", "emerged": "originated_in",
            "born_in": "originated_in",
            # causes / discovery
            "discover": "causes", "discovered": "causes",
            "invent": "causes", "invented": "causes",
            "formulate": "causes", "formulated": "causes",
            "propose": "causes", "proposed": "causes",
            "design": "causes", "designed": "causes",
            "publish": "causes", "published": "causes",
            "write": "causes", "wrote": "causes",
            "introduce": "causes", "introduced": "causes",
            "pioneer": "causes", "pioneered": "causes",
            "revolutionize": "causes", "revolutionized": "causes",
            "establish": "causes", "established": "causes",
            "found": "causes", "founded": "causes",
            "set_up": "causes", "create": "causes",
            "produce": "causes", "generate": "causes",
             "study": "causes", "studied": "causes",
             "research": "causes",
             "use": "causes", "used": "causes",
             "utilize": "causes",
             "connect": "causes", "connects": "causes",
             "work_on": "causes",
             "learn": "causes", "learned": "causes",
             "teach": "causes", "taught": "causes",
             "read": "causes", "reads": "causes",
             "analyze": "causes", "analyzed": "causes",
             "calculate": "causes", "calculated": "causes",
             "measure": "causes", "measured": "causes",
             "observe": "causes", "observed": "causes",
             "demonstrate": "causes", "demonstrated": "causes",
             "describe": "causes", "described": "causes",
             "explain": "causes", "explained": "causes",
             "discuss": "causes", "discussed": "causes",
             "report": "causes", "reported": "causes",
             "present": "causes", "presented": "causes",
             "announce": "causes", "announced": "causes",
             "illustrate": "causes", "illustrated": "causes",
             "perform": "causes", "performed": "causes",
             "conduct": "causes", "conducted": "causes",
            "known_for": "associated_with", "famous_for": "associated_with",
            "known_as": "is_a", "refer_to": "associated_with",
            "also_called": "associated_with", "known": "associated_with",
            "process": "causes", "processed": "causes",
            "works_at": "associated_with",
            # is_a (additional)
            "represent": "is_a", "represents": "is_a", "represented": "is_a",
            "consider": "is_a", "considered": "is_a",
            "define": "is_a", "defined": "is_a",
            "call": "is_a", "called": "is_a", "calls": "is_a",
            "mean": "is_a", "means": "is_a", "meant": "is_a",
            "constitute": "is_a", "constitutes": "is_a",
            # has_property (additional)
            "exhibit": "has_property", "exhibits": "has_property",
            "demonstrate": "has_property", "demonstrates": "has_property",
            "show": "has_property", "shows": "has_property",
            "display": "has_property", "displays": "has_property",
            "lack": "has_property", "lacks": "has_property",
            "require": "has_property", "requires": "has_property",
            "need": "has_property", "needs": "has_property",
            "offer": "has_property", "offers": "has_property",
            # causes (additional)
            "create": "causes", "creates": "causes", "created": "causes",
            "form": "causes", "forms": "causes", "formed": "causes",
            "make": "causes", "makes": "causes", "made": "causes",
            "build": "causes", "builds": "causes", "built": "causes",
            "construct": "causes", "constructs": "causes",
            "develop": "causes", "develops": "causes", "developed": "causes",
            "influence": "causes", "influences": "causes",
            "affect": "causes", "affects": "causes",
            "prevent": "causes", "prevents": "causes",
            "block": "causes", "blocks": "causes",
            "promote": "causes", "promotes": "causes",
            "encourage": "causes", "encourages": "causes",
            "force": "causes", "forces": "causes",
            "drive": "causes", "drives": "causes",
            "help": "causes", "helps": "causes", "helped": "causes",
            "boost": "causes", "boosts": "causes",
            "enhance": "causes", "enhances": "causes",
            "damage": "causes", "damages": "causes",
            "destroy": "causes", "destroys": "causes",
            "harm": "causes", "harms": "causes",
            "contribute_to": "causes", "contribute": "causes",
            # associated_with (additional)
            "teach": "associated_with", "taught": "associated_with",
            "learn": "associated_with", "learned": "associated_with",
            "read": "associated_with", "reads": "associated_with",
            "perform": "associated_with", "performed": "associated_with",
            "conduct": "associated_with", "conducted": "associated_with",
            "analyze": "associated_with", "analyzed": "associated_with",
            "calculate": "associated_with", "calculated": "associated_with",
            "measure": "associated_with", "measured": "associated_with",
            "observe": "associated_with", "observed": "associated_with",
            "describe": "associated_with", "described": "associated_with",
            "explain": "associated_with", "explained": "associated_with",
            "discuss": "associated_with", "discussed": "associated_with",
            "report": "associated_with", "reported": "associated_with",
            "announce": "associated_with", "announced": "associated_with",
            "present": "associated_with", "presented": "associated_with",
            "demonstrate": "associated_with", "demonstrated": "associated_with",
            "illustrate": "associated_with", "illustrated": "associated_with",
            "specialize_in": "associated_with",
            "focus_on": "associated_with",
            "work_on": "associated_with",
            "collaborate_with": "causes",
            "associate_with": "causes",
            "compete_with": "causes",
            "merge_with": "causes",
            "combine_with": "causes",
            "interact_with": "causes",
            "relate_to": "associated_with",
            "apply_to": "associated_with",
            "lead_to": "causes",
            "result_in": "causes",
            # part_of (additional)
            "part_of": "part_of",
            "belong": "part_of", "belongs": "part_of",
            "divide_into": "part_of",
            "split_into": "part_of",
            "separate_into": "part_of",
            "group_into": "part_of",
            "classify_as": "is_a",
            # compound location verbs (prep_obj pattern)
            "be_found_in": "located_in", "found_in": "located_in",
            "find_in": "located_in", "be_located_in": "located_in",
            "exist_in": "located_in", "exists_in": "located_in",
            "occur_in": "located_in", "occurs_in": "located_in",
            "take_place_in": "located_in",
            "be_born_in": "originated_in", "born_in": "originated_in",
            "grow_in": "located_in",
        }
        if relation_map:
            default_map.update(relation_map)
        self._relation_map = default_map
        self._antonym_triggers = antonym_triggers or [
            "opposite", "unlike", "contrary", "reverse", "versus", "antonym",
        ]
        self._synonym_triggers = synonym_triggers or [
            "also_known_as", "aka", "alias", "same_as", "synonym", "also_called",
        ]
        self._sbert_model_name = sbert_model_name
        self._sbert: Optional[SentenceTransformer] = None
        self._prototype_embs: Optional[np.ndarray] = None
        self._prototype_labels: List[str] = []

    def _lazy_init(self):
        if self._sbert is not None:
            return
        self._sbert = SentenceTransformer(self._sbert_model_name)
        unique_relations = list(set(self._relation_map.values()))
        self._prototype_labels = unique_relations
        prototype_sentences = {
            "is_a": "X is a type of Y",
            "has_property": "X has the property Y",
            "part_of": "X is part of Y",
            "causes": "X causes Y",
            "associated_with": "X is associated with Y",
            "antonym": "X is the opposite of Y",
            "synonym": "X is the same as Y",
            "located_in": "X is located in Y",
            "originated_in": "X originated in Y",
        }
        proto_texts = [prototype_sentences.get(r, f"X {r} Y") for r in unique_relations]
        self._prototype_embs = self._sbert.encode(proto_texts)
        self._prototype_embs = self._prototype_embs / (
            np.linalg.norm(self._prototype_embs, axis=1, keepdims=True) + 1e-8
        )

    def classify(self, subj: str, verb: str, obj: str, context: str = "") -> Tuple[str, float]:
        verb_lower = verb.lower().strip()
        for trigger in self._synonym_triggers:
            if trigger in context.lower() or trigger in verb_lower:
                return "synonym", 0.8
        for trigger in self._antonym_triggers:
            if trigger in context.lower() or trigger in verb_lower:
                return "antonym", 0.8
        verb_norm = verb_lower.replace(" ", "_")
        if verb_norm in self._relation_map:
            return self._relation_map[verb_norm], 0.85
        parts = verb_norm.split("_")
        for p in parts:
            if p in self._relation_map:
                return self._relation_map[p], 0.7
        self._lazy_init()
        sent_emb = self._sbert.encode(f"{subj} {verb} {obj}")
        sent_emb = sent_emb / (np.linalg.norm(sent_emb) + 1e-8)
        scores = np.dot(self._prototype_embs, sent_emb)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score > 0.28:
            return self._prototype_labels[best_idx], best_score
        return "associated_with", 0.3
