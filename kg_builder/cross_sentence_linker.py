import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CrossSentenceLinker:
    """
    Chains triples across adjacent sentences when they share entities.

    Sentence N: (India, is, country)       → entities: {india, country}
    Sentence N+1: (I, love, India)         → entities: {i, india}
    Chained:     (I, love→is, country)     → path: I →love→ India →is→ country

    The window tracks recent sentences. When a new sentence's triples share
    an entity with a previous sentence's triples, a chained triple is created.
    """

    def __init__(self, window_size: int = 1):
        self._window: List[Tuple[Set[str], List[Tuple]]] = []
        self._window_size = window_size

    def link(
        self, sentence_triples: List[Tuple], sentence_text: str = ""
    ) -> List[Tuple]:
        if not sentence_triples:
            self._window.append((set(), []))
            if len(self._window) > self._window_size:
                self._window.pop(0)
            return []

        current_entities: Set[str] = set()
        for t in sentence_triples:
            current_entities.add(t[0].lower().strip())
            current_entities.add(t[2].lower().strip())

        extended = list(sentence_triples)
        seen_keys: Set[Tuple[str, str, str]] = set()
        for t in sentence_triples:
            e1, rel, e2, conf, raw = t
            seen_keys.add((e1.lower().strip(), rel, e2.lower().strip()))

        for prev_entities, prev_triples in self._window:
            shared = current_entities & prev_entities
            if not shared:
                continue

            for pt in prev_triples:
                pt_e1, pt_rel, pt_e2, pt_conf, pt_raw = pt
                pt_e1_l = pt_e1.lower().strip()
                pt_e2_l = pt_e2.lower().strip()

                for ct in sentence_triples:
                    ct_e1, ct_rel, ct_e2, ct_conf, ct_raw = ct
                    ct_e1_l = ct_e1.lower().strip()
                    ct_e2_l = ct_e2.lower().strip()

                    if pt_e2_l == ct_e1_l:
                        chain_rel = f"{pt_rel}→{ct_rel}"
                        chain_e1 = pt_e1
                        chain_e2 = ct_e2
                        key = (chain_e1.lower().strip(), chain_rel, chain_e2.lower().strip())
                        if key not in seen_keys and chain_e1.lower().strip() != chain_e2.lower().strip():
                            seen_keys.add(key)
                            chain_conf = min(pt_conf, ct_conf) * 0.85
                            chain_raw = f"{pt_raw} {ct_raw}"
                            extended.append((chain_e1, chain_rel, chain_e2, chain_conf, chain_raw))

                    if ct_e2_l == pt_e1_l:
                        chain_rel = f"{ct_rel}→{pt_rel}"
                        chain_e1 = ct_e1
                        chain_e2 = pt_e2
                        key = (chain_e1.lower().strip(), chain_rel, chain_e2.lower().strip())
                        if key not in seen_keys and chain_e1.lower().strip() != chain_e2.lower().strip():
                            seen_keys.add(key)
                            chain_conf = min(ct_conf, pt_conf) * 0.85
                            chain_raw = f"{ct_raw} {pt_raw}"
                            extended.append((chain_e1, chain_rel, chain_e2, chain_conf, chain_raw))

        self._window.append((current_entities, sentence_triples))
        if len(self._window) > self._window_size:
            self._window.pop(0)

        return extended
