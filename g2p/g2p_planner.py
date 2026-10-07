"""Query-relation extractor (DEVIATION 9).

Replaces the 16-intent G2P/FFN planner. The extractor maps a natural-language
question to an ordered multi-relation chain (e.g. ["causes", "part_of"]) by
embedding the question clauses with a frozen SentenceTransformer and matching
them against a static relation descriptor bank. The chain drives the walker and
the template decoder. No training, no intents, no external LLM.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import G2PConfig
from .types import Plan, Subgraph

logger = logging.getLogger(__name__)


def collapse_runs(chain: List[str], collapse_consecutive_repeats: bool) -> List[str]:
    """Contract v3.3.2 section 8: `collapse_consecutive_repeats`.

    When enabled, ANY adjacent repeat is collapsed to a single occurrence
    (e.g. [causes, causes, part_of] -> [causes, part_of]). The previous
    `collapse_max` run-length cap was a different semantic and let duplicate
    relations survive into the chain, inflating walk length.
    """
    if not collapse_consecutive_repeats:
        return list(chain)
    out: List[str] = []
    for rel in chain:
        if out and out[-1] == rel:
            continue
        out.append(rel)
    return out


# Clauses that are pure connectors or punctuation contribute no question
# intent (a bare "and"/"?" fragment has looked up as a *high*-similarity match
# for example_of/has_property and polluted otherwise-correct chains).
ORPHAN_CLAUSE_WORDS = frozenset(
    {"and", "or", "so", "but", "because", "since", "therefore", "than"}
)

# Causal delimiters that split a declarative premise from the actual question
# (e.g. "Fire is hot, so what is the opposite of hot?"). Text before the LAST
# such delimiter is a premise, not a question clause, and is excluded from the
# chain. Requiring a leading comma keeps ordinary uses like "What is so hot?"
# untouched.
PREMISE_CAUSAL_DELIMITERS = ("so", "thus", "therefore", "hence")
_PREMISE_RE = re.compile(
    r",\s*(?:" + "|".join(re.escape(d) for d in PREMISE_CAUSAL_DELIMITERS) + r")\s+",
    flags=re.IGNORECASE,
)


class QueryRelationExtractor:
    def __init__(self, config: G2PConfig, label_map: Optional[Dict[int, str]] = None):
        self.config = config
        self.config.validate()
        self.label_map = label_map if label_map is not None else {}
        self._sentence_model = None
        self._variant_embeddings: Dict[str, List[np.ndarray]] = {}
        self._initialized = False
        self._compiled_descriptors = self._compile_descriptors()

    def _compile_descriptors(self) -> List[Tuple[str, "re.Pattern[str]", int]]:
        """Compile the descriptor bank into (relation, pattern, length) triples.

        Contract section 8 treats cue phrases and descriptors as ONE bank: the
        same `relation_variants` entries serve as the embedding bank and as the
        literal cue strings for literal matching. There is no separate cue
        bank.

        Literal matching runs on the FULL question before clause splitting,
        otherwise question-word-prefixed descriptors such as "what causes" are
        destroyed by the "what" clause splitter.
        """
        compiled: List[Tuple[str, "re.Pattern[str]", int]] = []
        for relation, variants in self.config.extraction.relation_variants.items():
            for variant in variants:
                variant = str(variant).strip().lower()
                if not variant:
                    continue
                pattern = r"(?<!\w)" + re.escape(variant).replace(r"\ ", r"\s+") + r"(?!\w)"
                compiled.append((relation, re.compile(pattern), len(variant)))
        return compiled

    def _literal_cue_relations(self, question: str) -> List[str]:
        """Relations whose descriptor appears literally, ordered by position.

        Contract section 8 fallback trigger: "No strong relation cue words or
        phrases from any descriptor bank are present in the question." This
        returns the relations whose descriptor literally matched; an empty
        result is the cue-absence condition that triggers fallback.

        Two ordering rules:

        1. Per relation, only the LONGEST matching descriptor counts, so a
           question containing both "is caused by" and "caused by" yields one
           caused_by rather than a duplicate from the shorter descriptor.
        2. Across relations, order by position in the question, so multi-hop
           questions keep clause order ("What causes rain and what is rain
           part of?").
        """
        text = question.strip().lower()
        best: Dict[str, Tuple[int, int]] = {}
        for relation, pattern, length in self._compiled_descriptors:
            m = pattern.search(text)
            if not m:
                continue
            current = best.get(relation)
            if current is None or (length, -m.start()) > (current[0], -current[1]):
                best[relation] = (length, m.start())
        if not best:
            return []
        ordered = sorted(best.items(), key=lambda kv: (kv[1][1], kv[0]))
        return [rel for rel, _ in ordered]

    def _get_sentence_model(self):
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for the query-relation extractor "
                    f"(model: {self.config.sentence_bert.model_name})"
                ) from exc
            self._sentence_model = SentenceTransformer(self.config.sentence_bert.model_name)
        return self._sentence_model

    def initialize(self, graph_relations: Optional[List[str]] = None) -> None:
        """Embed the relation descriptor bank. graph_relations filters to relations
        that actually exist in the current knowledge graph (optional)."""
        if self._initialized:
            return
        model = self._get_sentence_model()
        valid_relations = set(graph_relations) if graph_relations is not None else None
        for relation, variants in self.config.extraction.relation_variants.items():
            if valid_relations is not None and relation not in valid_relations:
                continue
            texts = list(dict.fromkeys([relation] + [str(v).lower().strip() for v in variants if v]))
            if not texts:
                continue
            embeddings = model.encode(
                texts,
                batch_size=self.config.sentence_bert.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            self._variant_embeddings[relation] = [np.asarray(e, dtype=np.float32) for e in embeddings]
        self._initialized = True
        logger.info("QueryRelationExtractor initialized with %d relations", len(self._variant_embeddings))

    def _split_clauses(self, question: str) -> List[str]:
        patterns = [p for p in self.config.extraction.clause_split if p and p.strip()]
        if not patterns:
            return [question.strip().lower()]
        joined = "|".join(re.escape(p) for p in patterns)
        word_boundary_pattern = r"\b(?:" + joined + r")\b"
        parts = re.split(word_boundary_pattern, question, flags=re.IGNORECASE)
        clauses = []
        for part in parts:
            clause = part.strip().lower()
            if clause:
                clauses.append(clause)
        return clauses or [question.strip().lower()]

    def _strip_premise(self, question: str) -> str:
        """Drop a declarative premise before a causal delimiter.

        "Fire is hot, so what is the opposite of hot?" -> the "fire is hot"
        clause is context, not a question; only the text after the last
        delimiter contributes relations. Ordinary uses of "so" without a
        leading comma (e.g. "What is so hot?") are left untouched.
        """
        matches = list(_PREMISE_RE.finditer(question))
        if not matches:
            return question
        return question[matches[-1].end():]

    def _is_orphan_clause(self, clause: str) -> bool:
        text = clause.strip().lower()
        if not text or not any(ch.isalnum() for ch in text):
            return True
        if len(text) < 4:
            return True
        if text in ORPHAN_CLAUSE_WORDS:
            return True
        return False

    def _best_relation_for_clause(self, clause: str) -> Tuple[Optional[str], float]:
        if not self._variant_embeddings:
            return None, 0.0
        model = self._get_sentence_model()
        query_emb = model.encode(
            clause,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        best_relation: Optional[str] = None
        best_sim = 0.0
        for relation, embeddings in self._variant_embeddings.items():
            for emb in embeddings:
                sim = float(np.dot(query_emb, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_relation = relation
        return best_relation, best_sim

    def _extract(self, question: str, graph_relations: Optional[List[str]] = None) -> Plan:
        """Map a question string to a Plan carrying an ordered relation_chain.

        Contract section 8 method, in order:
          1. Split question into clauses
          2. Match against relation descriptor banks and cue phrases
          3. Use literal matching + embedding similarity
          4. Build ordered relation_chain

        Fallback trigger (contract section 8 preferred_trigger): no strong
        relation cue words or phrases from any descriptor bank are present.
        Deliberately NOT "no chain" - the similarity score alone does not
        separate real from nonsense questions.
        """
        extraction = self.config.extraction

        # Literal matching over the descriptor bank. Matching runs on the full
        # question so question-word-prefixed descriptors survive clause
        # splitting. A literal match also carries the direction the frozen
        # encoder cannot distinguish (causes vs caused_by).
        cue_relations = self._literal_cue_relations(question)
        if graph_relations is not None:
            cue_relations = [r for r in cue_relations if r in graph_relations]
        cue_hit = bool(cue_relations)

        chain: List[str] = []
        sims: List[float] = []

        if cue_hit:
            chain.extend(cue_relations)
            sims.extend([1.0] * len(cue_relations))
        else:
            if not self._initialized:
                self.initialize(graph_relations)
            question_text = self._strip_premise(question)
            for clause in self._split_clauses(question_text):
                if self._is_orphan_clause(clause):
                    continue
                relation, sim = self._best_relation_for_clause(clause)
                if relation is None or sim < extraction.similarity_threshold:
                    continue
                if graph_relations is not None and relation not in graph_relations:
                    continue
                chain.append(relation)
                sims.append(sim)

        chain = collapse_runs(chain, extraction.collapse_consecutive_repeats)
        chain = chain[: extraction.max_chain_length]

        # Contract section 8 fallback: cue absence -> flag + default_chain.
        fallback = extraction.fallback.enabled and not cue_hit
        if fallback:
            chain = list(extraction.fallback.default_chain)

        if fallback:
            confidence = 0.6
        elif sims:
            confidence = float(np.clip(0.5 + 0.5 * float(np.mean(sims)), 0.0, 1.0))
        else:
            confidence = 0.6

        return Plan(
            intent_sequence=None,
            plan_confidence=round(confidence, 4),
            heuristic_fallback_used=fallback,
            intent_names=None,
            relation_chain=chain,
        )

    def extract(self, question: str, graph_relations: Optional[List[str]] = None) -> Plan:
        """Public entry point (contract section 8)."""
        return self._extract(question, graph_relations)

    def plan(self, subgraph: Subgraph, query_text: str = "") -> Plan:
        """Plan from a resonated subgraph.

        Contract section 8 makes literal descriptor matching authoritative, so
        the relation vocabulary is NOT narrowed to the relations present in the
        subgraph. An earlier version filtered by the subgraph's edge relations,
        which silently discarded a correct literal match whenever propagation
        had not happened to surface that relation: "What comes after spring?"
        matched `follows` literally, the subgraph carried only `causes`, the
        filter dropped `follows`, and the cue-absence fallback then replaced a
        correct chain with the default. That inverted the contract's ordering
        (literal match first) and is why such questions reported
        heuristic_fallback_used=True at hop 0.

        If the caller supplies an explicit vocabulary via extract(),
        graph_relations is honoured; plan() uses the full relation bank.
        """
        return self._extract(query_text, graph_relations=None)

    def plan_batch(self, subgraphs: List[Subgraph], query_text: str = "") -> List[Plan]:
        return [self.plan(sg, query_text=query_text) for sg in subgraphs]

    def get_plan_confidence(self, subgraph: Subgraph) -> float:
        activation_values = list(subgraph.node_activations.values())
        return float(np.mean(activation_values)) if activation_values else 0.5

    def mark_trained(self, *args, **kwargs) -> None:
        logger.info("QueryRelationExtractor requires no training (Deviation 9)")


# Backwards-compatible alias: G2PPlanner name replaced by QueryRelationExtractor.
G2PPlanner = QueryRelationExtractor