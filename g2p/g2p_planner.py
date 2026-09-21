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


def collapse_runs(chain: List[str], collapse_max: int) -> List[str]:
    """Collapse consecutive repeats above collapse_max (e.g. causes,causes -> causes)."""
    out: List[str] = []
    run = 0
    prev: Optional[str] = None
    for rel in chain:
        if rel == prev:
            run += 1
            if run >= collapse_max:
                continue
        else:
            run = 1
        out.append(rel)
        prev = rel
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
        joined = "|".join(re.escape(p) for p in patterns)
        parts = re.split(joined, question, flags=re.IGNORECASE) if joined else [question]
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

    def extract(self, question: str, graph_relations: Optional[List[str]] = None) -> Plan:
        """Map a question string to a Plan carrying an ordered relation_chain."""
        if not self._initialized:
            self.initialize(graph_relations)

        extraction = self.config.extraction
        chain: List[str] = []
        sims: List[float] = []
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

        chain = collapse_runs(chain, extraction.collapse_max)

        fallback = not chain
        if fallback:
            chain = list(extraction.default_chain)
        chain = chain[: extraction.max_chain_length]

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

    def plan(self, subgraph: Subgraph, query_text: str = "") -> Plan:
        edges = getattr(subgraph, "edges", None) or []
        graph_relations = sorted({rel for _, _, rel in edges}) or None
        return self.extract(query_text, graph_relations=graph_relations)

    def plan_batch(self, subgraphs: List[Subgraph]) -> List[Plan]:
        return [self.plan(sg) for sg in subgraphs]

    def get_plan_confidence(self, subgraph: Subgraph) -> float:
        activation_values = list(subgraph.node_activations.values())
        return float(np.mean(activation_values)) if activation_values else 0.5

    def mark_trained(self, *args, **kwargs) -> None:
        logger.info("QueryRelationExtractor requires no training (Deviation 9)")


# Backwards-compatible alias: G2PPlanner name replaced by QueryRelationExtractor.
G2PPlanner = QueryRelationExtractor