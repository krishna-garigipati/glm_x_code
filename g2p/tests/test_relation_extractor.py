"""DEVIATION 9: QueryRelationExtractor unit tests (fake embedding model).

Verifies the extractor maps clauses to an ordered relation chain without any
intent FFN. The sentence model is faked with a deterministic relation-axis
encoder so the tests are hermetic (no SentenceTransformer weights, no network).
"""

from __future__ import annotations

import unittest

import numpy as np

from g2p.config import G2PConfig, RelationExtractionConfig
from g2p.g2p_planner import (
    QueryRelationExtractor,
    G2PPlanner,
    collapse_runs,
)

RELATIONS = ["causes", "antonym", "part_of"]


class FakeRelationEncoder:
    """One-hot-style encoder: keyword lookup maps text to a relation axis."""

    _KEYWORDS = {
        "causes": ["cause", "causes", "leads to", "one thing causes another"],
        "antonym": ["antonym", "opposite", "contradict", "is the opposite of"],
        "part_of": ["part_of", "part of", "is a part of", "contained in", "contain"],
    }

    def __init__(self, *args, **kwargs):
        pass

    def _vec(self, text: str) -> np.ndarray:
        key = None
        for rel, keywords in self._KEYWORDS.items():
            if any(kw in text for kw in keywords):
                key = rel
                break
        vec = np.zeros(len(RELATIONS), dtype=np.float32)
        if key is not None:
            vec[RELATIONS.index(key)] = 1.0
        return vec

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        arr = np.stack([self._vec(t) for t in items]).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            arr = arr / norms
        return arr[0] if single else arr


def make_config() -> G2PConfig:
    cfg = G2PConfig()
    cfg.sentence_bert.model_name = "fake-encoder"
    cfg.sentence_bert.model_dim = len(RELATIONS)
    cfg.extraction = RelationExtractionConfig(
        similarity_threshold=0.35,
        max_chain_length=3,
        collapse_max=2,
        default_chain=["has_property"],
        clause_split=[" and ", ","],
        relation_variants={
            "has_property": ["has the property that"],
            "causes": ["one thing causes another", "leads to x"],
            "antonym": ["is the opposite of"],
            "part_of": ["is a part of", "is contained in"],
        },
    )
    return cfg


class TestRelationExtractorUnit(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        self.extractor = QueryRelationExtractor(self.config)
        self.extractor._sentence_model = FakeRelationEncoder()

    def test_initialize_prepares_variant_embeddings(self):
        self.extractor.initialize()
        self.assertTrue(self.extractor._initialized)
        self.assertIn("causes", self.extractor._variant_embeddings)
        self.assertIn("antonym", self.extractor._variant_embeddings)
        self.assertIn("part_of", self.extractor._variant_embeddings)

    def test_initialize_filters_by_graph_relations(self):
        self.extractor.initialize(graph_relations=["causes", "part_of"])
        self.assertEqual(sorted(self.extractor._variant_embeddings), ["causes", "part_of"])

    def test_extract_ordered_chain_from_question(self):
        plan = self.extractor.extract("What causes rain and what is rain part of?")
        self.assertEqual(plan.relation_chain, ["causes", "part_of"])
        self.assertIsNone(plan.intent_sequence)
        self.assertFalse(plan.heuristic_fallback_used)
        self.assertGreater(plan.plan_confidence, 0.5)

    def test_extract_filters_graph_relations(self):
        plan = self.extractor.extract(
            "What causes fires and what is smoke the opposite of?",
            graph_relations=["causes"],
        )
        self.assertEqual(plan.relation_chain, ["causes"])

    def test_extract_low_similarity_falls_back_to_default_chain(self):
        plan = self.extractor.extract("What time is it right now?")
        self.assertEqual(plan.relation_chain, ["has_property"])
        self.assertTrue(plan.heuristic_fallback_used)
        self.assertEqual(plan.plan_confidence, 0.6)

    def test_plan_delegates_to_extract(self):
        subgraph = object.__new__(object)
        plan = self.extractor.plan(subgraph, query_text="What is the opposite of hot?")
        self.assertEqual(plan.relation_chain, ["antonym"])

    def test_plan_batch_returns_plans(self):
        class Dummy:
            pass

        sgs = [Dummy(), Dummy()]
        plans = self.extractor.plan_batch(sgs)
        self.assertEqual(len(plans), 2)
        self.assertTrue(all(p.relation_chain == ["has_property"] for p in plans))


class TestCollapseRuns(unittest.TestCase):
    def test_collapses_repeats_above_limit(self):
        self.assertEqual(collapse_runs(["causes", "causes", "causes", "part_of"], 2), ["causes", "part_of"])
        self.assertEqual(collapse_runs(["causes", "causes", "causes"], 3), ["causes", "causes"])
        self.assertEqual(collapse_runs(["is_a", "causes", "causes", "part_of"], 2), ["is_a", "causes", "part_of"])

    def test_single_run_untouched(self):
        self.assertEqual(collapse_runs(["causes", "part_of"], 2), ["causes", "part_of"])


class TestG2PPlannerAlias(unittest.TestCase):
    def test_planner_name_alias(self):
        self.assertIs(G2PPlanner, QueryRelationExtractor)


if __name__ == "__main__":
    unittest.main()