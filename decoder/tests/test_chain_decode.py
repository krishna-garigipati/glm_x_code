"""DEVIATION 9: relation-chain template decoder tests.

The decoder now matches templates keyed by an exact ordered relation chain
(``chain: [...]``) in addition to the dormant intent-keyed definitions. Unknown
chains fall back to a grammatical chain-path renderer, and an honest
"no relation" answer exists for empty walks.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from decoder.template_decoder import TemplateDecoder

RELATION_PHRASES: Dict[str, str] = {
    "is_a": "is a",
    "has_property": "has",
    "causes": "causes",
    "caused_by": "is caused by",
    "antonym": "is the opposite of",
    "part_of": "is part of",
}

SENTENCE_STARTERS: List[str] = [
    "Therefore,",
    "Consequently,",
    "Specifically,",
]

FALLBACK_CFG: Dict[str, Any] = {
    "type": "concatenate",
    "separator": " ",
    "max_words": 200,
    "use_intent_prefix": False,
}

VALIDATION_CFG: Dict[str, Any] = {
    "min_output_length": 5,
    "max_output_length": 500,
    "require_node_mention": True,
    "max_repetitive_ngrams": 0,
}

CHAIN_RENDER_CFG: Dict[str, Any] = {
    "step_connector": ", and ",
    "no_relation_answer": "I don't have a relation in my knowledge graph that answers this question.",
}

TEMPLATES: List[Dict[str, Any]] = [
    {"chain": ["causes"], "template": "The reason is that {node0} {relation0} {node1}."},
    {"chain": ["antonym"], "template": "{node0} {relation0} {node1}."},
    {"chain": ["causes", "causes"], "template": "{node0} {relation0} {node1}, and this in turn {relation1} {node2}."},
    {"chain": ["is_a", "has_property"], "template": "{node0} is {node1}, and it has the property {node2}."},
]


class ChainDecodeTestBase(unittest.TestCase):
    def _decoder(self, templates: List[Dict[str, Any]] | None = None) -> TemplateDecoder:
        return TemplateDecoder(
            templates=list(templates) if templates is not None else list(TEMPLATES),
            relation_phrases=dict(RELATION_PHRASES),
            sentence_starters=list(SENTENCE_STARTERS),
            fallback_cfg=dict(FALLBACK_CFG),
            validation_cfg=dict(VALIDATION_CFG),
            chain_render_cfg=dict(CHAIN_RENDER_CFG),
        )


class TestChainTemplateMatch(ChainDecodeTestBase):
    def test_decode_matches_single_relation_chain_template(self):
        dec = self._decoder()
        text, ok = dec.decode(
            node_labels=["Smoke", "Cancer"],
            relation_labels=["causes"],
            chain=["causes"],
        )
        self.assertTrue(ok)
        self.assertIn("Smoke", text)
        self.assertIn("causes", text)
        self.assertIn("Cancer", text)

    def test_decode_matches_multi_relation_chain_template(self):
        dec = self._decoder()
        text, ok = dec.decode(
            node_labels=["Smoke", "Cancer", "Tumor"],
            relation_labels=["causes", "causes"],
            chain=["causes", "causes"],
        )
        self.assertTrue(ok)
        self.assertIn("this in turn causes", text)

    def test_antonym_chain_uses_relation_phrase(self):
        dec = self._decoder()
        text, ok = dec.decode(
            node_labels=["hot", "cold"],
            relation_labels=["antonym"],
            chain=["antonym"],
        )
        self.assertTrue(ok)
        self.assertIn("hot is the opposite of cold.", text)

    def test_chain_exactness_is_required(self):
        dec = self._decoder()
        text, ok = dec.decode(
            node_labels=["X", "Y"],
            relation_labels=["causes"],
            chain=["part_of"],  # no template matches; chain-path fallback uses walk labels
        )
        self.assertTrue(ok)
        self.assertIn("X causes Y.", text)

    def test_add_chain_template_registers_new_chain(self):
        dec = self._decoder()
        dec.add_chain_template(["part_of", "part_of"], "{node0} is part of {node1}, which is part of {node2}.")
        text, ok = dec.decode(
            node_labels=["Stock", "Capital", "City"],
            relation_labels=["part_of", "part_of"],
            chain=["part_of", "part_of"],
        )
        self.assertTrue(ok)
        self.assertIn("Stock is part of Capital, which is part of City.", text)


class TestChainPathFallback(ChainDecodeTestBase):
    def test_unknown_long_chain_renders_chain_path(self):
        dec = self._decoder()
        text, ok = dec.decode(
            node_labels=["Smoke", "Cancer", "Lung"],
            relation_labels=["causes", "part_of"],
            chain=["causes", "part_of"],
        )
        self.assertTrue(ok)
        self.assertIn("Smoke causes Cancer", text)
        self.assertIn("Cancer is part of Lung", text)
        self.assertIn(", and ", text)

    def test_starter_is_prepended_when_chain_path_used(self):
        dec = self._decoder()
        text, ok = dec.decode(
            node_labels=["Smoke", "Cancer"],
            relation_labels=["causes"],
            chain=["caused_by"],
        )
        self.assertTrue(ok)
        self.assertTrue(text.startswith("Therefore,"))

    def test_short_walk_returns_false(self):
        dec = self._decoder()
        text, ok = dec.decode(
            node_labels=["Lonely"],
            relation_labels=[],
            chain=["causes"],
        )
        self.assertEqual(text, "")
        self.assertFalse(ok)


class TestNoRelationAnswer(ChainDecodeTestBase):
    def test_render_no_relation_mentions_chain_and_concepts(self):
        dec = self._decoder()
        text = dec.render_no_relation(["Smoke", "Cancer"], chain=["causes"])
        self.assertIn("I don't have a relation", text)
        self.assertIn("Smoke, Cancer", text)
        self.assertIn("causes", text)

    def test_render_no_relation_without_labels(self):
        dec = self._decoder()
        text = dec.render_no_relation([], chain=["causes"])
        self.assertIn("I don't have a relation", text)
        self.assertNotIn("Closest concepts", text)


class TestChainRenderNoValidationGate(ChainDecodeTestBase):
    def test_render_chain_returns_without_validation_failure(self):
        dec = self._decoder()
        text = dec.render_chain(["Smoke", "Cancer"], ["causes"], ["causes"])
        self.assertIsNotNone(text)
        self.assertIn("Smoke", text)


if __name__ == "__main__":
    unittest.main()