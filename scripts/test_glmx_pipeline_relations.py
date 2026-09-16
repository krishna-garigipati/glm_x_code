"""DEVIATION 9: end-to-end relation-chain pipeline test (hermetic).

Builds a small in-memory graph, fakes SentenceTransformer with a deterministic
encoder, and runs the full GLM-X `ask()` path: extractor -> chain-guided walk ->
chain template decode -> Hebbian/ES feedback. Assertions target the relation
chain keys that replaced intent outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WS = Path(__file__).resolve().parent.parent
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))
SCRIPTS = str(WS / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


RELATIONS_16 = [
    "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
    "contradicts", "supports", "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident", "spatial_near", "linguistic_maps",
]

# Longest/most-specific first so "what is a dog" -> is_a, not has_property.
_KEYWORD_RULES = [
    ("is the opposite of", "antonym"),
    ("opposite of", "antonym"),
    ("antonym", "antonym"),
    ("contrary", "antonym"),
    ("another word for", "synonym"),
    ("synonym", "synonym"),
    ("same as", "synonym"),
    ("what causes", "causes"),
    ("leads to", "causes"),
    ("triggers", "causes"),
    ("results in", "causes"),
    ("brings about", "causes"),
    (" cause ", "causes"),
    ("causes", "causes"),
    ("cause", "causes"),
    (" is a ", "is_a"),
    ("a kind of", "is_a"),
    ("type of", "is_a"),
    ("kind of", "is_a"),
    ("category", "is_a"),
    ("belongs to the class", "is_a"),
    ("part of", "part_of"),
    ("makes up", "part_of"),
    ("component of", "part_of"),
    ("contained in", "part_of"),
    ("what is", "has_property"),
    ("what are", "has_property"),
    ("property", "has_property"),
    ("characterized by", "has_property"),
    ("possesses", "has_property"),
    ("contradicts", "contradicts"),
    ("disagrees with", "contradicts"),
    ("supports", "supports"),
    ("backed by", "supports"),
    ("evidence for", "supports"),
    ("associated with", "associated_with"),
    ("linked to", "associated_with"),
    ("related to", "associated_with"),
    ("an example of", "example_of"),
    ("for instance", "example_of"),
    ("near", "spatial_near"),
    ("close to", "spatial_near"),
    ("at the same time", "temporal_coincident"),
    ("simultaneously", "temporal_coincident"),
]


class FakeSBERT:
    def __init__(self, *args, **kwargs):
        pass

    def _classify(self, text: str):
        low = " " + text.lower().strip() + " "
        for keyword, relation in _KEYWORD_RULES:
            if keyword in low:
                return relation
        return None

    @staticmethod
    def _one_hot(relation: str) -> np.ndarray:
        vec = np.zeros(384, dtype=np.float32)
        vec[RELATIONS_16.index(relation) % 384] = 1.0
        return vec

    @staticmethod
    def _weak_vec() -> np.ndarray:
        rng = np.random.RandomState(7)
        v = rng.randn(384).astype(np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        return v * np.float32(0.05)  # below similarity_threshold

    @staticmethod
    def _node_emb(text: str) -> np.ndarray:
        rng = np.random.RandomState(hash(text) % (2 ** 32))
        v = rng.randn(384).astype(np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        return v

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        def _vec(t):
            relation = self._classify(t)
            return self._one_hot(relation) if relation is not None else self._weak_vec()

        if isinstance(texts, str):
            return _vec(texts)
        return np.stack([_vec(t) for t in texts])


@pytest.fixture(autouse=True)
def _fake_sbert(monkeypatch):
    import sentence_transformers
    import scripts.glmx_ask as _glmx

    # g2p_planner._get_sentence_model uses an inline `from sentence_transformers
    # import SentenceTransformer`, so patching the package-level name suffices.
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSBERT)
    monkeypatch.setattr(_glmx, "SentenceTransformer", FakeSBERT)


@pytest.fixture
def pipeline(_fake_sbert):
    import glmx_ask
    from graph.graph_component_implementation.dict_graph_store import DictGraphStore

    labels = ["smoking", "lung cancer", "DNA damage", "hot", "cold", "dog", "animal"]
    store = DictGraphStore()
    ids = {}
    for lab in labels:
        ids[lab] = store.add_node(label=lab, embedding=FakeSBERT._node_emb(lab), node_type="Concept")

    store.add_edge(ids["smoking"], ids["lung cancer"], "causes", strength=0.9, confidence=0.8)
    store.add_edge(ids["lung cancer"], ids["DNA damage"], "causes", strength=0.7, confidence=0.8)
    store.add_edge(ids["hot"], ids["cold"], "antonym", strength=0.9, confidence=0.8)
    store.add_edge(ids["dog"], ids["animal"], "is_a", strength=0.9, confidence=0.8)

    # Pin target entities onto their relation axis so the seed is deterministic:
    # the question embedding (one-hot on the same relation) wins the similarity.
    for lab, rel in (("smoking", "causes"), ("hot", "antonym"), ("dog", "is_a")):
        store._embeddings[ids[lab]] = FakeSBERT._one_hot(rel)

    p = glmx_ask.GLMXPipeline.__new__(glmx_ask.GLMXPipeline)
    p.sbert = FakeSBERT()
    p.graph_store = store
    p.load_models()
    return p


class TestRelationChainPipeline:
    def test_causal_question_uses_causes_chain(self, pipeline):
        result = pipeline.ask("Does smoking cause lung cancer?")
        assert result["relation_chain"] == ["causes"]
        assert result["heuristic_used"] is False
        assert result["answer"]
        assert result["n_walk_steps"] >= 1
        assert result["walk_path_edges"]

    def test_opposite_question_uses_antonym_chain(self, pipeline):
        result = pipeline.ask("What is the opposite of hot?")
        assert result["relation_chain"] == ["antonym"]
        assert result["heuristic_used"] is False
        assert result["answer"]
        assert "cold" in result["answer"]

    def test_definitional_question_uses_is_a_chain(self, pipeline):
        result = pipeline.ask("What is a dog?")
        assert result["relation_chain"] == ["is_a"]
        assert result["heuristic_used"] is False
        assert "animal" in result["answer"]

    def test_chain_of_causes_supports_multi_hop(self, pipeline):
        result = pipeline.ask("What does smoking cause?")
        assert "causes" in result["relation_chain"]
        assert len(result["walk_path_labels"]) >= 2

    def test_relation_details_len_matches_chain(self, pipeline):
        result = pipeline.ask("What is the opposite of hot?")
        assert len(result["relation_details"]) == len(result["relation_chain"])
        for line in result["relation_details"]:
            assert line.startswith("  step ")

    def test_unrelated_question_hits_default_chain_fallback(self, pipeline):
        result = pipeline.ask("What color is the moon?")
        assert result["heuristic_used"] is True
        assert result["relation_chain"]

    def test_result_has_no_intent_keys(self, pipeline):
        result = pipeline.ask("What is a dog?")
        assert "relation_chain" in result
        assert "intent_used" not in result
        assert "plan_intents" not in result


class TestPipelineLearning:
    def test_feedback_does_not_crash(self, pipeline):
        before = len(pipeline.graph_store._edges_raw)
        pipeline.ask("Does smoking cause lung cancer?")
        pipeline.ask("What is the opposite of hot?")
        assert len(pipeline.graph_store._edges_raw) >= before

    def test_es_theta_pushes_relation_biases(self, pipeline):
        walker = pipeline.walker
        snap = walker.relation_bias_snapshot()
        assert isinstance(snap, dict)
        assert set(snap) <= {
            "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
            "contradicts", "supports", "associated_with", "example_of", "part_of",
            "synonym", "antonym", "temporal_coincident", "spatial_near", "linguistic_maps",
        }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))