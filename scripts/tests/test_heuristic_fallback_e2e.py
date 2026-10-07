"""End-to-end test: the heuristic fallback must not fabricate a relation.

When the fallback substitutes `default_chain` (['has_property']) for a
question that matched no bank phrase, the pipeline's W4b honesty gate must
still refuse to answer when the anchored entity has no such edge.

This is the anti-fabrication guarantee. Without it, raising
`similarity_threshold` above the encoder's junk-similarity floor (~0.59)
would make the system confidently assert "X has <property>" for entities
that have no property at all.

Requires a real SentenceTransformer load, so it is slow. It also monkeypatches
`_best_relation_for_clause` to force the fallback, because the fallback is
otherwise unreachable at the shipped similarity_threshold (see
g2p/tests/test_heuristic_fallback.py).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB = ROOT / "test_results" / "tester-b" / "datasets" / "food_bio_small.db"

pytestmark = pytest.mark.slow


def _unmock_sentence_transformers() -> None:
    """Drop any mocked sentence_transformers so the real package is imported.

    Two modules pollute sys.modules['sentence_transformers'] with a MagicMock
    at *import* time and never restore it:

        g2p/tests/test_g2p_all.py   lines 18-19, 26
        g2p/tests/toy_pipeline.py   line 27

    That pollution is process-global. In a combined pytest run this module
    would otherwise build its pipeline on the mock, so query embeddings become
    random vectors whose width does not match the dataset's, raising
    "Mismatched embedding shapes" in
    graph/graph_component_implementation/utils.cosine_similarity.

    The mocks are always MagicMock instances (or classes derived from one), so
    drop any entry that is not a real module.

    Clearing sys.modules alone is not enough: scripts/glmx_ask.py line 33 does
    a module-level `from sentence_transformers import SentenceTransformer`, so
    the mock is already bound into that module's namespace by the time we get
    here. Rebind it to the freshly imported real class.
    """
    for name in list(sys.modules):
        if name != "sentence_transformers" and not name.startswith("sentence_transformers."):
            continue
        entry = sys.modules.get(name)
        if entry is None:
            continue
        if isinstance(entry, MagicMock) or (isinstance(entry, type) and issubclass(entry, MagicMock)):
            del sys.modules[name]

    import scripts.glmx_ask as glmx_ask_module
    from sentence_transformers import SentenceTransformer as RealSentenceTransformer

    if isinstance(glmx_ask_module.SentenceTransformer, MagicMock):
        glmx_ask_module.SentenceTransformer = RealSentenceTransformer


@pytest.fixture(scope="module")
def pipeline():
    if not DB.exists():
        pytest.skip(f"dataset missing: {DB}")

    _unmock_sentence_transformers()

    from scripts.glmx_ask import GLMXPipeline
    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

    pipe = GLMXPipeline()
    pipe.graph_store = SQLiteGraphStore.load_state(str(DB))
    pipe._seed = 0
    pipe._no_learning = True
    pipe.load_models()

    # Force the fallback: no clause ever clears the threshold.
    pipe.planner._best_relation_for_clause = lambda clause: (None, 0.0)
    return pipe


def test_fallback_refuses_when_anchor_lacks_relation(pipeline) -> None:
    """salmon has is_a but no has_property: must say "I don't know"."""
    result = pipeline.ask("Tell me about the migratory patterns of the salmon")

    assert result["heuristic_used"] is True
    assert result["relation_chain"] == ["has_property"]
    assert result["honest_no_relation"] is True
    assert "don't have a relation" in result["answer"]


def test_fallback_answers_when_anchor_has_relation(pipeline) -> None:
    """lemon does have a has_property edge, so the walk is legitimate."""
    result = pipeline.ask("Tell me about the flavour profile of the lemon")

    assert result["heuristic_used"] is True
    assert result["relation_chain"] == ["has_property"]
    assert result["honest_no_relation"] is False
    assert "sour" in result["answer"]


def test_fallback_answers_for_second_property_entity(pipeline) -> None:
    """honey is a second has_property anchor; guards against single-case luck."""
    result = pipeline.ask("Tell me about the dietary value of the honey")

    assert result["heuristic_used"] is True
    assert result["honest_no_relation"] is False
    assert "sweet" in result["answer"]


def test_fallback_does_not_fabricate_property_for_every_entity(pipeline) -> None:
    """The gate must discriminate, not blanket-refuse.

    If this ever fails by refusing the valid lemon/honey cases, the honesty
    gate has become over-eager and would suppress answerable questions.
    """
    refusing = pipeline.ask("Tell me about the migratory patterns of the salmon")
    answering = pipeline.ask("Tell me about the flavour profile of the lemon")

    assert refusing["honest_no_relation"] is True
    assert answering["honest_no_relation"] is False
