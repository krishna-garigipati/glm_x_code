"""Tests for the heuristic-fallback path of QueryRelationExtractor.

Contract v3.3.2 section 8 fallback.preferred_trigger: it fires on CUE ABSENCE
("no strong relation cue words or phrases from any descriptor bank are present
in the question"), NOT on a similarity threshold. The audit showed real and
nonsense questions have overlapping similarity distributions (junk text still
scores >=0.59 cosine), so a threshold-only trigger was unreachable (observed
0/99) and could never have separated real from nonsense.

Contract section 8 defines ONE descriptor bank (`relation_variants`). The same
entries serve as the embedding bank and as the literal cue strings. There is no
separate `cue_phrases` key and no `fallback.trigger` enum in the contract, so
neither is referenced here.

Literal cue matching is authoritative when a descriptor is present; embedding
similarity is consulted only for questions with no descriptor, which is how
relational questions still resolve through the frozen encoder.

These are unit tests: `_best_relation_for_clause` is stubbed so no
SentenceTransformer model is loaded, matching the stub pattern in
scripts/tests/test_glmx_ask_rescue.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from g2p.config import FallbackConfig, G2PConfig, RelationExtractionConfig  # noqa: E402
from g2p.g2p_planner import QueryRelationExtractor  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "config_g2p.yaml"

NONSENSE = [
    "What is the flurbomatic price of a zorptastic quibble?",
    "How many gribbles does a wompus contain?",
    "Who invented the snorpleflectometer?",
    "Where can I buy a frimblewhistle amplifier?",
]


def _config() -> G2PConfig:
    """Load the real config.

    A bare G2PConfig() fails validation ("relation_variants must be
    non-empty"), so tests must go through from_yaml.
    """
    config = G2PConfig.from_yaml(str(CONFIG_PATH))
    config.validate()
    return config


def _planner(config: G2PConfig) -> QueryRelationExtractor:
    """Build a planner without touching the encoder.

    `_best_relation_for_clause` is replaced per-test, so the frozen-encoder
    bank is never consulted and no model is downloaded.
    """
    planner = QueryRelationExtractor.__new__(QueryRelationExtractor)
    planner.config = config
    planner.label_map = {}
    planner._sentence_model = None
    planner._variant_embeddings = {}
    planner._initialized = True
    planner._compiled_descriptors = planner._compile_descriptors()
    return planner


def test_fallback_fires_when_question_has_no_descriptor() -> None:
    """Section 8: absence of any descriptor triggers fallback."""
    planner = _planner(_config())
    planner._best_relation_for_clause = lambda clause: (None, 0.0)

    plan = planner.extract("Tell me about the migratory patterns of the salmon")

    assert plan.heuristic_fallback_used is True
    assert plan.intent_sequence is None
    assert plan.intent_names is None


def test_fallback_substitutes_configured_default_chain() -> None:
    """The fallback chain comes from config, not a hardcoded literal."""
    config = _config()
    planner = _planner(config)
    planner._best_relation_for_clause = lambda clause: (None, 0.0)

    plan = planner.extract("Tell me about the migratory patterns of the salmon")

    assert plan.relation_chain == list(config.extraction.fallback.default_chain)


def test_fallback_assigns_fallback_confidence() -> None:
    """A fallback plan is low confidence and must not look like a match."""
    planner = _planner(_config())
    planner._best_relation_for_clause = lambda clause: (None, 0.0)

    plan = planner.extract("Tell me about the migratory patterns of the salmon")

    assert plan.plan_confidence == pytest.approx(0.6)


def test_fallback_not_used_when_descriptor_is_present() -> None:
    """Negative control: the flag discriminates rather than always being true.

    The embedding stub returns nothing, so the chain can only be produced by
    literal descriptor matching. A bare indefinite article is deliberately NOT
    a descriptor (otherwise "What is a blorptastic dendroidian?" would
    masquerade as a real question), so this uses a genuine relational
    descriptor.
    """
    planner = _planner(_config())
    planner._best_relation_for_clause = lambda clause: (None, 0.0)

    plan = planner.extract("What is the opposite of hot?")

    assert plan.heuristic_fallback_used is False
    assert plan.relation_chain == ["antonym"]


def test_descriptor_wins_over_stronger_embedding_for_opposite_relation() -> None:
    """Literal matching is authoritative for direction.

    The frozen encoder is direction-blind: it scores "What causes X?" and
    "What is X caused by?" almost identically, which previously inverted
    causes/caused_by. A descriptor must win even against a high-similarity
    stub.
    """
    planner = _planner(_config())
    planner._best_relation_for_clause = lambda clause: ("caused_by", 0.99)

    assert planner.extract("What causes rain?").relation_chain == ["causes"]
    assert planner.extract("What is rain caused by?").relation_chain == ["caused_by"]


def test_nonsense_questions_trigger_fallback() -> None:
    """The regression that motivated the descriptor-absence trigger."""
    config = _config()
    planner = _planner(config)
    planner._best_relation_for_clause = lambda clause: ("is_a", 0.72)

    for question in NONSENSE:
        plan = planner.extract(question)
        assert plan.heuristic_fallback_used is True, question
        assert plan.relation_chain == list(config.extraction.fallback.default_chain), question


def test_definitional_question_matches_is_a_without_fallback() -> None:
    """A definitional question carries an is_a descriptor, so no fallback.

    The is_a bank owns the bare "is a" form, which is what a definitional
    question ("What is a salmon?") literally contains. A previous revision used
    a `known_labels` subject anchor to force this case out of fallback; section
    8 defines no subject anchor, so that path was removed and the descriptor
    bank covers the phrasing instead.
    """
    planner = _planner(_config())
    planner._best_relation_for_clause = lambda clause: (None, 0.0)

    plan = planner.extract("What is a salmon?")

    assert plan.heuristic_fallback_used is False
    assert plan.relation_chain == ["is_a"]


def test_nonsense_subject_is_rejected_by_entity_gate_not_by_fallback_flag() -> None:
    """An invented subject is nonsense, but not at the extraction stage.

    "What is a blorptastic dendroidian?" is structurally a definitional
    question, so the is_a descriptor legitimately matches and the cue-absence
    trigger correctly does NOT fire. Rejecting an invented subject is the job of
    the entity-existence honesty gate in scripts/glmx_ask.py, which checks the
    anchored node against the graph before any walk is reported. This test
    pins that responsibility split so the extractor is not later "fixed" to
    guess subject validity from a label list it is not given.
    """
    planner = _planner(_config())
    planner._best_relation_for_clause = lambda clause: (None, 0.0)

    plan = planner.extract("What is a blorptastic dendroidian?")

    assert plan.relation_chain == ["is_a"]
    assert plan.heuristic_fallback_used is False


def test_default_chain_relations_must_be_banked() -> None:
    """A default_chain entry absent from the bank is a config error.

    Guards the fallback: it would otherwise emit a relation the walker can
    never traverse.
    """
    raw = {
        "relation_variants": {"is_a": ["is a"] * 5},
        "default_chain": ["has_property"],
    }
    config = RelationExtractionConfig.from_yaml(raw)

    with pytest.raises(ValueError, match="default_chain relation"):
        config.validate()


def test_default_chain_subset_of_bank_validates() -> None:
    raw = {
        "relation_variants": {
            "is_a": ["is a", "is a kind of", "is a type of", "kind of", "type of"],
            "has_property": ["possesses", "has a property", "is characterised by",
                             "has an attribute", "is known for"],
        },
        "default_chain": ["has_property"],
    }
    config = RelationExtractionConfig.from_yaml(raw)

    config.validate()
    assert config.default_chain == ["has_property"]


def test_shipped_config_default_chain_is_banked() -> None:
    """The real config's default_chain must resolve against its own bank."""
    config = _config()

    for relation in config.extraction.default_chain:
        assert relation in config.extraction.relation_variants


def test_threshold_alone_cannot_trigger_fallback() -> None:
    """Section 8 forbids relying on the similarity threshold.

    Junk text scores >=0.59 cosine against the frozen bank, above the 0.35
    threshold, so a threshold-only trigger is unreachable in practice. These
    questions carry no descriptor, so fallback must fire even though the
    embedding stub returns a score comfortably above threshold.
    """
    config = _config()
    planner = _planner(config)
    above = config.extraction.similarity_threshold + 0.4
    planner._best_relation_for_clause = lambda clause: ("is_a", above)

    plan = planner.extract("Tell me about the migratory patterns of the salmon")

    assert plan.heuristic_fallback_used is True


def test_no_separate_cue_bank_exists() -> None:
    """Section 8 defines one descriptor bank; `cue_phrases` is not a contract key."""
    config = _config()

    assert not hasattr(config.extraction, "cue_phrases")


def test_no_fallback_trigger_enum_exists() -> None:
    """Section 8 states the trigger as prose, not a config enum."""
    assert not hasattr(FallbackConfig(), "trigger")
    assert set(FallbackConfig.from_yaml({}).__dict__) == {
        "enabled", "flag", "default_chain",
    }


def test_descriptor_banks_meet_contract_expectation() -> None:
    """Section 8.8 expects at least 5-8 strong descriptors per relation."""
    config = _config()

    for relation, variants in config.extraction.relation_variants.items():
        assert len(variants) >= 5, relation


def test_direction_sensitive_banks_do_not_overlap() -> None:
    """Section 8.8 requires clearly distinct descriptors for inverse pairs.

    Shared descriptors between causes/caused_by, precedes/follows, or
    part_of/has_part defeat literal direction matching.
    """
    variants = _config().extraction.relation_variants

    for a, b in (("causes", "caused_by"), ("precedes", "follows"), ("part_of", "has_part")):
        if a not in variants or b not in variants:
            continue
        left = {v.strip().lower() for v in variants[a]}
        right = {v.strip().lower() for v in variants[b]}
        assert not (left & right), (a, b, sorted(left & right))


def test_short_descriptor_bank_warns_without_raising() -> None:
    """Section 8.8 is an expectation, not a hard failure.

    A thin bank must be reported but must not block execution, because the
    contract phrases it as "Minimum expectation".
    """
    raw = {
        "relation_variants": {
            "is_a": ["is a", "is a kind of", "is a type of", "kind of", "type of"],
            "has_property": ["possesses", "has a property"],
        }
    }
    config = RelationExtractionConfig.from_yaml(raw)

    config.validate()


def test_collapse_consecutive_repeats_is_boolean() -> None:
    """Section 8 replaces the old collapse_max run-length cap."""
    config = _config()

    assert config.extraction.collapse_consecutive_repeats is True
    assert isinstance(config.extraction.collapse_consecutive_repeats, bool)


def test_fallback_config_defaults_are_section_8_compliant() -> None:
    fallback = FallbackConfig()

    assert fallback.enabled is True
    assert fallback.flag == "heuristic_fallback_used"
    assert fallback.default_chain == ["has_property"]