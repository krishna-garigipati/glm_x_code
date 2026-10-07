"""Unit tests for GLMXPipeline._exact_label_match longest-label preference.

`_exact_label_match` decides which graph node a question is about. It used to
scan single tokens only, rightmost first, so when one node's label is a
contiguous sub-phrase of another's the SHORTER label always won:

    "What type of thing is the biological process?"  ->  'process'

and the system then answered confidently about the wrong entity. Measured on the
frozen Stage E PoC set that mis-resolution hit 13 of 208 questions and caused
all 11 failures.

The fix adds a phrase pass ahead of the token pass: longest contiguous n-gram
wins, rightmost breaks ties. These tests pin that behaviour and -- just as
importantly -- pin the cases that must NOT change, because a longest-match rule
that over-reaches would silently break multi-clause anchoring, which is what the
first-clause scope exists to protect.

Tested with a stub store so no SentenceTransformer model is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.glmx_ask import GLMXPipeline  # noqa: E402


class _StubStore:
    """Only the two things _exact_label_match touches."""

    def __init__(self, labels):
        self._label_to_id = {lbl: i for i, lbl in enumerate(labels)}
        self._ids = {i: lbl for lbl, i in self._label_to_id.items()}

    def get_label(self, nid):
        return self._ids[nid]


def _pipeline(labels):
    p = GLMXPipeline.__new__(GLMXPipeline)
    p.graph_store = _StubStore(labels)
    return p


def _anchor(labels, question):
    """The label the pipeline anchors on, or None."""
    nid = _pipeline(labels)._exact_label_match(question)
    return None if nid is None else _pipeline(labels).graph_store.get_label(nid)


# The nine sub-phrase collisions actually present in the frozen Stage E graph,
# with the label each question means.
SUB_PHRASE_LABELS = [
    "alarm", "fire alarm",
    "device", "safety device",
    "group", "musical group",
    "heavy", "heavy rain",
    "instrument", "string instrument",
    "moon", "moon crater record",
    "piano", "grand piano",
    "planet", "terrestrial planet",
    "process", "biological process",
]


def test_subphrase_collisions_resolve_to_the_longer_label():
    """Every collision pair must resolve to the longer, more specific label."""
    cases = [
        ("What type of thing is the biological process?", "biological process"),
        ("What type of thing is the fire alarm?", "fire alarm"),
        ("What type of thing is the musical group?", "musical group"),
        ("What type of thing is the string instrument?", "string instrument"),
        ("What type of thing is the terrestrial planet?", "terrestrial planet"),
        ("What is known for the grand piano?", "grand piano"),
        ("What is another word for the grand piano?", "grand piano"),
        ("What does the moon crater record support?", "moon crater record"),
        ("What contradicts the moon crater record?", "moon crater record"),
        ("What does the heavy rain cause?", "heavy rain"),
        ("What was the fire alarm caused by?", "fire alarm"),
        ("What is the string instrument an example of?", "string instrument"),
    ]
    labels = SUB_PHRASE_LABELS
    for question, expected in cases:
        got = _anchor(labels, question)
        assert got == expected, f"{question!r} anchored on {got!r}, want {expected!r}"


def test_shorter_label_still_wins_when_the_longer_is_absent():
    """Asking about 'piano' must still anchor on 'piano'.

    Longest-match must not become 'always pick a multi-word label'. If the
    question names only the short concept, that is the subject.
    """
    assert _anchor(["piano", "keyboard"], "What is known for the piano?") == "piano"
    assert _anchor(["alarm", "clock"], "What type of thing is the alarm?") == "alarm"


def test_first_clause_scope_still_protects_multi_hop_anchors():
    """The subject of the FIRST relation wins; a later clause must not steal it.

    This is the behaviour the first-clause scope was added for (Stage A: the
    'fin'/'fish' questions walked backwards off their own anchor and multi-hop
    collapsed to 1 hop). The phrase pass must not weaken it.
    """
    labels = ["fin", "fish", "shark", "animal", "string instrument", "instrument"]
    assert _anchor(labels, "What is the fin part of, and what is a fish?") == "fin"
    # A multi-word label in the SECOND clause must not win over a first-clause
    # subject, even though the phrase pass now prefers longer labels.
    assert _anchor(
        labels,
        "What is the fin part of, and what is the string instrument a part "
        "of?") == "fin"


def test_relation_word_before_subject_does_not_block_the_phrase():
    """A relation cue sitting between the cue and the subject must not hide it.

    The common real shape is "<relation words> <subject>": the subject is the
    LAST thing named, and it is often multi-word. Scanning n-grams of the whole
    token run handles this because the subject's own words are adjacent.

    All 11 phrase-pass hits in the frozen Stage E set are this shape
    ("What contradicts the moon crater record?", "What is known for the grand
    piano?"), and all 4 in the Stage D set.
    """
    labels = SUB_PHRASE_LABELS
    assert _anchor(labels, "What contradicts the moon crater record?") \
        == "moon crater record"
    assert _anchor(labels, "What evidence supports the moon crater record?") \
        == "moon crater record"


def test_known_residual_limitation_same_clause_object_phrase():
    """DOCUMENTS A PRE-EXISTING LIMITATION. Do not read this as desired behaviour.

    When a single-token subject and a longer label sit in the same clause with
    no relation cue between them, the rightmost-longest rule takes the longer
    label. "What is the fin part of the string instrument?" should anchor on
    `fin` but anchors on `string instrument`.

    This is NOT a regression from the phrase pass: the previous token scan
    anchored on `instrument` in exactly the same question, which is equally
    wrong. Fixing it properly needs the anchor to be attached to its nearest
    relation cue rather than found by position, which is a larger change than
    this fix and is out of scope.

    The shape does not occur in any frozen evaluation set: in Stages D and E
    every phrase-pass hit has the subject last and preceded by relation words,
    which is the shape the rule handles correctly. Pinned here so a future
    change to this behaviour is visible rather than silent.
    """
    labels = ["fin", "fish", "animal", "string instrument", "instrument"]
    assert _anchor(labels, "What is the fin part of the string instrument?") \
        == "string instrument"


def test_single_token_behaviour_is_unchanged():
    """Rightmost-first token preference, for labels that are not sub-phrases."""
    labels = ["wheel", "bicycle", "sail", "yacht"]
    assert _anchor(labels, "What is the wheel part of?") == "wheel"
    assert _anchor(labels, "What is the sail part of?") == "sail"


def test_question_filler_still_excluded():
    """Words like 'type'/'thing'/'part' must not enter the phrase n-grams.

    Without filler removal a phrase pass would try to match spans like
    'type of thing', and could anchor on a node whose label happens to contain
    question scaffolding.
    """
    labels = ["thing", "type", "wheel", "bicycle"]
    assert _anchor(labels, "What type of thing is the wheel?") == "wheel"
    assert _anchor(labels, "What is the part of the wheel?") == "wheel"


def test_no_match_returns_none():
    assert _anchor(["piano"], "What is known for the zebra?") is None


def test_empty_store_returns_none():
    assert _anchor([], "What type of thing is the grand piano?") is None


def test_deterministic_across_repeated_calls():
    """Anchoring must be a pure function of the question and the graph."""
    p = _pipeline(SUB_PHRASE_LABELS)
    question = "What is known for the grand piano?"
    first = p.graph_store.get_label(p._exact_label_match(question))
    for _ in range(20):
        nid = p._exact_label_match(question)
        assert p.graph_store.get_label(nid) == first


def test_longest_of_three_wins():
    """With a three-label nesting, the longest real label wins."""
    labels = ["record", "crater record", "moon crater record"]
    assert _anchor(labels, "What supports the moon crater record?") \
        == "moon crater record"


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)