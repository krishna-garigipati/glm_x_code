"""Stage D -- freeze the honesty, fallback and control question set.

Every graded pair here is DERIVED FROM the validated pair tables in
build_honesty_fallback_graph.py, and every question records the relation_chain,
anchor and expected_path it expects BEFORE anything is run. That is the
pre-registration rule from EXPERIMENTATION.md section 3.1: the golden list is
written down first and is not edited afterwards to raise a pass rate.

The previous version of this file did not do that. It hand-typed its own
`(anchor, relation, expected_node)` triples, which had drifted from the graph:

  * five "controls" carried `expected_node == anchor`, so the runner's
    `want in answer` test was satisfied by any answer that merely echoed the
    subject -- the check could not fail;
  * three of them asked for the BACKWARD read of a one-way relation
    ("What is an example of a bird?", "What supports the evolution theory?").
    The graph stores `sparrow example_of bird`, never `bird example_of
    sparrow`, so answering them requires exactly the illegal same-label
    mirroring the contract forbids. Refusing is the CORRECT behaviour, and
    grading refusal as a failure inverted the stage's whole purpose;
  * "What is the sparrow synonymous with?" matched no cue in the synonym
    descriptor bank, so the planner correctly fired cue-absence fallback and
    refused. It is reworded here to use a cue that exists.

Categories, and what each one is entitled to assert:

  control              a stored edge exists in the asked direction and is the
                       unique 1-hop target -> the system must ANSWER
  missing_relation     the anchor is in the graph but has no such relation at
                       all -> the system must REFUSE
  inverse_direction    the relation exists only in the OPPOSITE stored
                       direction, on a relation with no declared inverse ->
                       the system must REFUSE and must NOT name the far end
  honesty_out_of_graph the subject is absent from the graph -> must REFUSE
  nonsense_fallback    no relation cue from any descriptor bank -> the
                       heuristic fallback must FIRE

Usage:
    python test_results/stage_d/generate_questions.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STAGE_DIR = ROOT / "test_results" / "stage_d"
DB_PATH = STAGE_DIR / "honesty_fallback.db"
OUT_PATH = STAGE_DIR / "questions_frozen.json"

# The validated pair tables live with the graph builder. Importing them means
# the question set cannot disagree with the graph it grades: change an edge and
# the builder's validate() refuses to freeze, and change a pair here and the
# preflight in stage_d_runner.py refuses to run.
_spec = importlib.util.spec_from_file_location(
    "stage_d_graph", STAGE_DIR / "build_honesty_fallback_graph.py"
)
GRAPH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GRAPH)

QUESTIONS_VERSION = "1.1.0"

# --------------------------------------------------------------------------
# Natural-language phrasings, keyed by (anchor, relation).
#
# Each must (a) put the ANCHOR as the last non-filler noun token so exact-label
# anchoring lands on the intended node, and (b) contain a cue phrase that is
# LITERALLY present in configs/config_g2p.yaml, so the planner resolves the
# relation by literal match rather than by embedding similarity. The cue that
# fires is recorded per question as `chain` below and is asserted by preflight.
# --------------------------------------------------------------------------

# (anchor, relation) -> (question, expected_chain)
CONTROL_PHRASINGS: dict[tuple[str, str], tuple[str, list[str]]] = {
    ("cat", "is_a"):
        ("What is a cat?", ["is_a"]),
    ("hammer", "is_a"):
        ("What is a hammer?", ["is_a"]),
    ("wrench", "part_of"):
        ("What is the wrench part of?", ["part_of"]),
    ("screwdriver", "part_of"):
        ("What is the screwdriver part of?", ["part_of"]),
    ("storm", "causes"):
        ("What does the storm cause?", ["causes"]),
    ("flood", "caused_by"):
        ("What was the flood caused by?", ["caused_by"]),
    ("volcano", "causes"):
        ("What does the volcano cause?", ["causes"]),
    ("earthquake", "causes"):
        ("What does the earthquake cause?", ["causes"]),
    ("spore", "precedes"):
        ("What comes after the spore?", ["precedes"]),
    ("plantlet", "follows"):
        ("What comes before the plantlet?", ["follows"]),
    # Reached through the stored `plantlet follows sprout`, the declared inverse
    # of the asked `precedes`. "What comes after X" resolves to `precedes`
    # because the anchor is the subject and the walk goes forward to the answer.
    ("plantlet", "precedes"):
        ("What comes after the plantlet?", ["precedes"]),
    ("hot", "antonym"):
        ("What is the opposite of hot?", ["antonym"]),
    # Reworded from "What is the sparrow synonymous with?", which matched no cue
    # in the synonym bank. Uses the existing "what is another word for" cue, so
    # configs/config_g2p.yaml is untouched and Stages A/B/C carry no risk.
    ("sparrow", "synonym"):
        ("What is another word for the sparrow?", ["synonym"]),
    ("oak", "spatial_near"):
        ("What is near the oak?", ["spatial_near"]),
    ("cat", "has_property"):
        ("What is known for the cat?", ["has_property"]),
    ("dog", "linguistic_maps"):
        ("What do you call the dog in Spanish?", ["linguistic_maps"]),
    ("sparrow", "example_of"):
        ("What is the sparrow an example of?", ["example_of"]),
    ("tennis", "associated_with"):
        ("What is tennis associated with?", ["associated_with"]),
    ("paris", "part_of"):
        ("What is paris part of?", ["part_of"]),
    ("fossil record", "contradicts"):
        ("What contradicts the fossil record?", ["contradicts"]),
    ("fossil record", "supports"):
        ("What evidence supports the fossil record?", ["supports"]),
    ("trout", "example_of"):
        ("What is the trout an example of?", ["example_of"]),
    ("autumn", "temporal_coincident"):
        ("What happens at the same time as autumn?", ["temporal_coincident"]),
    ("microscope", "is_a"):
        ("What is a microscope?", ["is_a"]),
}

# (anchor, relation) -> (question, expected_chain) for the anchor-in-graph,
# relation-absent gaps.
MISSING_RELATION_PHRASINGS: dict[tuple[str, str], tuple[str, list[str]]] = {
    ("abacus", "part_of"):
        ("What is the abacus part of?", ["part_of"]),
    ("abacus", "causes"):
        ("What does the abacus cause?", ["causes"]),
    ("microscope", "spatial_near"):
        ("What is near the microscope?", ["spatial_near"]),
    ("marble", "synonym"):
        ("What is another word for the marble?", ["synonym"]),
    ("canoe", "antonym"):
        ("What is the opposite of a canoe?", ["antonym"]),
    ("kite", "has_property"):
        ("What is known for the kite?", ["has_property"]),
    ("harmonica", "temporal_coincident"):
        ("What happens at the same time as the harmonica?", ["temporal_coincident"]),
    ("oak", "synonym"):
        ("What is another word for the oak?", ["synonym"]),
    ("rose", "supports"):
        ("What is supported by the rose?", ["supports"]),
    ("chess", "example_of"):
        ("What is the chess an example of?", ["example_of"]),
    ("butterfly", "spatial_near"):
        ("What is near the butterfly?", ["spatial_near"]),
}

# (anchor, relation) -> (question, expected_chain, far_end_that_must_not_be_named)
#
# Each far end is the node sitting at the OTHER end of the stored edge. Naming
# it means the engine read the edge backwards, which for a relation with no
# declared inverse is the illegal same-label mirroring this stage exists to
# catch. These are the questions the previous set had wrongly filed as controls.
#
# "Give me an example of bird." rather than "What is an example of a bird?":
# the longer phrasing also contains "is an example of", which is an `is_a`
# descriptor, so the planner planned a two-relation chain and the question was
# no longer a clean single-relation backwards read. The short form carries only
# the `example_of` cue.
INVERSE_DIRECTION_PHRASINGS: dict[tuple[str, str], tuple[str, list[str], str]] = {
    ("evolution theory", "supports"):
        ("What supports the evolution theory?", ["supports"], "fossil record"),
    ("red", "has_property"):
        ("What has the property red?", ["has_property"], "rose"),
    ("insect", "example_of"):
        ("Give me an example of insect.", ["example_of"], "butterfly"),
    ("racket", "associated_with"):
        ("What is racket associated with?", ["associated_with"], "tennis"),
    ("young earth claim", "contradicts"):
        ("What claim contradicts young earth claim?", ["contradicts"], "fossil record"),
    ("bird", "example_of"):
        ("Give me an example of bird.", ["example_of"], "sparrow"),
    ("tall", "has_property"):
        ("What has the property tall?", ["has_property"], "oak"),
}

# Subjects that are absent from the graph. Each also carries a relation cue, so
# the planner resolves a REAL chain rather than falling back -- the refusal has
# to come from the entity gate, not from cue absence.
#
# `subject` is registered separately from the question text so the runner's
# preflight can verify absence MECHANICALLY against the store, rather than
# trusting that the word is unusual enough to look absent.
#
# Every one of these must also contain a descriptor-bank cue. Without one the
# planner fires cue-absence fallback, and the question would then be testing the
# fallback path rather than the entity gate -- two of the originals ("What is the
# quark made of?", "What is the klingon word for star?") carried no literal cue
# because "made of" and "word for" are not in the bank, and preflight rejected
# them. Both were reworded to phrasings that do.
OUT_OF_GRAPH: list[tuple[str, str, str]] = [
    ("o01", "unicorn", "What is a unicorn?"),
    ("o02", "zeppelin", "What is a zeppelin?"),
    ("o03", "quark", "What is the quark part of?"),
    ("o04", "telepathy", "What causes telepathy?"),
    ("o05", "klingon", "What is another word for klingon?"),
    ("o06", "dragon", "What is a dragon?"),
    ("o07", "narwhal", "What is a narwhal?"),
    ("o08", "thunderbolt", "What causes a thunderbolt?"),
]

# Cue-free questions. The fallback trigger in contract section 8 is cue ABSENCE,
# not a low similarity score, so every one of these must set
# heuristic_fallback_used. None of them may contain a descriptor-bank phrase;
# preflight re-checks that mechanically.
NONSENSE_CUE_FREE: list[str] = [
    "What color is the sound of Monday?",
    "How many electrons in a thought?",
    "Where does the sky taste salty?",
    "What shape is loneliness?",
    "How heavy is the smell of rain?",
]

# The disclosed-guess case, and the reason it exists.
#
# The five questions above are anchorless, so the entity honesty gate fires and
# the pipeline emits a refusal. That path never reaches the disclosure branch in
# glmx_ask.ask, so "fallback fired" would be proven while "the guess was
# disclosed as a guess" is never exercised at all -- the whole of contract
# section 8's fallback requirement would go untested while the stage still
# reported green.
#
# This question is cue-free too, but names a real anchor that HAS a has_property
# edge. The fallback fires, the default_chain is used, a real answer is produced
# from a real edge, and the disclosure prefix must appear on it.
#
# The specific wording matters and is not interchangeable. The walker drops a
# candidate whose node activation falls below walk.min_activation
# (walker/graph_walker.py, `_score_candidates`), and activation comes from the
# resonance seeded by the question embedding. "Anything worth knowing about the
# oak?" anchors correctly but resonates too weakly to activate `tall`, so the
# walk stops at the anchor and the honesty gate fires -- the disclosure branch is
# never reached. Naming the target property directly in the question activates
# it without introducing any relation cue, which is what makes this the only
# shape that actually exercises the path.
DISCLOSED_GUESS: str = "How tall is the oak?"


def _one_hop_target(anchor: str, rel: str) -> str:
    """The unique 1-hop target the builder already validated."""
    reach = GRAPH.reach_from(anchor, rel, hops=1)
    if len(reach) != 1:
        raise SystemExit(
            f"REFUSING TO FREEZE: ({anchor!r}, {rel!r}) is not 1-hop unique: {sorted(reach)}"
        )
    return next(iter(reach))


def _stored_far_end(anchor: str, rel: str) -> tuple[str, str]:
    """The other end of the stored `rel` edge touching `anchor`, plus its direction.

    Returns `(far_end, direction)` where direction is "out" when the stored edge
    leaves the anchor (`anchor rel far_end`) and "in" when it arrives at the
    anchor (`far_end rel anchor`).

    Direction matters. An `inverse_direction` question is only testing anything
    if the anchor is the ARRIVING end, because that is precisely the situation
    the walker cannot answer without illegally mirroring a one-way relation. A
    pair whose stored edge points the other way is an ordinary control that has
    been mislabelled, and it must not be smuggled into this category.
    """
    found: set[tuple[str, str]] = set()
    for src, edge_rel, tgt, *_ in GRAPH.EDGES:
        if edge_rel != rel:
            continue
        if src == anchor:
            found.add((tgt, "out"))
        if tgt == anchor:
            found.add((src, "in"))
    if len(found) != 1:
        raise SystemExit(
            f"REFUSING TO FREEZE: ({anchor!r}, {rel!r}) must touch exactly one "
            f"stored {rel} edge, found {sorted(found)}"
        )
    return next(iter(found))


def build() -> dict:
    questions: list[dict] = []

    # ---- control ---------------------------------------------------------
    for i, (anchor, rel) in enumerate(GRAPH.CONTROL_PAIRS, start=1):
        if (anchor, rel) not in CONTROL_PHRASINGS:
            raise SystemExit(f"REFUSING TO FREEZE: no phrasing registered for control ({anchor}, {rel})")
        text, chain = CONTROL_PHRASINGS[(anchor, rel)]
        target = _one_hop_target(anchor, rel)
        if target.lower() == anchor.lower():
            raise SystemExit(f"REFUSING TO FREEZE: control ({anchor}, {rel}) targets its own anchor")
        questions.append({
            "qid": f"c{i:02d}",
            "category": "control",
            "question": text,
            "anchor": anchor,
            "rel": rel,
            "expected_chain": chain,
            "expected_node": target,
            "expected_path": [anchor, target],
            "hops": 1,
            "note": "stored edge in the asked direction; must answer",
        })

    # ---- missing_relation -------------------------------------------------
    n = 0
    for anchor, rel in GRAPH.MUST_REFUSE["missing_relation"]:
        if (anchor, rel) not in MISSING_RELATION_PHRASINGS:
            raise SystemExit(f"REFUSING TO FREEZE: no phrasing for missing_relation ({anchor}, {rel})")
        text, chain = MISSING_RELATION_PHRASINGS[(anchor, rel)]
        n += 1
        questions.append({
            "qid": f"m{n:02d}",
            "category": "missing_relation",
            "question": text,
            "anchor": anchor,
            "rel": rel,
            "expected_chain": chain,
            "expected_node": None,
            "expected_path": None,
            "hops": 0,
            "note": "anchor in graph, relation absent entirely; must refuse",
        })

    # ---- inverse_direction ------------------------------------------------
    n = 0
    for anchor, rel in GRAPH.MUST_REFUSE["inverse_direction"]:
        if (anchor, rel) not in INVERSE_DIRECTION_PHRASINGS:
            raise SystemExit(f"REFUSING TO FREEZE: no phrasing for inverse_direction ({anchor}, {rel})")
        text, chain, far = INVERSE_DIRECTION_PHRASINGS[(anchor, rel)]
        stored_far, direction = _stored_far_end(anchor, rel)
        if stored_far != far:
            raise SystemExit(
                f"REFUSING TO FREEZE: ({anchor}, {rel}) far end is {stored_far!r}, "
                f"not the registered {far!r}"
            )
        if direction != "in":
            raise SystemExit(
                f"REFUSING TO FREEZE: ({anchor}, {rel}) is filed as inverse_direction "
                f"but its stored edge points OUT of the anchor ({anchor} {rel} "
                f"{stored_far}); that is a control, not a must-refuse"
            )
        n += 1
        questions.append({
            "qid": f"x{n:02d}",
            "category": "inverse_direction",
            "question": text,
            "anchor": anchor,
            "rel": rel,
            "expected_chain": chain,
            "expected_node": None,
            "expected_path": None,
            "forbidden_nodes": [far],
            "hops": 0,
            "note": "relation exists only in the opposite stored direction; "
                    "must refuse and must not name the far end",
        })

    # ---- honesty_out_of_graph ---------------------------------------------
    for qid, subject, text in OUT_OF_GRAPH:
        if subject.lower() in {lbl.lower() for lbl in GRAPH.CONCEPTS}:
            raise SystemExit(
                f"REFUSING TO FREEZE: {subject!r} IS a declared concept in the "
                f"graph, so {qid} is not an out-of-graph question"
            )
        questions.append({
            "qid": qid,
            "category": "honesty_out_of_graph",
            "question": text,
            "subject": subject,
            "anchor": None,
            "rel": None,
            "expected_chain": None,
            "expected_node": None,
            "expected_path": None,
            "hops": 0,
            "note": "subject absent from the graph; must refuse",
        })

    # ---- nonsense_fallback ------------------------------------------------
    for i, text in enumerate(NONSENSE_CUE_FREE, start=1):
        questions.append({
            "qid": f"n{i:02d}",
            "category": "nonsense_fallback",
            "question": text,
            "anchor": None,
            "rel": None,
            "expected_chain": None,
            "expected_node": None,
            "expected_path": None,
            "hops": 0,
            "expect_guess_disclosed": False,
            "note": "cue-free; heuristic fallback must fire",
        })
    questions.append({
        "qid": f"n{len(NONSENSE_CUE_FREE) + 1:02d}",
        "category": "nonsense_fallback",
        "question": DISCLOSED_GUESS,
        "anchor": "oak",
        "rel": "has_property",
        "expected_chain": ["has_property"],
        "expected_node": "tall",
        "expected_path": ["oak", "tall"],
        "hops": 1,
        "expect_guess_disclosed": True,
        "note": "cue-free but anchored on a real node that HAS a has_property "
                "edge: fallback must fire, the guess must be produced from a "
                "real edge, and it must be disclosed as a guess",
    })

    return questions


def main() -> int:
    problems = GRAPH.validate()
    if problems:
        print("REFUSING TO FREEZE THE QUESTION SET -- graph validation failed:")
        for kind, detail in problems:
            print(f"  {kind}: {detail}")
        return 1

    questions = build()
    cats: dict[str, int] = {}
    for q in questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1

    out = {
        "version": QUESTIONS_VERSION,
        "contract": "GLM-X v3.3.2 sections 6, 8 and 16",
        "graph": DB_PATH.name,
        "derived_from": "build_honesty_fallback_graph.py validated pair tables",
        "categories": cats,
        "questions": questions,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"  version {QUESTIONS_VERSION}, {len(questions)} questions: {cats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())