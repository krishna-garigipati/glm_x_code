"""Stage F: anti-vacuity / mutation test on the HELD-OUT frozen set.

A grader that passes everything is indistinguishable from a grader that checks
nothing. So every guard in the shared scorer is fired at a deliberately broken
input and counts as caught only when it names the SPECIFIC fault.

This reuses `test_results/stage_e/mutation_test.py` wholesale. Every mutation in
it is data-independent -- each one corrupts the spec or a synthetic row and
asserts a guard fires -- so the same battery is the right battery for Stage F.
Reusing it rather than copying it is the point: a hand-copied battery drifts, and
a drifted battery would quietly stop testing what Stage E tested.

TWO CHECKS ARE REBOUND, and only two, because they name graph labels:
`s_walker_mirror_reach` and `s_walker_no_fabricated_link` assert against Stage E's
`observatory`/`pyramid`. They are restated here against Stage F's `orange`/`peel`
(peel is stored `peel part_of orange`, so the section 9 mirror must make the
reverse walk legal) and `queso` (which has no `part_of` edge at all, so nothing
may be invented for it). The assertion STRENGTH is unchanged; only the fixture
labels differ.

THE FIXTURES ARE ALSO REBOUND, and this is the part that is not optional. The
shared battery's `_row()` / `ONE_HOP_Q` / `s_*` fixtures name Stage E labels
(`wheel`, `bicycle`, `pyramid`, `grand piano`). Pointed at Stage F's graph every
one of those labels is simply absent, so all nine scoring mutations -- each of
which asserts that a CORRUPTED row fails -- would report OK while their clean
baseline was already failing. That is a battery that passes by testing nothing.
So every label-bearing fixture is restated on Stage F's own stored edge
`seed part_of pumpkin`, and the shared `s_scoring_control` assertion (added for
this reason) proves the clean row scores a real pass before any mutation counts.

The shared module binds `stage_e_runner` at import time, so the Stage F runner's
already-rebound module is registered under that name first. That is what makes
the shared battery read Stage F's frozen set, graph and pinned hash rather than
Stage E's.

Usage:
    python test_results/stage_f/mutation_test.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE_DIR))
for sub in ("stage_a", "stage_b", "stage_c", "stage_d"):
    sys.path.insert(0, str(ROOT / "test_results" / sub))

import stage_f_runner  # noqa: E402  (rebinds the shared runner onto Stage F data)

# The shared mutation module does `import stage_e_runner as R`. Registering the
# rebound module under that name BEFORE loading it is the whole trick.
sys.modules["stage_e_runner"] = stage_f_runner.R

_SRC = ROOT / "test_results" / "stage_e" / "mutation_test.py"
_spec = importlib.util.spec_from_file_location("_stage_e_muttest", _SRC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load the shared mutation battery from {_SRC}")
M = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = M
_spec.loader.exec_module(M)

R = M.R
M.STAGE_DIR = STAGE_DIR
M.STAGE_KEY = "F"
M.MUTATION_REPORT = STAGE_DIR / "stage_f_mutation_test.json"


# --- the two label-bound checks, restated on Stage F's graph -----------------
def s_walker_mirror_reach() -> str:
    """part_of is stored forward only; the reverse walk is still legal.

    Stage F stores `peel part_of orange` and never the reverse, so a walker model
    that omits the section 9 mirror would return an empty set here and the
    direction would look unanswerable.
    """
    rev = sorted(R.walker_reach_from("orange", "part_of", M.TRIPLES, 1))
    sto = sorted(R.store_reachable_from("orange", "part_of", M.TRIPLES, 1))
    return f"reverse={rev} store_only={sto}"


def s_walker_no_fabricated_link() -> str:
    """A node's genuine neighbours, and nothing invented."""
    got = sorted(R.walker_reach_from("queso", "part_of", M.TRIPLES, 1))
    return f"unrelated={got}"


M.s_walker_mirror_reach = s_walker_mirror_reach
M.s_walker_no_fabricated_link = s_walker_no_fabricated_link
M.MIRROR_REACH_NEEDLE = "reverse=['peel']"
M.NO_FABRICATION_NEEDLE = "unrelated=[]"
M.SELECTED_LABEL_NEEDLE = "str_form=wheel dict_form=seed"


# --- fixtures restated on Stage F's own graph -------------------------------
# `seed part_of pumpkin` is a real stored Stage F edge, so the clean row below is
# a genuine one-hop answer rather than a row that fails for incidental reasons.
NL = M.NODE_LABELS
assert {"seed", "pumpkin", "peel", "orange", "milk", "liquid", "citrus fruit",
        "fruit", "sirloin", "beef", "flour", "powdery"} <= set(NL), \
    "Stage F fixtures must be real Stage F nodes"


def _row(**over):
    base = {
        "answer": "seed is part of pumpkin.",
        "relation_chain": ["part_of"],
        "path_labels": ["seed", "pumpkin"],
        "path_edges": ["part_of"],
        "selected_anchor": {"node_id": 1, "label": "seed"},
        "heuristic_used": False,
        "heuristic_disclosed": False,
        "honest_no_relation": False,
        "template_matched": True,
        "_invariant": {"invariant_ok": True, "checks": {}, "failed_checks": []},
        "_semantic": {"ok": True, "failed": [], "invented": 0, "provenance": []},
    }
    base.update(over)
    return base


M._row = _row
M.ONE_HOP_Q = {"qid": "x", "category": "simple_one_hop", "rel": "part_of",
               "anchor": "seed", "expected_node": "pumpkin",
               "expected_path": ["seed", "pumpkin"], "chain": ["part_of"],
               "hops": 1, "forbidden_nodes": []}


def s_wrong_anchor() -> str:
    return str(R.score_row(M.ONE_HOP_Q, _row(
        selected_anchor={"node_id": 2, "label": "zucchini"}), NL))


def s_fabricated_entity() -> str:
    """Answer asserts an entity that is not in the graph at all.

    `squash` is deliberately absent from Stage F's 147 nodes.
    """
    return str(R.score_row(M.ONE_HOP_Q, _row(
        answer="seed is part of pumpkin and also a squash."), NL))


def s_glue_words_are_not_fabrications() -> str:
    """Every word of an honest template must survive the vocabulary check."""
    return str(R.fabricated_entities(
        _row(answer="The seed is part of the pumpkin."), NL))


def s_disclosure_boilerplate_is_not_fabrication() -> str:
    """The section 8 prefix must be stripped before the claim is measured."""
    return str(R.fabricated_entities(
        _row(answer="Based on a heuristic guess (no relation cue matched): "
                    "milk is liquid.",
             heuristic_used=True, heuristic_disclosed=True), NL))


def s_refusal_prose_is_not_fabrication() -> str:
    """A refusal asserts nothing, so its prose must not be scored."""
    return str(R.fabricated_entities(
        _row(answer="I don't have a relation in my knowledge graph that "
                    "fully answers this question. Closest concepts I have: "
                    "pumpkin.",
             honest_no_relation=True), NL))


def s_honest_multiword_label_is_one_entity() -> str:
    """A multi-word label must be consumed whole, not fragmented.

    `citrus fruit` is a stored Stage F node with a stored `is_a fruit` edge.
    """
    return str(R.fabricated_entities(
        _row(answer="citrus fruit is a type of fruit."), NL))


def s_mirror_leak() -> str:
    """Naming the far end of a mirrorable pair fails a must-silence read.

    Stage F stores `sirloin example_of beef`; the reverse is only reachable
    through the section 9 mirror, which a must-silence read must not use.
    """
    q = {"qid": "m", "category": "mirror_silence", "rel": "is_a",
         "anchor": "sirloin", "expected_node": None, "expected_path": None,
         "chain": ["is_a"], "hops": 1, "forbidden_nodes": ["beef"]}
    return str(R.score_row(q, _row(
        answer="beef is a type of sirloin.",
        relation_chain=["is_a"], path_labels=["beef", "sirloin"],
        path_edges=["is_a"],
        selected_anchor={"node_id": 3, "label": "beef"}), NL))


def s_guess_not_disclosed() -> str:
    q = {"qid": "n", "category": "nonsense_fallback", "rel": None, "chain": [],
         "hops": 1, "anchor": "milk", "expected_node": None,
         "expected_path": None, "forbidden_nodes": [],
         "expect_guess_disclosed": True}
    return str(R.score_row(q, _row(
        answer="milk is liquid.", heuristic_used=True,
        heuristic_disclosed=False,
        selected_anchor={"node_id": 4, "label": "milk"}), NL))


def s_out_of_graph_answered() -> str:
    q = {"qid": "h", "category": "honesty_out_of_graph", "rel": None,
         "chain": None, "hops": 1, "anchor": "persimmon", "expected_node": None,
         "expected_path": None, "forbidden_nodes": []}
    return str(R.score_row(q, _row(
        answer="persimmon is a fruit.",
        selected_anchor={"node_id": 5, "label": "fruit"}), NL))


def s_declined_disclosed_guess() -> str:
    """The section 8 disclosure text contains 'no relation'; it is not a refusal."""
    row = _row(answer="Based on a heuristic guess (no relation cue matched): "
                      "milk is liquid.",
               heuristic_used=True, heuristic_disclosed=True)
    return f"declined={R.declined(row)}"


def s_declined_real_refusal() -> str:
    row = _row(answer="I don't have a relation in my knowledge graph that "
                      "fully answers this question.",
               honest_no_relation=True)
    return f"declined={R.declined(row)}"


for _fn in (s_wrong_anchor, s_fabricated_entity, s_glue_words_are_not_fabrications,
            s_disclosure_boilerplate_is_not_fabrication,
            s_refusal_prose_is_not_fabrication,
            s_honest_multiword_label_is_one_entity, s_mirror_leak,
            s_guess_not_disclosed, s_out_of_graph_answered,
            s_declined_disclosed_guess, s_declined_real_refusal):
    setattr(M, _fn.__name__, _fn)

# The two rebound checks are asserted with the same needles Stage E used.
_orig_main = M.main


def main() -> int:
    print("Stage F anti-vacuity battery (shared with Stage E; 2 label-bound "
          "fixtures restated on the Stage F graph)\n")
    return _orig_main()


if __name__ == "__main__":
    raise SystemExit(main())
