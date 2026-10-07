"""Stage E: prove the evaluation harness is not vacuous.

A grader that cannot fail is not a grader. Every check added for Stage E is
exercised here against a DELIBERATELY BROKEN input, and the mutation is only
considered caught when the specific guard reports the specific fault. Mutations
run against copies in memory or in a temp directory; the frozen artefacts are
never modified.

The distinction that matters: several of these guards were written because a
real bug slipped past an earlier version. The anchor check scored 113 correct
questions as failures because `selected_anchor` is a dict. `refused()` counted
the contract section 8 disclosure as a refusal because the disclosure text
contains the words "no relation". A Stage D helper counted mirror-derived
reachability as unreachable and rejected 7 valid frozen questions. Each of those
is now a mutation below, so the specific mistake cannot come back unnoticed.

Usage:
    python test_results/stage_e/mutation_test.py
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE_DIR))
for sub in ("stage_a", "stage_b", "stage_c", "stage_d"):
    sys.path.insert(0, str(ROOT / "test_results" / sub))

import stage_e_runner as R  # noqa: E402

SPEC = json.loads(R.QUESTIONS_PATH.read_text(encoding="utf-8"))
TRIPLES = R.load_store(R.DB_PATH)
NODE_LABELS = {s for s, _, _ in TRIPLES} | {t for _, _, t in TRIPLES}
PLANNER = R.load_planner()
CUES = R.load_cues()

PASSED: List[str] = []
FAILED: List[str] = []

# Report identity derived from the directory, so a later stage can reuse this
# module by rebinding STAGE_DIR without editing the output strings. Naming only.
STAGE_KEY = STAGE_DIR.name.rsplit("_", 1)[-1].upper()
MUTATION_REPORT = STAGE_DIR / f"{STAGE_DIR.name}_mutation_test.json"

# The two helper checks that walk the live graph name a concrete label, so their
# expected value is data-bound. Kept as overridable globals so a later stage that
# reuses this battery on its own graph restates the FIXTURE, not the assertion.
MIRROR_REACH_NEEDLE = "reverse=['telescope']"
NO_FABRICATION_NEEDLE = "unrelated=['giza']"
SELECTED_LABEL_NEEDLE = "str_form=wheel dict_form=wheel"


def run_preflight(spec: Dict[str, Any], db: Path | None = None) -> str:
    """Preflight outcome as text, or 'RAN CLEAN' if it raised SystemExit(0).

    preflight() exits the process on problems, so a mutation is 'caught' when it
    does NOT exit -- it raises SystemExit(1) after printing.
    """
    import io
    import contextlib
    old = R.DB_PATH
    if db is not None:
        R.DB_PATH = db
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            try:
                R.preflight(spec, TRIPLES, NODE_LABELS, CUES, PLANNER)
            except SystemExit as exc:
                return buf.getvalue()
        return "RAN CLEAN"
    finally:
        R.DB_PATH = old


def expect_caught(name: str, mutate: Callable[[], str],
                  must_contain: str) -> None:
    out = mutate()
    if must_contain.lower() in out.lower():
        PASSED.append(name)
        print(f"{name:44} OK")
    else:
        FAILED.append(f"{name}: expected {must_contain!r}, got: "
                      f"{out.strip().splitlines()[:3]}")
        print(f"{name:44} NOT CAUGHT")


def clean() -> str:
    """Control: the unmutated frozen set must pass preflight."""
    return run_preflight(copy.deepcopy(SPEC))


# ---------------------------------------------------------------------------
def mutate(spec: Dict[str, Any]) -> None:
    return spec


def m_gold_wrong() -> str:
    s = copy.deepcopy(SPEC)
    q = next(q for q in s["questions"] if q["category"] == "simple_one_hop")
    q["expected_node"] = "a node that is not the answer"
    return run_preflight(s)


def m_repeated_relation() -> str:
    s = copy.deepcopy(SPEC)
    q = next(q for q in s["questions"]
             if q["category"] == "short_multi_hop" and len(q["chain"]) == 2)
    q["chain"] = [q["chain"][0], q["chain"][0]]
    return run_preflight(s)


def m_chain_length() -> str:
    s = copy.deepcopy(SPEC)
    q = next(q for q in s["questions"] if q["category"] == "short_multi_hop")
    q["expected_path"] = q["expected_path"] + ["extra"]
    return run_preflight(s)


def m_chain_too_long() -> str:
    """A 4-relation chain that is otherwise entirely valid.

    Every other preflight guard is satisfied on purpose, so the ONLY thing that
    can reject this is `max_chain_length: 3` from configs/config_g2p.yaml. The
    path is a real walkable one (`egypt -> cairo -> giza -> city -> country`,
    4 pairwise-distinct relations) and the question cues exactly those four
    relations in that order, so the cue guard, the distinctness guard, the
    reachability guard and the store-membership guard all pass. Without the
    length check this question would be scored.
    """
    s = copy.deepcopy(SPEC)
    s["questions"].append({
        "qid": "mh999", "category": "short_multi_hop", "rel": "part_of",
        "anchor": "egypt",
        "question": ("What is the egypt part of, what is near egypt, what type "
                     "of thing is that, and what is that associated with?"),
        "chain": ["part_of", "spatial_near", "is_a", "associated_with"],
        "expected_path": ["egypt", "cairo", "giza", "city", "country"],
        "expected_node": "country",
        "hops": 4, "forbidden_nodes": [], "sibling_qid": None,
        "derivation": "MUTATION: valid except for hop count",
    })
    return run_preflight(s)


def m_mirror_answerable() -> str:
    """Make a must-refuse row actually answerable by adding the reverse edge."""
    s = copy.deepcopy(SPEC)
    q = next(q for q in s["questions"] if q["category"] == "mirror_silence")
    global TRIPLES
    saved = TRIPLES
    TRIPLES = set(TRIPLES) | {(q["anchor"], q["rel"], q["forbidden_nodes"][0])}
    try:
        return run_preflight(s)
    finally:
        TRIPLES = saved


def m_mirror_forbidden_no_edge() -> str:
    s = copy.deepcopy(SPEC)
    q = next(q for q in s["questions"] if q["category"] == "mirror_silence")
    q["forbidden_nodes"] = ["a node unrelated to the question"]
    return run_preflight(s)


def m_out_of_graph_in_store() -> str:
    s = copy.deepcopy(SPEC)
    q = next(q for q in s["questions"] if q["category"] == "honesty_out_of_graph")
    real = sorted(NODE_LABELS)[0]
    q["subject"] = real
    return run_preflight(s)


def m_expected_path_absent() -> str:
    s = copy.deepcopy(SPEC)
    q = next(q for q in s["questions"] if q["category"] == "simple_one_hop")
    q["expected_path"] = [q["anchor"], "a node that does not exist"]
    return run_preflight(s)


def m_pair_same_gold() -> str:
    """Both directions expecting the same node cannot test direction."""
    s = copy.deepcopy(SPEC)
    fwd = next(q for q in s["questions"] if q.get("sibling_qid"))
    rev = next(q for q in s["questions"] if q["qid"] == fwd["sibling_qid"])
    rev["expected_node"] = fwd["expected_node"]
    rev["expected_path"] = [rev["anchor"], fwd["expected_node"]]
    return run_preflight(s)


def m_category_minimum() -> str:
    s = copy.deepcopy(SPEC)
    s["questions"] = [q for q in s["questions"] if q["category"] != "simple_one_hop"]
    return run_preflight(s)


def m_declared_chain_mismatch() -> str:
    s = copy.deepcopy(SPEC)
    q = next(q for q in s["questions"]
             if q["category"] == "simple_one_hop" and q["rel"] == "is_a")
    q["chain"] = ["causes"]
    return run_preflight(s)


def m_graph_hash_drift() -> str:
    """Point the frozen set at a different graph than the one on disk."""
    tmp = Path(tempfile.mkdtemp()) / "tampered.db"
    shutil.copy(R.DB_PATH, tmp)
    s = copy.deepcopy(SPEC)
    s["graph"]["sha256"] = "0" * 64
    try:
        return run_preflight(s, db=tmp)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def m_question_file_edited() -> str:
    """Edit one gold answer in the frozen file after freezing.

    This is the mutation the freeze exists to prevent. The graph hash lives
    inside the question file, so an edited set still agrees with itself; only
    the independently pinned FROZEN_SET_SHA256 catches it. Writes to a temp copy
    and points the runner at it -- the real frozen file is never touched.
    """
    import os
    tmp = Path(tempfile.mkdtemp()) / "poc_questions_frozen.json"
    spec = json.loads(R.QUESTIONS_PATH.read_text(encoding="utf-8"))
    for q in spec["questions"]:
        if q["category"] == "simple_one_hop":
            q["expected_node"] = "definitely easier"
            break
    tmp.write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    old = R.QUESTIONS_PATH
    R.QUESTIONS_PATH = tmp
    try:
        return run_preflight(SPEC)
    finally:
        R.QUESTIONS_PATH = old
        shutil.rmtree(tmp.parent, ignore_errors=True)


# --- scoring mutations -----------------------------------------------------
def _row(**over: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "answer": "wheel is part of bicycle.",
        "relation_chain": ["part_of"],
        "path_labels": ["wheel", "bicycle"],
        "path_edges": ["part_of"],
        "selected_anchor": {"node_id": 1, "label": "wheel"},
        "heuristic_used": False,
        "heuristic_disclosed": False,
        "honest_no_relation": False,
        "template_matched": True,
        "_invariant": {"invariant_ok": True, "checks": {}, "failed_checks": []},
        "_semantic": {"ok": True, "failed": [], "invented": 0, "provenance": []},
    }
    base.update(over)
    return base


ONE_HOP_Q = {"qid": "x", "category": "simple_one_hop", "rel": "part_of",
             "anchor": "wheel", "expected_node": "bicycle",
             "expected_path": ["wheel", "bicycle"], "chain": ["part_of"],
             "hops": 1, "forbidden_nodes": []}


def s_scoring_control() -> str:
    """The unmutated fixture row must PASS, or every mutation below is vacuous.

    Each scoring mutation asserts that a corrupted row FAILS. If the clean
    fixture row already failed -- because its labels are not in the graph this
    battery is pointed at -- then all nine would "pass" while testing nothing.
    This is the assertion that makes them non-vacuous.
    """
    return str(R.score_row(ONE_HOP_Q, _row(), NODE_LABELS))


def s_wrong_anchor() -> str:
    return str(R.score_row(ONE_HOP_Q, _row(
        selected_anchor={"node_id": 2, "label": "sail"}), NODE_LABELS))


def s_fabricated_entity() -> str:
    """Answer asserts an entity that is not in the graph at all."""
    return str(R.score_row(ONE_HOP_Q, _row(
        answer="wheel is part of bicycle and also a unicycle."), NODE_LABELS))


def s_glue_words_are_not_fabrications() -> str:
    """Every word of an honest template must survive the vocabulary check."""
    return str(R.fabricated_entities(
        _row(answer="The wheel is part of the bicycle."), NODE_LABELS))


def s_disclosure_boilerplate_is_not_fabrication() -> str:
    """The section 8 prefix must be stripped before the claim is measured."""
    return str(R.fabricated_entities(
        _row(answer="Based on a heuristic guess (no relation cue matched): "
                    "pyramid is wooden.",
             heuristic_used=True, heuristic_disclosed=True), NODE_LABELS))


def s_refusal_prose_is_not_fabrication() -> str:
    """A refusal asserts nothing, so its prose must not be scored."""
    return str(R.fabricated_entities(
        _row(answer="I don't have a relation in my knowledge graph that "
                    "fully answers this question. Closest concepts I have: "
                    "bicycle.",
             honest_no_relation=True), NODE_LABELS))


def s_honest_multiword_label_is_one_entity() -> str:
    """A multi-word label must be consumed whole, not fragmented."""
    return str(R.fabricated_entities(
        _row(answer="grand piano is a type of piano."), NODE_LABELS))


def s_invariant_violation() -> str:
    return str(R.score_row(ONE_HOP_Q, _row(
        _invariant={"invariant_ok": False, "checks": {},
                    "failed_checks": ["chain_matches_walk"]}), NODE_LABELS))


def s_illegal_hop() -> str:
    return str(R.score_row(ONE_HOP_Q, _row(
        _semantic={"ok": True, "failed": [], "invented": 2,
                   "provenance": []}), NODE_LABELS))


def s_decoder_silent() -> str:
    return str(R.score_row(ONE_HOP_Q, _row(
        answer="I am not sure."), NODE_LABELS))


def s_mirror_leak() -> str:
    q = {"qid": "m", "category": "mirror_silence", "rel": "is_a",
         "anchor": "alarm", "expected_node": None, "expected_path": None,
         "chain": ["is_a"], "hops": 1, "forbidden_nodes": ["fire alarm"]}
    return str(R.score_row(q, _row(
        answer="fire alarm is a type of alarm.",
        relation_chain=["is_a"], path_labels=["fire alarm", "alarm"],
        path_edges=["is_a"],
        selected_anchor={"node_id": 3, "label": "fire alarm"}), NODE_LABELS))


def s_fallback_off() -> str:
    q = {"qid": "n", "category": "nonsense_fallback", "rel": None, "chain": [],
         "hops": 0, "anchor": None, "expected_node": None,
         "expected_path": None, "forbidden_nodes": []}
    return str(R.score_row(q, _row(
        answer="no relation", heuristic_used=False), NODE_LABELS))


def s_guess_not_disclosed() -> str:
    q = {"qid": "n", "category": "nonsense_fallback", "rel": None, "chain": [],
         "hops": 1, "anchor": "pyramid", "expected_node": None,
         "expected_path": None, "forbidden_nodes": [],
         "expect_guess_disclosed": True}
    return str(R.score_row(q, _row(
        answer="pyramid is wooden.", heuristic_used=True,
        heuristic_disclosed=False,
        selected_anchor={"node_id": 4, "label": "pyramid"}), NODE_LABELS))


def s_out_of_graph_answered() -> str:
    q = {"qid": "h", "category": "honesty_out_of_graph", "rel": None,
         "chain": None, "hops": 1, "anchor": "unicorn", "expected_node": None,
         "expected_path": None, "forbidden_nodes": []}
    return str(R.score_row(q, _row(
        answer="a unicorn is a horse.",
        selected_anchor={"node_id": 5, "label": "horse"}), NODE_LABELS))


def s_declined_disclosed_guess() -> str:
    """The section 8 disclosure text contains 'no relation'; it is not a refusal."""
    row = _row(answer="Based on a heuristic guess (no relation cue matched): "
                      "pyramid is wooden.",
               heuristic_used=True, heuristic_disclosed=True)
    return f"declined={R.declined(row)}"


def s_declined_real_refusal() -> str:
    row = _row(answer="I don't have a relation in my knowledge graph that "
                      "fully answers this question.",
               honest_no_relation=True)
    return f"declined={R.declined(row)}"


def s_selected_label_str_form() -> str:
    return f"str_form={R.selected_label(_row(selected_anchor='wheel'))} " \
           f"dict_form={R.selected_label(_row())}"


def s_walker_mirror_reach() -> str:
    """part_of is stored forward only; the reverse walk is still legal."""
    rev = sorted(R.walker_reach_from("observatory", "part_of", TRIPLES, 1))
    sto = sorted(R.store_reachable_from("observatory", "part_of", TRIPLES, 1))
    return f"reverse={rev} store_only={sto}"


def s_walker_no_fabricated_link() -> str:
    """A node's genuine neighbours, and nothing invented."""
    got = sorted(R.walker_reach_from("pyramid", "part_of", TRIPLES, 1))
    return f"unrelated={got}"


def s_pair_same_answer() -> str:
    fwd = {"id": "a", "expected_node": "fire alarm",
           "answer": "smoke causes fire alarm."}
    rev = {"id": "b", "expected_node": "smoke",
           "answer": "smoke causes fire alarm."}
    return str(R.check_direction_pair(fwd, rev))


def s_pair_genuinely_different() -> str:
    fwd = {"id": "a", "expected_node": "fire alarm",
           "answer": "smoke causes fire alarm."}
    rev = {"id": "b", "expected_node": "smoke",
           "answer": "The reason is that smoke causes fire alarm."}
    return str(R.check_direction_pair(fwd, rev))


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"{'mutation':44} {'result'}")
    print("-" * 78)

    ctl = clean()
    if ctl != "RAN CLEAN":
        print("CONTROL FAILED: the unmutated frozen set does not pass preflight:")
        print(ctl)
        return 1
    print(f"{'[control] unmutated set passes preflight':44} OK")

    # --- preflight guards
    expect_caught("gold answers the wrong node", m_gold_wrong,
                  "not the single expected")
    expect_caught("chain repeats a relation", m_repeated_relation,
                  "repeats a relation")
    expect_caught("chain/path length mismatch", m_chain_length, "length mismatch")
    expect_caught("chain longer than max_chain_length", m_chain_too_long,
                  "longer than max_chain_length")
    expect_caught("must-refuse read is actually answerable", m_mirror_answerable,
                  "is answerable")
    expect_caught("forbidden node has no stored edge", m_mirror_forbidden_no_edge,
                  "no stored")
    expect_caught("out-of-graph subject is in the store", m_out_of_graph_in_store,
                  "IS in the store")
    expect_caught("expected_path names an absent node", m_expected_path_absent,
                  "absent from the store")
    expect_caught("direction pair with identical golds", m_pair_same_gold,
                  "cannot test direction")
    expect_caught("category below contract minimum", m_category_minimum,
                  "contract minimum")
    expect_caught("declared chain != cued chain", m_declared_chain_mismatch,
                  "but question cues")
    expect_caught("graph hash drift vs frozen set", m_graph_hash_drift,
                  "graph hash drift")
    expect_caught("frozen question file edited after freezing",
                  m_question_file_edited, "question set hash drift")

    # --- scoring guards
    ctl_row = s_scoring_control()
    if "'pass': True" in ctl_row:
        PASSED.append("clean fixture row passes (mutations are not vacuous)")
        print(f"{'[control] clean fixture row scores a pass':44} OK")
    else:
        FAILED.append("clean fixture row does not pass: the scoring mutations "
                      f"below would be vacuous: {ctl_row}")
        print(f"{'[control] clean fixture row scores a pass':44} FAILED")

    scoring = [
        ("wrong anchor fails the row", s_wrong_anchor, "'pass': False"),
        ("fabricated entity fails the row", s_fabricated_entity,
         "fabricated_entities"),
        ("invariant violation fails the row", s_invariant_violation,
         "'pass': False"),
        ("illegal/fabricated hop fails the row", s_illegal_hop,
         "invented_hops"),
        ("answer that never states the node fails", s_decoder_silent,
         "'pass': False"),
        ("mirror silence naming the far end fails", s_mirror_leak,
         "named_forbidden"),
        ("cue-free question with no fallback fails", s_fallback_off,
                  "fallback_did_not_trigger"),
        ("undisclosed guess fails", s_guess_not_disclosed, "guess_not_disclosed"),
        ("out-of-graph subject answered fails", s_out_of_graph_answered,
         "answered_an_out_of_graph_subject"),
    ]
    for name, fn, needle in scoring:
        out = fn()
        if needle.lower() in out.lower():
            PASSED.append(name)
            print(f"{name:44} OK")
        else:
            FAILED.append(f"{name}: expected {needle!r}, got {out}")
            print(f"{name:44} NOT CAUGHT")

    # --- helper behaviour that earlier bugs actually broke
    helpers = [
        ("disclosed guess is not a refusal", s_declined_disclosed_guess,
         "declined=false"),
        ("genuine refusal is a refusal", s_declined_real_refusal,
         "declined=true"),
        ("selected_label handles dict and str", s_selected_label_str_form,
         SELECTED_LABEL_NEEDLE),
        ("walker reach includes the section 9 mirror", s_walker_mirror_reach,
         MIRROR_REACH_NEEDLE),
        ("store-only model misses that same step", s_walker_mirror_reach,
         "store_only=[]"),
        ("walker invents no unstored link", s_walker_no_fabricated_link,
         NO_FABRICATION_NEEDLE),
        ("direction pair with identical answers fails", s_pair_same_answer,
         "'ok': False"),
        ("direction pair answering differently passes", s_pair_genuinely_different,
         "'ok': True"),
    ]
    for name, fn, needle in helpers:
        out = fn()
        if needle.lower() in out.lower():
            PASSED.append(name)
            print(f"{name:44} OK")
        else:
            FAILED.append(f"{name}: expected {needle!r}, got {out}")
            print(f"{name:44} NOT CAUGHT")

    # --- false-positive guards: the detector must not fire on honest answers
    quiet = [
        ("template glue is not a fabrication", s_glue_words_are_not_fabrications,
         "[]"),
        ("disclosure boilerplate is not a fabrication",
         s_disclosure_boilerplate_is_not_fabrication, "[]"),
        ("refusal prose is not a fabrication", s_refusal_prose_is_not_fabrication,
         "[]"),
        ("multi-word label stays one entity",
         s_honest_multiword_label_is_one_entity, "[]"),
    ]
    for name, fn, needle in quiet:
        out = fn()
        if needle.lower() in out.lower():
            PASSED.append(name)
            print(f"{name:44} OK")
        else:
            FAILED.append(f"{name}: expected {needle!r}, got {out}")
            print(f"{name:44} FALSE POSITIVE")

    print("-" * 78)
    print(f"  mutations caught : {len(PASSED)}")
    print(f"  missed           : {len(FAILED)}")
    for f in FAILED:
        print(f"    MISSED {f}")
    total = len(PASSED) + len(FAILED)
    print(f"\n  ANTI_VACUITY = {not FAILED}  ({len(PASSED)}/{total})")

    report = MUTATION_REPORT
    report.write_text(json.dumps({
        "stage": STAGE_KEY,
        "anti_vacuity": not FAILED,
        "mutations_total": total,
        "mutations_caught": len(PASSED),
        "missed": FAILED,
        "caught": PASSED,
        "graph_sha256": R.sha256(R.DB_PATH),
        "question_set_sha256": R.sha256(R.QUESTIONS_PATH),
        "method": ("each guard is run against a deliberately broken input; a "
                   "mutation counts as caught only when the specific guard "
                   "reports the specific fault. Frozen artefacts are never "
                   "modified -- file-level mutations write to a temp copy."),
    }, indent=2), encoding="utf-8")
    print(f"  wrote {report}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())