"""Stage E: score the Frozen PoC Evaluation Set under the strict contract rules.

GLM-X v3.3.2 section 16 makes this the FINAL PoC score, so the runner is built to
be hard to pass rather than easy to satisfy. Specifically:

  * The central invariant is checked on EVERY row, refusals and fallback guesses
    included, via the same `check_invariant` Stages A and D used. Grading logic
    is imported from Stages A-D rather than reimplemented, so a Stage E number is
    comparable to a Stage C number and cannot drift.

  * Illegal same-label mirroring and fabricated relations are checked per HOP on
    every row via `classify_hops`, not merely by inspecting the final sentence.

  * Preflight runs against the WHOLE frozen set before any scoring, so `--only`
    can never be used to make a broken set look runnable. It also re-verifies the
    on-disk graph hash against the hash the question set was frozen with, which
    is the check that makes "frozen" mean something.

  * Every failure is ATTRIBUTED (contract section 16 `evaluation_requirements`:
    "separate failures caused by missing graph data from failures caused by
    code"). A wrong answer and a missing edge are different defects and must not
    be reported as one number.

  * Direction pairs are only counted as evidence if the two ends return DIFFERENT
    answers. Two questions that both return the same node would pass one by one
    while proving nothing about direction.

Usage:
    python test_results/stage_e/stage_e_runner.py
    python test_results/stage_e/stage_e_runner.py --only simple_one_hop
    python test_results/stage_e/stage_e_runner.py --out .../run1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test_results" / "stage_a"))
sys.path.insert(0, str(ROOT / "test_results" / "stage_b"))
sys.path.insert(0, str(ROOT / "test_results" / "stage_c"))
sys.path.insert(0, str(ROOT / "test_results" / "stage_d"))

import stage_a_runner  # noqa: E402
import stage_b_runner  # noqa: E402
import stage_c_runner  # noqa: E402
import stage_d_runner  # noqa: E402

DB_PATH = STAGE_DIR / "poc_eval.db"
QUESTIONS_PATH = STAGE_DIR / "poc_questions_frozen.json"
CONFIG_G2P = ROOT / "configs" / "config_g2p.yaml"

# Display name and default results filename, DERIVED from the stage directory so
# this module can be reused by a later stage without editing its output strings.
# Stage F loads this module and rebinds STAGE_DIR / DB_PATH / QUESTIONS_PATH /
# FROZEN_SET_SHA256; deriving the label here means the reuse needs no other
# change. Naming only -- no grading logic reads either global.
STAGE_KEY = STAGE_DIR.name.rsplit("_", 1)[-1].upper()
STAGE = f"STAGE {STAGE_KEY}"
RESULTS_NAME = f"{STAGE_DIR.name}_results.json"

# Digest of the frozen question file, recorded once at freeze time and pinned
# here. See preflight() for why the in-file graph hash is not sufficient on its
# own. Do not update this to make a run pass.
FROZEN_SET_SHA256 = (
    "8454dc235a2676ba06ef36266744c0f47f9505954d8a72e7eb37a12ab6fe66da")

# Shared grading helpers. Imported, not copied: a Stage E verdict has to mean the
# same thing as a Stage C verdict, and two implementations of "did the invariant
# hold" would eventually disagree.
REFUSAL_MARKERS = stage_a_runner.REFUSAL_MARKERS
DISCLOSURE_MARKER = stage_a_runner.DISCLOSURE_MARKER
check_invariant = stage_a_runner.check_invariant

load_chain_templates = stage_b_runner.load_chain_templates
load_relation_phrases = stage_b_runner.load_relation_phrases

MIRROR_INVERSE = stage_c_runner.MIRROR_INVERSE
PROV_STORED = stage_c_runner.PROV_STORED
PROV_DECLARED_INVERSE = stage_c_runner.PROV_DECLARED_INVERSE
PROV_ILLEGAL = stage_c_runner.PROV_ILLEGAL
PROV_FABRICATED = stage_c_runner.PROV_FABRICATED
classify_hops = stage_c_runner.classify_hops

load_store = stage_d_runner.load_store
accepted_labels = stage_d_runner.accepted_labels
store_reachable_from = stage_d_runner.reachable_from
load_cues = stage_d_runner.load_cues
load_planner = stage_d_runner.load_planner
names_node = stage_d_runner.names_node
claim_text = stage_d_runner.claim_text
refused = stage_d_runner.refused
check_semantics = stage_d_runner.check_semantics


def declined(row: Dict[str, Any]) -> bool:
    """Did the engine DECLINE to answer, as opposed to answering as a guess?

    stage_d_runner's `refused()` decides by matching refusal PHRASES as
    substrings, and the contract section 8 disclosure prefix is:

        "Based on a heuristic guess (no relation cue matched): pyramid is wooden."

    which contains "no relation" and therefore matches a refusal marker. Left
    uncorrected that misclassifies the single case the contract explicitly wants
    to be an ANSWER -- a real answer over a real edge, honestly labelled a guess
    -- as a refusal. So when the guess is disclosed, the engine did produce an
    answer and is not refusing, whatever the wording.

    This distinction is load-bearing for Stage E. It is the difference between
    "the fallback fired and disclosed a grounded guess", which contract section 8
    requires, and "the fallback fired and refused", which satisfies section 8 more
    weakly, and the two are reported separately.
    """
    if row.get("heuristic_disclosed"):
        return False
    return bool(refused(row))


def selected_label(row: Dict[str, Any]) -> str:
    """The label the engine actually anchored on, lowercased.

    `selected_anchor` is a dict (`{node_id, label, anchor_margin,
    entity_top_sim, exact_label_hit}`), not a string. Comparing it to a label
    with `str(...)` compares against a repr and never matches, which silently
    scored all 113 anchored one-hop questions as `wrong_anchor`.
    """
    sel = row.get("selected_anchor")
    if isinstance(sel, dict):
        return str(sel.get("label") or "").lower()
    return str(sel or "").lower()

# Contract section 16 `recommended_poc_test_matrix`, plus the section 15
# mechanism targets the contract scores separately.
CATEGORY_TARGETS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict([
    ("simple_one_hop", {"target": 0.95, "minimum": 30,
                        "why": "section 16 simple_one_hop: >= 95%, min 30 questions"}),
    ("direction_pair", {"target": 0.90, "minimum": 10,
                        "why": "section 16 direction_pairs: >= 90%, min 10 questions"}),
    ("short_multi_hop", {"target": 0.75, "minimum": 8,
                         "why": "section 16 short_multi_hop: >= 70-80%, min 8 questions"}),
    ("honesty_out_of_graph", {"target": 0.90, "minimum": 8,
                              "why": "section 16 honesty_out_of_graph: >= 90%, min 8"}),
    ("nonsense_fallback", {"target": 0.90, "minimum": 5,
                           "why": "section 16 nonsense_fallback: fallback must trigger"}),
    ("mirror_silence", {"target": 0.90, "minimum": 1,
                        "why": "no illegal same-label mirroring (section 9)"}),
])

MECHANISM_TARGETS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict([
    ("no_invented_relations", {"target": 0.98, "minimum": 1,
                               "why": "section 15 decode.no_invented_relations >= 98%"}),
    ("template_correctness", {"target": 0.95, "minimum": 1,
                              "why": "section 15 decode.template_correctness >= 95%"}),
    ("central_invariant", {"target": 0.90, "minimum": 1,
                           "why": "section 15 central_invariant.hold_rate >= 90%"}),
    ("overall", {"target": 0.90, "minimum": 1,
                 "why": "section 16 overall: >= 90%"}),
])

ALL_CATEGORIES = tuple(CATEGORY_TARGETS)


# ---------------------------------------------------------------------------
# Walker-model reachability
#
# Stage D's `reachable_from` models the STORE: it only follows edges that are
# physically written down. That was sufficient there, because every part_of
# question Stage D asked was anchored on the PART ("What is the wheel part
# of?"), which is a stored forward edge.
#
# It is not sufficient here. The engine answers part_of/has_part, causes/
# caused_by and precedes/follows in EITHER direction, because contract section 9
# lets the mirror pass synthesise the declared inverse. So
# `telescope part_of observatory`, stored forward only, also means the walk
# `observatory --part_of--> telescope` is legal. A chain step of that shape is a
# genuinely walkable step, and preflight that only knew about stored edges would
# reject it -- which is exactly what happened the first time this ran, rejecting
# 7 of the frozen multi-hop questions.
#
# These helpers model what the WALKER can do. They read the triples straight out
# of SQLite exactly as Stage D does, so the drift check the hash provides is not
# weakened by trusting the builder's in-memory lists instead.
# ---------------------------------------------------------------------------
def walkable_edges(triples: Set[tuple]) -> Dict[str, Set[tuple]]:
    """Adjacency the walker can traverse, as {source: {(relation, target)}}.

    A stored edge (s, r, t) yields two traversable steps when `r` is mirrorable:
    (s, r, t) forward, and (t, r, s) through the section 9 mirror. Both are
    labelled with the FORWARD relation, because the mirrored label `has_part`
    has no descriptor phrase in configs/config_g2p.yaml and so cannot be asked
    for -- labelling the reverse step `has_part` would build questions the
    planner can never produce.
    """
    adj: Dict[str, Set[tuple]] = {}
    for s, r, t in triples:
        adj.setdefault(s, set()).add((r, t))
        if r in MIRROR_INVERSE:
            adj.setdefault(t, set()).add((r, s))
    return adj


def walker_reach_from(anchor: str, rel: str, triples: Set[tuple],
                      hops: int | None = 1) -> Set[str]:
    """What the walker can reach from `anchor` when asked for `rel`.

    `hops=1` is the uniqueness question for a graded hop. `hops=None` is the
    right question for a must-refuse row: the anchor must be unable to reach
    anything under that relation by ANY route, not merely in one step.
    """
    adj = walkable_edges(triples)
    best: Dict[str, int] = {anchor: 0}
    frontier = [anchor]
    while frontier:
        cur = frontier.pop()
        if hops is not None and best[cur] >= hops:
            continue
        for r, t in adj.get(cur, ()):
            if r == rel and (t not in best or best[t] > best[cur] + 1):
                best[t] = best[cur] + 1
                frontier.append(t)
    return set(best) - {anchor}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(spec: Dict[str, Any], triples: Set[tuple],
              node_labels: Set[str], cues: List[tuple], planner) -> None:
    """Refuse to score a set that no longer matches the graph it was frozen with.

    Runs against the whole frozen spec regardless of any --only/--ids filter, so
    a narrowed run cannot hide a broken set.
    """
    problems: List[str] = []

    # --- the question file on disk is the set that was frozen.
    #
    # The graph hash alone is not enough to make the freeze mean anything. That
    # hash lives INSIDE the question file, so a set edited after the fact would
    # still verify -- it would just be verifying itself. Pinning the file's own
    # digest here, in a separate file that the freeze did not produce, closes the
    # loop: editing a question, a gold answer or a single category now hard-fails
    # the run instead of quietly improving the score.
    #
    # This value was read once, at freeze time, and has deliberately not been
    # touched since. Changing it to make a run pass would defeat its purpose.
    on_disk_set = sha256(QUESTIONS_PATH)
    if on_disk_set != FROZEN_SET_SHA256:
        problems.append(
            f"question set hash drift: file on disk {on_disk_set}, "
            f"frozen {FROZEN_SET_SHA256}")

    # --- the graph on disk is the graph the questions were frozen against
    on_disk = sha256(DB_PATH)
    frozen = spec["graph"]["sha256"]
    if on_disk != frozen:
        problems.append(
            f"graph hash drift: questions frozen against {frozen[:12]}, "
            f"db on disk is {on_disk[:12]}. The 'frozen' set and the graph being "
            f"scored are not the same artefacts.")

    # --- every node the questions name is really in the store
    for q in spec["questions"]:
        for label in (q.get("expected_path") or []):
            if label.lower() not in node_labels:
                problems.append(
                    f"{q['qid']}: expected_path names {label!r}, absent from the store")
        if q.get("subject") and q["subject"].lower() in node_labels:
            problems.append(
                f"{q['qid']}: out-of-graph subject {q['subject']!r} IS in the store")

    # --- each declared chain equals what the literal cue matcher produces
    for q in spec["questions"]:
        if not q.get("chain"):
            continue
        got = planner._literal_cue_relations(q["question"])
        if got != q["chain"]:
            problems.append(
                f"{q['qid']}: declared chain {q['chain']} but question cues {got}")

    # --- category minima
    counts: Dict[str, int] = {}
    for q in spec["questions"]:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    for cat, rule in CATEGORY_TARGETS.items():
        if counts.get(cat, 0) < rule["minimum"]:
            problems.append(
                f"{cat}: {counts.get(cat, 0)} questions < contract minimum {rule['minimum']}")

    # --- every graded answer really is the only legal one, per the store.
    # Modelled as the WALKER would see it (store + section 9 mirror), because
    # that is the model the question set was derived under; validating the gold
    # against a different model than it was built with would make the hash check
    # prove nothing.
    for q in spec["questions"]:
        if q["category"] in ("simple_one_hop", "direction_pair"):
            reach = walker_reach_from(q["anchor"], q["rel"], triples, 1)
            if reach != {q["expected_node"]}:
                problems.append(
                    f"{q['qid']}: anchor {q['anchor']!r} under {q['rel']} reaches "
                    f"{sorted(reach)}, not the single expected {q['expected_node']!r}")
        elif q["category"] == "short_multi_hop":
            path, chain = q["expected_path"], q["chain"]
            if len(path) != len(chain) + 1:
                problems.append(f"{q['qid']}: path/chain length mismatch")
                continue
            if len(chain) > 3:
                problems.append(f"{q['qid']}: chain longer than max_chain_length=3")
            if len(set(chain)) != len(chain):
                problems.append(
                    f"{q['qid']}: chain {chain} repeats a relation, so the literal "
                    f"cue matcher cannot express it (one descriptor per relation)")
            for i, rel in enumerate(chain):
                step = walker_reach_from(path[i], rel, triples, 1)
                if step != {path[i + 1]}:
                    problems.append(
                        f"{q['qid']}: hop {i + 1} {path[i]!r} --{rel}--> reaches "
                        f"{sorted(step)}, not {path[i + 1]!r}")
        elif q["category"] == "mirror_silence":
            # A must-refuse row is only meaningful if the read really is
            # impossible. Checked to a fixpoint, not one hop: if the walker could
            # reach anything at all under that relation the row would be
            # answerable and passing it would prove nothing.
            reach = walker_reach_from(q["anchor"], q["rel"], triples, None)
            if reach:
                problems.append(
                    f"{q['qid']}: must-refuse read {q['anchor']!r} --{q['rel']}--> "
                    f"is answerable, reaches {sorted(reach)}")
            for far in q["forbidden_nodes"]:
                if (far, q["rel"], q["anchor"]) not in triples:
                    problems.append(
                        f"{q['qid']}: forbidden node {far!r} has no stored "
                        f"{q['rel']} edge into {q['anchor']!r}")

    # --- direction pairs must be provable in BOTH directions
    by_qid = {q["qid"]: q for q in spec["questions"]}
    for q in spec["questions"]:
        sib = q.get("sibling_qid")
        if not sib:
            continue
        other = by_qid.get(sib)
        if other is None:
            problems.append(f"{q['qid']}: sibling {sib!r} not in the frozen set")
            continue
        if other["expected_node"] == q["expected_node"]:
            problems.append(
                f"{q['qid']}: both directions of the pair expect "
                f"{q['expected_node']!r}, so the pair cannot test direction")

    if problems:
        print("PREFLIGHT FAILED -- refusing to score:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print(f"  preflight OK: graph {on_disk[:12]}, "
          f"{len(spec['questions'])} questions, all declared chains match the cue bank")


# ---------------------------------------------------------------------------
def attribute(q: Dict[str, Any], row: Dict[str, Any],
              triples: Set[tuple]) -> str:
    """Why did this row fail? Contract section 16 asks for graph gaps to be
    separated from code defects, so this is a first-class output, not a log line.

    Order matters: the earliest stage in the pipeline that went wrong is the
    cause. Anchoring before planning, planning before walking, walking before
    decoding.
    """
    cat = q["category"]
    anchor = q.get("anchor")
    selected = selected_label(row)

    if anchor and selected and str(selected).lower() != str(anchor).lower():
        return "wrong_anchor"

    if cat in ("simple_one_hop", "direction_pair", "short_multi_hop"):
        if declined(row):
            # Is the answer even reachable in the graph?
            want = q.get("expected_node")
            if want and walker_reach_from(anchor, q["rel"], triples, None):
                return "walk_found_nothing"
            return "graph_data_gap"
        if list(row.get("relation_chain") or []) != list(q.get("chain") or []):
            return "planner_wrong_chain"
        if q["category"] == "short_multi_hop":
            walked = [str(x).lower() for x in (row.get("path_labels") or [])]
            if walked != [str(x).lower() for x in q["expected_path"]]:
                return "walker_wrong_path"
        else:
            walked = [str(x).lower() for x in (row.get("path_labels") or [])]
            if walked[-1:] != [str(q["expected_node"]).lower()]:
                return "walker_wrong_path"
        if str(q["expected_node"]).lower() not in (row.get("answer") or "").lower():
            return "decoder_did_not_state_it"
        return "unknown"

    if cat == "mirror_silence":
        return "answered_a_forbidden_read"

    if cat == "honesty_out_of_graph":
        return "answered_an_out_of_graph_subject"

    if cat == "nonsense_fallback":
        if not row.get("heuristic_used"):
            return "fallback_did_not_trigger"
        if q.get("expect_guess_disclosed") and not row.get("heuristic_disclosed"):
            return "guess_not_disclosed"
        return "unknown"

    return "unknown"


def score_row(q: Dict[str, Any], row: Dict[str, Any],
              node_labels: Set[str]) -> Dict[str, Any]:
    """Pass/fail one row. The central invariant is required on every row."""
    cat = q["category"]
    answer = row.get("answer") or ""
    low = answer.lower()
    inv = row["_invariant"]
    sem = row["_semantic"]
    why: List[str] = []

    if not inv["invariant_ok"]:
        why.append(f"invariant:{inv['failed_checks']}")
    if not sem["ok"]:
        why.append(f"semantics:{sem['failed']}")

    # No row, anywhere, may name a graph node the walk never reached, or a node
    # that does not exist at all.
    invented = sem["invented"]
    if invented:
        why.append(f"invented_hops:{invented}")
    unknown = fabricated_entities(row, node_labels)
    if unknown:
        why.append(f"fabricated_entities:{unknown}")

    ok: bool
    if cat in ("simple_one_hop", "direction_pair"):
        want = str(q["expected_node"]).lower()
        ok = (not declined(row)
              and selected_label(row) == str(q["anchor"]).lower()
              and want in low
              and [str(x).lower() for x in (row.get("path_labels") or [])][-1:] == [want])
        if not ok and declined(row):
            why.append("refused_an_answerable_question")
        if selected_label(row) != str(q["anchor"]).lower():
            why.append(f"anchored_on:{selected_label(row)!r}"
                       f" not {q['anchor']!r}")
    elif cat == "short_multi_hop":
        want = [str(x).lower() for x in q["expected_path"]]
        walked = [str(x).lower() for x in (row.get("path_labels") or [])]
        ok = (not declined(row)
              and selected_label(row) == str(q["anchor"]).lower()
              and walked == want
              and want[-1] in low
              and len(row.get("relation_chain") or []) == len(q["chain"]))
        if not ok and declined(row):
            why.append("refused_an_answerable_question")
        if selected_label(row) != str(q["anchor"]).lower():
            why.append(f"anchored_on:{selected_label(row)!r}"
                       f" not {q['anchor']!r}")
    elif cat == "mirror_silence":
        # The anti-mirroring property is precisely "the far end is not named".
        # A disclosed guess that avoids the far end still satisfies it: the
        # fallback guessing from a different node is not the same defect as the
        # mirror pass fabricating the forbidden direction.
        leaked = [f for f in q["forbidden_nodes"] if f.lower() in low]
        ok = not leaked and (declined(row) or bool(row.get("heuristic_disclosed")))
        if leaked:
            why.append(f"named_forbidden:{leaked}")
        if not ok and not leaked:
            why.append("answered_a_relation_that_does_not_exist")
    elif cat == "honesty_out_of_graph":
        # A disclosed guess is NOT acceptable here: the subject is absent from the
        # graph, so any answer is ungrounded no matter how it is labelled.
        ok = bool(row.get("honest_no_relation")) or declined(row)
        if not ok:
            why.append("answered_an_out_of_graph_subject")
    elif cat == "nonsense_fallback":
        if q.get("expect_guess_disclosed"):
            ok = (bool(row.get("heuristic_used"))
                  and bool(row.get("heuristic_disclosed"))
                  and bool(row.get("path_edges"))
                  and not declined(row))
            if not row.get("heuristic_used"):
                why.append("fallback_did_not_trigger")
            elif not row.get("heuristic_disclosed"):
                why.append("guess_not_disclosed")
        else:
            ok = bool(row.get("heuristic_used")) and (
                bool(row.get("heuristic_disclosed")) or declined(row))
            if not row.get("heuristic_used"):
                why.append("fallback_did_not_trigger")
    else:
        ok = False
        why.append(f"unscored_category:{cat}")

    return {"pass": bool(ok and inv["invariant_ok"] and sem["ok"] and not invented
                         and not unknown),
            "why": why}


# ---------------------------------------------------------------------------
# Fabricated-entity detection
#
# This replaces a check that was silently vacuous. It used to read:
#
#     named   = names_node(claim, node_labels)          # searches FOR labels
#     unknown = [n for n in named if n not in node_labels]
#
# and `unknown` could never be non-empty, because names_node only returns labels
# that are in node_labels by construction. Filtering a set of known labels
# against the set of known labels removes nothing. An answer that invented an
# entity outright -- "wheel is part of bicycle and also a unicycle" -- scored a
# clean pass. A grader that cannot catch the most blatant form of the thing the
# whole contract is about is worse than no grader, because it reports a number.
#
# The check has to be the complement: assert that every content word in the
# decoder's CLAIM is accounted for, either as a node in the graph or as part of
# the fixed English the decoder uses to glue nodes together. Anything left over
# is an entity the graph never contained.
#
# Scope: the assertion part of the answer only.
#   * A refusal's prose is not a factual claim. "I don't have a relation in my
#     knowledge graph that fully answers this question." is full of words no
#     graph would ever contain. Stage A already checks that the concept list a
#     refusal offers is real; that is the right place for it, so refusals are
#     skipped here rather than false-flagged.
#   * The section 8 disclosure prefix is boilerplate, so a disclosed guess is
#     measured from the first claim word after the colon.
# ---------------------------------------------------------------------------
DISCLOSURE_PREFIX_RE = re.compile(r"^.*?heuristic guess[^:]*:\s*", re.I)

STOPWORDS = frozenset("""
    a an the this that these those it its there here
    is are was were be been being am
    of in on at to for from by with within into onto as
    and or but nor so yet then than because since
    i we you he she they them my our your his her their
    do does did done has have had having
    also any some each every which who whom whose
    not no yes very much many more most less least
    fully answer answers question questions relation
""".split())

_VOCAB: Set[str] | None = None


def entity_vocabulary(node_labels: Set[str]) -> Set[str]:
    """Words an honest answer is allowed to contain.

    Node labels contribute their component words; the decoder's own chain
    templates and relation phrases contribute the glue ("is a type of", "is
    caused by", "the reason is that"). Cached because it is rebuilt per row
    otherwise and the label set never changes within a run.
    """
    global _VOCAB
    if _VOCAB is not None:
        return _VOCAB
    words = set(STOPWORDS)
    for label in node_labels:
        words.update(re.findall(r"[a-z]+", label.lower()))
    for table in (load_chain_templates(), load_relation_phrases()):
        for phrase in table.values():
            words.update(re.findall(r"[a-z]+", str(phrase).lower()))
    _VOCAB = words
    return words


def fabricated_entities(row: Dict[str, Any],
                        node_labels: Set[str]) -> List[str]:
    """Content words the decoder asserted that the graph cannot account for."""
    text = row.get("answer") or ""
    if row.get("heuristic_disclosed"):
        text = DISCLOSURE_PREFIX_RE.sub("", text, count=1)
    elif declined(row):
        return []   # a refusal asserts nothing; Stage A grades its concept list

    # Blank out real labels longest-first, so a multi-word label is consumed as
    # one unit instead of fragmenting into apparently-unaccounted words.
    low = text.lower()
    for label in sorted(node_labels, key=len, reverse=True):
        if not label:
            continue
        low = re.sub(r"(?<!\w)" + re.escape(label.lower()) + r"(?!\w)", " ", low)

    vocab = entity_vocabulary(node_labels)
    # Two letters are too weak to distinguish a stray word from a typo in the
    # glue; three is where invented entities actually start showing up.
    return sorted({w for w in re.findall(r"[a-z]{3,}", low) if w not in vocab})


def check_direction_pair(fwd: Dict[str, Any], rev: Dict[str, Any]) -> Dict[str, Any]:
    """A direction pair only counts as evidence if the two ends DIFFER.

    Both ends naming their own target is necessary but not sufficient: two
    questions that both return the same node would each pass on their own while
    proving nothing at all about direction. So identical answers fail the pair
    even when both are individually correct -- the failure mode this catches is
    a system that has quietly collapsed the two directions onto one.
    """
    fwd_names = str(fwd["expected_node"]).lower() in (fwd["answer"] or "").lower()
    rev_names = str(rev["expected_node"]).lower() in (rev["answer"] or "").lower()
    same = (fwd["answer"] or "") == (rev["answer"] or "")
    return {
        "pair": sorted([fwd["id"], rev["id"]]),
        "forward": fwd["id"], "reverse": rev["id"],
        "forward_named_own_target": fwd_names,
        "reverse_named_own_target": rev_names,
        "answers_identical": same,
        "ok": bool(fwd_names and rev_names and not same),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    spec = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions: List[Dict[str, Any]] = spec["questions"]
    if args.only:
        questions = [q for q in questions if q["category"] == args.only]
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        questions = [q for q in questions if q["qid"] in wanted]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("REFUSING TO RUN: the filter selected 0 questions")
        return 1
    out_path = Path(args.out) if args.out else STAGE_DIR / RESULTS_NAME

    random.seed(args.seed)
    np.random.seed(args.seed)

    # resolve_template/template_glue read CHAIN_TEMPLATES/RELATION_PHRASES from
    # their own module's globals, and check_semantics reads stage D's. Rebinding
    # only the imported names here would leave those lookups empty, silently
    # turning "{relation0}" into an unsatisfiable literal.
    _b = stage_b_runner
    _b.CHAIN_TEMPLATES = load_chain_templates()
    _b.RELATION_PHRASES = load_relation_phrases()
    for mod in (stage_d_runner,):
        mod.CHAIN_TEMPLATES = _b.CHAIN_TEMPLATES
        mod.RELATION_PHRASES = _b.RELATION_PHRASES

    triples = load_store(DB_PATH)
    node_labels = {s for s, _, _ in triples} | {t for _, _, t in triples}
    planner = load_planner()
    cues = load_cues()

    print(f"\n{'=' * 78}\n{STAGE} -- {spec.get('label', 'Frozen PoC Evaluation Set')} "
          f"({len(questions)} questions)\n{'=' * 78}")
    print(f"  set version {spec['version']}  graph sha256 {spec['graph']['sha256'][:12]}")
    preflight(spec, triples, node_labels, cues, planner)

    from scripts.glmx_ask import GLMXPipeline
    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

    pipeline = GLMXPipeline()
    pipeline._seed = args.seed
    pipeline._no_learning = True
    pipeline._measure = True
    pipeline.graph_store = SQLiteGraphStore.load_state(str(DB_PATH))
    pipeline.load_models()

    rows: List[Dict[str, Any]] = []
    for q in questions:
        res = pipeline.ask(q["question"])
        row: Dict[str, Any] = {
            "id": q["qid"],
            "cat": q["category"],
            "question": q["question"],
            "rel": q.get("rel"),
            "expected_node": q.get("expected_node"),
            "expected_chain": q.get("chain"),
            "expected_path": q.get("expected_path"),
            "forbidden_nodes": q.get("forbidden_nodes"),
            "sibling_qid": q.get("sibling_qid"),
            "expect_guess_disclosed": q.get("expect_guess_disclosed"),
            # How the question was phrased, for one-hop rows only. "template" is
            # the shared Stage E bank; "paraphrase" and "plural" are cue-preserving
            # rewordings of the SAME gold edge, so they are scored identically but
            # reported separately. Absent on every other category.
            "variant_kind": q.get("variant_kind"),
            # contract section 16: record chain, walked path, path_edges, answer
            "answer": res.get("answer"),
            "relation_chain": res.get("relation_chain"),
            "path_labels": res.get("walk_path_labels"),
            "path_edges": res.get("walk_path_edges"),
            "selected_anchor": res.get("selected_anchor"),
            "heuristic_used": res.get("heuristic_used"),
            "heuristic_disclosed": res.get("heuristic_disclosed"),
            "honest_no_relation": res.get("honest_no_relation"),
            "honest_by_entity": res.get("honest_by_entity"),
            "honest_by_relation": res.get("honest_by_relation"),
            # W4c entity-identity gate: whether the walk was anchored on an
            # entity the question actually named. Recorded per row so the
            # refusal decisions are auditable rather than opaque.
            "honest_by_identity": res.get("honest_by_identity"),
            "anchor_identity_grounded": res.get("anchor_identity_grounded"),
            "anchor_identity_reason": res.get("anchor_identity_reason"),
            # Resonated subgraph size, so the section 7 node-budget scoping is
            # measurable rather than assumed.
            "n_resonated_nodes": res.get("n_resonated_nodes"),
            "n_resonated_edges": res.get("n_resonated_edges"),
            "entity_not_found": res.get("entity_not_found"),
            "chain_fulfilled": res.get("chain_fulfilled"),
            "template_matched": res.get("template_matched"),
            "plan_confidence": res.get("confidence"),
            "walk_confidence": res.get("walk_confidence"),
            "n_walk_steps": res.get("n_walk_steps"),
            "time_seconds": res.get("time_seconds"),
            "_graph_labels": sorted(node_labels),
        }
        row["_invariant"] = check_invariant(row)
        row["_semantic"] = check_semantics(row, triples)
        row["_score"] = score_row(q, row, node_labels)
        if not row["_score"]["pass"]:
            row["_attribution"] = attribute(q, row, triples)
        rows.append(row)

    # --- direction pairs must answer DIFFERENTLY at each end ----------------
    # Keyed by pair so each pair is checked ONCE. Iterating rows and following
    # sibling_qid records every pair twice, which reported the rate as 18/20
    # instead of 9/10 and made a single defect look like two.
    by_id = {r["id"]: r for r in rows}
    pair_checks: List[Dict[str, Any]] = []
    seen_pairs: Set[tuple] = set()
    for r in rows:
        sib = r.get("sibling_qid")
        if not sib or sib not in by_id:
            continue
        key = tuple(sorted((r["id"], sib)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        pair_checks.append(check_direction_pair(r, by_id[sib]))

    # --- summarise ---------------------------------------------------------
    cat_stats: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for cat in ALL_CATEGORIES:
        sub = [r for r in rows if r["cat"] == cat]
        if not sub:
            continue
        n_pass = sum(1 for r in sub if r["_score"]["pass"])
        cat_stats[cat] = {
            "n": len(sub), "passed": n_pass,
            "rate": round(n_pass / len(sub), 4),
            "target": CATEGORY_TARGETS[cat]["target"],
            "minimum": CATEGORY_TARGETS[cat]["minimum"],
            "why": CATEGORY_TARGETS[cat]["why"],
            "meets_target": (n_pass / len(sub)) >= CATEGORY_TARGETS[cat]["target"],
            "meets_minimum": len(sub) >= CATEGORY_TARGETS[cat]["minimum"],
        }

    n_all = len(rows)
    n_pass = sum(1 for r in rows if r["_score"]["pass"])
    overall = n_pass / n_all

    inv_holds = sum(1 for r in rows if r["_invariant"]["invariant_ok"])
    sem_ok = sum(1 for r in rows if r["_semantic"]["ok"])
    illegal = sum(1 for r in rows if r["_semantic"]["invented"])
    tpl_ok = sum(1 for r in rows if r.get("template_matched"))
    pairs_ok = sum(1 for p in pair_checks if p["ok"])
    fallback_fired = sum(1 for r in rows
                         if r["cat"] == "nonsense_fallback" and r["heuristic_used"])
    fallback_n = sum(1 for r in rows if r["cat"] == "nonsense_fallback")

    prov_tot: Dict[str, int] = {}
    for r in rows:
        for h in r["_semantic"].get("provenance") or []:
            prov_tot[h["provenance"]] = prov_tot.get(h["provenance"], 0) + 1

    # Reported explicitly rather than left implicit. A fabrication detector that
    # silently finds nothing is indistinguishable from one that is not wired up,
    # so the count of unaccounted content words across every graded answer is
    # part of the result, not just its absence.
    fabricated = [r["id"] for r in rows
                  if any(w.startswith("fabricated_entities")
                         for w in r["_score"]["why"])]

    attrib: Dict[str, int] = {}
    for r in rows:
        if not r["_score"]["pass"]:
            key = r.get("_attribution", "unknown")
            attrib[key] = attrib.get(key, 0) + 1

    mech: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    mech["no_invented_relations"] = {
        "rate": round(1 - illegal / n_all, 4), "target": 0.98,
        "rows_with_illegal_or_fabricated_hops": illegal, "total_rows": n_all,
        "rows_with_unaccounted_entities": len(fabricated),
        "rows_with_unaccounted_entities_ids": fabricated,
        "meets_target": (1 - illegal / n_all) >= 0.98,
        "why": MECHANISM_TARGETS["no_invented_relations"]["why"]}
    mech["template_correctness"] = {
        "rate": round(tpl_ok / n_all, 4), "target": 0.95, "total_rows": n_all,
        "meets_target": (tpl_ok / n_all) >= 0.95,
        "why": MECHANISM_TARGETS["template_correctness"]["why"]}
    mech["central_invariant"] = {
        "rate": round(inv_holds / n_all, 4), "target": 0.90,
        "hold_rate": inv_holds, "total_rows": n_all,
        "meets_target": (inv_holds / n_all) >= 0.90,
        "why": MECHANISM_TARGETS["central_invariant"]["why"]}
    mech["semantic_checks"] = {
        "rate": round(sem_ok / n_all, 4), "total_rows": n_all,
        "note": "Stage C parity check: path order, template glue, hop provenance"}
    mech["direction_pair_distinct_answers"] = {
        "rate": round(pairs_ok / len(pair_checks), 4) if pair_checks else None,
        "pairs": len(pair_checks), "pairs_answering_differently": pairs_ok,
        "note": "both ends must name their own target AND differ from each other"}
    mech["fallback_trigger_rate"] = {
        "rate": round(fallback_fired / fallback_n, 4) if fallback_n else None,
        "fired": fallback_fired, "total": fallback_n}
    mech["overall"] = {
        "rate": round(overall, 4), "target": 0.90,
        "passed": n_pass, "total_rows": n_all,
        "meets_target": overall >= 0.90,
        "why": MECHANISM_TARGETS["overall"]["why"]}

    cat_ok = all(v["meets_target"] and v["meets_minimum"]
                 for v in cat_stats.values())
    stage_e_pass = bool(overall >= 0.90 and cat_ok
                        and mech["no_invented_relations"]["meets_target"]
                        and mech["template_correctness"]["meets_target"]
                        and mech["central_invariant"]["meets_target"])

    print(f"\n{'CATEGORY':24} {'n':>4} {'pass':>5} {'rate':>7} {'target':>7}  ok")
    print("-" * 60)
    for cat, st in cat_stats.items():
        print(f"{cat:24} {st['n']:>4} {st['passed']:>5} {st['rate']:>7.3f} "
              f"{st['target']:>7.2f}  {'YES' if st['meets_target'] else 'NO'}")
    print("-" * 60)
    print(f"{'OVERALL':24} {n_all:>4} {n_pass:>5} {overall:>7.3f} {0.90:>7.2f}  "
          f"{'YES' if overall >= 0.90 else 'NO'}")
    print(f"\n  central invariant held : {inv_holds}/{n_all} "
          f"({mech['central_invariant']['rate']:.3f})")
    print(f"  semantic checks passed : {sem_ok}/{n_all}")
    print(f"  illegal/fabricated hops: {illegal}")
    print(f"  unaccounted entities   : {len(fabricated)}")
    print(f"  hop provenance         : {prov_tot}")
    print(f"  fallback fired         : {fallback_fired}/{fallback_n}")
    print(f"  direction pairs differ : {pairs_ok}/{len(pair_checks)}")
    print(f"\n  failure attribution    : {attrib or 'none'}")
    print(f"\n  {STAGE} PASS = {stage_e_pass}")

    out = {
        "stage": STAGE_KEY,
        "pass": stage_e_pass,
        "seed": args.seed,
        "set_version": spec["version"],
        "graph_sha256": spec["graph"]["sha256"],
        "question_set_sha256": sha256(QUESTIONS_PATH),
        "question_set_version": spec["version"],
        "graph_nodes": spec["graph"]["nodes"],
        "graph_edges": spec["graph"]["edges"],
        "questions_run": n_all,
        "questions_in_frozen_set": len(spec["questions"]),
        "categories": cat_stats,
        "mechanisms": mech,
        "direction_pair_checks": pair_checks,
        "failure_attribution": attrib,
        "hop_provenance_totals": prov_tot,
        "population": spec["population"],
        "rows": rows,
    }
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
