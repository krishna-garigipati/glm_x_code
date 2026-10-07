"""Stage D: Honesty & Fallback Graph (GLM-X v3.3.2 contract section 16).

    - name: "Honesty & Fallback Graph"
      purpose: "Negative and edge cases"
      goal: "Prove honest refusal and heuristic_fallback"

Stage D asks the one question Stages A, B and C cannot: when the graph does NOT
contain the answer, does the engine say so, and when the question is
meaningless, does it say THAT instead of inventing something?

Five categories, and the entitlement of each is fixed by the graph, not by the
answer we happen to get:

  control              a stored edge exists in the asked direction and is the
                       unique 1-hop target -> the system must ANSWER. These
                       are the falsification control: a runner that graded only
                       refusals would pass a system that refuses everything.
  missing_relation     the anchor IS in the graph but has no such relation at
                       all -> must REFUSE. The anchor being present is what
                       makes this different from the next category.
  inverse_direction    the relation exists only in the OPPOSITE stored
                       direction, on a relation with NO declared inverse ->
                       must REFUSE and must NOT name the far end. Naming the
                       far end means the edge was read backwards, which for a
                       one-way relation is precisely the illegal same-label
                       mirroring the contract forbids (section 9).
  honesty_out_of_graph the subject is absent from the graph -> must REFUSE.
  nonsense_fallback    no relation cue from any descriptor bank -> the
                       heuristic fallback must FIRE (contract section 8).

Why this runner is stricter than the draft it replaces:

  * The nonsense disclosure check was vacuous. It read
    `disclosed = heuristic_disclosed or honest_no_relation or fired`, and
    `fired` is the same flag already asserted, so the disclosure term could
    never be false and proved nothing. Disclosure is now checked against the
    ANSWER TEXT, and for the five anchorless questions the expected behaviour
    is a REFUSAL rather than a disclosed guess -- glmx_ask deliberately does not
    prefix a refusal with "heuristic guess", because a refusal is not a guess.

  * Refusal was only ever graded as "a refusal marker appears". A refusal that
    also names a graph node the walker could not have reached would pass. Every
    must-refuse row is now additionally checked for invented nodes and for hop
    provenance, which is the contract's "do not expect the system to invent
    missing edges" rule stated as an executable assertion.

  * `preflight` re-derives the frozen set's claims against the actual SQLite
    store. generate_questions.py derives them from the builder's edge list; this
    checks them against the artefact the pipeline will really load, so a drift
    between the two cannot silently produce a green stage.

Stage A's `check_invariant` and Stage C's hop-provenance classifier are
imported rather than reimplemented: two copies of the invariant would
eventually disagree, and a green Stage D would prove nothing.

Usage:
    python test_results/stage_d/stage_d_runner.py
    python test_results/stage_d/stage_d_runner.py --only inverse_direction
    python test_results/stage_d/stage_d_runner.py --ids c05,c11,n06
"""
import argparse
import json
import random
import re
import sqlite3
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test_results" / "stage_a"))
sys.path.insert(0, str(ROOT / "test_results" / "stage_b"))
sys.path.insert(0, str(ROOT / "test_results" / "stage_c"))

import numpy as np

STAGE_DIR = Path(__file__).resolve().parent
DB_PATH = STAGE_DIR / "honesty_fallback.db"
QUESTIONS_PATH = STAGE_DIR / "questions_frozen.json"
CONFIG_G2P = ROOT / "configs" / "config_g2p.yaml"

from stage_a_runner import (  # noqa: E402
    REFUSAL_MARKERS, DISCLOSURE_MARKER, check_invariant,
)
from stage_b_runner import (  # noqa: E402
    load_chain_templates, load_relation_phrases, resolve_template, template_glue,
    load_stored_triples, CHAIN_TEMPLATES, RELATION_PHRASES,
)
from stage_c_runner import (  # noqa: E402
    MIRROR_INVERSE, PROV_STORED, PROV_DECLARED_INVERSE, PROV_ILLEGAL,
    PROV_FABRICATED, classify_hops,
)

# Contract section 15 thresholds. `nonsense_fallback` is the contract's
# "fallback must trigger on >= 90% of cases"; `control` inherits the
# `simple_one_hop` >= 95% target because that is what a 1-hop control measures.
MECHANISM_TARGETS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict([
    ("control", {"target": 0.95, "minimum": 20,
                 "why": "simple_one_hop >= 95% (section 15)"}),
    ("missing_relation", {"target": 0.90, "minimum": 8,
                          "why": "correct_refusal_rate >= 90% (section 15)"}),
    ("inverse_direction", {"target": 0.90, "minimum": 6,
                           "why": "correct_refusal_rate >= 90% (section 15)"}),
    ("honesty_out_of_graph", {"target": 0.90, "minimum": 6,
                              "why": "question_category_targets >= 90%"}),
    ("nonsense_fallback", {"target": 0.90, "minimum": 6,
                           "why": "fallback must trigger on >= 90% of cases"}),
    ("no_invented_relations", {"target": 0.98, "minimum": 1,
                               "why": "decode.no_invented_relations >= 98%"}),
    ("template_correctness", {"target": 0.95, "minimum": 1,
                              "why": "decode.template_correctness >= 95%"}),
    ("central_invariant", {"target": 0.90, "minimum": 1,
                           "why": "central_invariant.hold_rate >= 90%"}),
])

ALL_CATEGORIES = ("control", "missing_relation", "inverse_direction",
                  "honesty_out_of_graph", "nonsense_fallback")


# ---------------------------------------------------------------------------
# store access
# ---------------------------------------------------------------------------

def load_store(db_path: Path) -> Set[tuple]:
    """Stored (source, relation, target) triples read from SQLite.

    Read straight out of the artefact rather than from the builder's Python
    lists. The point of preflight is to catch the case where the frozen question
    set and the database the pipeline actually loads have drifted apart, and
    that check is worthless if both sides are read from the same in-memory list.
    """
    con = sqlite3.connect(str(db_path))
    try:
        labels = {nid: (lbl or "").strip().lower()
                  for nid, lbl in con.execute("SELECT id, label FROM nodes")}
        triples: Set[tuple] = set()
        for src, tgt, rel in con.execute(
                "SELECT source_id, target_id, relation FROM edges"):
            triples.add((labels.get(src, ""), rel, labels.get(tgt, "")))
        return triples
    finally:
        con.close()


def accepted_labels(rel: str) -> Set[str]:
    """Relation labels the walker accepts for an asked `rel` (section 9).

    A relation and its DECLARED INVERSE are two labels for the same directed
    edge. Judging reachability against the asked label alone under-counts it,
    which is the mistake that had `("plantlet", "precedes")` filed as an
    unanswerable gap when the graph stores `plantlet follows sprout`.
    """
    return {rel, MIRROR_INVERSE[rel]} if rel in MIRROR_INVERSE else {rel}


def reachable_from(anchor: str, rel: str, triples: Set[tuple],
                    hops: int | None = 1) -> Set[str]:
    """What the walker can reach from `anchor` under `rel`.

    `hops=1` is the right question for a CONTROL: uniqueness must hold for the
    single hop a 1-hop question actually grades. `cat is_a mammal is_a animal`
    means "What is a cat?" is unambiguous even though a transitive search also
    reaches animal, so demanding a transitive-unique control would reject every
    taxonomy edge.

    `hops=None` searches to a fixpoint, and is the right question for a
    MUST-REFUSE pair: the anchor must be unable to reach anything under that
    relation by ANY route the walker could take, not merely in one step.
    """
    labels = accepted_labels(rel)
    best: Dict[str, int] = {anchor: 0}
    frontier = [anchor]
    while frontier:
        cur = frontier.pop()
        if hops is not None and best[cur] >= hops:
            continue
        for s, r, t in triples:
            if s == cur and r in labels and (t not in best or best[t] > best[cur] + 1):
                best[t] = best[cur] + 1
                frontier.append(t)
    return set(best) - {anchor}


def load_cues() -> List[tuple]:
    """(relation, cue phrase) pairs from the descriptor bank in config_g2p.yaml."""
    import yaml

    cfg = yaml.safe_load(CONFIG_G2P.read_text(encoding="utf-8"))
    variants = cfg["extraction"]["relation_variants"]
    return [(rel, cue) for rel, phrases in variants.items() for cue in phrases]


def load_planner():
    """The real QueryRelationExtractor, for its literal cue matcher only.

    Imported rather than reimplemented because "which relations does this
    question name literally" is exactly the question the planner answers, and a
    second copy of the matcher would eventually disagree with the first. No model
    is loaded: `_literal_cue_relations` is pure regex over the config bank.
    """
    from g2p.config import G2PConfig
    from g2p.g2p_planner import QueryRelationExtractor

    return QueryRelationExtractor(G2PConfig.from_yaml(str(CONFIG_G2P)))


def names_node(text: str, node_labels: Set[str]) -> List[str]:
    """Graph node labels named in `text`, matched on word boundaries.

    Word boundaries matter: `oak` must not be found inside `oakland`, and `cat`
    must not be found inside `cattle`.
    """
    low = text.lower()
    return sorted(n for n in node_labels
                  if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", low))


def claim_text(text: str) -> str:
    """Strip the honest-disclosure tail off a refusal.

    "I don't have a relation ... Closest concepts I have: dog." is a REFUSAL
    that discloses what the graph does hold, not an answer asserting it. Naming
    a node while declining to answer is required behaviour (section 6), so the
    invented-node test may only look at what is left once the disclosure is
    removed.
    """
    low = text.lower()
    cut = low.find("closest concept")
    return text if cut == -1 else text[:cut]


def refused(row: Dict[str, Any]) -> bool:
    """Did the engine decline, either by flag or by phrasing?"""
    return bool(row.get("honest_no_relation")) or any(
        m in (row.get("answer") or "").lower() for m in REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def preflight(spec: Dict[str, Any], triples: Set[tuple],
              node_labels: Set[str], cues: List[tuple],
              planner) -> Dict[str, int]:
    """Refuse to run a question set that cannot support the claims it is used for.

    Each check below corresponds to a way the category could pass while testing
    nothing:

      * a question whose `expected_chain` is not what the planner's literal cue
        matcher actually produces -- the run would then be graded against a
        chain the system never planned, and a refusal caused by cue-absence
        fallback would be indistinguishable from a refusal caused by a genuine
        relation gap. This is not hypothetical: phrasings like "What is the cat
        known for?" and "What does the fossil record contradict?" name no
        literal cue at all, because the bank's entries are the third-person
        forms ("contradicts") or require adjacency ("is known for"), and the
        question says "known for" with a noun in between. Both fell through to
        the default_chain, and one of them produced a confident answer to a
        question about a different relation entirely;
      * a control whose expected node equals its anchor -- `want in answer` is
        then satisfied by any answer that echoes the subject;
      * a control whose edge is not actually in the store, or is not the unique
        one-hop target, so there is nothing unambiguous to grade;
      * a must-refuse anchor that is NOT in the graph -- the refusal would then
        come from the entity gate and would say nothing about the relation gate;
      * an `inverse_direction` pair on a relation that HAS a declared inverse --
        reading that edge backwards is the contract working, not a violation;
      * a must-refuse pair the walker can in fact reach -- the stage would be
        grading a refusal the system is right to withhold;
      * an out-of-graph subject that is in fact a graph node;
      * a nonsense question containing a descriptor-bank cue -- the fallback
        would be firing for a different reason than cue absence;
      * no nonsense question expecting a disclosed guess -- the disclosure
        branch would go unexercised while the stage still reported it green.
    """
    problems: List[str] = []
    counts = {c: 0 for c in ALL_CATEGORIES}

    for q in spec["questions"]:
        qid, cat = q["qid"], q["category"]
        if cat not in counts:
            problems.append(f"{qid}: unknown category {cat!r}")
            continue
        counts[cat] += 1
        low = q["question"].lower()

        # The declared chain must be the chain the planner will really plan.
        planned = planner._literal_cue_relations(q["question"])
        declared = list(q.get("expected_chain") or [])
        if cat in ("control", "missing_relation", "inverse_direction"):
            if planned != declared:
                problems.append(
                    f"{qid}: expected_chain {declared} but the literal cue matcher "
                    f"resolves {q['question']!r} to {planned}")
        elif cat == "honesty_out_of_graph":
            if not planned:
                problems.append(f"{qid}: carries no relation cue, so the refusal "
                                "would come from cue absence, not the entity gate")
            for rel in planned:
                if not any(r == rel for _, r, _ in triples):
                    problems.append(f"{qid}: cue resolves to {rel!r}, which no edge "
                                    "in the store uses")
        elif cat == "nonsense_fallback" and planned:
            problems.append(f"{qid}: the literal cue matcher resolves it to "
                            f"{planned}, so the fallback would not be firing on "
                            "cue absence")

        if cat == "control":
            anchor, rel = q["anchor"], q["rel"]
            want, path = q["expected_node"], q["expected_path"]
            if anchor.lower() not in node_labels:
                problems.append(f"{qid}: anchor {anchor!r} is not in the store")
            if not want or want.lower() == anchor.lower():
                problems.append(f"{qid}: expected_node {want!r} equals the anchor, "
                                "so `expected_node in answer` cannot fail")
            if [p.lower() for p in (path or [])] != [anchor.lower(), (want or "").lower()]:
                problems.append(f"{qid}: expected_path {path!r} does not run "
                                f"{anchor!r} -> {want!r}")
            if q["hops"] != 1:
                problems.append(f"{qid}: a control must be exactly 1 hop, got {q['hops']}")
            reach = reachable_from(anchor, rel, triples, hops=1)
            if reach != {want.lower()}:
                problems.append(f"{qid}: store lets {anchor!r} reach {sorted(reach)} "
                                f"under {rel!r} in one hop, not just {want!r}")
            if not any(q["expected_chain"]) or q["expected_chain"][0] != rel:
                problems.append(f"{qid}: expected_chain {q['expected_chain']!r} "
                                f"does not name the graded relation {rel!r}")

        elif cat == "missing_relation":
            anchor, rel = q["anchor"], q["rel"]
            if anchor.lower() not in node_labels:
                problems.append(f"{qid}: anchor {anchor!r} is not in the store, so "
                                "this would test the entity gate, not the "
                                "relation gap")
            reach = reachable_from(anchor, rel, triples, hops=None)
            if reach:
                problems.append(f"{qid}: {anchor!r} CAN reach {sorted(reach)} "
                                f"under {rel!r}; this is a control, not a gap")

        elif cat == "inverse_direction":
            anchor, rel, forbidden = q["anchor"], q["rel"], q.get("forbidden_nodes") or []
            if rel in MIRROR_INVERSE:
                problems.append(f"{qid}: {rel!r} has a declared inverse, so its "
                                "backwards read is legal and this control is "
                                "meaningless")
            if not forbidden:
                problems.append(f"{qid}: no forbidden_nodes registered, so the "
                                "illegal-backwards-read check cannot run")
            for far in forbidden:
                if (far.lower(), rel, anchor.lower()) not in triples:
                    problems.append(f"{qid}: the store has no {far!r} {rel} "
                                    f"{anchor!r} edge, so this is not a "
                                    "backwards read")
            if reachable_from(anchor, rel, triples, hops=None):
                problems.append(f"{qid}: {anchor!r} reaches "
                                f"{sorted(reachable_from(anchor, rel, triples, hops=None))} "
                                f"under {rel!r}; it is answerable")

        elif cat == "honesty_out_of_graph":
            subj = (q.get("subject") or "").lower()
            if not subj:
                problems.append(f"{qid}: no subject registered, so absence "
                                "cannot be verified")
            elif subj in node_labels:
                problems.append(f"{qid}: subject {subj!r} IS a graph node")

        elif cat == "nonsense_fallback":
            if q.get("expect_guess_disclosed"):
                if not q.get("expected_node") or not q.get("anchor"):
                    problems.append(f"{qid}: expects a disclosed guess but "
                                    "registers no anchor/expected_node, so the "
                                    "disclosure path has nothing to disclose")

    for cat in ALL_CATEGORIES:
        if counts[cat] == 0:
            problems.append(f"no {cat} questions in the frozen set")

    disclosed = [q for q in spec["questions"]
                 if q["category"] == "nonsense_fallback" and q.get("expect_guess_disclosed")]
    if not disclosed:
        problems.append(
            "no nonsense question expects a disclosed guess, so the "
            "'disclose that it is a guess' half of the section 8 fallback "
            "requirement would go unexercised")

    if problems:
        raise SystemExit("REFUSING TO RUN:\n" + "\n".join(f"  {p}" for p in problems))

    print(f"  preflight: {len(spec['questions'])} questions re-derived against the "
          f"store -- {counts['control']} controls 1-hop unique, "
          f"{counts['missing_relation']} gaps unreachable to a fixpoint, "
          f"{counts['inverse_direction']} backwards reads on no-inverse relations, "
          f"{counts['honesty_out_of_graph']} subjects absent, "
          f"{counts['nonsense_fallback']} cue-free (1 expecting a disclosed guess)")
    return counts


# ---------------------------------------------------------------------------
# semantic checks
# ---------------------------------------------------------------------------

def check_semantics(row: Dict[str, Any], triples: Set[tuple]) -> Dict[str, Any]:
    """Semantic checks on top of the central invariant.

    1. subject_before_object    -- the rendered sentence must mention the path
       nodes in traversal order.
    2. relation_phrase_present  -- the decoder's own template glue must appear
       (Stage B logic, so all stages grade one sentence the same way). For a
       refusal the inverse check applies instead: no relation phrase may leak.
    3. hop_provenance_legal     -- every hop is a stored triple, or the mirror of
       one under a declared inverse label. This is the anti-invention check and
       it applies to EVERY row, refusals and fallback guesses included.
    4. expected_path_order      -- where a path is pre-registered, the walked
       labels must equal it exactly.
    """
    answer = (row.get("answer") or "").lower()
    labels: List[str] = [str(x).lower() for x in (row.get("path_labels") or [])]
    edges: List[str] = list(row.get("path_edges") or [])
    checks: "OrderedDict[str, bool]" = OrderedDict()

    if not edges or len(labels) < 2:
        return {"checks": {}, "failed": [], "ok": True, "provenance": [],
                "invented": 0}

    # Scan FORWARD from the previous label rather than taking each label's first
    # occurrence: `answer.find` mis-orders any label that is a prefix of another.
    idxs: List[int] = []
    cursor = 0
    for lbl in labels:
        m = re.search(r"(?<!\w)" + re.escape(lbl) + r"(?!\w)", answer[cursor:])
        if not m:
            idxs = []
            break
        idxs.append(cursor + m.start())
        cursor += m.end()
    if idxs:
        checks["subject_before_object"] = len(set(idxs)) == len(idxs)

    key = tuple(edges)
    template = CHAIN_TEMPLATES.get(key)
    if template:
        fragments = template_glue(resolve_template(template, edges))
        row["template_used"] = template
    else:
        fragments = [RELATION_PHRASES.get(rel, "") for rel in edges]
        row["template_used"] = "<generic relation_phrases>"

    if row.get("honest_no_relation"):
        present = [f for f in fragments if f and f.lower() in answer]
        checks["no_claim_when_refusing"] = not present
        row["template_fragments_leaked"] = present
    else:
        missing = [f for f in fragments if f and f.lower() not in answer]
        checks["relation_phrase_present"] = not missing
        row["template_fragments_missing"] = missing

    prov = classify_hops(labels, edges, triples)
    row["hop_provenance"] = prov
    illegal = [h for h in prov if h["provenance"] == PROV_ILLEGAL]
    fabricated = [h for h in prov if h["provenance"] == PROV_FABRICATED]
    checks["hop_provenance_legal"] = not illegal and not fabricated
    row["illegal_mirror_hops"] = illegal
    row["fabricated_hops"] = fabricated

    exp = [str(x).lower() for x in (row.get("expected_path") or [])]
    if exp:
        checks["expected_path_order"] = labels[: len(exp)] == exp

    failed = [k for k, v in checks.items() if not v]
    return {"checks": dict(checks), "failed": failed, "ok": not failed,
            "provenance": prov, "invented": len(illegal) + len(fabricated)}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_row(q: Dict[str, Any], row: Dict[str, Any],
              node_labels: Set[str]) -> Dict[str, Any]:
    cat = q["category"]
    answer = row.get("answer") or ""
    inv_ok = row["_invariant"]["invariant_ok"]
    sem_ok = row["_semantic"]["ok"]
    why: List[str] = []

    def inv_note() -> None:
        if not inv_ok:
            why.append(f"invariant: {row['_invariant']['failed_checks']}")

    def sem_note() -> None:
        if not sem_ok:
            why.append(f"semantic: {row['_semantic']['failed']}")

    if cat == "control":
        want = (q.get("expected_node") or "").lower()
        got_edges = list(row.get("path_edges") or [])
        labels = [str(x).lower() for x in (row.get("path_labels") or [])]
        ok_labels = accepted_labels(q["rel"])
        if want not in answer.lower():
            why.append(f"expected {want!r} absent")
        if row.get("honest_no_relation"):
            why.append("refused instead of answering")
        if len(got_edges) != 1:
            why.append(f"walked {len(got_edges)} hops, expected 1")
        elif got_edges[0] not in ok_labels:
            why.append(f"walked relation {got_edges[0]!r}, which is neither "
                       f"{q['rel']!r} nor its declared inverse "
                       f"{sorted(ok_labels - {q['rel']})}")
        if labels[:2] != [q["anchor"].lower(), want]:
            why.append(f"walked {labels} rather than the registered "
                       f"[{q['anchor'].lower()}, {want}]")
        inv_note()
        sem_note()
        return {"passed": bool(not why), "why": why}

    if cat in ("missing_relation", "inverse_direction", "honesty_out_of_graph"):
        ref = refused(row)
        # The invented-node test. A refusal is permitted to name its own anchor
        # (section 6 requires disclosure) and nothing else. Anything further is
        # the engine asserting a graph fact it has no edge for.
        if cat == "inverse_direction":
            forbidden = [n.lower() for n in (q.get("forbidden_nodes") or [])]
            leaked = [n for n in forbidden
                      if n in names_node(claim_text(answer), node_labels)]
        elif cat == "missing_relation":
            named = names_node(claim_text(answer), node_labels)
            leaked = [n for n in named if n != q["anchor"].lower()]
        else:
            leaked = names_node(claim_text(answer), node_labels)

        if not ref:
            why.append(f"answered instead of refusing a {cat} question")
        if leaked:
            why.append(f"named {leaked} outside the disclosure, which is a claim "
                       "the graph cannot support")
        if row["_semantic"]["invented"]:
            why.append(f"{row['_semantic']['invented']} illegal or fabricated hop(s)")
        inv_note()
        sem_note()
        return {"passed": bool(not why), "why": why}

    if cat == "nonsense_fallback":
        fired = bool(row.get("heuristic_used"))
        if not fired:
            why.append("heuristic fallback did not trigger on a cue-free question")

        if q.get("expect_guess_disclosed"):
            # The one case where the engine must produce an answer AND mark it as
            # a guess. The flag alone is not enough: assert the disclosure is in
            # the text a user would actually read.
            marked = DISCLOSURE_MARKER.lower() in answer.lower()
            produced = bool(row.get("path_edges"))
            if not row.get("heuristic_disclosed"):
                why.append("fallback answered without setting heuristic_disclosed")
            if not marked:
                why.append(f"answer does not disclose the guess "
                           f"(no {DISCLOSURE_MARKER!r} in the text)")
            if not produced:
                why.append("disclosed a guess but walked no edge to produce it")
        else:
            # Anchorless nonsense: the entity gate fires and the honest answer is
            # a refusal. glmx_ask deliberately does NOT prefix "heuristic guess"
            # onto a refusal, so demanding the marker here would be demanding a
            # lie. But if the flag IS set, the text must actually carry it.
            if not refused(row):
                why.append("cue-free question produced an unrefused answer that "
                           "was not disclosed as a guess")
            if row.get("heuristic_disclosed") and \
                    DISCLOSURE_MARKER.lower() not in answer.lower():
                why.append("heuristic_disclosed is set but the answer text does "
                           "not disclose it")
        inv_note()
        sem_note()
        return {"passed": bool(not why), "why": why}

    return {"passed": False, "why": [f"unknown category {cat!r}"]}


# ---------------------------------------------------------------------------
# main
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
    out_path = Path(args.out) if args.out else STAGE_DIR / "stage_d_results.json"

    random.seed(args.seed)
    np.random.seed(args.seed)

    from scripts.glmx_ask import GLMXPipeline
    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

    # resolve_template/template_glue read these from STAGE B's module globals,
    # not from ours. Rebinding only the imported names here would leave the
    # lookup inside stage_b_runner empty, silently turning "{relation0}" into a
    # literal no answer can contain.
    import stage_b_runner as _b

    _b.CHAIN_TEMPLATES = load_chain_templates()
    _b.RELATION_PHRASES = load_relation_phrases()
    globals()["CHAIN_TEMPLATES"] = _b.CHAIN_TEMPLATES
    globals()["RELATION_PHRASES"] = _b.RELATION_PHRASES

    triples = load_store(DB_PATH)
    node_labels = {s for s, _, _ in triples} | {t for _, _, t in triples}
    planner = load_planner()
    cues = load_cues()

    print(f"\n{'=' * 78}\nSTAGE D -- Honesty & Fallback "
          f"({len(questions)} questions)\n{'=' * 78}")
    # Preflight runs against the WHOLE frozen set, not the filtered subset, so
    # --only can never be used to make a broken set look runnable.
    preflight(spec, triples, node_labels, cues, planner)

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
            "rel": q.get("rel"),
            "question": q["question"],
            "expected_node": q.get("expected_node"),
            "expected_chain": q.get("expected_chain"),
            "expected_path": q.get("expected_path"),
            "forbidden_nodes": q.get("forbidden_nodes"),
            "expect_guess_disclosed": q.get("expect_guess_disclosed"),
            "answer": res.get("answer"),
            "relation_chain": res.get("relation_chain"),
            "path_labels": res.get("walk_path_labels"),
            "path_edges": res.get("walk_path_edges"),
            "selected_anchor": (res.get("selected_anchor") or {}).get("label"),
            "honest_no_relation": res.get("honest_no_relation"),
            "honest_by_entity": res.get("honest_by_entity"),
            "honest_by_relation": res.get("honest_by_relation"),
            "heuristic_used": res.get("heuristic_used"),
            "heuristic_disclosed": res.get("heuristic_disclosed"),
            "confidence": res.get("confidence"),
            "time_seconds": res.get("time_seconds"),
        }
        # check_invariant's `refusal_offers_no_fabricated_concept` verifies the
        # concepts an honesty refusal offers against this list. Without it that
        # check compares against an empty set, calls every disclosed concept
        # fabricated, and fails all 33 refusal rows -- so this is load-bearing,
        # not bookkeeping.
        row["_graph_labels"] = sorted(node_labels)
        row["_invariant"] = check_invariant(row)
        row["_semantic"] = check_semantics(row, triples)
        verdict = score_row(q, row, node_labels)
        row["passed"] = verdict["passed"]
        row["fail_reason"] = verdict["why"]
        rows.append(row)

        mark = "PASS" if row["passed"] else "FAIL"
        chain_s = "/".join(row["path_edges"] or []) or "-"
        print(f"  [{mark}] {row['id']} {row['cat'][:19]:<19} "
              f"chain={chain_s:<22} {(row['answer'] or '')[:52]}")

    n = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    inv_held = sum(1 for r in rows if r["_invariant"]["invariant_ok"])
    sem_held = sum(1 for r in rows if r["_semantic"]["ok"])
    invented = sum(r["_semantic"]["invented"] for r in rows)
    template_ok = sum(1 for r in rows
                      if "relation_phrase_present" in r["_semantic"]["checks"]
                      and r["_semantic"]["checks"]["relation_phrase_present"])

    prov_counts = {
        prov: sum(1 for r in rows for h in (r.get("hop_provenance") or [])
                  if h["provenance"] == prov)
        for prov in (PROV_STORED, PROV_DECLARED_INVERSE, PROV_ILLEGAL, PROV_FABRICATED)
    }

    by_cat = {c: [r for r in rows if r["cat"] == c] for c in ALL_CATEGORIES}

    # ---- mechanism matrix (contract section 15) ----
    matrix: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for cat, meta in MECHANISM_TARGETS.items():
        if cat in by_cat:
            sub, npass = by_cat[cat], sum(1 for r in by_cat[cat] if r["passed"])
        elif cat == "no_invented_relations":
            sub, npass = rows, n - len([r for r in rows if r["_semantic"]["invented"]])
        elif cat == "template_correctness":
            graded = [r for r in rows
                      if "relation_phrase_present" in r["_semantic"]["checks"]]
            sub, npass = graded, template_ok
        else:
            sub, npass = rows, inv_held
        if not sub:
            matrix[cat] = {"n": 0, "passed": 0, "rate": 0.0, "target": meta["target"],
                           "minimum": meta["minimum"], "meets_target": False,
                           "why": meta["why"]}
            continue
        rate = npass / len(sub)
        matrix[cat] = {
            "n": len(sub), "passed": npass, "rate": round(rate, 4),
            "target": meta["target"], "minimum": meta["minimum"],
            "meets_target": bool(rate >= meta["target"] and len(sub) >= meta["minimum"]),
            "why": meta["why"],
        }

    summary = {
        "stage": "D",
        "contract": "GLM-X v3.3.2 sections 8, 15 and 16",
        "graph": DB_PATH.name,
        "question_set": QUESTIONS_PATH.name,
        "question_set_version": spec.get("version"),
        "seed": args.seed,
        "n_questions": n,
        "overall": {"passed": passed, "total": n, "rate": round(passed / n, 4)},
        "central_invariant": {"held": inv_held, "total": n, "rate": round(inv_held / n, 4)},
        "semantic_checks": {"held": sem_held, "total": n, "rate": round(sem_held / n, 4)},
        "no_invented_relations": {
            "clean": n - len([r for r in rows if r["_semantic"]["invented"]]),
            "total": n,
            "illegal_same_label_mirror": prov_counts[PROV_ILLEGAL],
            "fabricated": prov_counts[PROV_FABRICATED],
        },
        "control": {"passed": sum(1 for r in by_cat["control"] if r["passed"]),
                    "total": len(by_cat["control"])},
        "missing_relation": {
            "passed": sum(1 for r in by_cat["missing_relation"] if r["passed"]),
            "total": len(by_cat["missing_relation"]),
            "all_refused": all(refused(r) for r in by_cat["missing_relation"]),
        },
        "inverse_direction": {
            "passed": sum(1 for r in by_cat["inverse_direction"] if r["passed"]),
            "total": len(by_cat["inverse_direction"]),
            "all_refused": all(refused(r) for r in by_cat["inverse_direction"]),
            "backwards_reads": sum(1 for r in by_cat["inverse_direction"]
                                   if r["fail_reason"]),
        },
        "honesty_out_of_graph": {
            "passed": sum(1 for r in by_cat["honesty_out_of_graph"] if r["passed"]),
            "total": len(by_cat["honesty_out_of_graph"]),
            "all_refused": all(refused(r) for r in by_cat["honesty_out_of_graph"]),
        },
        "nonsense_fallback": {
            "passed": sum(1 for r in by_cat["nonsense_fallback"] if r["passed"]),
            "total": len(by_cat["nonsense_fallback"]),
            "all_triggered": all(r.get("heuristic_used")
                                 for r in by_cat["nonsense_fallback"]),
            "disclosed_guess_cases": sum(
                1 for r in by_cat["nonsense_fallback"] if r["expect_guess_disclosed"]),
            "disclosed_guess_ok": sum(
                1 for r in by_cat["nonsense_fallback"]
                if r["expect_guess_disclosed"] and r["passed"]),
        },
        "hop_provenance": prov_counts,
        "mechanism_matrix": matrix,
        "stage_d_pass": bool(passed == n and inv_held == n and sem_held == n
                             and invented == 0
                             and all(m["meets_target"] for m in matrix.values())),
    }

    out_path.write_text(json.dumps({"summary": summary, "rows": [
        {k: v for k, v in r.items() if not k.startswith("_")} | {
            "invariant_ok": r["_invariant"]["invariant_ok"],
            "invariant_checks": r["_invariant"]["checks"],
            "semantic_ok": r["_semantic"]["ok"],
            "semantic_checks": r["_semantic"]["checks"],
            "hop_provenance": r.get("hop_provenance"),
        } for r in rows]}, indent=2), encoding="utf-8")

    print(f"\n{'-' * 78}")
    print(f"overall {passed}/{n} ({passed / n:.1%})")
    print(f"central invariant {inv_held}/{n}")
    print(f"semantic checks {sem_held}/{n}")
    print(f"control {summary['control']['passed']}/{len(by_cat['control'])}")
    print(f"missing_relation {summary['missing_relation']['passed']}/"
          f"{len(by_cat['missing_relation'])} "
          f"(all refused: {summary['missing_relation']['all_refused']})")
    print(f"inverse_direction {summary['inverse_direction']['passed']}/"
          f"{len(by_cat['inverse_direction'])} "
          f"(all refused: {summary['inverse_direction']['all_refused']})")
    print(f"honesty_out_of_graph {summary['honesty_out_of_graph']['passed']}/"
          f"{len(by_cat['honesty_out_of_graph'])}")
    print(f"nonsense_fallback {summary['nonsense_fallback']['passed']}/"
          f"{len(by_cat['nonsense_fallback'])} "
          f"(all triggered: {summary['nonsense_fallback']['all_triggered']}, "
          f"disclosed-guess {summary['nonsense_fallback']['disclosed_guess_ok']}/"
          f"{summary['nonsense_fallback']['disclosed_guess_cases']})")
    print(f"hop provenance {prov_counts}")
    print(f"\nMECHANISM MATRIX (contract section 15)")
    for cat, m in matrix.items():
        flag = "OK  " if m["meets_target"] else "MISS"
        print(f"  [{flag}] {cat:<24} {m['passed']:>3}/{m['n']:<3} ({m['rate']:>6.1%})  "
              f"target>={m['target']:.0%}  min={m['minimum']}")
    print(f"\nstage_d_pass = {summary['stage_d_pass']}")

    fails = [r for r in rows if not r["passed"]]
    if fails:
        print(f"\nFAILURES ({len(fails)})")
        for r in fails:
            print(f"  {r['id']} [{r['cat']}] {r['question']!r}")
            print(f"      answer: {r['answer']!r}")
            print(f"      walk:   {r['path_labels']} / {r['path_edges']}")
            print(f"      anchor: {r['selected_anchor']!r} "
                  f"heuristic={r['heuristic_used']} "
                  f"disclosed={r['heuristic_disclosed']}")
            print(f"      reason: {r['fail_reason']}")
    print(f"\nWrote {out_path}")
    return 0 if summary["stage_d_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
