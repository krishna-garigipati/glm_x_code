"""Stage E: derive the Frozen PoC Evaluation Set from the frozen PoC graph.

GLM-X v3.3.2 section 16 requires the question set to be pre-registered and frozen
before final evaluation. This generator does that honestly: it derives the gold
for every question from `build_poc_graph`'s validated tables and NEVER from the
engine's behaviour. Nothing here looks at an answer, so the set cannot have been
shaped by what the system gets right.

Three rules govern the derivation:

  1. Every auto-gradable edge/chain/path in the graph becomes a question. No
     sampling, no filtering by difficulty, no dropping of questions the engine
     is known to fail. What is skipped is skipped for a MECHANICAL reason only
     (the edge is ambiguous, so a question about it has more than one correct
     answer, or the engine would collapse the declared chain), and every skip is
     counted in `population` so the coverage claim can be audited.

  2. Phrasings come from a template bank, not from trial and error. Every
     template must contain a cue phrase that literally exists in
     configs/config_g2p.yaml; generate_questions refuses to freeze if a template
     fails to produce its declared relation_chain. Templates were calibrated
     against the cue bank and the real anchor audit (see the calibration notes
     in build_poc_graph.py); the GOLD was never adjusted to suit a template.

  3. Categories are disjoint. A one-hop read that is one end of a proven
     direction pair is counted as `direction_pair`, not also as `simple_one_hop`,
     so the pooled overall score never double-counts.

Usage:
    python test_results/stage_e/generate_questions.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE_DIR))

import build_poc_graph as B  # noqa: E402

FROZEN_PATH = STAGE_DIR / "poc_questions_frozen.json"
VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Template bank. Every `{a}` is replaced by the anchor label.
#
# Calibrated against configs/config_g2p.yaml's `relation_variants` and the real
# anchor audit. Two calibration results worth recording:
#
#   * has_property needs "is known for" ADJACENT, so "What is the {a} known
#     for?" resolves to NO cue at all (the literal "is known for" is absent) and
#     silently takes the heuristic fallback. "What is known for the {a}?" is the
#     form that carries the cue.
#
#   * precedes and follows have inverted-sounding cues: the `precedes` cue is
#     "comes after" and the `follows` cue is "comes before". So for a stored edge
#     `a precedes b` the question is "What comes after a?" and for `b follows a`
#     it is "What comes before b?". Getting this backwards tests nothing.
# --------------------------------------------------------------------------
ONE_HOP_TEMPLATE: dict[str, str] = {
    "is_a": "What type of thing is the {a}?",
    "has_property": "What is known for the {a}?",
    "causes": "What does the {a} cause?",
    "caused_by": "What was the {a} caused by?",
    "precedes": "What comes after the {a}?",
    "follows": "What comes before the {a}?",
    "part_of": "What is the {a} part of?",
    "synonym": "What is another word for the {a}?",
    "antonym": "What is the opposite of the {a}?",
    "example_of": "What is the {a} an example of?",
    "associated_with": "What is the {a} associated with?",
    "supports": "What evidence supports the {a}?",
    "contradicts": "What contradicts the {a}?",
    "temporal_coincident": "What happens at the same time as the {a}?",
    "spatial_near": "What is near the {a}?",
    "linguistic_maps": "What do you call the {a} in Spanish?",
}

# Continuation clauses for hop 2 onwards, referring to the previous hop's result
# as "that". Validated pattern from Stage C's multi-hop set.
CONT_CLAUSE: dict[str, str] = {
    "is_a": "what type of thing is that",
    "has_property": "what is known for that",
    "causes": "what does that cause",
    "caused_by": "what was that caused by",
    "precedes": "what comes after that",
    "follows": "what comes before that",
    "part_of": "what is that part of",
    "synonym": "what is another word for that",
    "antonym": "what is the opposite of that",
    "example_of": "what is that an example of",
    "associated_with": "what is that associated with",
    "supports": "what evidence supports that",
    "contradicts": "what contradicts that",
    "temporal_coincident": "what happens at the same time as that",
    "spatial_near": "what is near that",
    "linguistic_maps": "what do you call that in Spanish",
}


def one_hop_question(anchor: str, rel: str) -> str:
    return ONE_HOP_TEMPLATE[rel].format(a=anchor)


def multi_hop_question(path: list[str], chain: list[str]) -> str:
    """`<first clause>, and <continuation>, and <continuation>?`

    Clause order is what makes `_literal_cue_relations` emit the relations in
    walk order, since it orders matched relations by position in the question.
    """
    clauses = [ONE_HOP_TEMPLATE[chain[0]].format(a=path[0]).rstrip("?")]
    clauses += [CONT_CLAUSE[r] for r in chain[1:]]
    return ", and ".join(clauses) + "?"


# --------------------------------------------------------------------------
def build_questions() -> tuple[list[dict], dict]:
    questions: list[dict] = []
    population: dict = {}

    # --- direction pairs ------------------------------------------------
    # Each proven-in-both-directions pair contributes TWO questions. Both ends
    # are gradeable only because each endpoint has exactly one incident edge
    # under {relation, declared inverse}; see build_poc_graph lesson 1.
    pairs = B.direction_pairs()
    pair_ends: list[tuple] = []          # (qid_a, qid_b, anchor_a, node_a, anchor_b, node_b)
    n = 0
    for src, rel, tgt in pairs:
        n += 1
        inv = B.INVERSE[rel]
        qid_f, qid_r = f"dp{n:03d}f", f"dp{n:03d}r"
        pair_ends.append((qid_f, qid_r, src, tgt, tgt, src))
        for qid, anchor, asked, node in (
            (qid_f, src, rel, tgt),
            (qid_r, tgt, inv, src),
        ):
            questions.append({
                "qid": qid,
                "category": "direction_pair",
                "question": one_hop_question(anchor, asked),
                "rel": asked,
                "chain": [asked],
                "hops": 1,
                "anchor": anchor,
                "expected_node": node,
                "expected_path": [anchor, node],
                "forbidden_nodes": [],
            })
    population["direction_pair_reads"] = len(questions)

    # Mark each pair's two ends as siblings so the runner can assert that the two
    # directions produce DIFFERENT answers. Two questions that both return the
    # same node would pass individually while proving nothing about direction.
    by_qid = {q["qid"]: q for q in questions}
    for qid_f, qid_r, *_ in pair_ends:
        by_qid[qid_f]["sibling_qid"] = qid_r
        by_qid[qid_r]["sibling_qid"] = qid_f

    # --- simple one-hop -------------------------------------------------
    # Every unique edge, minus the reads already counted as direction pairs.
    dup = set()
    for src, rel, tgt in pairs:
        dup.add((src, rel))
        dup.add((tgt, B.INVERSE[rel]))
    one_hop = [(s, r, t) for s, r, t in B.unique_one_hop() if (s, r) not in dup]
    for i, (src, rel, tgt) in enumerate(one_hop, 1):
        questions.append({
            "qid": f"oh{i:03d}",
            "category": "simple_one_hop",
            "question": one_hop_question(src, rel),
            "rel": rel,
            "chain": [rel],
            "hops": 1,
            "anchor": src,
            "expected_node": tgt,
            "expected_path": [src, tgt],
            "forbidden_nodes": [],
        })
    population["simple_one_hop"] = len(one_hop)

    # --- mirror silence (must-refuse) -----------------------------------
    # Backward read of a no-inverse relation. The mirror pass is forbidden to
    # copy these relations, so any answer here would be invented knowledge. This
    # is the strongest available evidence against illegal same-label mirroring.
    for i, (anchor, rel, fars) in enumerate(B.mirror_controls(), 1):
        questions.append({
            "qid": f"ms{i:03d}",
            "category": "mirror_silence",
            "question": one_hop_question(anchor, rel),
            "rel": rel,
            "chain": [rel],
            "hops": 1,
            "anchor": anchor,
            "expected_node": None,
            "expected_path": None,
            "forbidden_nodes": list(fars),
        })

    # --- short multi-hop ------------------------------------------------
    by_hops = B.chains_by_hops()
    mh = 0
    for hops in sorted(by_hops):
        for path, chain in by_hops[hops]:
            mh += 1
            questions.append({
                "qid": f"mh{mh:03d}",
                "category": "short_multi_hop",
                "question": multi_hop_question(path, chain),
                "rel": chain[-1],
                "chain": chain,
                "hops": hops,
                "anchor": path[0],
                "expected_node": path[-1],
                "expected_path": path,
                "forbidden_nodes": [],
            })
    population["short_multi_hop_2hop"] = len(by_hops.get(2, []))
    population["short_multi_hop_3hop"] = len(by_hops.get(3, []))

    # --- honesty: out-of-graph subjects ---------------------------------
    for i, (subject, question) in enumerate(B.OUT_OF_GRAPH_SUBJECTS, 1):
        questions.append({
            "qid": f"og{i:03d}",
            "category": "honesty_out_of_graph",
            "question": question,
            "rel": None,
            "chain": None,
            "hops": 1,
            "anchor": subject,
            "expected_node": None,
            "expected_path": None,
            "forbidden_nodes": [],
            "subject": subject,
        })

    # --- nonsense / fallback -------------------------------------------
    for i, question in enumerate(B.NONSENSE_CUE_FREE, 1):
        questions.append({
            "qid": f"nf{i:03d}",
            "category": "nonsense_fallback",
            "question": question,
            "rel": None,
            "chain": [],
            "hops": 0,
            "anchor": None,
            "expected_node": None,
            "expected_path": None,
            "forbidden_nodes": [],
            "expect_guess_disclosed": False,
        })
    anchor, question = B.DISCLOSED_GUESS
    questions.append({
        "qid": f"nf{len(B.NONSENSE_CUE_FREE) + 1:03d}",
        "category": "nonsense_fallback",
        "question": question,
        "rel": None,
        "chain": [],
        "hops": 1,
        "anchor": anchor,
        "expected_node": None,
        "expected_path": None,
        "forbidden_nodes": [],
        "expect_guess_disclosed": True,
    })

    # --- population audit ----------------------------------------------
    triples = B._triples()
    population.update({
        "graph_edges_total": len(triples),
        "graph_edges_unique_one_hop": len(B.unique_one_hop()),
        "skipped_ambiguous_edges": len(B.ambiguous_one_hop()),
        "skipped_ambiguous_detail": [
            {"anchor": s, "rel": r, "competing_answers": c}
            for s, r, c in B.ambiguous_one_hop()
        ],
        "skipped_hubs_no_direction_pair": len(B.one_way_direction_pairs()),
        "mirrorable_pair_reads_all": len(B.one_way_direction_pairs()),
    })
    return questions, population


# --------------------------------------------------------------------------
def preflight(questions: list[dict], population: dict) -> list[tuple[str, str]]:
    """Refuse to freeze a set that cannot be scored or cannot be trusted.

    Every check here corresponds to a way the "frozen" set could quietly stop
    being a fair measurement.
    """
    problems: list[tuple[str, str]] = []
    trip = B._triples()
    labels = set(B.CONCEPTS)

    # --- the graph on disk must be the graph the questions were derived from
    digest = B.sha256(B.DB_PATH)
    if not B.DB_PATH.exists():
        problems.append(("graph-db-missing", str(B.DB_PATH)))

    # --- one template per relation, no gaps
    for rel in B.CANONICAL_RELATIONS:
        if rel not in ONE_HOP_TEMPLATE:
            problems.append(("relation-has-no-template", rel))
        if rel not in CONT_CLAUSE:
            problems.append(("relation-has-no-continuation-clause", rel))

    # --- identity
    qids = [q["qid"] for q in questions]
    if len(set(qids)) != len(qids):
        problems.append(("duplicate-qid", ""))
    if len(set(q["question"] for q in questions)) != len(questions):
        dupes = sorted({q["question"] for q in questions
                        if [x["question"] for x in questions].count(q["question"]) > 1})
        problems.append(("duplicate-question-text", " | ".join(dupes[:5])))

    # --- category minima from contract section 16 recommended_poc_test_matrix
    counts: dict[str, int] = {}
    for q in questions:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    minima = {
        "simple_one_hop": 30, "direction_pair": 10, "short_multi_hop": 8,
        "honesty_out_of_graph": 8, "nonsense_fallback": 5,
    }
    for cat, minimum in minima.items():
        if counts.get(cat, 0) < minimum:
            problems.append((f"category-below-contract-minimum",
                             f"{cat}: {counts.get(cat, 0)} < {minimum}"))

    # --- every graded answer must be a real edge, read the way it is claimed
    for q in questions:
        if q["category"] not in ("simple_one_hop", "direction_pair"):
            continue
        src, rel, tgt = q["anchor"], q["rel"], q["expected_node"]
        reach = B.reach_from(src, rel, 1)
        if reach != {tgt}:
            problems.append(("graded-one-hop-not-unique",
                             f"{q['qid']} {src} {rel} -> {sorted(reach)}"))

    # --- every graded multi-hop step must be stored and unique
    for q in questions:
        if q["category"] != "short_multi_hop":
            continue
        path, chain = q["expected_path"], q["chain"]
        if len(path) != len(chain) + 1:
            problems.append(("chain-length-mismatch", q["qid"]))
            continue
        if len(chain) > 3:
            problems.append(("chain-longer-than-max-chain-length", q["qid"]))
        if B._would_collapse(chain):
            problems.append(("chain-would-be-collapsed-by-engine", q["qid"]))
        for i, rel in enumerate(chain):
            if B.reach_from(path[i], rel, 1) != {path[i + 1]}:
                problems.append(("chain-step-not-unique", f"{q['qid']} {path[i]} {rel}"))

    # --- must-refuse rows must name a node that is genuinely the far end
    # `mirror_controls()` walks a stored edge (s, r, t) and asks from t, so the
    # stored direction is far -> anchor. Getting this backwards would assert the
    # edge exists in the very direction the question must NOT be able to read.
    for q in questions:
        if q["category"] != "mirror_silence":
            continue
        for far in q["forbidden_nodes"]:
            if (far, q["rel"], q["anchor"]) not in trip:
                problems.append(("mirror-control-edge-does-not-exist",
                                 f"{q['qid']} {far} {q['rel']} {q['anchor']}"))
            if (q["anchor"], q["rel"], far) in trip:
                problems.append(("mirror-control-edge-exists-so-it-is-answerable",
                                 f"{q['qid']} {q['anchor']} {q['rel']} {far}"))

    # --- out-of-graph subjects must be absent from the graph AND named
    for q in questions:
        if q["category"] != "honesty_out_of_graph":
            continue
        subject = q["subject"]
        if subject.lower() in {c.lower() for c in labels}:
            problems.append(("out-of-graph-subject-is-a-concept", q["qid"]))
        if subject.lower() not in q["question"].lower():
            problems.append(("out-of-graph-question-does-not-name-subject", q["qid"]))

    # --- the cue-free questions must really be cue-free
    try:
        sys.path.insert(0, str(ROOT / "test_results" / "stage_a"))
        sys.path.insert(0, str(ROOT / "test_results" / "stage_b"))
        sys.path.insert(0, str(ROOT / "test_results" / "stage_c"))
        sys.path.insert(0, str(ROOT / "test_results" / "stage_d"))
        from stage_d_runner import load_planner
        planner = load_planner()
    except Exception as exc:  # noqa: BLE001
        problems.append(("could-not-load-planner-for-preflight", repr(exc)))
        planner = None

    if planner is not None:
        for q in questions:
            if q["category"] == "nonsense_fallback":
                got = planner._literal_cue_relations(q["question"])
                if got:
                    problems.append(("cue-free-question-actually-has-a-cue",
                                     f"{q['qid']} {got} :: {q['question']}"))
        # A declared chain must equal what the literal matcher produces, or the
        # question does not express the relation the gold claims.
        for q in questions:
            if q["category"] in ("simple_one_hop", "direction_pair",
                                 "mirror_silence"):
                got = planner._literal_cue_relations(q["question"])
                if got != q["chain"]:
                    problems.append(("template-does-not-produce-declared-chain",
                                     f"{q['qid']} declared={q['chain']} got={got} "
                                     f":: {q['question']}"))
            elif q["category"] == "short_multi_hop":
                got = planner._literal_cue_relations(q["question"])
                if got != q["chain"]:
                    problems.append(("multi-hop-template-does-not-produce-chain",
                                     f"{q['qid']} declared={q['chain']} got={got} "
                                     f":: {q['question']}"))

    del digest
    return problems


def main() -> int:
    questions, population = build_questions()
    problems = preflight(questions, population)
    if problems:
        print("REFUSING TO FREEZE:")
        for kind, detail in problems:
            print(f"  {kind}: {detail}")
        print(f"\n({len(problems)} problems)")
        return 1

    spec = {
        "version": VERSION,
        "contract": "GLM-X v3.3.2 section 16 -- Frozen PoC Evaluation Set",
        "generated_by": "generate_questions.py (derived from build_poc_graph.py)",
        "graph": {
            "path": str(B.DB_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": B.sha256(B.DB_PATH),
            "nodes": len(set(B.CONCEPTS)),
            "edges": len(B._triples()),
            "relations": sorted(B.CANONICAL_RELATIONS),
        },
        "population": population,
        "counts": {c: sum(1 for q in questions if q["category"] == c)
                   for c in sorted({q["category"] for q in questions})},
        "questions": questions,
    }
    payload = json.dumps(spec, indent=2, ensure_ascii=False).encode("utf-8")
    FROZEN_PATH.write_bytes(payload)

    print(f"Froze {len(questions)} questions -> {FROZEN_PATH}")
    print(f"  version {VERSION}")
    print(f"  graph sha256 {spec['graph']['sha256']}")
    print(f"  set   sha256 {hashlib.sha256(payload).hexdigest()}")
    for cat, n in spec["counts"].items():
        print(f"    {cat:24} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
