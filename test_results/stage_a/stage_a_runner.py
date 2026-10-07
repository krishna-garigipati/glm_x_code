"""Stage A: prove the full GLM-X pipeline end-to-end on a clean Toy Graph.

This is the contract's first formal validation layer (section 15 / 16):

    "Prove the full pipeline works end-to-end: clean graph, correct answer,
     correct provenance, honest failure. A toy test set only."

Unlike the existing 90/99 diagnostic golden run, this runner

  * scores against a FROZEN question set whose per-category counts are fixed
    in toy_questions_frozen.json,
  * checks every mechanism target from section 16 separately,
  * and, most importantly, verifies the CENTRAL INVARIANT for every single
    row: relation_chain == walked path_edges == rendered sentence.

The invariant is the whole point of the architecture. A wrong answer that
has correct provenance is a decoder bug; a correct answer with no provenance
is a fabrication. Both fail here.

Usage:
    python test_results/stage_a/stage_a_runner.py
    python test_results/stage_a/stage_a_runner.py --out results.json
"""
import argparse
import json
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

STAGE_DIR = Path(__file__).resolve().parent
DB_PATH = STAGE_DIR / "toy_graph.db"
QUESTIONS_PATH = STAGE_DIR / "toy_questions_frozen.json"

# Phrases the contract allows a refusal to use (section 6 honesty behaviour).
REFUSAL_MARKERS = (
    "don't have", "do not have", "don't know", "do not know",
    "not in my", "no relationship", "no relation",
)
DISCLOSURE_MARKER = "heuristic guess"

# The honesty gate offers the nearest concepts it does hold, e.g.
#   "Closest concepts I have: sweet, honey. (No has_property relation found.)"
_CLOSEST_CONCEPTS = re.compile(
    r"closest concepts i have:\s*(.*?)(?:\.\s*\(|\.$|\Z)", re.IGNORECASE | re.DOTALL
)


def _offered_concepts(answer: str) -> List[str]:
    """Concepts an honesty refusal offers the user."""
    m = _CLOSEST_CONCEPTS.search(answer or "")
    if not m:
        return []
    raw = m.group(1)
    return [c.strip(" .").lower() for c in re.split(r",| and ", raw) if c.strip(" .")]


# --------------------------------------------------------------------------
# CENTRAL INVARIANT
# --------------------------------------------------------------------------
def check_invariant(row: Dict[str, Any]) -> Dict[str, Any]:
    """Verify relation_chain ~= walked path_edges ~= rendered sentence.

    Returns a verdict dict. Every check is mechanically decidable from the
    trace, so this can never pass by vibes.
    """
    chain: List[str] = list(row.get("relation_chain") or [])
    edges: List[str] = list(row.get("path_edges") or [])
    answer: str = (row.get("answer") or "").lower()
    path_labels: List[str] = list(row.get("path_labels") or [])
    honest = bool(row.get("honest_no_relation"))
    heuristic = bool(row.get("heuristic_used"))
    disclosed = bool(row.get("heuristic_disclosed"))

    checks: "OrderedDict[str, bool]" = OrderedDict()

    # 1. CHAIN ~= WALK. The walk must follow the plan exactly. A walk that
    #    deviates from the chain means the answer was produced by a relation
    #    nobody asked for.
    #
    #    Inverse-tolerance, and why it is required rather than optional: the
    #    walker treats a relation and its declared inverse as labels for the SAME
    #    directed edge (walker/graph_walker.py:inverse_relations). "What causes
    #    thunder?" asks for `causes`, but the only edge leaving the anchor thunder
    #    is the mirrored reverse of the stored `lightning causes thunder`, and that
    #    mirrored edge is labelled `caused_by`. The walk is therefore correct while
    #    reporting the inverse label. Demanding a byte-identical label would make
    #    that question unaskable whenever the mirror is what reaches the answer,
    #    which is the common case for a correctly-anchored "what causes X?".
    #
    #    This is a reading of the contract's "relation_chain ~= path_edges", not a
    #    loosening of it: only the CLOSED inverse set is tolerated, and any row
    #    that relies on it is reported in `walk_used_inverse_label` so the
    #    relaxation stays visible instead of silent.
    if chain or edges:
        inverse_of = {
            "causes": "caused_by", "caused_by": "causes",
            "precedes": "follows", "follows": "precedes",
            "part_of": "has_part", "has_part": "part_of",
            "is_a": "is_a",
        }
        exact = list(chain[: len(edges)]) == list(edges)
        # Each hop must match EXACTLY or be that hop's declared inverse
        # independently of the others. The previous form required every hop to be
        # inverted at once, so a chain that mixes an exact hop with an inverted
        # one could never be tolerated -- which is exactly the shape the mirror
        # pass produces on a multi-hop walk (observed on Stage C mh68: planned
        # ['causes','part_of','is_a'], walked ['caused_by','part_of','is_a'],
        # where only hop 1 took the inverse edge). Widening this cannot turn a
        # previously-passing row into a failure, and any row that relies on it is
        # still reported in `walk_used_inverse_label`.
        tolerated = (not exact) and len(chain) == len(edges) and all(
            c == e or inverse_of.get(c) == e for c, e in zip(chain, edges)
        )
        checks["chain_matches_walk"] = exact or tolerated
        row["walk_used_inverse_label"] = tolerated

    # 2. WALK ~= SENTENCE. Every node the walk reached must actually be said
    #    out loud. This is what stops the decoder from dropping a hop.
    if edges and path_labels:
        final = path_labels[-1].lower()
        checks["final_node_stated"] = final in answer

    # 3. NO UNWALKED CLAIM. The sentence must not introduce a graph label that
    #    the walk never reached -- the classic silent-hallucination failure.
    #    Matched on word boundaries: a bare substring test fires on ordinary
    #    words ("sea" inside "seaside") and reports fiction that never happened.
    #
    #    Subsumption also has to be handled, and this is a real failure, not a
    #    hypothetical one. `ice` and `ice mass` are both graph labels, and the
    #    walk reached `ice mass`; the answer "glacier is a type of ice mass."
    #    therefore mentions `ice` only as a piece of a label it did reach. A
    #    word-boundary match still fires on that piece and accuses the sentence
    #    of asserting an unwalked relation. So a label that occurs INSIDE a
    #    reached label is subsumed by it and is not a leak. Longer labels are
    #    tested first so that `ice mass` is credited before `ice` is judged.
    if edges and path_labels:
        reached = {lbl.lower() for lbl in path_labels}
        # Longest first: a reached label must be able to shield a shorter
        # label nested inside it, so subsumption is decided on the most
        # specific label that actually appears in the answer.
        shields = sorted(
            (lbl for lbl in reached if lbl in answer), key=len, reverse=True
        )
        leaked = [
            lbl for lbl in row.get("_graph_labels", [])
            if lbl.lower() not in reached
            and re.search(r"\b" + re.escape(lbl.lower()) + r"\b", answer)
            and not any(lbl.lower() in shield for shield in shields)
        ]
        checks["no_unwalked_claim"] = not leaked
        row["leaked_labels"] = leaked

    # 4. HONESTY IS CLEAN. A refusal must say so explicitly, and every concept
    #    it offers the user must be a concept that really exists in the graph.
    #
    #    An earlier version of this check also required the walk to be empty.
    #    That was wrong, and it is worth recording why: 14 of the 15 honest
    #    rows here DO walk nothing, but nf02 ("Tell me about sugar", a node
    #    absent from the graph) traverses sweet -> honey first and then refuses,
    #    because the traversal is what produces the "Closest concepts I have:
    #    sweet, honey" disclosure. Contract section 6 wants exactly that
    #    behaviour, so a non-empty path on a refusal is not by itself a
    #    violation -- presenting that path as the ANSWER would be. The property
    #    that actually matters, and that a dishonest system would fail, is
    #    offering the user a concept that is not in the graph.
    if honest:
        checks["refusal_is_explicit"] = any(m in answer for m in REFUSAL_MARKERS)
        offered = _offered_concepts(answer)
        known = {lbl.lower() for lbl in row.get("_graph_labels", [])}
        fabricated = [c for c in offered if c and c not in known]
        checks["refusal_offers_no_fabricated_concept"] = not fabricated
        row["fabricated_concepts"] = fabricated

    # 5. HEURISTICS ARE DISCLOSED. A guess must never be passed off as a walk.
    if heuristic:
        checks["guess_disclosed"] = bool(disclosed) or honest

    verdict = all(checks.values()) if checks else False
    return {
        "invariant_ok": verdict,
        "checks": dict(checks),
        "failed_checks": [k for k, v in checks.items() if not v],
    }


# --------------------------------------------------------------------------
# SCORING
# --------------------------------------------------------------------------
def score_row(q: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    """Decide pass/fail for one question, per its category's rule."""
    cat = q["cat"]
    answer = (r.get("answer") or "").lower()
    invariant_ok = r["_invariant"]["invariant_ok"]

    if cat in ("one_hop", "direction_pairs", "short_multi_hop"):
        want = (q.get("node") or "").lower()
        hops = int(q.get("hops", 1))
        got_chain = list(r.get("relation_chain") or [])
        got_edges = list(r.get("path_edges") or [])
        ok = (
            want in answer
            and not r.get("honest_no_relation")
            and len(got_edges) == hops
            and list(got_chain[: hops]) == got_edges
            and invariant_ok
        )
        reason = []
        if want not in answer:
            reason.append(f"expected target {want!r} absent")
        if r.get("honest_no_relation"):
            reason.append("refused instead of answering")
        if len(got_edges) != hops:
            reason.append(f"walked {len(got_edges)} hops, expected {hops}")
        if got_chain and list(got_chain[: hops]) != got_edges:
            reason.append(f"chain {got_chain} does not match walk {got_edges}")
        if not invariant_ok:
            reason.append(f"invariant: {r['_invariant']['failed_checks']}")
        return {"passed": bool(ok), "why": reason}

    if cat == "honesty_out_of_graph":
        refused = bool(r.get("honest_no_relation")) or any(m in answer for m in REFUSAL_MARKERS)
        no_claim = any(m in answer for m in REFUSAL_MARKERS)
        ok = refused and no_claim and invariant_ok
        reason = []
        if not refused:
            reason.append("did not refuse an out-of-graph entity")
        if not no_claim:
            reason.append("refused but still asserted a fact")
        if not invariant_ok:
            reason.append(f"invariant: {r['_invariant']['failed_checks']}")
        return {"passed": bool(ok), "why": reason}

    if cat == "nonsense_fallback":
        # Per contract section 8 the fallback MUST trigger, and any guess it
        # makes must be disclosed to the user.
        fired = bool(r.get("heuristic_used"))
        ok = fired and (bool(r.get("heuristic_disclosed")) or bool(r.get("honest_no_relation")))
        reason = []
        if not fired:
            reason.append("heuristic fallback did not trigger")
        if not (r.get("heuristic_disclosed") or r.get("honest_no_relation")):
            reason.append("guess was not disclosed and not refused")
        if not invariant_ok:
            reason.append(f"invariant: {r['_invariant']['failed_checks']}")
        return {"passed": bool(ok), "why": reason}

    return {"passed": False, "why": [f"unknown category {cat!r}"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Override the output path.")
    ap.add_argument("--only", default=None,
                    help="Run only one category (one_hop, direction_pairs, "
                         "short_multi_hop, honesty_out_of_graph, nonsense_fallback).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N questions (smoke test).")
    args = ap.parse_args()

    spec = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions: List[Dict[str, Any]] = spec["questions"]
    if args.only:
        questions = [q for q in questions if q["cat"] == args.only]
    if args.limit:
        questions = questions[: args.limit]
    out_path = Path(args.out) if args.out else STAGE_DIR / "stage_a_results.json"

    random.seed(args.seed)
    np.random.seed(args.seed)

    from scripts.glmx_ask import GLMXPipeline
    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

    pipeline = GLMXPipeline()
    pipeline._seed = args.seed
    pipeline._no_learning = True   # contract section 11: inference is frozen
    pipeline._measure = True
    pipeline.graph_store = SQLiteGraphStore.load_state(str(DB_PATH))
    pipeline.load_models()

    # Every label in the graph, for the "no unwalked claim" invariant check.
    # Use get_all_nodes(): this store numbers nodes from 1, so enumerating
    # range(get_node_count()) is off by one and reports the wrong label for the
    # last id. That bug made no_unwalked_claim fail spuriously on oh23/dp05/dp06.
    graph_labels = set()
    for node in pipeline.graph_store.get_all_nodes():
        if node.label:
            graph_labels.add(node.label)

    rows: List[Dict[str, Any]] = []
    print(f"\n{'=' * 78}\nSTAGE A -- Toy Graph end-to-end ({len(questions)} questions)\n{'=' * 78}")

    for q in questions:
        res = pipeline.ask(q["q"])
        row = {
            "id": q["id"],
            "cat": q["cat"],
            "question": q["q"],
            "expected_node": q.get("node"),
            "expected_chain": q.get("chain"),
            "expected_hops": q.get("hops"),
            "answer": res.get("answer"),
            "answer_level": res.get("answer_level"),
            "relation_chain": res.get("relation_chain"),
            "path_labels": res.get("walk_path_labels"),
            "path_edges": res.get("walk_path_edges"),
            "selected_anchor": (res.get("selected_anchor") or {}).get("label"),
            "entity_not_found": res.get("entity_not_found"),
            "entity_top_sim": res.get("entity_top_sim"),
            "honest_no_relation": res.get("honest_no_relation"),
            "honest_by_entity": res.get("honest_by_entity"),
            "honest_by_relation": res.get("honest_by_relation"),
            "heuristic_used": res.get("heuristic_used"),
            "heuristic_disclosed": res.get("heuristic_disclosed"),
            "template_matched": res.get("template_matched"),
            "chain_fulfilled": res.get("chain_fulfilled"),
            "confidence": res.get("confidence"),
            "walk_confidence": res.get("walk_confidence"),
            "time_seconds": res.get("time_seconds"),
            "_graph_labels": sorted(graph_labels),
        }
        row["_invariant"] = check_invariant(row)
        verdict = score_row(q, row)
        row["passed"] = verdict["passed"]
        row["fail_reason"] = verdict["why"]
        rows.append({k: v for k, v in row.items() if not k.startswith("_")} | {
            "invariant_ok": row["_invariant"]["invariant_ok"],
            "invariant_checks": row["_invariant"]["checks"],
        })

        mark = "PASS" if row["passed"] else "FAIL"
        chain_s = "/".join(row["path_edges"] or []) or "-"
        print(f"  [{mark}] {q['id']} {q['cat'][:14]:<14} chain={chain_s:<26} "
              f"{(row['answer'] or '')[:46]}")

    # ---- mechanism matrix (contract section 16) ----
    targets = spec["categories"]
    matrix: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for cat, meta in targets.items():
        if cat == "overall":
            continue
        sub = [r for r in rows if r["cat"] == cat]
        if not sub:
            matrix[cat] = {"n": 0, "passed": 0, "rate": 0.0, "target": meta,
                           "meets_target": False, "note": "no questions in this category"}
            continue
        npass = sum(1 for r in sub if r["passed"])
        rate = npass / len(sub)
        want = meta.get("target")
        if isinstance(want, float):
            meets = rate >= want and len(sub) >= meta.get("minimum", 0)
        else:
            meets = len(sub) >= meta.get("minimum", 0)
        matrix[cat] = {
            "n": len(sub), "passed": npass, "rate": round(rate, 4),
            "minimum": meta.get("minimum"), "target": want,
            "meets_target": bool(meets),
        }

    total_pass = sum(1 for r in rows if r["passed"])
    overall_rate = total_pass / len(rows) if rows else 0.0
    inv_pass = sum(1 for r in rows if r["invariant_ok"])

    print(f"\n{'-' * 78}\nMECHANISM MATRIX (contract section 16)\n{'-' * 78}")
    for cat, m in matrix.items():
        flag = "OK  " if m["meets_target"] else "MISS"
        tgt = m["target"]
        tgt_s = f"{tgt:.0%}" if isinstance(tgt, float) else str(tgt)
        print(f"  [{flag}] {cat:<22} {m['passed']:>3}/{m['n']:<3} "
              f"({m['rate']:>6.1%})  target>={tgt_s}  min={m.get('minimum')}")
    print(f"\n  overall: {total_pass}/{len(rows)} ({overall_rate:.1%}) "
          f"target>={targets['overall']['target']:.0%}")
    print(f"  central invariant held: {inv_pass}/{len(rows)} ({inv_pass / len(rows):.1%})")

    failures = [r for r in rows if not r["passed"]]
    if failures:
        print(f"\n{'-' * 78}\nFAILURES ({len(failures)})\n{'-' * 78}")
        for r in failures:
            print(f"  {r['id']} [{r['cat']}] {r['question']}")
            print(f"      answer: {r['answer']}")
            print(f"      reason: {r['fail_reason']}")

    all_ok = all(m["meets_target"] for m in matrix.values()) and \
        overall_rate >= targets["overall"]["target"]

    out = {
        "stage": "A",
        "contract": spec["contract"],
        "graph": str(DB_PATH),
        "question_set": str(QUESTIONS_PATH),
        "question_set_version": spec["version"],
        "n_questions": len(rows),
        "seed": args.seed,
        "learning_enabled": False,
        "overall": {
            "passed": total_pass, "total": len(rows), "rate": round(overall_rate, 4),
            "target": targets["overall"]["target"], "meets_target": bool(
                overall_rate >= targets["overall"]["target"]),
        },
        "central_invariant": {
            "held": inv_pass, "total": len(rows),
            "rate": round(inv_pass / len(rows), 4) if rows else 0.0,
            "checks": [
                "chain_matches_walk",
                "final_node_stated",
                "no_unwalked_claim",
                "refusal_is_explicit",
                "refusal_offers_no_fabricated_concept",
                "guess_disclosed",
            ],
        },
        "mechanism_matrix": matrix,
        "stage_a_pass": bool(all_ok),
        "rows": rows,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"STAGE A: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())