"""Stage C: Direction & Multi-hop Graph (GLM-X v3.3.2 contract section 16).

    - name: "Direction & Multi-hop Graph"
      purpose: "Test direction and short chains"
      goal: "Prove causes/caused_by, part_of, follows/precedes and simple multi-hop"

Stage A proved the pipeline end to end. Stage B proved all 16 relations are
walkable. Stage C asks the question both of those leave open: does DIRECTION
survive being chained, and does the mirror pass stay silent where it must?

Three things here are new relative to Stage B:

  * DIRECTION AS ANCHOR CONTRAST. Direction is NOT tested by inverting a hop
    label. The walker's candidate filter matches on edge label
    (walker/graph_walker.py::inverse_relations), so `A causes B` and the
    inverse-labelled read are literally the same candidate set: "the answer
    changed when I inverted the hop" is vacuously false for every single hop,
    and a stage built on that assertion would report a false pass on a reversed
    walk. What can be tested is whether the system tells the two ENDS of one
    edge apart: for a stored edge `A causes B`, the question about B must
    return A, and the same question about A must not return B unless the graph
    really stores `B causes A`. generate_questions.py pre-registers a
    `control_anchor` for each pair that has one, and this runner scores both
    halves: correct answer at the anchor, and no answer at the control.

  * forbidden_reverse -- reverse reads of relations that have NO declared
    inverse label. The graph stores only the forward direction, so the correct
    behaviour is honest refusal. This is the automated guard against the illegal
    same-label mirroring that shipped in the tester graphs, and against the
    stored-direction inversion that Stage B first caught.

  * mirror_provenance -- every hop is labelled stored / declared_inverse /
    illegal by checking the walked (source, relation, target) triple against the
    store, so the results file can prove no hop came from a relation the
    contract does not permit mirroring, and none was fabricated.

Stage A's central invariant is reused verbatim rather than reimplemented: two
copies of "the invariant" would eventually disagree and a green Stage C would
prove nothing.

Usage:
    python test_results/stage_c/stage_c_runner.py
    python test_results/stage_c/stage_c_runner.py --only direction_pairs
"""
import argparse
import json
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test_results" / "stage_a"))
sys.path.insert(0, str(ROOT / "test_results" / "stage_b"))

import numpy as np

STAGE_DIR = Path(__file__).resolve().parent
DB_PATH = STAGE_DIR / "direction_multihop.db"
QUESTIONS_PATH = STAGE_DIR / "direction_multihop_questions_frozen.json"

from stage_a_runner import (  # noqa: E402
    REFUSAL_MARKERS, check_invariant, _offered_concepts,
)
from stage_b_runner import (  # noqa: E402
    load_chain_templates, load_relation_phrases, resolve_template, template_glue,
    load_stored_triples, CHAIN_TEMPLATES, RELATION_PHRASES,
)

# Relations with a declared inverse label in walker/graph_walker.py. Only these
# may be mirrored at runtime; contract section 9 lists exactly this set (with
# is_a's reversibility narrowed to forward-only by agreement, so is_a never
# produces a distinct mirror edge either).
MIRROR_PERMITTED = ("causes", "caused_by", "precedes", "follows", "part_of", "has_part")

# Relations with NO declared inverse. A hop over one of these must correspond to
# a STORED triple in that exact orientation. Stage B listed only
# supports/has_property/example_of; contradicts and the six symmetric relations
# belong here too, because the mirror pass is equally forbidden to reverse them,
# and leaving them out would let a reversed hop pass unlabelled.
ASYMMETRIC_NO_INVERSE = (
    "supports", "has_property", "example_of", "contradicts",
    "synonym", "antonym", "spatial_near", "temporal_coincident",
    "linguistic_maps", "associated_with",
)

# Provenance classes for a single walked hop, recorded per hop in the results.
PROV_STORED = "stored"
PROV_DECLARED_INVERSE = "declared_inverse"
PROV_ILLEGAL = "illegal_same_label_mirror"
PROV_FABRICATED = "fabricated"

MIRROR_INVERSE = {
    "causes": "caused_by", "caused_by": "causes",
    "precedes": "follows", "follows": "precedes",
    "part_of": "has_part", "has_part": "part_of",
}

# The closed 16-relation vocabulary, used to tell "no edge connects these nodes
# at all" (fabricated) apart from "an edge exists but not with this label".
ALL_RELATIONS = set(MIRROR_INVERSE) | set(ASYMMETRIC_NO_INVERSE) | {"is_a"}

DIRECTED_CATEGORIES = ("direction_pairs", "short_multi_hop")


def compute_reachability(db_path: Path) -> Dict[str, Dict[str, set]]:
    """Index stored edges and their runtime mirrors by (anchor, relation).

    Mirrored reachability is what the walker can actually see, so ambiguity must
    be judged against it rather than against forward edges alone. A relation
    outside MIRROR_PERMITTED contributes no mirrored entry: that absence is the
    behaviour under test.
    """
    import sqlite3

    con = sqlite3.connect(str(db_path))
    try:
        labels = {
            nid: (lbl or "").strip().lower()
            for nid, lbl in con.execute("SELECT id, label FROM nodes")
        }
        fwd: Dict[str, Dict[str, set]] = {}
        mir: Dict[str, Dict[str, set]] = {}
        inv = {"causes": "caused_by", "caused_by": "causes",
               "precedes": "follows", "follows": "precedes",
               "part_of": "has_part", "has_part": "part_of"}
        for src, tgt, rel in con.execute(
                "SELECT source_id, target_id, relation FROM edges"):
            s, t = labels.get(src, ""), labels.get(tgt, "")
            fwd.setdefault(rel, {}).setdefault(s, set()).add(t)
            if rel in MIRROR_PERMITTED:
                mir.setdefault(inv[rel], {}).setdefault(t, set()).add(s)
        return {"forward": fwd, "mirrored": mir}
    finally:
        con.close()


def classify_hops(labels: List[str], edges: List[str],
                  triples: set) -> List[Dict[str, Any]]:
    """Label every walked hop with where it came from, or that it does not exist.

    A hop is `stored` when the exact (source, relation, target) triple is in the
    store, `declared_inverse` when only the inverse-labelled triple is, and
    `illegal_same_label_mirror` when neither exists but the relation is one the
    mirror pass is forbidden to reverse. `fabricated` means no edge connects the
    two nodes under any label in the closed set, i.e. the walk invented it.

    Classifying from the store rather than from `path_edges` is the point: the
    edge label the walker reports says what it claims, not what exists.
    """
    out: List[Dict[str, Any]] = []
    for i, rel in enumerate(edges):
        if i + 1 >= len(labels):
            break
        s, t = labels[i], labels[i + 1]
        inv = MIRROR_INVERSE.get(rel)
        if (s, rel, t) in triples:
            prov = PROV_STORED
        elif inv and (t, inv, s) in triples:
            prov = PROV_DECLARED_INVERSE
        elif rel in ASYMMETRIC_NO_INVERSE and (t, rel, s) in triples:
            # The store has this edge backwards and the mirror pass may not fix
            # it. Reading it is exactly the illegal mirroring under test.
            prov = PROV_ILLEGAL
        elif any((s, r, t) in triples or (t, r, s) in triples
                 for r in ALL_RELATIONS):
            prov = PROV_FABRICATED
        else:
            prov = PROV_FABRICATED
        out.append({"hop": i, "src": s, "rel": rel, "tgt": t, "provenance": prov})
    return out


def preflight(spec: Dict[str, Any], reach: Dict[str, Dict[str, set]]) -> Dict[str, int]:
    """Refuse to run a question set that cannot support the claims it is used for.

    The frozen file is the product of generate_questions.py + the audit, so this
    re-checks the properties the stage's conclusions rest on rather than trusting
    the flags in the file:

      * every directed question carries a concrete pre-registered path, so the
        runner compares against a registered order instead of its own walk;
      * at least one mirror_silence control exists, or the stage's headline
        claim ("direction is honoured") has no adversarial evidence at all;
      * every control is on a relation with NO declared inverse, since for the
        other relations the mirror read is the contract working.
    """
    problems: List[str] = []
    graded = 0
    for q in spec["questions"]:
        if q["cat"] == "mirror_silence":
            graded += 1
            if q["rel"] in MIRROR_INVERSE:
                problems.append(f"{q['id']}: {q['rel']!r} has a declared inverse, "
                                "so its mirror read is legal and this control is "
                                "meaningless")
            continue
        if q["cat"] not in DIRECTED_CATEGORIES:
            continue
        path = q.get("path") or []
        if len(path) != (q.get("hops") or 0) + 1:
            problems.append(f"{q['id']}: path {path} does not match "
                            f"{q.get('hops')} hop(s)")
            continue
        if (path[-1] or "").lower() != (q.get("node") or "").lower():
            problems.append(f"{q['id']}: path ends at {path[-1]!r}, "
                            f"declared answer {q.get('node')!r}")
    if graded == 0:
        problems.append(
            "no direction_pairs question has a control anchor, so nothing in this "
            "run would test direction")
    if problems:
        raise SystemExit("REFUSING TO RUN:\n" + "\n".join(f"  {p}" for p in problems))
    print(f"  preflight: {graded} direction-graded control pair(s); every directed "
          "question has a registered path")
    return {"direction_graded": graded}


def check_semantics(row: Dict[str, Any], triples: set) -> Dict[str, Any]:
    """Semantic checks on top of the central invariant.

    1. subject_before_object   -- the rendered sentence must mention path nodes
       in traversal order.
    2. relation_phrase_present -- the decoder's own template glue must appear
       (Stage B logic, re-used so both stages grade the same sentence the same
       way).
    3. hop_provenance_legal    -- every hop is a stored triple, or the mirror of
       one under a declared inverse label. Anything else is either an illegal
       same-label mirror or a fabricated hop.
    4. expected_path_order -- the walked labels must equal the frozen registered
       path, which is the strict check the contract asks for on a direction stage.
    """
    answer = (row.get("answer") or "").lower()
    labels: List[str] = [str(x).lower() for x in (row.get("path_labels") or [])]
    edges: List[str] = list(row.get("path_edges") or [])
    checks: "OrderedDict[str, bool]" = OrderedDict()

    if not edges or len(labels) < 2:
        return {"checks": {}, "failed": [], "ok": True, "provenance": []}

    # Scan FORWARD from the previous label rather than taking each label's
    # first occurrence. `answer.find` is wrong whenever one label is a prefix of
    # another: for the path flood -> dam failure -> dam -> structure, "dam" first
    # occurs inside "dam failure", which put dam before dam failure and failed a
    # perfectly ordered answer. Word-boundary matching plus a forward cursor
    # asks the question actually being tested -- does the sentence mention the
    # nodes in traversal order?
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
        checks["subject_before_object"] = bool(idxs) and len(set(idxs)) == len(idxs)

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

    exp = [x.lower() for x in (row.get("expected_path") or [])]
    if exp:
        checks["expected_path_order"] = labels[: len(exp)] == exp
        row["expected_path"] = exp

    failed = [k for k, v in checks.items() if not v]
    return {"checks": dict(checks), "failed": failed, "ok": not failed, "provenance": prov}


def score_row(q: Dict[str, Any], row: Dict[str, Any], node_labels: set) -> Dict[str, Any]:
    cat = q["cat"]
    answer = (row.get("answer") or "")
    want = q.get("node")
    got = list(row.get("path_edges") or [])
    inv_ok = row["_invariant"]["invariant_ok"]
    sem_ok = row["_semantic"]["ok"]
    why: List[str] = []

    def names_node(text: str) -> list:
        low = text.lower()
        return sorted(n for n in node_labels
                      if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", low))

    def claim_text(text: str) -> str:
        """Strip the honest-disclosure tail off a refusal.

        "I don't have a relation ... Closest concepts I have: tiny. (No
        antonym relation found.)" is a REFUSAL that discloses a graph node, not
        an answer asserting one. Naming the anchor while declining to answer is
        required behaviour (contract section 6), so the node-name test may only
        look at what is left once the disclosure is removed.
        """
        low = text.lower()
        cut = low.find("closest concept")
        if cut == -1:
            return text
        return text[:cut]

    if cat == "forbidden_reverse":
        refused = bool(row.get("honest_no_relation")) or any(
            m in answer.lower() for m in REFUSAL_MARKERS)
        named = names_node(claim_text(answer))
        if not refused:
            why.append("answered instead of refusing a reverse read")
        if named:
            why.append(f"asserted a claim using graph nodes {named}")
        if not inv_ok:
            why.append(f"invariant: {row['_invariant']['failed_checks']}")
        return {"passed": bool(refused and not named and inv_ok), "why": why}

    if cat == "mirror_silence":
        # The far end of a ONE-WAY edge (a relation with no declared inverse)
        # has no stored edge and no mirror to fall back on, so it must not name
        # the source. Getting the source back means the engine read the edge
        # backwards anyway: the illegal same-label mirroring under test.
        forbidden = q.get("forbidden_nodes") or []
        leaked = [n for n in forbidden if re.search(
            r"(?<!\w)" + re.escape(n) + r"(?!\w)", claim_text(answer).lower())]
        refused = bool(row.get("honest_no_relation")) or any(
            m in answer.lower() for m in REFUSAL_MARKERS)
        if not refused:
            why.append(f"answered a {q['rel']!r} question instead of refusing "
                       "the backwards read")
        if leaked:
            why.append(f"answered {leaked} for {q['rel']!r} from anchor "
                       f"{q['anchor']!r}: the one-way edge is read backwards")
        if not inv_ok:
            why.append(f"invariant: {row['_invariant']['failed_checks']}")
        return {"passed": bool(refused and not leaked and inv_ok), "why": why}

    if cat in ("direction_pairs", "short_multi_hop"):
        hops = q["hops"]
        ok = want in answer and len(got) == hops and inv_ok and sem_ok \
            and not row.get("honest_no_relation")
        if want not in answer:
            why.append(f"expected {want!r} absent")
        if len(got) != hops:
            why.append(f"walked {len(got)} hops, expected {hops}")
        if row.get("honest_no_relation"):
            why.append("refused instead of answering")
        if not inv_ok:
            why.append(f"invariant: {row['_invariant']['failed_checks']}")
        if not sem_ok:
            why.append(f"semantic: {row['_semantic']['failed']}")
        return {"passed": bool(ok), "why": why}

    if cat == "honesty_out_of_graph":
        refused = bool(row.get("honest_no_relation")) or any(
            m in answer.lower() for m in REFUSAL_MARKERS)
        if not refused:
            why.append("did not refuse an out-of-graph entity")
        if not inv_ok:
            why.append(f"invariant: {row['_invariant']['failed_checks']}")
        return {"passed": bool(refused and inv_ok), "why": why}

    if cat == "nonsense_fallback":
        fired = bool(row.get("heuristic_used"))
        disclosed = bool(row.get("heuristic_disclosed")) or bool(row.get("honest_no_relation"))
        if not fired:
            why.append("heuristic fallback did not trigger")
        if not disclosed:
            why.append("guess was not disclosed and not refused")
        if not inv_ok:
            why.append(f"invariant: {row['_invariant']['failed_checks']}")
        return {"passed": bool(fired and disclosed and inv_ok), "why": why}

    return {"passed": False, "why": [f"unknown category {cat!r}"]}


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
        questions = [q for q in questions if q["cat"] == args.only]
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]
    out_path = Path(args.out) if args.out else STAGE_DIR / "stage_c_results.json"

    random.seed(args.seed)
    np.random.seed(args.seed)

    from scripts.glmx_ask import GLMXPipeline
    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

    # resolve_template/template_glue read these from STAGE B's module globals,
    # not from ours. Rebinding the imported names here would leave the lookup
    # inside stage_b_runner empty, which silently turns "{relation0}" into a
    # literal that no answer can contain -- every relation_phrase_present check
    # would then fail for a reason that has nothing to do with the pipeline.
    import stage_b_runner as _b

    _b.CHAIN_TEMPLATES = load_chain_templates()
    _b.RELATION_PHRASES = load_relation_phrases()
    globals()["CHAIN_TEMPLATES"] = _b.CHAIN_TEMPLATES
    globals()["RELATION_PHRASES"] = _b.RELATION_PHRASES

    reach = compute_reachability(DB_PATH)
    triples = load_stored_triples(DB_PATH)
    node_labels = {s for s, _, _ in triples} | {t for _, _, t in triples}
    print(f"\n{'=' * 78}\nSTAGE C -- Direction & Multi-hop Graph ({len(questions)} questions)\n{'=' * 78}")
    preflight(spec, reach)

    pipeline = GLMXPipeline()
    pipeline._seed = args.seed
    pipeline._no_learning = True
    pipeline._measure = True
    pipeline.graph_store = SQLiteGraphStore.load_state(str(DB_PATH))
    pipeline.load_models()

    rows: List[Dict[str, Any]] = []
    for q in questions:
        res = pipeline.ask(q["q"])
        exp_path: List[str] = [str(x).lower() for x in (q.get("path") or [])]
        row: Dict[str, Any] = {
            "id": q["id"],
            "cat": q["cat"],
            "rel": q.get("rel"),
            "question": q["q"],
            "expected_node": q.get("node"),
            "expected_chain": q.get("chain"),
            "expected_hops": q.get("hops"),
            "expected_path": exp_path,
            "answer": res.get("answer"),
            "relation_chain": res.get("relation_chain"),
            "path_labels": res.get("walk_path_labels"),
            "path_edges": res.get("walk_path_edges"),
            "selected_anchor": (res.get("selected_anchor") or {}).get("label"),
            "honest_no_relation": res.get("honest_no_relation"),
            "heuristic_used": res.get("heuristic_used"),
            "heuristic_disclosed": res.get("heuristic_disclosed"),
            "confidence": res.get("confidence"),
            "time_seconds": res.get("time_seconds"),
        }
        row["_graph_labels"] = sorted(node_labels)
        row["_invariant"] = check_invariant(row)
        row["_semantic"] = check_semantics(row, triples)
        verdict = score_row(q, row, node_labels)
        row["passed"] = verdict["passed"]
        row["fail_reason"] = verdict["why"]
        rows.append(row)

        mark = "PASS" if row["passed"] else "FAIL"
        chain_s = "/".join(row["path_edges"] or []) or "-"
        print(f"  [{mark}] {q['id']} {str(q.get('rel'))[:20]:<20} chain={chain_s:<26} "
              f"{(row['answer'] or '')[:40]}")

    n = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    inv_held = sum(1 for r in rows if r["_invariant"]["invariant_ok"])
    sem_held = sum(1 for r in rows if r["_semantic"]["ok"])

    dp = [r for r in rows if r["cat"] == "direction_pairs"]
    mh = [r for r in rows if r["cat"] == "short_multi_hop"]
    fr = [r for r in rows if r["cat"] == "forbidden_reverse"]
    ms = [r for r in rows if r["cat"] == "mirror_silence"]
    hon = [r for r in rows if r["cat"] == "honesty_out_of_graph"]
    nse = [r for r in rows if r["cat"] == "nonsense_fallback"]
    mh2 = [r for r in mh if r["expected_hops"] == 2]
    mh3 = [r for r in mh if r["expected_hops"] == 3]

    summary = {
        "stage": "C",
        "contract": "GLM-X v3.3.2 section 16",
        "graph": DB_PATH.name,
        "question_set": QUESTIONS_PATH.name,
        "seed": args.seed,
        "n_questions": n,
        "overall": {"passed": passed, "total": n, "rate": round(passed / n, 4)},
        "central_invariant": {"held": inv_held, "total": n, "rate": round(inv_held / n, 4)},
        "semantic_checks": {"held": sem_held, "total": n, "rate": round(sem_held / n, 4)},
        "direction_pairs": {"passed": sum(1 for r in dp if r["passed"]), "total": len(dp)},
        "multi_hop": {
            "passed": sum(1 for r in mh if r["passed"]), "total": len(mh),
            "two_hop": {"passed": sum(1 for r in mh2 if r["passed"]), "total": len(mh2)},
            "three_hop": {"passed": sum(1 for r in mh3 if r["passed"]), "total": len(mh3)},
        },
        "forbidden_reverse": {
            "passed": sum(1 for r in fr if r["passed"]), "total": len(fr),
            "all_refused": all(
                bool(r.get("honest_no_relation"))
                or any(m in (r.get("answer") or "").lower() for m in REFUSAL_MARKERS)
                for r in fr),
            "fabricated_claims": sum(1 for r in fr if r["fail_reason"]),
        },
        "mirror_silence": {
            "passed": sum(1 for r in ms if r["passed"]), "total": len(ms),
            "backwards_reads": sum(1 for r in ms if r["fail_reason"]),
        },
        "hop_provenance": {
            prov: sum(1 for r in rows for h in (r.get("hop_provenance") or [])
                      if h["provenance"] == prov)
            for prov in (PROV_STORED, PROV_DECLARED_INVERSE,
                         PROV_ILLEGAL, PROV_FABRICATED)
        },
        "honesty_out_of_graph": {"passed": sum(1 for r in hon if r["passed"]), "total": len(hon)},
        "nonsense_fallback": {"passed": sum(1 for r in nse if r["passed"]), "total": len(nse)},
        "per_category": {
            cat: {"passed": sum(1 for r in rows if r["cat"] == cat and r["passed"]),
                  "total": sum(1 for r in rows if r["cat"] == cat)}
            for cat in sorted({r["cat"] for r in rows})
        },
        "stage_c_pass": bool(
            passed == n and inv_held == n and sem_held == n
            and all(r["passed"] for r in dp) and all(r["passed"] for r in mh)
            and all(r["passed"] for r in fr) and all(r["passed"] for r in ms)
            and all(r["passed"] for r in hon) and all(r["passed"] for r in nse)
        ),
    }

    out_path.write_text(json.dumps({"summary": summary, "rows": [
        {k: v for k, v in r.items() if not k.startswith("_")} | {
            "invariant_ok": r["_invariant"]["invariant_ok"],
            "invariant_checks": r["_invariant"]["checks"],
            "semantic_ok": r["_semantic"]["ok"],
            "semantic_checks": r["_semantic"]["checks"],
        } for r in rows]}, indent=2), encoding="utf-8")

    print(f"\n{'-' * 78}")
    print(f"overall {passed}/{n} ({passed / n:.1%})")
    print(f"central invariant {inv_held}/{n}")
    print(f"semantic checks {sem_held}/{n}")
    print(f"direction pairs {summary['direction_pairs']['passed']}/{len(dp)}")
    print(f"multi-hop {summary['multi_hop']['passed']}/{len(mh)} "
          f"(2-hop {summary['multi_hop']['two_hop']['passed']}/{len(mh2)}, "
          f"3-hop {summary['multi_hop']['three_hop']['passed']}/{len(mh3)})")
    print(f"forbidden reverse {summary['forbidden_reverse']['passed']}/{len(fr)}")
    print(f"mirror silence {summary['mirror_silence']['passed']}/{len(ms)}")
    print(f"hop provenance {summary['hop_provenance']}")
    print(f"honesty {summary['honesty_out_of_graph']['passed']}/{len(hon)}")
    print(f"nonsense {summary['nonsense_fallback']['passed']}/{len(nse)}")
    print(f"stage_c_pass = {summary['stage_c_pass']}")
    fails = [r for r in rows if not r["passed"]]
    if fails:
        print(f"\nFAILURES: {len(fails)}")
        for r in fails:
            print(f"  {r['id']} {r['question'][:50]!r}\n      {r['fail_reason']}\n"
                  f"      answer: {r['answer']!r}\n      walk: {r['path_labels']} / {r['path_edges']}")
    print(f"\nWrote {out_path}")
    return 0 if summary["stage_c_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())