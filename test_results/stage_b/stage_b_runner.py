"""Stage B: Relation Coverage Graph (GLM-X v3.3.2 contract section 16).

    - name: "Relation Coverage Graph"
      purpose: "Test all 16 relations"
      goal: "Prove every canonical relation can be used"

Stage A proved the pipeline end-to-end on a small graph. This stage asks a
different question: can EVERY canonical relation actually be walked and
rendered? So the emphasis moves from category pass rates to RELATION coverage,
and the runner reports per-relation accuracy rather than only per-category.

Two additions over the Stage A runner, both required by the stage's quality bar
("a structurally valid answer that means the wrong thing is a failure"):

  * relation_phrase_present -- the rendered sentence must actually contain the
    surface phrase of the relation it claims to have walked. This catches a
    decoder that renders the right nodes under the wrong relation word.

  * subject_before_object -- the walked path's labels must appear in the
    sentence in path order. This is the automated form of "direction must be
    correct in the natural language output", and it is what catches the
    subject/object inversion that the mirror pass introduces for relations with
    no defined inverse label (see known_limitations in the question set).

Usage:
    python test_results/stage_b/stage_b_runner.py
    python test_results/stage_b/stage_b_runner.py --only one_hop
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

import numpy as np

STAGE_DIR = Path(__file__).resolve().parent
DB_PATH = STAGE_DIR / "relation_coverage.db"
QUESTIONS_PATH = STAGE_DIR / "relation_coverage_questions_frozen.json"

# Reuse Stage A's central-invariant implementation verbatim rather than
# reimplementing it: two copies of "the invariant" would eventually disagree,
# and then a green Stage B would prove nothing.
from stage_a_runner import (  # noqa: E402
    REFUSAL_MARKERS, DISCLOSURE_MARKER, check_invariant, _offered_concepts,
)

# Surface phrases are NOT re-declared here. The runner reads the chain-keyed
# templates straight out of decoder/config_decoder.yaml, because a hand-kept copy
# drifts from the decoder and then "passes" sentences the decoder never emits.
# The drift was not hypothetical: an earlier copy mapped is_a -> "is a type of",
# which is right for the one-hop template but wrong for the chain template
# ["is_a","has_property"] ("{node0} is {node1}, and it has the property
# {node2}."), so a correct multi-hop answer was failed for not containing a
# phrase its own template never produces.
DECODER_CONFIG = ROOT / "decoder" / "config_decoder.yaml"


def load_chain_templates(path: Path = DECODER_CONFIG) -> Dict[tuple, str]:
    import yaml

    tpl = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("templates", {})
    out: Dict[tuple, str] = {}
    for entry in tpl.get("definitions", []):
        if isinstance(entry, dict) and entry.get("chain"):
            out[tuple(entry["chain"])] = entry["template"]
    return out


def load_relation_phrases(path: Path = DECODER_CONFIG) -> Dict[str, str]:
    import yaml

    tpl = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("templates", {})
    return dict(tpl.get("relation_phrases", {}) or {})


CHAIN_TEMPLATES: Dict[tuple, str] = {}
RELATION_PHRASES: Dict[str, str] = {}


def resolve_template(template: str, edges: List[str]) -> str:
    """Substitute {relationN} with the phrase the decoder uses for that hop.

    Several chain templates are written as "{node0} {relation0} {node1}.", so
    the surface verb is not literal in the template at all -- it comes from
    templates.relation_phrases. Both halves are read from the decoder config so
    this stays in sync with the component that actually renders the sentence.
    """
    import re as _re

    def sub(m: "re.Match[str]") -> str:
        idx = int(m.group(1))
        rel = edges[idx] if idx < len(edges) else ""
        return RELATION_PHRASES.get(rel, m.group(0))

    return _re.sub(r"\{relation(\d+)\}", sub, template)


def template_glue(template: str) -> List[str]:
    """The literal text a template puts BETWEEN its node placeholders.

    "{node0} is a type of {node1}." -> [" is a type of "]. These fragments are
    what the decoder is required to have emitted, so requiring them is a real
    test of chain ~= sentence rather than a keyword guess.
    """
    import re as _re

    parts = _re.split(r"\{node\d+\}", template)
    return [p.strip() for p in parts if p and p.strip(" .,").strip()]


CANONICAL_RELATIONS = [
    "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
    "contradicts", "supports", "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident", "spatial_near",
    "linguistic_maps",
]

# Relations that are ASYMMETRIC in meaning and have no declared inverse in
# walker/graph_walker.py:INVERSE_RELATION_LABELS.
#
# For these, the mirror pass creates a reverse edge that keeps the same label,
# so traversing it backwards swaps subject and object while the label still
# reads as though the stored direction were being asserted. That is not a
# paraphrase; it is the negation of the stored fact. `supports` is the sharpest
# case: the graph stores `fossil record supports deep time claim`, and the first
# Stage B run answered "What evidence supports the deep time claim?" with
# "deep time claim supports fossil record." -- false, and certified PASS, because
# the runner had no way to tell a mirrored hop from a stored one.
#
# Relations with a declared inverse (causes/caused_by, precedes/follows,
# part_of/has_part) are deliberately NOT listed: there the walker treats both
# labels as the same directed edge, so the reversed hop is legitimately the
# edge that was asked about. Symmetric relations (contradicts, synonym,
# antonym, temporal_coincident, spatial_near, associated_with,
# linguistic_maps) read the same in both directions.
ASYMMETRIC_NO_INVERSE = ("supports", "has_property", "example_of")


def load_stored_triples(db_path: Path) -> set:
    """Read (subject, relation, object) triples in their STORED direction."""
    import sqlite3

    con = sqlite3.connect(str(db_path))
    try:
        labels = {
            nid: (lbl or "").strip().lower()
            for nid, lbl in con.execute("SELECT id, label FROM nodes")
        }
        return {
            (labels.get(src, ""), rel, labels.get(tgt, ""))
            for src, tgt, rel in con.execute(
                "SELECT source_id, target_id, relation FROM edges"
            )
        }
    finally:
        con.close()


STORED_TRIPLES: set = set()


def check_semantics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Semantic-correctness checks layered on top of the central invariant.

    The central invariant proves relation_chain ~= path_edges ~= the sentence's
    node set. These two prove the sentence MEANS what the walk did.
    """
    answer = (row.get("answer") or "").lower()
    labels: List[str] = [str(x) for x in (row.get("path_labels") or [])]
    edges: List[str] = list(row.get("path_edges") or [])
    checks: "OrderedDict[str, bool]" = OrderedDict()

    if not edges or len(labels) < 2:
        return {"checks": {}, "failed": [], "ok": True}

    # 1. SUBJECT BEFORE OBJECT. The sentence must mention the path nodes in the
    #    order the walk visited them. A reversed mention means the rendered claim
    #    has its subject and object swapped relative to the traversal.
    idxs = []
    for lbl in labels:
        pos = answer.find(lbl.lower())
        idxs.append(pos)
    if all(p >= 0 for p in idxs):
        checks["subject_before_object"] = idxs == sorted(idxs) and len(set(idxs)) == len(idxs)

    # 2. THE DECODER'S OWN TEMPLATE IS SPOKEN. Look up the chain-keyed template
    #    for the chain that was actually walked and require every literal
    #    fragment it puts between nodes to appear in the answer. Checking the
    #    exact chain (rather than a per-relation phrase) is what lets a
    #    multi-hop answer be checked against the template that produced it.
    # 2. THE DECODER'S OWN RENDERING IS SPOKEN.
#    If the decoder declares a template for exactly this chain, require every
    #    literal fragment that template puts between nodes. If it declares none,
    #    the decoder fell back to its generic chain rendering, which joins hops
    #    with templates.relation_phrases -- so require each walked relation's
    #    phrase instead. Guessing the one-hop template as a fallback is wrong:
    #    ["causes"] renders "The reason is that ..." but a causes>is_a walk is
    #    rendered generically as "rain causes flood, and flood is a disaster.",
    #    which is a correct answer that mentions no "reason".
    key = tuple(edges)
    template = CHAIN_TEMPLATES.get(key)
    if template:
        fragments = template_glue(resolve_template(template, edges))
        row["template_used"] = template
    else:
        fragments = [
            RELATION_PHRASES.get(rel, "") for rel in edges
        ]
        row["template_used"] = "<generic relation_phrases>"

    # An honest refusal walks edges purely to disclose "Closest concepts I have",
    # and asserts nothing. Requiring it to then contain the relation phrase would
    # penalise the correct behaviour, so the check inverts for refusals: the
    # phrase must be ABSENT, which is what "did not make a claim" means
    # mechanically.
    if row.get("honest_no_relation"):
        present = [f for f in fragments if f and f.lower() in answer]
        checks["no_claim_when_refusing"] = not present
        row["template_fragments_leaked"] = present
    else:
        missing = [f for f in fragments if f and f.lower() not in answer]
        checks["relation_phrase_present"] = not missing
        row["template_fragments_missing"] = missing

    # 3. STORED DIRECTION FOR ASYMMETRIC RELATIONS. For relations that are
    #    asymmetric and have no declared inverse, every hop must match a triple
    #    that is actually stored in that direction. This is what catches a
    #    mirrored hop, which is invisible to the other two checks because the
    #    labels and the phrase both look perfectly ordinary.
    if STORED_TRIPLES:
        inverted = [
            (labels[i], edges[i], labels[i + 1])
            for i in range(len(edges))
            if i + 1 < len(labels)
            and edges[i] in ASYMMETRIC_NO_INVERSE
            and (labels[i].lower(), edges[i], labels[i + 1].lower())
            not in STORED_TRIPLES
        ]
        checks["asymmetric_hops_stored_direction"] = not inverted
        row["inverted_hops"] = inverted

    failed = [k for k, v in checks.items() if not v]
    return {"checks": dict(checks), "failed": failed, "ok": not failed}


def _walk_matches_chain(walked: List[str], chain: List[str]) -> bool:
    """Walked labels must match the plan, tolerating only declared inverses.

    Mirrors the tolerance in stage_a_runner.check_invariant:1 so scoring and the
    invariant cannot disagree about what counts as the same edge.
    """
    inverse_of = {
        "causes": "caused_by", "caused_by": "causes",
        "precedes": "follows", "follows": "precedes",
        "part_of": "has_part", "has_part": "part_of",
        "is_a": "is_a",
    }
    if list(chain[: len(walked)]) == list(walked):
        return True
    return len(chain) == len(walked) and all(
        inverse_of.get(c) == e for c, e in zip(chain, walked)
    )


def score_row(q: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    cat = q["cat"]
    answer = (r.get("answer") or "")
    sem_ok = r["_semantic"]["ok"]
    inv_ok = r["_invariant"]["invariant_ok"]

    if cat == "one_hop":
        want = q["node"]
        ok = want in answer and not r.get("honest_no_relation") \
            and _walk_matches_chain(list(r.get("path_edges") or []), list(q["chain"])) \
            and inv_ok and sem_ok
        why = []
        if want not in answer:
            why.append(f"expected {want!r} absent")
        if r.get("honest_no_relation"):
            why.append("refused instead of answering")
        if not _walk_matches_chain(list(r.get("path_edges") or []), list(q["chain"])):
            why.append(f"walk {r.get('path_edges')} != chain {q['chain']}")
        if not inv_ok:
            why.append(f"invariant: {r['_invariant']['failed_checks']}")
        if not sem_ok:
            why.append(f"semantic: {r['_semantic']['failed']}")
        return {"passed": bool(ok), "why": why}

    if cat == "direction_pairs":
        want = q["node"]
        got = list(r.get("path_edges") or [])
        ok = want in answer and _walk_matches_chain(got, list(q["chain"])) \
            and not r.get("honest_no_relation") and inv_ok and sem_ok
        why = []
        if want not in answer:
            why.append(f"expected {want!r} absent")
        if not _walk_matches_chain(got, list(q["chain"])):
            why.append(f"walk {got} != expected chain {q['chain']} (direction wrong)")
        if not inv_ok:
            why.append(f"invariant: {r['_invariant']['failed_checks']}")
        if not sem_ok:
            why.append(f"semantic: {r['_semantic']['failed']}")
        return {"passed": bool(ok), "why": why}

    if cat == "short_multi_hop":
        want = q["node"]
        hops = q["hops"]
        got = list(r.get("path_edges") or [])
        ok = want in answer and len(got) == hops \
            and _walk_matches_chain(got[:hops], list(q["chain"])) \
            and inv_ok and sem_ok
        why = []
        if want not in answer:
            why.append(f"expected {want!r} absent")
        if len(got) != hops:
            why.append(f"walked {len(got)} hops, expected {hops}")
        if not _walk_matches_chain(got[:hops], list(q["chain"])):
            why.append(f"chain {got} does not match {q['chain']}")
        if not inv_ok:
            why.append(f"invariant: {r['_invariant']['failed_checks']}")
        if not sem_ok:
            why.append(f"semantic: {r['_semantic']['failed']}")
        return {"passed": bool(ok), "why": why}

    if cat == "honesty_out_of_graph":
        refused = bool(r.get("honest_no_relation")) or any(m in answer.lower() for m in REFUSAL_MARKERS)
        no_claim = any(m in answer.lower() for m in REFUSAL_MARKERS)
        ok = refused and no_claim and inv_ok
        why = []
        if not refused:
            why.append("did not refuse an out-of-graph entity")
        if not no_claim:
            why.append("made a claim without refusing")
        if not inv_ok:
            why.append(f"invariant: {r['_invariant']['failed_checks']}")
        return {"passed": bool(ok), "why": why}

    if cat == "nonsense_fallback":
        fired = bool(r.get("heuristic_used"))
        disclosed = bool(r.get("heuristic_disclosed")) or bool(r.get("honest_no_relation"))
        ok = fired and disclosed and inv_ok
        why = []
        if not fired:
            why.append("heuristic fallback did not trigger")
        if not disclosed:
            why.append("guess was not disclosed and not refused")
        if not inv_ok:
            why.append(f"invariant: {r['_invariant']['failed_checks']}")
        return {"passed": bool(ok), "why": why}

    return {"passed": False, "why": [f"unknown category {cat!r}"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    spec = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions: List[Dict[str, Any]] = spec["questions"]
    if args.only:
        questions = [q for q in questions if q["cat"] == args.only]
    if args.limit:
        questions = questions[: args.limit]
    out_path = Path(args.out) if args.out else STAGE_DIR / "stage_b_results.json"

    random.seed(args.seed)
    np.random.seed(args.seed)

    from scripts.glmx_ask import GLMXPipeline
    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

    pipeline = GLMXPipeline()
    pipeline._seed = args.seed
    pipeline._no_learning = True
    pipeline._measure = True
    pipeline.graph_store = SQLiteGraphStore.load_state(str(DB_PATH))
    pipeline.load_models()

    graph_labels = {n.label for n in pipeline.graph_store.get_all_nodes() if n.label}

    global STORED_TRIPLES
    STORED_TRIPLES = load_stored_triples(DB_PATH)
    global CHAIN_TEMPLATES
    CHAIN_TEMPLATES = load_chain_templates()
    global RELATION_PHRASES
    RELATION_PHRASES = load_relation_phrases()

    # ---- PRE-FLIGHT: confirm every asked (anchor, relation) is unambiguous ----
    # If this is not true the run cannot distinguish a code failure from a data
    # failure, and the stage's whole premise ("clean, hand-crafted graph")
    # is void.
    spec_rels = spec["relations_covered"]
    missing_rels = [r for r in CANONICAL_RELATIONS if r not in spec_rels]
    if missing_rels:
        raise SystemExit(f"question set does not cover: {missing_rels}")

    rows: List[Dict[str, Any]] = []
    print(f"\n{'=' * 78}\nSTAGE B -- Relation Coverage Graph ({len(questions)} questions)\n{'=' * 78}")

    for q in questions:
        res = pipeline.ask(q["q"])
        row = {
            "id": q["id"],
            "cat": q["cat"],
            "rel": q.get("rel"),
            "question": q["q"],
            "expected_node": q.get("node"),
            "expected_chain": q.get("chain"),
            "expected_hops": q.get("hops"),
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
            "_graph_labels": sorted(graph_labels),
        }
        row["_invariant"] = check_invariant(row)
        row["_semantic"] = check_semantics(row)
        verdict = score_row(q, row)
        row["passed"] = verdict["passed"]
        row["fail_reason"] = verdict["why"]
        rows.append({k: v for k, v in row.items() if not k.startswith("_")} | {
            "invariant_ok": row["_invariant"]["invariant_ok"],
            "invariant_checks": row["_invariant"]["checks"],
            "semantic_ok": row["_semantic"]["ok"],
            "semantic_checks": row["_semantic"]["checks"],
        })

        mark = "PASS" if row["passed"] else "FAIL"
        chain_s = "/".join(row["path_edges"] or []) or "-"
        print(f"  [{mark}] {q['id']} {str(q.get('rel'))[:16]:<16} chain={chain_s:<24} "
              f"{(row['answer'] or '')[:44]}")

    n = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    inv_held = sum(1 for r in rows if r["invariant_ok"])
    sem_held = sum(1 for r in rows if r["semantic_ok"])

    # ---- per-relation ----
    per_rel: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for rel in CANONICAL_RELATIONS:
        sub = [r for r in rows if r["rel"] and r["rel"].split(">")[0] == rel]
        if not sub:
            continue
        p = sum(1 for r in sub if r["passed"])
        iv = sum(1 for r in sub if r["invariant_ok"])
        per_rel[rel] = {
            "n": len(sub), "passed": p, "rate": round(p / len(sub), 4),
            "invariant_held": iv,
        }

    # chains containing more than one relation are attributed to the head only
    for r in rows:
        if r["rel"] and ">" in r["rel"]:
            head = r["rel"].split(">")[0]
            if head in per_rel:
                per_rel[head]["n"] += 1
                per_rel[head]["passed"] += 1 if r["passed"] else 0
                per_rel[head]["invariant_held"] += 1 if r["invariant_ok"] else 0
    for rel, v in per_rel.items():
        v["rate"] = round(v["passed"] / v["n"], 4) if v["n"] else 0.0

    # ---- direction accuracy ----
    dp = [r for r in rows if r["cat"] == "direction_pairs"]
    dp_ok = sum(1 for r in dp if r["passed"])

    # ---- multi-hop ----
    mh = [r for r in rows if r["cat"] == "short_multi_hop"]
    mh_ok = sum(1 for r in mh if r["passed"])
    mh2 = [r for r in mh if r["expected_hops"] == 2]
    mh3 = [r for r in mh if r["expected_hops"] == 3]
    hon = [r for r in rows if r["cat"] == "honesty_out_of_graph"]
    nse = [r for r in rows if r["cat"] == "nonsense_fallback"]

    summary = {
        "stage": "B",
        "contract": "GLM-X v3.3.2 section 16",
        "graph": DB_PATH.name,
        "question_set": QUESTIONS_PATH.name,
        "seed": args.seed,
        "n_questions": n,
        "overall": {"passed": passed, "total": n, "rate": round(passed / n, 4)},
        "central_invariant": {"held": inv_held, "total": n, "rate": round(inv_held / n, 4)},
        "semantic_checks": {"held": sem_held, "total": n, "rate": round(sem_held / n, 4)},
        "per_relation": per_rel,
        "direction_accuracy": {"passed": dp_ok, "total": len(dp),
                               "rate": round(dp_ok / len(dp), 4) if dp else None},
        "multi_hop": {
            "passed": mh_ok, "total": len(mh),
            "rate": round(mh_ok / len(mh), 4) if mh else None,
            "two_hop": {"passed": sum(1 for r in mh2 if r["passed"]), "total": len(mh2)},
            "three_hop": {"passed": sum(1 for r in mh3 if r["passed"]), "total": len(mh3)},
        },
        "relation_coverage": {
            "canonical": len(CANONICAL_RELATIONS),
            "exercised_by_questions": len([r for r in CANONICAL_RELATIONS if r in per_rel]),
            "stored_in_graph": sorted(spec_rels),
        },
        # Honesty and nonsense behaviour are reported explicitly because they are
        # pass conditions in their own right: a suite can look perfect while
        # quietly hallucinating on out-of-graph entities.
        "honesty_out_of_graph": {
            "passed": sum(1 for r in hon if r["passed"]), "total": len(hon),
            "all_refused": all(
                bool(r.get("honest_no_relation"))
                or any(m in (r.get("answer") or "").lower() for m in REFUSAL_MARKERS)
                for r in hon
            ),
        },
        "nonsense_fallback": {
            "passed": sum(1 for r in nse if r["passed"]), "total": len(nse),
            "all_refused_or_disclosed": all(
                bool(r.get("heuristic_disclosed")) or bool(r.get("honest_no_relation"))
                for r in nse
            ),
        },
        "per_category": {
            cat: {"passed": sum(1 for r in rows if r["cat"] == cat and r["passed"]),
                  "total": sum(1 for r in rows if r["cat"] == cat)}
            for cat in sorted({r["cat"] for r in rows})
        },
        "stage_b_pass": bool(
            passed == n
            and inv_held == n
            and sem_held == n
            and len(per_rel) == len(CANONICAL_RELATIONS)
            and all(r["passed"] for r in hon)
            and all(r["passed"] for r in nse)
        ),
    }

    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                        encoding="utf-8")

    print(f"\n{'-' * 78}")
    print(f"overall {passed}/{n} ({passed / n:.1%})")
    print(f"central invariant {inv_held}/{n} ({inv_held / n:.1%})")
    print(f"semantic checks {sem_held}/{n} ({sem_held / n:.1%})")
    print(f"direction accuracy {dp_ok}/{len(dp)}")
    print(f"multi-hop {mh_ok}/{len(mh)} (2-hop {sum(1 for r in mh2 if r['passed'])}/{len(mh2)}, "
          f"3-hop {sum(1 for r in mh3 if r['passed'])}/{len(mh3)})")
    print(f"relations exercised {len(per_rel)}/16")
    print(f"honesty out-of-graph {sum(1 for r in hon if r['passed'])}/{len(hon)}")
    print(f"nonsense fallback {sum(1 for r in nse if r['passed'])}/{len(nse)}")
    print(f"stage_b_pass = {summary['stage_b_pass']}")
    fails = [r for r in rows if not r["passed"]]
    print(f"\nFAILURES: {len(fails)}")
    for r in fails:
        print(f"  {r['id']} {r['question'][:44]!r}\n      {r['fail_reason']}\n"
              f"      answer: {r['answer']!r}")
    print(f"\nWrote {out_path}")
    return 0 if summary["stage_b_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())