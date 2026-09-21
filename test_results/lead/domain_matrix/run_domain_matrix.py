"""Phase 3 - cross-domain smoke matrix.

Runs a curated, forward-edge-verified question set against four domain DBs
(nature_weather_small, toy_eval, toy, food_bio_small) with 3 seeded
no-learning pipelines each. Reports, per domain:

  - pass rate (node-in-path / honest grading, same semantics as the medium probe)
  - anchored-sim AND margin histograms (W4 calibration inputs)
  - honest-trigger counts (by_entity / by_relation)
  - seed determinism (seed0/1/2 must be identical)

This is a DRIFT/HEALTH matrix: per-domain pass is informational, not a gate.
W4 thresholds (sim_floor/margin_min) are calibrated from the merged histograms
here, never to a single domain's score distribution.

    python test_results/lead/domain_matrix/run_domain_matrix.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

HERE = Path(__file__).resolve().parent

DOMAINS = {
    "nature_weather_small": {
        "db": ROOT / "test_results" / "tester-a" / "datasets" / "nature_weather_small.db",
        "questions": [
            {"id": "nw01", "q": "What does storm cause?", "node": "rain", "relation": "causes"},
            {"id": "nw02", "q": "What does lightning cause?", "node": "fire", "relation": "causes"},
            {"id": "nw03", "q": "What does fire cause?", "node": "smoke", "relation": "causes"},
            {"id": "nw04", "q": "What does cloud cause?", "node": "rain", "relation": "causes"},
            {"id": "nw05", "q": "What comes after summer?", "node": "autumn", "relation": "follows"},
            {"id": "nw06", "q": "What comes before summer?", "node": "spring", "relation": "precedes"},
            {"id": "nw07", "q": "What comes after spring?", "node": "summer", "relation": "follows"},
            {"id": "nw08", "q": "What does sun cause?", "node": "light", "relation": "causes"},
            {"id": "nw09", "q": "What is rain?", "honest": True, "relation": "is_a_absent"},
        ],
    },
    "toy_eval": {
        "db": ROOT / "test_results" / "lead" / "datasets" / "toy_eval.db",
        "questions": [
            {"id": "te01", "q": "What is a dog?", "node": "animal", "relation": "is_a"},
            {"id": "te02", "q": "What is the opposite of hot?", "node": "cold", "relation": "antonym"},
            {"id": "te03", "q": "What is the opposite of warm?", "node": "cool", "relation": "antonym"},
            {"id": "te04", "q": "What does rain cause?", "node": "flood", "relation": "causes"},
            {"id": "te05", "q": "What property does car have?", "node": "fast", "relation": "has_property"},
            {"id": "te06", "q": "What property does ice have?", "node": "cold", "relation": "has_property"},
            {"id": "te07", "q": "What property does water have?", "node": "liquid", "relation": "has_property"},
            {"id": "te08", "q": "What is the wheel a part of?", "node": "car", "relation": "part_of"},
            {"id": "te09", "q": "What does fire cause?", "node": "smoke", "relation": "causes"},
            {"id": "te10", "q": "What is a wheel?", "honest": True, "relation": "is_a_absent"},
        ],
    },
    "toy": {
        "db": ROOT / "test_results" / "lead" / "datasets" / "toy.db",
        "questions": [
            {"id": "ty01", "q": "What is the opposite of hot?", "node": "cold", "relation": "antonym"},
            {"id": "ty02", "q": "What does rain cause?", "node": "flood", "relation": "causes"},
            {"id": "ty03", "q": "What property does water have?", "node": "liquid", "relation": "has_property"},
            {"id": "ty04", "q": "What is a dog?", "node": "animal", "relation": "is_a"},
            {"id": "ty05", "q": "What is the wheel a part of?", "node": "car", "relation": "part_of"},
            {"id": "ty06", "q": "What property does ice have?", "node": "cold", "relation": "has_property"},
            {"id": "ty07", "q": "What is a wheel?", "honest": True, "relation": "is_a_absent"},
        ],
    },
    "food_bio_small": {
        "db": ROOT / "test_results" / "tester-b" / "datasets" / "food_bio_small.db",
        "questions": [
            {"id": "fb01", "q": "What is a dog?", "node": "mammal", "relation": "is_a"},
            {"id": "fb02", "q": "What is an apple?", "node": "fruit", "relation": "is_a"},
            {"id": "fb03", "q": "What is Rice?", "node": "grain", "relation": "is_a_case_dup"},
            {"id": "fb04", "q": "What is the yolk a part of?", "node": "egg", "relation": "part_of"},
            {"id": "fb05", "q": "What is the petal a part of?", "node": "flower", "relation": "part_of"},
            {"id": "fb06", "q": "What property does honey have?", "node": "sweet", "relation": "has_property"},
            {"id": "fb07", "q": "What property does lemon have?", "node": "sour", "relation": "has_property"},
            {"id": "fb08", "q": "What property does chili have?", "node": "spicy", "relation": "has_property"},
            {"id": "fb09", "q": "What is the opposite of hot?", "node": "cold", "relation": "antonym"},
            {"id": "fb10", "q": "What is another word for happy?", "node": "glad", "relation": "synonym"},
            {"id": "fb11", "q": "What is another word for small?", "node": "little", "relation": "synonym"},
            {"id": "fb12", "q": "What is photosynthesis?", "honest": True, "relation": "is_a_absent"},
        ],
    },
}

SEEDS = [0, 1, 2]


def make_pipeline(db: Path, seed: int | None):
    p = GLMXPipeline()
    p.graph_store = SQLiteGraphStore.load_state(str(db))
    p._seed = seed
    p._no_learning = True
    p.load_models()
    return p


def grade(g: dict, r: dict) -> tuple[bool, list[str]]:
    reasons = []
    if g.get("honest"):
        ok = bool(r.get("honest_no_relation")) and bool(r.get("template_matched"))
        if not ok:
            reasons.append(
                f"honest_no_relation={r.get('honest_no_relation')} "
                f"template_matched={r.get('template_matched')} answer={r.get('answer')!r}"
            )
        return ok, reasons
    path = [str(x) for x in (r.get("walk_path_labels") or [])]
    answer = str(r.get("answer") or "")
    node_hit = g["node"] in path[1:] or g["node"].lower() in answer.lower()
    if not node_hit:
        reasons.append(f"node {g['node']!r} not in path {path} / answer {answer!r}")
    if r.get("honest_no_relation"):
        reasons.append("honest_no_relation unexpectedly True")
    return node_hit and not r.get("honest_no_relation"), reasons


def main() -> None:
    summary = {}
    report_rows = []
    line_chunks = []
    all_rows = {}

    for domain, spec in DOMAINS.items():
        db = spec["db"]
        questions = spec["questions"]
        pipelines = {s: make_pipeline(db, s) for s in SEEDS}

        rows = {}
        for s, pl in pipelines.items():
            rrows = {}
            for g in questions:
                r = pl.ask(g["q"])
                rrows[g["id"]] = {
                    "answer": str(r.get("answer")),
                    "chain": list(r.get("relation_chain") or []),
                    "hops": len(r.get("walk_path_edges") or []),
                    "path": [str(x) for x in (r.get("walk_path_labels") or [])],
                    "honest": bool(r.get("honest_no_relation")),
                    "honest_no_relation": bool(r.get("honest_no_relation")),
                    "honest_by_entity": bool(r.get("honest_by_entity")),
                    "honest_by_relation": bool(r.get("honest_by_relation")),
                    "template_matched": bool(r.get("template_matched")),
                    "entity_top_sim": float(r.get("entity_top_sim") or 0.0),
                    "anchor_margin": r.get("anchor_margin"),
                }
            rows[s] = rrows
        all_rows[domain] = rows

        # grade seeds
        fails = []
        pass_count = 0
        for g in questions:
            outcomes = [grade(g, rows[s][g["id"]])[0] for s in SEEDS]
            ok = all(outcomes)
            pass_count += int(ok)
            if not ok:
                fails.append(g["id"])

        # determinism (answers + chains identical across seeds)
        det_diff = []
        for g in questions:
            gid = g["id"]
            answers = {s: rows[s][gid]["answer"] for s in SEEDS}
            chains = {s: tuple(rows[s][gid]["chain"]) for s in SEEDS}
            if len(set(answers.values())) != 1 or len(set(chains.values())) != 1:
                det_diff.append(gid)

        # sim/margin histograms + honest triggers (seed0 reference)
        sims = [r["entity_top_sim"] for r in rows[0].values()]
        margins = [r["anchor_margin"] for r in rows[0].values() if r["anchor_margin"] is not None]
        honest_triggers = Counter()
        for r in rows[0].values():
            if r["honest"]:
                if r["honest_by_entity"]:
                    honest_triggers["by_entity"] += 1
                elif r["honest_by_relation"]:
                    honest_triggers["by_relation"] += 1
                else:
                    honest_triggers["no_path"] += 1

        rate = pass_count / len(questions) * 100
        summary[domain] = {
            "questions": len(questions),
            "pass": pass_count,
            "rate_pct": round(rate, 1),
            "fails": fails,
            "determinism_ok": not det_diff,
            "determinism_variants": det_diff,
            "sim_median": round(statistics.median(sims), 4),
            "sim_min": round(min(sims), 4),
            "sim_floor_hits": sum(1 for v in sims if v < 0.55),
            "margin_median": round(statistics.median(margins), 4) if margins else None,
            "margin_min_hits": sum(1 for v in margins if v < 0.04) if margins else None,
            "honest_triggers": dict(honest_triggers),
        }
        report_rows.append((domain, rate, len(questions), fails, not det_diff))

        line_chunks.append(
            f"\n{'='*72}\nDOMAIN: {domain}  ({db.name}, {len(questions)} questions)\n"
            f"{'='*72}"
        )
        for g in questions:
            r = rows[0][g["id"]]
            st = "PASS" if all(grade(g, rows[s][g["id"]])[0] for s in SEEDS) else "FAIL"
            line_chunks.append(
                f"[{st}] {g['id']} {g['q']}\n"
                f"     ans={r['answer']!r} chain={r['chain']} hop={r['hops']} "
                f"hon={r['honest']} sim={r['entity_top_sim']:.3f} "
                f"margin={'-' if r['anchor_margin'] is None else round(r['anchor_margin'],4)}"
            )
        line_chunks.append(
            f"-> pass {pass_count}/{len(questions)} ({rate:.1f}%)  fails={fails or '-'}\n"
            f"-> determinism (seed0/1/2): "
            f"{'IDENTICAL' if not det_diff else 'VARIANTS: ' + ','.join(det_diff)}\n"
            f"-> sim median {summary[domain]['sim_median']} "
            f"(<0.55 floor hits: {summary[domain]['sim_floor_hits']})\n"
            f"-> margin median {summary[domain]['margin_median']} "
            f"(<0.04 margin hits: {summary[domain]['margin_min_hits']})\n"
            f"-> honest trigger breakdown: {summary[domain]['honest_triggers']}"
        )

    merge_sims = [v for d in summary.values() for v in [d["sim_median"]]]
    merge_margins = [v for d in summary.values() if d["margin_median"] is not None
                     for v in [d["margin_median"]]]
    line_chunks.append("\n" + "=" * 72)
    line_chunks.append("W4 CALIBRATION NOTE (informational - thresholds NOT auto-changed)")
    line_chunks.append("=" * 72)
    line_chunks.append(
        f"defaults sim_floor=0.55 margin_min=0.04 vs per-domain medians: "
        f"sim {merge_sims}, margin {merge_margins}"
    )
    line_chunks.append(
        "if any domain's non-honest anchors sit below the floor, calibrate "
        "per-domain _honesty_defaults; a drift signal, not a global gate."
    )

    text = "\n".join(line_chunks)
    print(text)

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "domain_matrix.txt").write_text(text + "\n", encoding="utf-8")
    (HERE / "domain_matrix.json").write_text(
        json.dumps({"summary": summary, "domains": DOMAINS, "rows": all_rows},
            indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved {HERE / 'domain_matrix.txt'} and {HERE / 'domain_matrix.json'}")


if __name__ == "__main__":
    main()