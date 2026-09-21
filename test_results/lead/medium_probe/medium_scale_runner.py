"""Medium-scale probe runner (Phase 5, medium level testing).

Runs the pre-registered expanded medium QA set (pre_registered_set.json) against
food_bio_medium deterministically:

  - 3 fresh pipelines (seed 0/1/2) + 1 unseeded pipeline, all --no-learning
  - every question answered by every pipeline (latency captured per call)
  - grading is node-only (or honest flag) exactly like unified_golden_runner.py
  - determinism verdict: answers must be byte-identical across the 3 seeded runs
  - within-pipeline stability: two probe questions re-asked in the same pipeline
  - unseeded pass is a diagnostic only (variance is NOT a failure)

Usage:
    python test_results/lead/medium_probe/medium_scale_runner.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

HERE = Path(__file__).resolve().parent
SET = HERE / "pre_registered_set.json"
DB = ROOT / "test_results" / "tester-b" / "datasets" / "food_bio_medium.db"
OUT = HERE / "medium_probe_results"

SEEDS = [0, 1, 2, None]  # None == unseeded (diagnostic)
REASK_STABILITY = ["mp16", "mp45"]  # same-pipeline re-run check


def make_pipeline(seed: int | None):
    p = GLMXPipeline()
    p.graph_store = SQLiteGraphStore.load_state(str(DB))
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
    data = json.loads(SET.read_text(encoding="utf-8"))
    questions = data["questions"]
    lines = []
    lines.append("=" * 78)
    lines.append("MEDIUM-SCALE PROBE  -  food_bio_medium (%d questions)" % len(questions))
    lines.append(f"dataset db: {DB}")
    lines.append("pre-registered set: pre_registered_set.json (frozen before any run)")
    lines.append("=" * 78)

    pipelines = {}
    seed_rows = {}
    for seed in SEEDS:
        tag = "unseeded" if seed is None else f"seed{seed}"
        lines.append(f"building pipeline: {tag} (no-learning) ...")
        pl = make_pipeline(seed)
        pipelines[tag] = pl

    # seed 0/1/2 + unseeded runs over every question
    latency = {tag: {} for tag in pipelines}
    rows = {}
    for tag, pl in pipelines.items():
        rrows = {}
        for g in questions:
            t0 = time.perf_counter()
            r = pl.ask(g["q"])
            latency[tag][g["id"]] = time.perf_counter() - t0
            rrows[g["id"]] = {
                "answer": str(r.get("answer")),
                "chain": list(r.get("relation_chain") or []),
                "hops": len(r.get("walk_path_edges") or []),
                "path": [str(x) for x in (r.get("walk_path_labels") or [])],
                "honest": bool(r.get("honest_no_relation")),
                "honest_no_relation": bool(r.get("honest_no_relation")),
                "template_matched": bool(r.get("template_matched")),
                "entity_not_found": bool(r.get("entity_not_found")),
                "entity_top_sim": float(r.get("entity_top_sim") or 0.0),
            }
        rows[tag] = rrows

    # same-pipeline stability re-ask
    stability = {}
    for gid in REASK_STABILITY:
        g = next(x for x in questions if x["id"] == gid)
        pl = pipelines["seed0"]
        r1 = pl.ask(g["q"])
        r2 = pl.ask(g["q"])
        stability[gid] = {
            "first": str(r1.get("answer")),
            "second": str(r2.get("answer")),
            "stable": str(r1.get("answer")) == str(r2.get("answer")),
        }

    # grade seeded runs; a q fails if ANY seeded run fails
    fails = []
    seeded_tags = [s for s in rows if s != "unseeded"]
    lines.append("\n" + "-" * 78)
    lines.append("GRADING (node-only / honest) - three seeded runs, PASS only if all agree")
    lines.append("-" * 78)
    pass_count = 0
    for g in questions:
        outcomes = {}
        all_ok = True
        for tag in seeded_tags:
            ok, reasons = grade(g, rows[tag][g["id"]])
            outcomes[tag] = ok
            all_ok = all_ok and ok
        pass_count += int(all_ok)
        status = "PASS" if all_ok else "FAIL"
        seed0 = rows["seed0"][g["id"]]
        sim = seed0.get("entity_top_sim", 0.0)
        pad = (g["id"], g["q"])
        lines.append(f"[{status}] {g['id']} {g['q']}")
        lines.append(
            f"      seed0: ans={seed0['answer']!r} chain={seed0['chain']} "
            f"hop={seed0['hops']} hon={seed0['honest']} sim={sim:.3f}"
        )
        if not all_ok:
            fails.append(g["id"])
            for tag in seeded_tags:
                ok2, reasons = grade(g, rows[tag][g["id"]])
                if not ok2:
                    lines.append(f"      {tag}: {'; '.join(reasons)}")
    lines.append(f"\nPASS {pass_count}/{len(questions)} across all 3 seeded runs")

    # within-pipeline stability
    lines.append("\n" + "-" * 78)
    lines.append("WITHIN-PIPELINE STABILITY (same seed0 pipeline, re-asked)")
    lines.append("-" * 78)
    for gid, s in stability.items():
        mark = "OK" if s["stable"] else "DRIFT"
        lines.append(f"[{mark}] {gid}: first={s['first']!r} second={s['second']!r}")

    # determinism across seeded pipelines
    lines.append("\n" + "-" * 78)
    lines.append("DETERMINISM ACROSS SEEDED PIPELINES (answers must be identical)")
    lines.append("-" * 78)
    det_ok = True
    for g in questions:
        gid = g["id"]
        answers = {tag: rows[tag][gid]["answer"] for tag in seeded_tags}
        chains = {tag: tuple(rows[tag][gid]["chain"]) for tag in seeded_tags}
        same = len(set(answers.values())) == 1 and len(set(chains.values())) == 1
        det_ok = det_ok and same
        if not same:
            lines.append(f"[VAR] {gid}: answers={answers}")
    lines.append("DETERMINISM: IDENTICAL across seed0/1/2" if det_ok else "DETERMINISM: VARIANTS DETECTED")

    # unseeded variance diagnostic
    lines.append("\n" + "-" * 78)
    lines.append("UNSEEDED VARIANCE DIAGNOSTIC (informational, not a failure)")
    lines.append("-" * 78)
    counted = 0
    for g in questions:
        gid = g["id"]
        a0 = rows["seed0"][gid]["answer"]
        au = rows["unseeded"][gid]["answer"]
        if a0 != au:
            counted += 1
            lines.append(f"  {gid}: seed0={a0!r} unseeded={au!r}")
    lines.append(f"  … {counted}/{len(questions)} questions differ between seed0 and unseeded")

    # latency + per-relation coverage
    lines.append("\n" + "-" * 78)
    lines.append("LATENCY (per question, seeded runs only)")
    lines.append("-" * 78)
    ts = [v for tag in seeded_tags for v in latency[tag].values()]
    lines.append(f"  total time all seeded runs: {sum(ts):.2f}s across {len(ts)} asks "
                 f"(== {len(ts) / sum(ts):.1f} asks/sec)")
    lines.append(f"  median {statistics.median(ts) * 1000:.1f} ms | P90 "
                 f"{statistics.quantiles(ts, n=10)[8] * 1000:.1f} ms | max {max(ts) * 1000:.1f} ms")
    lines.append(f"  load_models dominant cost amortized across {len(questions)} asks/pipeline")

    lines.append("\n" + "-" * 78)
    lines.append("COVERAGE PER RELATION (pre-registered fresh questions)")
    lines.append("-" * 78)
    from collections import defaultdict

    cov = defaultdict(list)
    for g in questions:
        if not g.get("honest"):
            cov[g["relation"]].append(g["id"])
    for rel, ids in sorted(cov.items()):
        ok_ids = [i for i in ids if i not in fails]
        lines.append(f"  {rel:>18}  {len(ok_ids)}/{len(ids)}  {', '.join(ids)}")

    # zero-crash assertion
    assert all(len(rows[t]) == len(questions) for t in rows)
    lines.append("\nZERO-CRASH: all questions returned a result in every pipeline")

    lines.append("\n" + "=" * 78)
    rate = pass_count / len(questions) * 100
    lines.append(f"OVERALL: {pass_count}/{len(questions)} ({rate:.1f}%) threshold 90% "
                 f"-> {'MET' if rate >= 90 else 'NOT MET'}")
    lines.append("=" * 78)

    text = "\n".join(lines)
    print(text)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(
        json.dumps({
            "questions": questions,
            "rows": rows,
            "latency": latency,
            "stability": stability,
            "pass_count": pass_count,
            "total": len(questions),
            "fails": fails,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved {OUT.with_suffix('.txt')} and {OUT.with_suffix('.json')}")


if __name__ == "__main__":
    main()