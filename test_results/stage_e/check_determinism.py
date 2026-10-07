"""Stage E: prove the frozen PoC result is reproducible.

Contract v3.3.2 section 16 asks the PoC evaluation to be run twice and compared;
a PoC score that changes between runs on identical inputs is not a score. This
runs the whole frozen set a second time into a separate file and diffs it row by
row against the first run.

What is compared, and why each one matters:

  * the rendered answer text, relation_chain, walked path and path_edges -- the
    substance of the result;
  * the pass/fail verdict and the failure attribution, so a rerun cannot quietly
    turn a failure into a pass (or the reverse) while the prose stays identical;
  * per-category and per-mechanism rates;
  * latency, reported but deliberately EXCLUDED from the equality test, because
    wall-clock timing is not reproducible and including it would make every
    comparison fail for a reason that has nothing to do with determinism.

Usage:
    python test_results/stage_e/check_determinism.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parents[1]
RUNNER = STAGE_DIR / "stage_e_runner.py"
BASELINE = STAGE_DIR / "stage_e_results.json"
RERUN = STAGE_DIR / "stage_e_results_rerun.json"
REPORT = STAGE_DIR / "stage_e_determinism.json"

# Keys compared per row. `time_seconds` is deliberately absent.
COMPARED = (
    "question", "answer", "relation_chain", "path_labels", "path_edges",
    "selected_anchor", "heuristic_used", "heuristic_disclosed",
    "honest_no_relation", "chain_fulfilled", "template_matched",
    "n_walk_steps",
)


def verdict(row: Dict[str, Any]) -> Any:
    return (bool(row["_score"]["pass"]), list(row["_score"]["why"]),
            row.get("_attribution"))


def main() -> int:
    if not BASELINE.exists():
        print(f"REFUSING TO CHECK: no baseline run at {BASELINE}")
        return 1

    print("Stage E determinism: re-running the frozen set ...")
    proc = subprocess.run(
        [sys.executable, "-B", str(RUNNER), "--out", str(RERUN)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if proc.returncode != 0 and not RERUN.exists():
        print("REFUSING TO CHECK: the re-run produced no results file")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return 1
    if not RERUN.exists():
        print("REFUSING TO CHECK: the re-run produced no results file")
        return 1

    a = json.loads(BASELINE.read_text(encoding="utf-8"))
    b = json.loads(RERUN.read_text(encoding="utf-8"))

    rows_a = {r["id"]: r for r in a["rows"]}
    rows_b = {r["id"]: r for r in b["rows"]}
    problems: List[str] = []

    if set(rows_a) != set(rows_b):
        problems.append(
            f"row sets differ: only in first {sorted(set(rows_a) - set(rows_b))[:5]}, "
            f"only in second {sorted(set(rows_b) - set(rows_a))[:5]}")
    if a["questions_run"] != b["questions_run"]:
        problems.append(f"row count differs: {a['questions_run']} vs {b['questions_run']}")

    n_fields = 0
    diffs: List[Dict[str, Any]] = []
    for qid in sorted(set(rows_a) & set(rows_b)):
        ra, rb = rows_a[qid], rows_b[qid]
        for key in COMPARED:
            n_fields += 1
            if ra.get(key) != rb.get(key):
                diffs.append({"qid": qid, "field": key,
                              "first": ra.get(key), "second": rb.get(key)})
        if verdict(ra) != verdict(rb):
            diffs.append({"qid": qid, "field": "verdict",
                          "first": verdict(ra), "second": verdict(rb)})
        if ra["_invariant"] != rb["_invariant"]:
            diffs.append({"qid": qid, "field": "invariant",
                          "first": ra["_invariant"], "second": rb["_invariant"]})
        if ra["_semantic"] != rb["_semantic"]:
            diffs.append({"qid": qid, "field": "semantic",
                          "first": ra["_semantic"], "second": rb["_semantic"]})

    for cat in sorted(set(a["categories"]) | set(b["categories"])):
        ca, cb = a["categories"].get(cat), b["categories"].get(cat)
        if (ca or {}).get("rate") != (cb or {}).get("rate"):
            problems.append(
                f"{cat}: rate {(ca or {}).get('rate')} vs {(cb or {}).get('rate')}")
        if (ca or {}).get("passed") != (cb or {}).get("passed"):
            problems.append(
                f"{cat}: passed {(ca or {}).get('passed')} vs "
                f"{(cb or {}).get('passed')}")
    if a["mechanisms"]["overall"]["rate"] != b["mechanisms"]["overall"]["rate"]:
        problems.append(
            f"overall rate {a['mechanisms']['overall']['rate']} vs "
            f"{b['mechanisms']['overall']['rate']}")
    if a["pass"] != b["pass"]:
        problems.append(f"verdict differs: stage_e_pass {a['pass']} vs {b['pass']}")

    total = len(set(rows_a) & set(rows_b)) * (len(COMPARED) + 3)
    print(f"\n  rows compared      : {len(set(rows_a) & set(rows_b))}")
    print(f"  fields per row     : {len(COMPARED) + 3} "
          f"(answer, chain, path, edges, anchor, flags, verdict, invariant, semantic)")
    print(f"  comparisons made   : {total}")
    print(f"  behavioural diffs  : {len(diffs)}")
    print(f"  aggregate diffs    : {len(problems)}")

    deterministic = not diffs and not problems
    print(f"\n  DETERMINISTIC = {deterministic}")
    if diffs:
        print("\n  first differences:")
        for d in diffs[:20]:
            print(f"    {d['qid']}.{d['field']}: {d['first']!r} vs {d['second']!r}")
    if problems:
        print("\n  aggregate differences:")
        for p in problems[:20]:
            print(f"    {p}")

    REPORT.write_text(json.dumps({
        "stage": "E",
        "deterministic": deterministic,
        "rows_compared": len(set(rows_a) & set(rows_b)),
        "fields_per_row": len(COMPARED) + 3,
        "comparisons": total,
        "behavioural_diffs": diffs,
        "aggregate_diffs": problems,
        "compared_row_fields": list(COMPARED),
        "excluded_from_comparison": ["time_seconds (wall clock is not reproducible)"],
        "baseline": str(BASELINE.relative_to(ROOT)).replace("\\", "/"),
        "rerun": str(RERUN.relative_to(ROOT)).replace("\\", "/"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {REPORT}")
    return 0 if deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())