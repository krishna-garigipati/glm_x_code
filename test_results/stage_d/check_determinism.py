"""Contract section 12 determinism, Stage D edition.

Same graph + same question must give a byte-identical answer. Stage D is the
stage where that matters most, because most of its rows end in a refusal or a
disclosed guess rather than a walk, and both of those are assembled from several
independently-tie-broken pieces: the anchor comes from an embedding similarity
search, the closest-concepts disclosure from a ranked neighbour list, and the
heuristic branch from a fallback chain. Any of them could drift between runs
without changing the headline pass/fail count, so the comparison is on the row
fields themselves rather than on the summary.

    python test_results/stage_d/stage_d_runner.py --out test_results/stage_d/_run1.json
    python test_results/stage_d/stage_d_runner.py
    python test_results/stage_d/check_determinism.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Every field that must be reproducible. `time_seconds` is deliberately absent:
# it is wall-clock and is expected to move.
KEYS = ["answer", "relation_chain", "path_edges", "path_labels",
        "selected_anchor", "honest_no_relation", "honest_by_entity",
        "honest_by_relation", "heuristic_used", "heuristic_disclosed",
        "confidence", "invariant_ok", "semantic_ok", "passed"]

a_path = HERE / "_run1.json"
b_path = HERE / "stage_d_results.json"
for p in (a_path, b_path):
    if not p.exists():
        print(f"REFUSING TO COMPARE: {p.name} is missing.\n"
              f"  Run the stage twice, writing the first pass to "
              f"{p.name} and the second to stage_d_results.json.")
        sys.exit(1)

a = json.loads(a_path.read_text(encoding="utf-8"))
b = json.loads(b_path.read_text(encoding="utf-8"))

print("run1:", a["summary"]["n_questions"], "rows | run2:",
      b["summary"]["n_questions"], "rows")

ra = {r["id"]: r for r in a["rows"]}
rb = {r["id"]: r for r in b["rows"]}

if set(ra) != set(rb):
    print("ROW SET DIFFERS:", sorted(set(ra) ^ set(rb)))
    sys.exit(1)
print("row ids identical:", len(ra))

diffs = []
for qid in sorted(ra):
    for k in KEYS:
        if ra[qid].get(k) != rb[qid].get(k):
            diffs.append((qid, k, ra[qid].get(k), rb[qid].get(k)))

print("\ndiffering fields:", len(diffs))
for qid, k, x, y in diffs[:40]:
    print(f"  {qid} . {k}\n      run1: {x!r}\n      run2: {y!r}")

# The summary is derived from the rows, so an identical row set must produce an
# identical summary. Checking it too catches a harness bug that happens to
# recompute rather than read the rows.
sa = {k: v for k, v in a["summary"].items() if k != "seed"}
sb = {k: v for k, v in b["summary"].items() if k != "seed"}
summary_diffs = [k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k)]
print("summary fields differing:", len(summary_diffs), summary_diffs)

print("\nDETERMINISTIC" if not diffs and not summary_diffs else "\nNON-DETERMINISTIC")

t1 = sum(r.get("time_seconds") or 0 for r in a["rows"])
t2 = sum(r.get("time_seconds") or 0 for r in b["rows"])
print(f"wall time run1={t1:.1f}s run2={t2:.1f}s (expected to differ; not an answer field)")

sys.exit(0 if not diffs and not summary_diffs else 1)
