"""REGRESSION REFERENCE ONLY -- not the official success metric.

The official evaluation is the staged testing strategy: Stage A -> Stage C ->
Stage D -> Stage E. This script does not gate those stages. It only answers one
question: "did anything change that did not mean to?"

Purpose: detect NEW breakage. The 17 recorded failures below are known and
explained; anything outside that set is a regression to be investigated.

Two baselines are kept:
  * historical (90/99) -- predates the strict-mirroring contract fix, kept for
    reference so the direction of travel stays visible.
  * strict-mirroring (82/99) -- the current gate.

Classification of the 17 (established by direct per-question traces against the
stored graphs, not inference from the output text):
  A. Invalid goldens -- satisfiable ONLY by an undeclared same-label reverse,
     which the contract forbids. The honest refusal is correct behaviour and the
     question as written is unanswerable. (8)
       example_of: fbs07 fbs08 fbm10 fbm11 zoo11 sc11 gg09
       supports:   zoo02
  B. Two defensible answers out of the anchor; the expected one was previously
     selected by the removed activation lift, not by any contract rule. (2)
       nws10  winter has legal follows -> spring (stored) and autumn
             (legal mirror of stored `autumn precedes winter`)
       zoo14  lion has two STORED associated_with edges -> zebra and savanna
  C. Known retrieval limitation at contract-mandated tier1 top_k=64. Deferred,
     explicitly not blocking. Revisit during or after Stage C. (1)
       wc11   `hurricane part_of eye` is stored and resonates at 0.0104, but the
              rank gate keeps the top 64 of a 76-node candidate set and drops it.
              Root cause: _gate runs inside every _propagate call while resonate()
              replaces `activations` wholesale, so gated-out nodes are erased
              permanently. top_k=64 is mandated by contract section 7.
  D. Ambiguity that predates this work (6)
       fbs15 fbm20 nws07 nws08 wc13 gg11   (gg11 = same shape as zoo14)

NOTE: test_results/tester-a/datasets/science_evidence.db is an empty 0-node
file; the populated copy lives in tester-b. Always confirm which copy a probe
loaded before drawing conclusions from it.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# unified_golden_runner.py writes its real artifact to
# test_results/lead/unified_golden_results.txt. The _gold*.log files are ad-hoc
# shell redirections from manual runs and live in two different directories, so
# globbing for them compared a fresh run against whichever stale copy sorted
# last. Prefer the runner's own output; fall back to the newest log only if it
# is missing.
CANONICAL = HERE.parent / "lead" / "unified_golden_results.txt"
if CANONICAL.is_file():
    log_path = CANONICAL
else:
    logs = sorted(HERE.glob("_gold*.log"), key=lambda p: (p.stat().st_mtime, p.name))
    if not logs:
        raise SystemExit(
            "no golden results found; run unified_golden_runner.py --strict first"
        )
    log_path = logs[-1]
print(f"reading: {log_path}")
raw = log_path.read_bytes()
# PowerShell 5.1 `*>` redirection writes UTF-16LE, so sniff the encoding
# instead of assuming utf-8 (which yields NUL-interleaved garbage).
for enc in ("utf-16", "utf-8-sig", "utf-8"):
    try:
        text = raw.decode(enc)
    except UnicodeDecodeError:
        continue
    if "OVERALL" in text:
        break
log = text

HISTORICAL_FAILURES = {
    "fbs15", "fbm20", "nws07", "wc13", "nws02", "nws08", "sc05", "sc06", "gg11",
}

# Group A: only satisfiable via an undeclared same-label reverse.
INVALID_GOLDENS = {
    "fbs07", "fbs08", "fbm10", "fbm11", "zoo11", "sc11", "gg09", "zoo02",
}
# Group B: two defensible answers; previously decided by the removed lift.
AMBIGUOUS_ANSWERS = {"nws10", "zoo14"}
# Group C: retrieval limitation at contract-mandated tier1 top_k=64.
TOPK_LIMITATION = {"wc11"}
# Group D: still failing from the historical baseline. nws02, sc05 and sc06 were
# fixed by the strict-mirroring work and are deliberately not carried forward.
PRE_EXISTING_STILL_FAILING = {"fbs15", "fbm20", "nws07", "nws08", "wc13", "gg11"}

STRICT_MIRROR_FAILURES = (
    INVALID_GOLDENS
    | AMBIGUOUS_ANSWERS
    | TOPK_LIMITATION
    | PRE_EXISTING_STILL_FAILING
)

overall = re.search(r"OVERALL:\s*(\d+)/(\d+)", log)
print("current:", overall.group(0) if overall else "??")
print("strict-mirroring baseline: {}/{}".format(
    99 - len(STRICT_MIRROR_FAILURES), 99))
print("historical baseline      : 90/99 (pre-Stage-A, predates the contract fix)")
if overall:
    print("delta vs strict baseline:",
          int(overall.group(1)) - (99 - len(STRICT_MIRROR_FAILURES)), "net questions")
print()

fails = []
for m in re.finditer(r"\[FAIL\]\s+(\S+)\s+Q:\s*(.*)", log):
    fails.append((m.group(1), m.group(2).strip()))


def classify(qid):
    if qid in INVALID_GOLDENS:
        return "INVALID-GOLDEN"
    if qid in AMBIGUOUS_ANSWERS:
        return "AMBIGUOUS"
    if qid in TOPK_LIMITATION:
        return "TOPK-LIMIT"
    if qid in PRE_EXISTING_STILL_FAILING:
        return "pre-existing"
    return "REGRESSION"


print("current failures: {}".format(len(fails)))
regressions = []
for qid, q in fails:
    tag = classify(qid)
    if tag == "REGRESSION":
        regressions.append((qid, q))
    print("  {:<8} {:<15} {}".format(qid, tag, q[:70]))

print("\nfixed since historical baseline:",
      sorted(HISTORICAL_FAILURES - {q for q, _ in fails}) or "none")
print("NEWLY broken ({}):".format(len(regressions)))
for qid, q in regressions:
    print("  {}  {}".format(qid, q))

# Show the failure detail lines for each regression.
lines = log.splitlines()
for qid, _ in regressions:
    print("\n--- {} detail ---".format(qid))
    for i, ln in enumerate(lines):
        if "[FAIL] {}".format(qid) in ln:
            for out in lines[i:i + 7]:
                print("   " + out)
            break

print("\nREFERENCE ONLY -- official metric is the staged strategy "
      "(A -> B -> C -> D -> E).")
print("Gate is STRICT_MIRROR_BASELINE (82/99). Exit {}.".format(
    "1" if regressions else "0"))
sys.exit(1 if regressions else 0)