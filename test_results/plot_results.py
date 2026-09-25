"""Generate PNG scorecards from tester per-question JSON results (resolves IS-13).

Every tester runner (tester-b small/medium, tester-c, tester-b-real-graph) writes
one `<suite>_<question>.json` per golden question carrying stable metadata keys:

    _golden_id     question id (e.g. "fbs01", "trg06")
    _golden_gold   expected answer node(s)
    _golden_chain  expected relation chain
    _golden_hops   expected hop count
    _pass          PASS/FAIL for this question
    question       full question text
    answer         emitted answer text
    relation_chain chain actually extracted
    honest_no_relation  true when the pipeline emitted an honest fallback

This script scans those JSONs, aggregates PASS/FAIL and per-relation accuracy per
suite, and renders:

    <suite dir>/<suite name>_scorecard.png    per-question status + per-relation accuracy
    test_results/scorecard_overview.png       all suites + all relations aggregated

Usage:
    python test_results/plot_results.py                 # all suites
    python test_results/plot_results.py --only real-graph
    python test_results/plot_results.py --no-overview
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

SUITES: Dict[str, str] = {
    "real-graph": "tester-b-real-graph/tester_b_real_graph_trg*.json",
    "tester-b small": "tester-b/food_bio_small_fbs*.json",
    "tester-b medium": "tester-b/food_bio_medium_fbm*.json",
    "tester-c small": "tester-c/health_science_small_hsc*.json",
}

COLOR_PASS = "#2ca02c"
COLOR_FAIL = "#d62728"
COLOR_HONEST = "#ffbf00"
COLOR_REL = "#1f77b4"


def load_suite(rel_glob: str) -> List[dict]:
    records: List[dict] = []
    for fp in sorted((ROOT / rel_glob).parent.glob((ROOT / rel_glob).name)):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skip {fp.name}: {exc}", file=sys.stderr)
            continue
        if "_pass" not in data:
            continue
        chain = data.get("_golden_chain") or data.get("relation_chain") or []
        records.append({
            "id": str(data.get("_golden_id", fp.stem)),
            "question": str(data.get("question", "")),
            "gold": str(data.get("_golden_gold", "")),
            "chain": list(chain),
            "relation": str(chain[0]) if chain else "unknown",
            "passed": bool(data["_pass"]),
            "honest": bool(data.get("honest_no_relation", False)),
            "answer": str(data.get("answer", "")),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
        })
    return records


def status_color(rec: dict) -> str:
    if rec["passed"]:
        return COLOR_PASS
    if rec["honest"]:
        return COLOR_HONEST
    return COLOR_FAIL


def relation_accuracy(records: List[dict]) -> List[dict]:
    acc: Dict[str, List[int]] = {}
    for rec in records:
        acc.setdefault(rec["relation"], [0, 0])
        acc[rec["relation"]][1] += 1
        acc[rec["relation"]][0] += 1 if rec["passed"] else 0
    rows = sorted(
        ({"relation": r, "pass": p, "n": n, "rate": p / n} for r, (p, n) in acc.items()),
        key=lambda row: (row["rate"], -row["n"]),
        reverse=True,
    )
    return rows


def render_suite(name: str, records: List[dict], out: Path) -> None:
    n = len(records)
    if n == 0:
        print(f"  {name}: no results found, skipped")
        return
    passed = sum(1 for r in records if r["passed"])

    fig, (axq, axr) = plt.subplots(
        1, 2, figsize=(max(11, 0.55 * n + 7), max(4, 0.42 * n + 1.5))
    )

    ids = [r["id"] for r in records]
    colors = [status_color(r) for r in records]
    y = list(range(n - 1, -1, -1))
    axq.barh(y, [1] * n, color=colors, edgecolor="#33475b", linewidth=0.6)
    axq.set_yticks(y)
    axq.set_yticklabels(ids, fontsize=8)
    axq.set_xlim(0, 1.45)
    axq.set_xticks([])
    for yi, rec in zip(y, records):
        marker = " PASS" if rec["passed"] else (" honest" if rec["honest"] else " FAIL")
        axq.text(0.03, yi, rec["question"][:60] + marker, va="center", fontsize=7)
    axq.set_title("per-question result (green=PASS red=FAIL amber=honest)", fontsize=10)
    axq.axis("off")

    rows = relation_accuracy(records)
    rels = [r["relation"] for r in rows][::-1]
    rates = [r["rate"] * 100 for r in rows][::-1]
    labels = [f"{r['pass']}/{r['n']} ({r['rate'] * 100:.0f}%)" for r in rows][::-1]
    axr.barh(rels, rates, color=COLOR_REL, edgecolor="#33475b", linewidth=0.6)
    for yi, label in zip(range(len(rels)), labels):
        axr.text(rates[yi] + 1, yi, label, va="center", fontsize=8)
    axr.set_xlim(0, 108)
    axr.set_xlabel("accuracy %")
    axr.set_title("accuracy per expected relation (pass/total)", fontsize=10)
    axr.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        f"{name}  —  {passed}/{n} passed"
        + (f"   ({max(r['rate'] for r in rows) * 100:.0f}% best relation)" if rows else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def render_overview(suite_rows: Dict[str, int]) -> None:
    fig, (axs, axr) = plt.subplots(1, 2, figsize=(14, 5))
    names = list(suite_rows.keys())
    passes = [suite_rows[n][0] for n in names]
    totals = [suite_rows[n][1] for n in names]
    y = list(range(len(names) - 1, -1, -1))
    rates = [p / t * 100 if t else 0 for p, t in zip(passes, totals)]
    axs.barh(y, [100.0] * len(y), color="#e8e8e8", edgecolor="#33475b")
    axs.barh(y, rates, color=COLOR_PASS, edgecolor="#33475b")
    labels = [f"{passes[i]}/{totals[i]} ({rates[i]:.0f}%)" for i in range(len(names))]
    for yi, label in zip(y, labels):
        axs.text(101, yi, label, va="center", fontsize=9)
    axs.set_yticks(y)
    axs.set_yticklabels(names, fontsize=9)
    axs.set_xlim(0, 130)
    axs.set_xlabel("pass %")
    axs.set_title("pass rate per suite", fontsize=10)

    all_rows: List[dict] = []
    for name in names:
        all_rows.extend(_SUITE_RELS.get(name, []))
    rel_acc = relation_accuracy(all_rows)
    rels = [r["relation"] for r in rel_acc][::-1]
    rates = [r["rate"] * 100 for r in rel_acc][::-1]
    nbs = [f"{r['pass']}/{r['n']} ({r['rate'] * 100:.0f}%)" for r in rel_acc][::-1]
    axr.barh(rels, rates, color=COLOR_REL, edgecolor="#33475b")
    for yi, label in zip(range(len(rels)), nbs):
        axr.text(rates[yi] + 1, yi, label, va="center", fontsize=8)
    axr.set_xlim(0, 108)
    axr.set_xlabel("accuracy %")
    axr.set_title("accuracy per relation (all suites)", fontsize=10)
    axr.tick_params(axis="y", labelsize=8)

    fig.suptitle("GLMX tester scorecard overview", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "scorecard_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


_SUITE_RELS: Dict[str, List[dict]] = {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Scorecards from tester per-question JSONs (IS-13)")
    ap.add_argument("--only", choices=sorted(SUITES), help="render only this suite")
    ap.add_argument("--no-overview", action="store_true", help="skip the combined overview")
    args = ap.parse_args()

    selected = {k: v for k, v in SUITES.items() if args.only is None or k == args.only}

    for name, rel_glob in selected.items():
        records = load_suite(rel_glob)
        _SUITE_RELS[name] = records
        print(f"[{name}] {sum(1 for r in records if r['passed'])}/{len(records)} passed")
        for row in relation_accuracy(records):
            print(f"    {row['relation']:<18} {row['pass']}/{row['n']} "
                  f"({row['rate'] * 100:.0f}%)")
        subdir = (ROOT / rel_glob).parent
        out = subdir / f"{name.replace(' ', '_')}_scorecard.png"
        render_suite(name, records, out)

    if not args.no_overview:
        suite_rows = {n: (sum(1 for r in _SUITE_RELS[n] if r["passed"]), len(_SUITE_RELS[n]))
                      for n in _SUITE_RELS}
        render_overview(suite_rows)


if __name__ == "__main__":
    main()