"""Unified golden runner for all tester datasets (Phase 5, no branch merges).

Grades every pre-registered golden deterministically (seed + no-learning) in a
single process, mirroring:
    python scripts/glmx_ask.py --db <db> -q "<q>" --seed 0 --no-learning --measure

The golden lists below are FROZEN snapshots copied verbatim from the committed
tester pre-registrations (food_bio_small_golden.md, food_bio_medium_golden.md,
nature_weather_small_golden.md). They are not edited after runs.

Tiering:
    gate  - tester-b sets count toward Experimentation Gate (Section 7.2).
            food_bio_small must reach >=90% of non-edge-case goldens.
    probe - tester-a set; dataset rebuilt from its 53-triple JSON on 2026-09-25
            (IS-04/05 reconcile, 9 canonical relations, in-band), answers graded
            verbatim (no relation-filter relabeling).

Usage:
    python test_results/lead/unified_golden_runner.py [--out <prefix>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.glmx_ask import GLMXPipeline


def _small_fb() -> list[dict]:
    return [
        {"id": "fbs01", "q": "What is a salmon?", "node": "fish", "chain": ["is_a"], "hops": 1},
        {"id": "fbs02", "q": "What is a robin?", "node": "bird", "chain": ["is_a"], "hops": 1},
        {"id": "fbs03", "q": "What is the yolk a part of?", "node": "egg", "chain": ["part_of"], "hops": 1},
        {"id": "fbs04", "q": "What is the petal a part of?", "node": "flower", "chain": ["part_of"], "hops": 1},
        {"id": "fbs05", "q": "What property does honey have?", "node": "sweet", "chain": ["has_property"], "hops": 1},
        {"id": "fbs06", "q": "What property does lemon have?", "node": "sour", "chain": ["has_property"], "hops": 1},
        {"id": "fbs07", "q": "Give me an example of a bird", "node": "robin", "chain": ["example_of"], "hops": 1},
        {"id": "fbs08", "q": "Give me an example of a fish", "node": "salmon", "chain": ["example_of"], "hops": 1},
        {"id": "fbs09", "q": "What is another word for happy?", "node": "glad", "chain": ["synonym"], "hops": 1},
        {"id": "fbs10", "q": "What is another word for small?", "node": "little", "chain": ["synonym"], "hops": 1},
        {"id": "fbs11", "q": "What is the opposite of hot?", "node": "cold", "chain": ["antonym"], "hops": 1},
        {"id": "fbs12", "q": "What is the opposite of sweet?", "node": "sour", "chain": ["antonym"], "hops": 1},
        {"id": "fbs13", "q": "What leads to tooth decay?", "node": "sugar", "chain": ["causes"], "hops": 1},
        {"id": "fbs14", "q": "What follows photosynthesis?", "honest": True},
        {"id": "fbs15", "q": "What is rice?", "node": "grain", "chain": ["is_a"], "hops": 1},
    ]


def _medium_fb() -> list[dict]:
    return [
        {"id": "fbm01", "q": "What is a mammal?", "node": "animal", "chain": ["is_a"], "hops": 1},
        {"id": "fbm02", "q": "What is a penguin?", "node": "bird", "chain": ["is_a"], "hops": 1},
        {"id": "fbm03", "q": "What is a salmon?", "node": "fish", "chain": ["is_a"], "hops": 1},
        {"id": "fbm04", "q": "What is the yolk a part of?", "node": "egg", "chain": ["part_of"], "hops": 1},
        {"id": "fbm05", "q": "What is the petal a part of?", "node": "flower", "chain": ["part_of"], "hops": 1},
        {"id": "fbm06", "q": "What is the wing a part of?", "node": "bird", "chain": ["part_of"], "hops": 1},
        {"id": "fbm07", "q": "What property does honey have?", "node": "sweet", "chain": ["has_property"], "hops": 1},
        {"id": "fbm08", "q": "What property does chili have?", "node": "spicy", "chain": ["has_property"], "hops": 1},
        {"id": "fbm09", "q": "What property does snow have?", "node": "cold", "chain": ["has_property"], "hops": 1},
        {"id": "fbm10", "q": "Give me an example of a bird", "node": "robin", "chain": ["example_of"], "hops": 1},
        {"id": "fbm11", "q": "Give me an example of a fish", "node": "salmon", "chain": ["example_of"], "hops": 1},
        {"id": "fbm12", "q": "What is another word for big?", "node": "large", "chain": ["synonym"], "hops": 1},
        {"id": "fbm13", "q": "What is another word for quick?", "node": "fast", "chain": ["synonym"], "hops": 1},
        {"id": "fbm14", "q": "What is the opposite of dark?", "node": "light", "chain": ["antonym"], "hops": 1},
        {"id": "fbm15", "q": "What is the opposite of wet?", "node": "dry", "chain": ["antonym"], "hops": 1},
        {"id": "fbm16", "q": "What leads to tooth decay?", "node": "sugar", "chain": ["causes"], "hops": 1},
        {"id": "fbm17", "q": "What is tooth decay caused by?", "node": "sugar", "chain": ["caused_by"], "hops": 1},
        {"id": "fbm18", "q": "Tell me something associated with water", "node": "rain", "chain": ["associated_with"], "hops": 1},
        {"id": "fbm19", "q": "What organism lives close to plankton?", "honest": True},
        {"id": "fbm20", "q": "What is corn?", "node": "grain", "chain": ["is_a"], "hops": 1},
    ]


def _small_nw() -> list[dict]:
    return [
        {"id": "nws01", "q": "What causes flood?", "node": "rain", "chain": ["causes"], "hops": 1},
        {"id": "nws02", "q": "What causes rain?", "node": "cloud", "chain": ["caused_by"], "hops": 1},
        {"id": "nws03", "q": "What comes after summer?", "node": "autumn", "chain": ["follows"], "hops": 1},
        {"id": "nws04", "q": "What comes before summer?", "node": "spring", "chain": ["precedes"], "hops": 1},
        {"id": "nws05", "q": "What is the opposite of sun?", "node": "cloud", "chain": ["antonym"], "hops": 1},
        {"id": "nws06", "q": "What is associated with rain?", "node": "cloud", "chain": ["associated_with"], "hops": 1},
        {"id": "nws07", "q": "What is rain?", "node": "water", "chain": ["is_a"], "hops": 1},
        {"id": "nws08", "q": "What is part of a storm?", "node": "lightning", "chain": ["part_of"], "hops": 1},
        {"id": "nws09", "q": "What does lightning cause?", "node": "fire", "chain": ["causes"], "hops": 1},
        {"id": "nws10", "q": "What comes before winter?", "node": "autumn", "chain": ["precedes"], "hops": 1},
    ]


SUITES = {
    "food_bio_small": {
        "db": str(ROOT / "test_results" / "tester-b" / "datasets" / "food_bio_small.db"),
        "tier": "gate",
        "goldens": _small_fb(),
    },
    "food_bio_medium": {
        "db": str(ROOT / "test_results" / "tester-b" / "datasets" / "food_bio_medium.db"),
        "tier": "gate",
        "goldens": _medium_fb(),
    },
    "nature_weather_small": {
        "db": str(ROOT / "test_results" / "tester-a" / "datasets" / "nature_weather_small.db"),
        "tier": "probe",
        "goldens": _small_nw(),
    },
}

HONEST_TEXT = "don't have a relation"


def grade(g, r: dict) -> tuple[bool, list[str]]:
    reasons = []
    if g.get("honest"):
        ok = bool(r.get("honest_no_relation")) and bool(r.get("template_matched"))
        if not ok:
            reasons.append(f"honest_no_relation={r.get('honest_no_relation')} "
                           f"template_matched={r.get('template_matched')} "
                           f"answer={r.get('answer')!r}")
        return ok, reasons
    hops = len(r.get("walk_path_edges") or [])
    chain = list(r.get("relation_chain") or [])
    path = [str(x) for x in (r.get("walk_path_labels") or [])]
    answer = str(r.get("answer") or "")
    node_hit = g["node"] in path[1:] or g["node"].lower() in answer.lower()
    checks = {
        "hops": hops == g["hops"],
        "node": node_hit,
    }
    chain_ok = chain == g["chain"]
    for name, ok in checks.items():
        if not ok:
            reasons.append(f"{name}: got {path if name == 'node' else hops}")
    if not chain_ok:
        reasons.append(f"chain: got {chain} (golden {g['chain']})")
    ok = all(checks.values()) and not r.get("honest_no_relation")
    if r.get("honest_no_relation") and not ok:
        reasons.append("honest_no_relation unexpectedly True")
    return ok, reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suites", default=",".join(SUITES.keys()),
                        help="Comma-separated suite names (default: all)")
    parser.add_argument("--out", default=str(ROOT / "test_results" / "lead" / "unified_golden_results"),
                        help="Output prefix (.txt and .json)")
    args = parser.parse_args()

    selected = [s.strip() for s in args.suites.split(",") if s.strip()]
    lines = []
    rows_all = {}
    overall_ok = overall_total = 0

    lines.append("=" * 78)
    lines.append("UNIFIED GOLDEN RUNNER (Phase 5) - deterministic: "
                 f"--seed {args.seed} --no-learning")
    lines.append("=" * 78)

    for name in selected:
        suite = SUITES[name]
        db = Path(suite["db"])
        lines.append("\n" + "-" * 78)
        lines.append(f"SUITE: {name}  [{suite['tier']}]  db={db}")
        if not db.exists():
            lines.append("  SKIPPED: db missing")
            continue

        pipeline = GLMXPipeline()
        from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore
        pipeline.graph_store = SQLiteGraphStore.load_state(str(db))
        pipeline._seed = args.seed
        pipeline._no_learning = True
        pipeline.load_models()

        goldens = suite["goldens"]
        rows = []
        ok_count = 0
        for g in goldens:
            r = pipeline.ask(g["q"])
            ok, reasons = grade(g, r)
            ok_count += int(ok)
            overall_ok += int(ok)
            overall_total += 1
            rows.append({
                "id": g["id"], "q": g["q"], "ok": ok,
                "reasons": reasons,
                "answer": str(r.get("answer")),
                "chain": list(r.get("relation_chain") or []),
                "hops": len(r.get("walk_path_edges") or []),
                "path": list(r.get("walk_path_labels") or []),
                "honest": bool(r.get("honest_no_relation")),
            })
        rows_all[name] = rows
        lines.append(f"RESULT: {ok_count}/{len(goldens)} passed")
        for row in rows:
            mark = "PASS" if row["ok"] else "FAIL"
            lines.append(f"[{mark}] {row['id']} Q: {row['q']}")
            if row["ok"]:
                lines.append(f"      -> {row['answer']}")
            else:
                lines.append(f"      -> {row['chain']} hop={row['hops']} hon={row['honest']}")
                lines.append(f"        path={row['path']}  reason: {'; '.join(row['reasons'])}")
                lines.append(f"        answer: {row['answer']}")

    lines.append("\n" + "=" * 78)
    lines.append(f"OVERALL: {overall_ok}/{overall_total} passed (deterministic seed={args.seed})")
    lines.append("=" * 78)

    text = "\n".join(lines)
    print(text)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    out_prefix.with_suffix(".json").write_text(
        json.dumps({"rows": rows_all, "seed": args.seed}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved {out_prefix.with_suffix('.txt')} and {out_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()