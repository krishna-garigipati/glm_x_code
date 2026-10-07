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
    probe - tester-a set; dataset violates Section 5.4 (3 relations), answers
            graded verbatim (no relation-filter relabeling).

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
        {"id": "nws02", "q": "What is the flood caused by?", "node": "rain", "chain": ["caused_by"], "hops": 1},
{"id": "nws03", "q": "What comes after summer?", "node": "spring", "chain": ["precedes"], "hops": 1},
    {"id": "nws04", "q": "What comes before summer?", "node": "autumn", "chain": ["follows"], "hops": 1},
        {"id": "nws05", "q": "What is the opposite of sun?", "node": "cloud", "chain": ["antonym"], "hops": 1},
        {"id": "nws06", "q": "What is associated with rain?", "node": "cloud", "chain": ["associated_with"], "hops": 1},
        {"id": "nws07", "q": "What is rain?", "node": "water", "chain": ["is_a"], "hops": 1},
        {"id": "nws08", "q": "What is part of a storm?", "node": "lightning", "chain": ["part_of"], "hops": 1},
        {"id": "nws09", "q": "What does lightning cause?", "node": "fire", "chain": ["causes"], "hops": 1},
        {"id": "nws10", "q": "What comes before winter?", "node": "spring", "chain": ["follows"], "hops": 1},
    ]


def _zoology_large() -> list[dict]:
    # Phase C: first-ever probe of supports / contradicts / linguistic_maps.
    return [
        {"id": "zoo01", "q": "What supports the theory of evolution?", "node": "fossils", "chain": ["supports"], "hops": 1},
        {"id": "zoo02", "q": "What supports photosynthesis?", "node": "sunlight", "chain": ["supports"], "hops": 1},
        {"id": "zoo03", "q": "What contradicts the belief that sugar is healthy?", "node": "tooth decay", "chain": ["contradicts"], "hops": 1},
        {"id": "zoo04", "q": "What contradicts the statement that smoking is safe?", "node": "lung cancer", "chain": ["contradicts"], "hops": 1},
        {"id": "zoo05", "q": "What supports natural selection?", "node": "adaptation", "chain": ["supports"], "hops": 1},
        {"id": "zoo06", "q": "What do you call a baby cat?", "node": "kitten", "chain": ["linguistic_maps"], "hops": 1},
        {"id": "zoo07", "q": "What is a lion?", "node": "mammal", "chain": ["is_a"], "hops": 1},
        {"id": "zoo08", "q": "What is the wing a part of?", "node": "bird", "chain": ["part_of"], "hops": 1},
        {"id": "zoo09", "q": "What is the opposite of nocturnal?", "node": "diurnal", "chain": ["antonym"], "hops": 1},
        {"id": "zoo10", "q": "What is another word for carnivore?", "node": "predator", "chain": ["synonym"], "hops": 1},
        {"id": "zoo11", "q": "Give me an example of a fish", "node": "salmon", "chain": ["example_of"], "hops": 1},
        {"id": "zoo12", "q": "What property does a whale have?", "node": "ocean", "chain": ["has_property"], "hops": 1},
        {"id": "zoo13", "q": "What causes tooth decay?", "node": "sugar", "chain": ["causes"], "hops": 1},
        {"id": "zoo14", "q": "What is associated with a lion?", "node": "savanna", "chain": ["associated_with"], "hops": 1},
    ]


def _science_evidence() -> list[dict]:
    # Phase C: evidence-style supports/contradicts + mitosis follows/precedes chains.
    return [
        {"id": "sc01", "q": "What evidence supports exercise?", "node": "fitness", "chain": ["supports"], "hops": 1},
        {"id": "sc02", "q": "What evidence supports vaccination?", "node": "prevention", "chain": ["supports"], "hops": 1},
        {"id": "sc03", "q": "What contradicts the statement that sugar is safe?", "node": "diabetes", "chain": ["contradicts"], "hops": 1},
        {"id": "sc04", "q": "What contradicts the statement that junk food is safe?", "node": "obesity", "chain": ["contradicts"], "hops": 1},
{"id": "sc05", "q": "What comes after prophase?", "node": "metaphase", "chain": ["precedes"], "hops": 1},
    {"id": "sc06", "q": "What comes after metaphase?", "node": "anaphase", "chain": ["precedes"], "hops": 1},
{"id": "sc07", "q": "What comes before telophase?", "node": "anaphase", "chain": ["follows"], "hops": 1},
    {"id": "sc08", "q": "What comes after interphase?", "node": "prophase", "chain": ["precedes"], "hops": 1},
        {"id": "sc09", "q": "What causes fermentation?", "node": "yeast", "chain": ["causes"], "hops": 1},
        {"id": "sc10", "q": "What is fermentation caused by?", "node": "yeast", "chain": ["caused_by"], "hops": 1},
        {"id": "sc11", "q": "Give me an example of a fungus", "node": "yeast", "chain": ["example_of"], "hops": 1},
        {"id": "sc12", "q": "What is another word for a doctor?", "node": "physician", "chain": ["synonym"], "hops": 1},
        {"id": "sc13", "q": "What is another word for a disease?", "node": "illness", "chain": ["synonym"], "hops": 1},
        {"id": "sc14", "q": "What is the opposite of sick?", "node": "healthy", "chain": ["antonym"], "hops": 1},
    ]


def _weather_climate_large() -> list[dict]:
    # Phase C: first-ever probe of temporal_coincident / spatial_near.
    return [
        {"id": "wc01", "q": "What happens at the same time as harvest?", "node": "autumn", "chain": ["temporal_coincident"], "hops": 1},
        {"id": "wc02", "q": "What occurs during monsoons?", "node": "flooding", "chain": ["temporal_coincident"], "hops": 1},
        {"id": "wc03", "q": "What is located near the equator?", "node": "tropics", "chain": ["spatial_near"], "hops": 1},
        {"id": "wc04", "q": "What is close to the coast?", "node": "ocean", "chain": ["spatial_near"], "hops": 1},
{"id": "wc05", "q": "What comes after spring?", "node": "summer", "chain": ["precedes"], "hops": 1},
    {"id": "wc06", "q": "What comes before winter?", "node": "autumn", "chain": ["follows"], "hops": 1},
        {"id": "wc07", "q": "What causes thunder?", "node": "lightning", "chain": ["causes"], "hops": 1},
        {"id": "wc08", "q": "What is thunder caused by?", "node": "lightning", "chain": ["caused_by"], "hops": 1},
        {"id": "wc09", "q": "What type of storm is a cyclone?", "node": "storm", "chain": ["is_a"], "hops": 1},
        {"id": "wc10", "q": "What property does hail have?", "node": "cold", "chain": ["has_property"], "hops": 1},
        {"id": "wc11", "q": "What is part of a hurricane?", "node": "eye", "chain": ["part_of"], "hops": 1},
        {"id": "wc12", "q": "What is the opposite of wet?", "node": "dry", "chain": ["antonym"], "hops": 1},
        {"id": "wc13", "q": "What is rain?", "node": "precipitation", "chain": ["is_a"], "hops": 1},
        {"id": "wc14", "q": "What is another word for breeze?", "node": "wind", "chain": ["synonym"], "hops": 1},
    ]


def _geo_glossary() -> list[dict]:
    # Phase C: glossary "term for" mapping + spatial/temporal riders.
    return [
        {"id": "gg01", "q": "What do you call a large farm in Spanish?", "node": "hacienda", "chain": ["linguistic_maps"], "hops": 1},
        {"id": "gg02", "q": "What is the term for a mountain lake in Scotland?", "node": "loch", "chain": ["linguistic_maps"], "hops": 1},
        {"id": "gg03", "q": "What is near the Mediterranean?", "node": "europe", "chain": ["spatial_near"], "hops": 1},
        {"id": "gg04", "q": "What occurs during the monsoon?", "node": "floods", "chain": ["temporal_coincident"], "hops": 1},
        {"id": "gg05", "q": "What is the opposite of urban?", "node": "rural", "chain": ["antonym"], "hops": 1},
        {"id": "gg06", "q": "What is a delta?", "node": "landform", "chain": ["is_a"], "hops": 1},
        {"id": "gg07", "q": "What is part of a mountain?", "node": "peak", "chain": ["part_of"], "hops": 1},
        {"id": "gg08", "q": "What property does the Sahara have?", "node": "arid", "chain": ["has_property"], "hops": 1},
        {"id": "gg09", "q": "Give me an example of a volcano", "node": "vesuvius", "chain": ["example_of"], "hops": 1},
        {"id": "gg10", "q": "What is another word for a brook?", "node": "stream", "chain": ["synonym"], "hops": 1},
        {"id": "gg11", "q": "What is associated with a glacier?", "node": "iceberg", "chain": ["associated_with"], "hops": 1},
        {"id": "gg12", "q": "What is a strait?", "node": "waterway", "chain": ["is_a"], "hops": 1},
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
    "zoology_large": {
        "db": str(ROOT / "test_results" / "tester-b" / "datasets" / "zoology_large.db"),
        "tier": "scale",
        "goldens": _zoology_large(),
    },
    "science_evidence": {
        "db": str(ROOT / "test_results" / "tester-b" / "datasets" / "science_evidence.db"),
        "tier": "scale",
        "goldens": _science_evidence(),
    },
    "weather_climate_large": {
        "db": str(ROOT / "test_results" / "tester-a" / "datasets" / "weather_climate_large.db"),
        "tier": "scale",
        "goldens": _weather_climate_large(),
    },
    "geo_glossary": {
        "db": str(ROOT / "test_results" / "tester-a" / "datasets" / "geo_glossary.db"),
        "tier": "scale",
        "goldens": _geo_glossary(),
    },
}

HONEST_TEXT = "don't have a relation"


_STRICT = False


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
    if _STRICT:
        # Strict: the expected node must actually be on the walked path. The
        # legacy substring-anywhere fallback in the answer text lets an
        # unrelated answer pass as long as the word appears somewhere in it.
        node_hit = g["node"] in path[1:]
    else:
        node_hit = g["node"] in path[1:] or g["node"].lower() in answer.lower()
    checks = {
        "hops": hops == g["hops"],
        "node": node_hit,
    }
    chain_ok = chain == g["chain"]
    if _STRICT:
        # Strict: the relation chain the system planned must match the golden.
        # Legacy mode reports a chain mismatch in `reasons` but never lets it
        # affect the verdict, so a correct answer reached by the wrong relation
        # scores as a pass.
        checks["chain"] = chain_ok
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
    parser.add_argument("--strict", action="store_true",
                        help="Also require relation_chain == golden chain and the expected "
                             "node to be on the walked path (no answer-substring fallback). "
                             "Off by default so the legacy 95/99 metric is preserved; write "
                             "strict results to a separate --out prefix.")
    args = parser.parse_args()

    global _STRICT
    _STRICT = bool(args.strict)

    selected = [s.strip() for s in args.suites.split(",") if s.strip()]
    lines = []
    rows_all = {}
    overall_ok = overall_total = 0

    lines.append("=" * 78)
    lines.append("UNIFIED GOLDEN RUNNER (Phase 5) - deterministic: "
                 f"--seed {args.seed} --no-learning")
    if args.strict:
        lines.append("STRICT MODE: chain must equal golden chain; node must be on walked path")
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
                "heuristic_used": bool(r.get("heuristic_used")),
                "plan_confidence": float(r.get("confidence") or 0.0),
            })
        rows_all[name] = rows
        lines.append(f"RESULT: {ok_count}/{len(goldens)} passed")
        for row in rows:
            mark = "PASS" if row["ok"] else "FAIL"
            lines.append(f"[{mark}] {row['id']} Q: {row['q']}")
            if row["ok"]:
                lines.append(f"      -> {row['answer']}")
            else:
                lines.append(f"      -> {row['chain']} hop={row['hops']} hon={row['honest']} "
                 f"heur={row['heuristic_used']} conf={row['plan_confidence']:.2f}")
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