"""Cross-domain generalization probe (Phase G, tester-c).

Cold -> Hot -> Reload experiment proving the extractor goes from a static
relation descriptor bank to an accretive, per-KG learnable lexicon:

  cold   - static bank only (dynamic lexicon disabled)
  hot    - teach the learned lexicon from the verified(golden) mapping of the
           4 taught goldens per domain, then ask all 6
  reload - persist the learned lexicon into each KG's metadata, reload fresh
           pipelines, re-ask all 6 (proves persistence, not in-memory tricks)

Datasets are all brand-new domains (Astronomy, Computing, Music, Human Body,
History) with disjoint vocabulary from every earlier test tier. The golden
lists below are a FROZEN snapshot (they are not edited after runs).

Usage:
    python test_results/tester-c/generation_probe_runner.py
    python test_results/tester-c/generation_probe_runner.py --modes cold,hot,reload
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.glmx_ask import GLMXPipeline
from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

logger = logging.getLogger(__name__)

D = Path(__file__).resolve().parent / "datasets"


def _astronomy() -> list[dict]:
    return [
        {"id": "ast01", "q": "What is the sun?", "node": "main sequence star", "chain": ["is_a"], "hops": 1, "teach": True},
        {"id": "ast02", "q": "What is the moon a part of?", "node": "solar system", "chain": ["part_of"], "hops": 1, "teach": True},
        {"id": "ast03", "q": "What property does Mars have?", "node": "red", "chain": ["has_property"], "hops": 1, "teach": True},
        {"id": "ast04", "q": "Name the planet that lies closest to the sun.", "node": "mercury", "chain": ["spatial_near"], "hops": 1, "teach": False},
        {"id": "ast05", "q": "Which planet is red?", "node": "mars", "chain": ["has_property"], "hops": 1, "teach": False},
        {"id": "ast06", "q": "What causes tides?", "node": "moon", "chain": ["causes"], "hops": 1, "teach": True},
    ]


def _computing() -> list[dict]:
    return [
        {"id": "com01", "q": "What is a smartphone?", "node": "personal computer", "chain": ["is_a"], "hops": 1, "teach": True},
        {"id": "com02", "q": "What is the keyboard a part of?", "node": "laptop", "chain": ["part_of"], "hops": 1, "teach": True},
        {"id": "com03", "q": "What property does a firewall have?", "node": "secure", "chain": ["has_property"], "hops": 1, "teach": True},
        {"id": "com04", "q": "What causes data loss?", "node": "phishing", "chain": ["causes"], "hops": 1, "teach": True},
        {"id": "com05", "q": "What is a malware infection caused by?", "node": "virus", "chain": ["caused_by"], "hops": 1, "teach": False},
        {"id": "com06", "q": "A phone is portable, so what is the opposite of online?", "node": "offline", "chain": ["antonym"], "hops": 1, "teach": False},
    ]


def _music() -> list[dict]:
    return [
        {"id": "mus01", "q": "What is a trumpet?", "node": "brass instrument", "chain": ["is_a"], "hops": 1, "teach": True},
        {"id": "mus02", "q": "What is the reed a part of?", "node": "clarinet", "chain": ["part_of"], "hops": 1, "teach": True},
        {"id": "mus03", "q": "Give me an example of a brass instrument", "node": "trumpet", "chain": ["example_of"], "hops": 1, "teach": True},
        {"id": "mus04", "q": "What is the opposite of sharp?", "node": "flat", "chain": ["antonym"], "hops": 1, "teach": True},
        {"id": "mus05", "q": "What causes hearing loss?", "node": "excessive volume", "chain": ["causes"], "hops": 1, "teach": False},
        {"id": "mus06", "q": "What do you call a violin in Italian?", "node": "violino", "chain": ["linguistic_maps"], "hops": 1, "teach": False},
    ]


def _human_body() -> list[dict]:
    return [
        {"id": "hum01", "q": "What is the hand a part of?", "node": "arm", "chain": ["part_of"], "hops": 1, "teach": True},
        {"id": "hum02", "q": "What is a red blood cell?", "node": "blood cell", "chain": ["is_a"], "hops": 1, "teach": True},
        {"id": "hum03", "q": "What property does bone have?", "node": "hard", "chain": ["has_property"], "hops": 1, "teach": True},
        {"id": "hum04", "q": "What leads to a headache?", "node": "dehydration", "chain": ["causes"], "hops": 1, "teach": True},
        {"id": "hum05", "q": "What is lung damage caused by?", "node": "smoking", "chain": ["caused_by"], "hops": 1, "teach": False},
        {"id": "hum06", "q": "What is another word for abdomen?", "node": "belly", "chain": ["synonym"], "hops": 1, "teach": False},
    ]


def _history() -> list[dict]:
    return [
        {"id": "his01", "q": "What is a pharaoh?", "node": "ruler", "chain": ["is_a"], "hops": 1, "teach": True},
        {"id": "his02", "q": "What is the acropolis a part of?", "node": "athens", "chain": ["part_of"], "hops": 1, "teach": True},
        {"id": "his03", "q": "What comes after the stone age?", "node": "bronze age", "chain": ["follows"], "hops": 1, "teach": True},
        {"id": "his04", "q": "What property does Sparta have?", "node": "warlike", "chain": ["has_property"], "hops": 1, "teach": True},
        {"id": "his05", "q": "What comes before the iron age?", "node": "bronze age", "chain": ["precedes"], "hops": 1, "teach": False},
        {"id": "his06", "q": "Sparta was warlike, so what appears after the bronze age?", "node": "iron age", "chain": ["follows"], "hops": 1, "teach": False},
    ]


SUITES = {
    "astronomy_small": {"db": D / "astronomy_small.db", "goldens": _astronomy()},
    "computing_small": {"db": D / "computing_small.db", "goldens": _computing()},
    "music_small": {"db": D / "music_small.db", "goldens": _music()},
    "human_body_small": {"db": D / "human_body_small.db", "goldens": _human_body()},
    "history_civilizations": {"db": D / "history_civilizations.db", "goldens": _history()},
}

LEXICON_META_KEY = "learned_lexicon"


def _grade(g: dict, r: dict) -> tuple[bool, list[str]]:
    reasons = []
    hops = len(r.get("walk_path_edges") or [])
    chain = list(r.get("relation_chain") or [])
    path = [str(x) for x in (r.get("walk_path_labels") or [])]
    answer = str(r.get("answer") or "")
    node_hit = g["node"].lower() in [str(x).lower() for x in path[1:]] or g["node"].lower() in answer.lower()
    checks = {"hops": hops == g["hops"], "node": node_hit}
    for name, ok in checks.items():
        if not ok:
            reasons.append(f"{name}: got {path if name == 'node' else hops}")
    if chain != g["chain"]:
        reasons.append(f"chain: got {chain} (golden {g['chain']})")
    ok = all(checks.values())
    if r.get("honest_no_relation"):
        ok = False
        reasons.append("honest_no_relation unexpectedly True")
    return ok, reasons


def _enable_dynamic(pipeline: GLMXPipeline) -> None:
    dl = pipeline.planner.config.dynamic_lexicon
    dl.enabled = True
    dl.graph_grounded = True
    labels = [
        pipeline.graph_store.get_label(nid)
        for nid in range(1, pipeline.graph_store.get_node_count() + 1)
    ]
    pipeline.planner.set_label_vocab(labels)


def run_mode(mode: str, suites: dict, out_prefix: Path) -> dict:
    lines = []
    rows_by_domain = {}
    learned_sizes = {}
    for name, suite in suites.items():
        db = Path(suite["db"])
        lines.append(f"--- {name} [{mode}] db={db}")
        if not db.exists():
            lines.append("  SKIPPED: db missing")
            continue

        pipeline = GLMXPipeline()
        store = SQLiteGraphStore.load_state(str(db))
        pipeline.graph_store = store
        pipeline._seed = 0
        pipeline._no_learning = True
        pipeline.load_models()

        goldens = suite["goldens"]

        if mode in ("hot", "reload") and mode == "hot":
            _enable_dynamic(pipeline)
            for g in goldens:
                if not g.get("teach"):
                    continue
                clause = pipeline.planner.canonical_clause(g["q"])
                relation = g["chain"][0]
                pipeline.planner.learn_success(clause, relation)
            learned = pipeline.planner.get_learned_lexicon()
            learned_sizes[name] = len(learned)
            store.set_metadata(LEXICON_META_KEY, json.dumps(learned))
            store.save_state(str(db))

        if mode == "reload":
            _enable_dynamic(pipeline)
            raw = store.get_metadata(LEXICON_META_KEY, "[]")
            try:
                entries = json.loads(raw)
            except json.JSONDecodeError:
                entries = []
            pipeline.planner.load_learned_lexicon(entries)
            learned_sizes[name] = len(entries)

        # cold == frozen static-bank baseline: dynamic lexicon stays OFF (the
        # exact same behavior the unified_golden_runner relies on).

        rows = []
        ok_count = 0
        for g in goldens:
            r = pipeline.ask(g["q"])
            ok, reasons = _grade(g, r)
            ok_count += int(ok)
            rows.append({
                "id": g["id"], "q": g["q"], "ok": ok, "teach": g.get("teach", False),
                "reasons": reasons,
                "answer": str(r.get("answer")),
                "chain": list(r.get("relation_chain") or []),
                "hops": len(r.get("walk_path_edges") or []),
                "path": list(r.get("walk_path_labels") or []),
                "honest": bool(r.get("honest_no_relation")),
                "heuristic_used": bool(r.get("heuristic_used")),
                "plan_confidence": float(r.get("confidence") or 0.0),
            })
        rows_by_domain[name] = rows
        lines.append(f"  RESULT: {ok_count}/{len(goldens)}")

    text = "\n".join(lines)
    print(text)
    out_p = Path(str(out_prefix) + f"_{mode}")
    out_p.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    out_p.with_suffix(".json").write_text(
        json.dumps({"mode": mode, "rows": rows_by_domain, "learned_lexicon_sizes": learned_sizes},
                   indent=2, default=str), encoding="utf-8")
    return {"mode": mode, "rows": rows_by_domain, "learned_lexicon_sizes": learned_sizes}


def _summarize(mode_results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("CROSS-DOMAIN PROBE - cold vs hot vs reload")
    print("=" * 78)
    by_domain = {}
    for res in mode_results:
        mode_label = res["mode"]
        for domain, rows in res["rows"].items():
            by_domain.setdefault(domain, {})[mode_label] = rows
    domains = list(by_domain.keys())
    print(f"{'domain':<24}{'cold':>6}{'hot':>6}{'reload':>8}")
    totals = {m: [0, 0] for m in ("cold", "hot", "reload")}
    for domain in domains:
        parts = []
        for mode in ("cold", "hot", "reload"):
            rows = by_domain[domain].get(mode)
            if rows is None:
                parts.append("   -  ")
                continue
            ok = sum(1 for r in rows if r["ok"])
            total = len(rows)
            totals[mode][0] += ok
            totals[mode][1] += total
            parts.append(f"{ok:>4}/{total}")
        print(f"{domain:<24}{'  '.join(p.lstrip() for p in parts).rjust(22)}")
    print("-" * 78)
    tots = "  ".join(f"{totals[m][0]:>3}/{totals[m][1]}" for m in ("cold", "hot", "reload"))
    print(f"{'TOTAL':<24}{tots.rjust(22)}")

    print("\nTaught vs held-out (hot):")
    for mode in ("hot", "reload"):
        for domain in domains:
            rows = by_domain[domain].get(mode)
            if not rows:
                continue
            taught = [r for r in rows if r["teach"]]
            held = [r for r in rows if not r["teach"]]
            print(f"  {mode:>6} {domain:<24} taught {sum(r['ok'] for r in taught)}/{len(taught)} "
                  f"held-out {sum(r['ok'] for r in held)}/{len(held)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="cold,hot,reload")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "generation_probe_results"))
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    mode_results = []
    for mode in modes:
        if mode == "reload":  # reload needs the persisted metadata written by hot
            for name, suite in SUITES.items():
                db = Path(suite["db"])
                if not db.exists():
                    continue
                store = SQLiteGraphStore.load_state(str(db))
                if not store.get_metadata(LEXICON_META_KEY, ""):
                    print(f"reload: {name} has no persisted lexicon - run hot first; running cold baseline for it")
        res = run_mode(mode, SUITES, out_prefix)
        mode_results.append(res)
    _summarize(mode_results)


if __name__ == "__main__":
    main()