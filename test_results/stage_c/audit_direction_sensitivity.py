"""Verify the frozen Stage C question set against the frozen graph.

Run AFTER generate_questions.py and after the builder has written
direction_multihop.db. This is a read-only check: it re-derives every claim from
the database and fails loudly on any disagreement. It never rewrites the question
set, because the file is called FROZEN -- a checker that repairs its own input
cannot fail, and a frozen set that silently changes is not frozen.

What it verifies, per question:

  direction_pairs     the declared `node` is the ONLY thing reachable in one hop
                      from the declared anchor under walker-faithful candidate
                      matching, so the question has exactly one right answer.
  control_anchor      the OTHER endpoint of the same stored edge is not also
                      reachable in one hop under the same label. This is the
                      direction claim: the graph must not contain a same-label
                      edge in both orientations for a directional relation.
  short_multi_hop     exactly ONE traversal of the declared chain exists from
                      the anchor. Several converging paths would grade a
                      tie-break between them rather than the chain.
  forbidden_reverse   the anchor has no incident edge of that label in either
                      orientation, so refusal is the only correct output.
  honesty / nonsense  the anchor text appears nowhere in the graph, otherwise
                      the expected refusal would be a miss.

Candidate matching mirrors walker/graph_walker.py: _collect_candidates only ever
traverses an edge FROM ITS SOURCE, and inverse_relations supplies the labels
{asked, declared inverse}. So the reachable set from a node is its outgoing
stored edges under either label, plus the edges the mirror pass emits from it
(incoming stored edges whose inverse label is one of the two). is_a is absent
from that map, so is_a edges are only reachable in their stored direction
(agreed 2026-10-02).

Usage:
    python test_results/stage_c/audit_direction_sensitivity.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
DB_PATH = STAGE_DIR / "direction_multihop.db"
QUESTIONS_PATH = STAGE_DIR / "direction_multihop_questions_frozen.json"

# Must match walker/graph_walker.py::INVERSE_RELATION_LABELS. is_a is absent by
# agreement; this module asserts the two agree at run time.
EXPECTED_INVERSE = {
    "causes": "caused_by", "caused_by": "causes",
    "precedes": "follows", "follows": "precedes",
    "part_of": "has_part", "has_part": "part_of",
}
INVERSE = dict(EXPECTED_INVERSE)


def load_store(db_path: Path):
    con = sqlite3.connect(str(db_path))
    try:
        labels = {nid: (lbl or "").strip().lower()
                  for nid, lbl in con.execute("SELECT id, label FROM nodes")}
        edges = [(labels.get(s, ""), r, labels.get(t, ""))
                 for s, t, r in con.execute(
                     "SELECT source_id, target_id, relation FROM edges")]
        return labels, edges
    finally:
        con.close()


def check_inverse_matches_walker() -> None:
    """The walker's inverse map is the contract for reachability; do not drift."""
    sys.path.insert(0, str(STAGE_DIR.parents[1]))
    from walker.graph_walker import INVERSE_RELATION_LABELS

    walker_map = {k: v for k, v in INVERSE_RELATION_LABELS.items() if k != "is_a"}
    if walker_map != EXPECTED_INVERSE:
        raise SystemExit(
            f"inverse map drift: walker={walker_map} audit={EXPECTED_INVERSE}. "
            "Reachability here would not model the real walker.")


def main() -> int:
    check_inverse_matches_walker()
    spec = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    labels, edges = load_store(DB_PATH)
    node_labels = set(labels.values())

    fwd: dict[tuple[str, str], set[str]] = {}
    mirrored: dict[tuple[str, str], set[str]] = {}
    for s, r, t in edges:
        fwd.setdefault((s, r), set()).add(t)
        if r in INVERSE:
            mirrored.setdefault((t, INVERSE[r]), set()).add(s)

    def cand(node: str, rel: str) -> set:
        """Walker-faithful reach: stored edges leaving `node`, plus the edges the
        mirror pass emits from it. The walker only traverses an edge from its
        source, so an incoming edge is reachable only through its mirror, which
        exists only for relations with a declared inverse."""
        labels = {rel}
        if rel in INVERSE:
            labels.add(INVERSE[rel])
        out: set = set()
        for label in labels:
            out |= fwd.get((node, label), set())
            out |= mirrored.get((node, label), set())
        return out

    def walks(node: str, chain: list[str]) -> list[list[str]]:
        frontier = [[node]]
        for rel in chain:
            nxt = [p + [c] for p in frontier for c in sorted(cand(p[-1], rel))]
            if not nxt:
                return []
            frontier = nxt
        return frontier

    problems: list[str] = []

    def fail(qid: str, msg: str) -> None:
        problems.append(f"{qid}: {msg}")

    for q in spec["questions"]:
        cat = q["cat"]

        if cat == "direction_pairs":
            anchor = q["path"][0]
            rel = q["rel"]
            reach = cand(anchor, rel)
            if reach != {q["node"]}:
                fail(q["id"], f"one hop from {anchor!r} under {rel!r} reaches "
                              f"{sorted(reach)}, declared {q['node']!r}")
            control_anchor = q.get("control_anchor")
            control_node = q.get("control_node")
            if control_anchor and control_node is None:
                # The other endpoint is asked about under the same relation; it
                # must resolve elsewhere, otherwise the edge is mirrored.
                same = cand(control_anchor, rel)
                if q["node"] in same:
                    fail(q["id"],
                         f"{control_anchor!r} also reaches {q['node']!r} under "
                         f"{rel!r}: the edge exists in both orientations")
            elif control_node is not None and control_node == q["node"]:
                fail(q["id"], f"control anchor {control_anchor!r} returns the same "
                              f"node {q['node']!r}, so the pair does not test direction")

        elif cat == "mirror_silence":
            reach = cand(q["anchor"], q["rel"])
            leaked = [n for n in q["forbidden_nodes"] if n in reach]
            if leaked:
                fail(q["id"], f"anchor {q['anchor']!r} reaches {leaked} under "
                              f"{q['rel']!r}: the one-way edge is readable backwards")
            if q["rel"] in INVERSE:
                fail(q["id"], f"{q['rel']!r} has a declared inverse, so its mirror "
                              "read is legal and this control is meaningless")

        elif cat == "short_multi_hop":
            chain = list(q["chain"])
            anchor = q["path"][0]
            found = [tuple(p) for p in walks(anchor, chain)]
            if tuple(q["path"]) not in found:
                fail(q["id"], f"declared path {q['path']} is not walkable via "
                              f"{'>'.join(chain)}; reachable: {sorted(set(found))[:4]}")
            elif len(set(found)) > 1:
                fail(q["id"], f"{len(set(found))} distinct traversals exist: "
                              f"{sorted(set(found))[:4]} -- the question does not "
                              "pin down one path")

        elif cat == "forbidden_reverse":
            reach = cand(q["anchor"], q["rel"])
            if reach:
                fail(q["id"], f"forbidden_reverse anchor {q['anchor']!r} is reachable "
                              f"to {sorted(reach)}")

        elif cat in ("honesty_out_of_graph", "nonsense_fallback"):
            named = [n for n in node_labels if n in q["q"].lower()]
            if named:
                fail(q["id"], f"expects a refusal but the question names graph "
                              f"nodes: {named}")

    by_cat = Counter(q["cat"] for q in spec["questions"])
    print(f"graph: {len(node_labels)} nodes, {len(edges)} edges")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<24} {n}")

    graded = [q for q in spec["questions"] if q["cat"] == "mirror_silence"]
    print(f"\nmirror-silence controls: {len(graded)}")

    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nPASS: every frozen question matches the frozen graph")
    return 0


if __name__ == "__main__":
    sys.exit(main())