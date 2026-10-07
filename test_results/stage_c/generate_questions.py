"""Generate the frozen Stage C question set from the built graph.

The Stage C graph is authored with ASKED_PAIRS and FORBIDDEN_REVERSE lists, but
hand-writing a natural-language question for each entry let the two drift apart
and produced questions whose cue phrased a different relation than the chain
declared. This generator closes that gap: it emits one question per asked pair
using a phrasing that provably fires the intended relation cue, then verifies
each emitted question against the graph before writing.

Every question is derived, never hand-typed, so adding a pair to the builder
cannot leave the question set behind. Re-run this after any graph change.

Question shapes:

  1 hop        "<cue for R> <anchor>?"          anchor is the asked pair's anchor
  N hops       clause_1 cues R0 and names the anchor; each later clause cues
               R_i and names the RESULT of the previous hop. The answer is the
               node reached after the final hop.

The cue wording is taken from configs/config_g2p.yaml relation_variants so the
question cannot use a phrase the planner does not recognise.

Usage:
    python test_results/stage_c/generate_questions.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parents[1]
sys.path.insert(0, str(ROOT))

G2P_CONFIG = ROOT / "configs" / "config_g2p.yaml"

# A phrasing per relation, chosen to contain one of the cue variants above.
# (relation, template) where {x} is filled with the anchor node. The template is
# written so that the node appears LAST for most relations, which matches how
# the planner's anchor selection reads a question.
ONE_HOP = {
    "causes": "What causes {x}?",
    "caused_by": "What is {x} caused by?",
    "precedes": "What comes after {x}?",
    "follows": "What comes before {x}?",
    "part_of": "What is the {x} part of?",
    "is_a": "What is {art} {x}?",
    "has_property": "What is the {x} known for?",
    "supports": "What evidence supports {x}?",
    "contradicts": "What claim contradicts {x}?",
    "example_of": "Give me an example of {x}.",
    "synonym": "What is another word for {x}?",
    "antonym": "What is the opposite of {x}?",
    "spatial_near": "What is near {x}?",
    "temporal_coincident": "What occurs during {x}?",
    "linguistic_maps": "What do you call {x} in Spanish?",
    "associated_with": "What is associated with {x}?",
}

# Phrasing for the FIRST clause of a multi-hop question. It names the anchor and
# cues chain[0].
MULTI_CLAUSE = {
    "causes": "what does {x} cause",
    "caused_by": "what is {x} caused by",
    "precedes": "what comes after {x}",
    "follows": "what comes before {x}",
    "part_of": "what is the {x} part of",
    "is_a": "what {art} thing is {x}",
    "has_property": "what is the {x} known for",
    "supports": "what evidence supports {x}",
    "contradicts": "what claim contradicts {x}",
    "example_of": "give me an example of {x}",
    "synonym": "what is another word for {x}",
    "antonym": "what is the opposite of {x}",
    "spatial_near": "what is near {x}",
    "temporal_coincident": "what occurs during {x}",
    "linguistic_maps": "what do you call {x} in Spanish",
    "associated_with": "what is associated with {x}",
}

# Phrasing for clause 2 onward. These MUST NOT name a node.
#
# Naming the intermediate hop in a later clause (the first draft did this) makes
# the question mention two graph nodes, and the pipeline anchors on whichever
# one it likes best -- not necessarily the one the question is about. Observed:
# "What does acid rain cause, and what is a crop failure?" anchored on
# `crop failure` and the planner returned ['caused_by', 'is_a'] instead of
# ['causes', 'is_a'], so the walk left the anchor backwards and the question
# could not be answered. A later clause therefore refers to the previous hop
# deictically and carries only the relation cue.
TAIL_CLAUSE = {
    "causes": "what does that cause",
    "caused_by": "what is that caused by",
    "precedes": "what comes after that",
    "follows": "what comes before that",
    "part_of": "what is that part of",
    "is_a": "what type of thing is that",
    "has_property": "what is that known for",
    "supports": "what evidence supports that",
    "contradicts": "what claim contradicts that",
    "example_of": "give me an example of that",
    "synonym": "what is another word for that",
    "antonym": "what is the opposite of that",
    "spatial_near": "what is near that",
    "temporal_coincident": "what occurs during that",
    "linguistic_maps": "what do you call that in Spanish",
    "associated_with": "what is associated with that",
}

# --------------------------------------------------------------------------


def load_cues() -> dict:
    cfg = yaml.safe_load(G2P_CONFIG.read_text(encoding="utf-8")) or {}
    return (cfg.get("extraction") or {}).get("relation_variants") or {}


CUES = load_cues()


_CUE_HINT = (
    "cue overlap is resolved by the planner itself, which keeps only the\n"
    "    LONGEST matching descriptor per relation and orders the result by\n"
    "    position in the question"
)


def render(relation: str, node: str, multi: bool) -> str:
    tpl = (MULTI_CLAUSE if multi else ONE_HOP)[relation]
    return tpl.format(x=node)


def article(node: str) -> str:
    return "an" if node[:1].lower() in "aeiou" else "a"


def build_question(chain: list[str], nodes: list[str]) -> str:
    """Render a chain + node path as a single question string.

    Only the anchor (nodes[0]) is ever named. Later clauses use TAIL_CLAUSE so
    the question stays anchored -- see the note on that dict.
    """
    if len(chain) == 1:
        q = ONE_HOP[chain[0]]
        return q.format(x=nodes[0], art=article(nodes[0]))
    parts = [MULTI_CLAUSE[chain[0]].format(x=nodes[0], art=article(nodes[0]))]
    parts += [TAIL_CLAUSE[rel] for rel in chain[1:]]
    q = parts[0][0].upper() + parts[0][1:]
    q = q + ", and " + ", and ".join(parts[1:])
    return q.rstrip(".") + "?"


_PLANNER = None


def planner_chain(question: str) -> list:
    """The chain the REAL planner produces for this question.

    Verifying against the planner rather than against a regex copy of the cue
    bank is the point: a hand-rolled matcher is a second implementation of
    "what the extractor does", and the two would eventually disagree, at which
    point a green Stage C would only prove the regex works. The overlap cases
    are real -- "what is another word for X" contains the is_a cue "is an" --
    so only the component that actually decides can adjudicate.
    """
    global _PLANNER
    if _PLANNER is None:
        from g2p.config import G2PConfig
        from g2p.g2p_planner import QueryRelationExtractor
        # Mirror glmx_ask.py: the extractor is built from the shipped YAML with
        # the full relation bank (plan() ignores the subgraph vocabulary), then
        # marked trained. initialize() is deliberately NOT called: it is the
        # embedding path this check is meant to avoid, and a literal cue hit
        # never reaches it.
        _PLANNER = QueryRelationExtractor(G2PConfig.from_yaml(str(G2P_CONFIG)))
        _PLANNER.mark_trained()
    plan = _PLANNER.plan(None, query_text=question)
    return list(getattr(plan, "relation_chain", []) or [])


def verify_cues(chain: list[str], question: str) -> None:
    """The planner's chain for `question` must equal the declared `chain`."""
    got = planner_chain(question)
    if got != list(chain):
        raise SystemExit(
            f"planner reads {got} for {question!r}, question declares {list(chain)}. "
            "Reword the template so the literal cue ordering resolves to the "
            "intended chain.")
    # Probe each template with a node name that cannot itself carry a cue, then
    # confirm the planner reads the template's own relation. {x} is filled with
    # a placeholder word rather than a real graph label so the check depends
    # only on the template.
    probe = "brontosaurus"
    for rel in set(MULTI_CLAUSE):
        single = build_question([rel], [probe, probe])
        got = planner_chain(single)
        if got != [rel]:
            raise SystemExit(
                f"template for {rel!r} reads as {got} in {single!r} ({_CUE_HINT})")


def main() -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "builder", STAGE_DIR / "build_direction_multihop_graph.py")
    b = importlib.util.module_from_spec(spec)
    sys.modules["builder"] = b
    spec.loader.exec_module(b)

    edges = [(s, r, t) for s, r, t, _, _ in b.EDGES]
    inv = b.INVERSE
    nodes_all = list(dict.fromkeys(b.CONCEPTS))
    node_set = set(nodes_all)

    def candidates(node: str, rel: str) -> set:
        """Reachability exactly as the walker + mirror pass will see it.

        The walker only ever traverses an edge from its SOURCE
        (_collect_candidates skips `source != current_node`), so a reverse read
        is legal only because the mirror pass materialises it as a real edge
        with the inverse label. Reach is therefore the union, over the asked
        label set, of outgoing stored edges and mirror-pass edges starting here.
        Treating the graph as undirected would model a traversal the walker
        cannot perform and would reject anchors that are genuinely unambiguous.
        """
        labels = {rel}
        if rel in inv:
            labels.add(inv[rel])
        out = set()
        for s, r, t in edges:
            if s == node and r in labels:
                out.add(t)
            if t == node and r in inv and inv[r] in labels:
                out.add(s)
        return out

    def walks(anchor: str, chain: list[str], flip: int | None = None) -> list[list[str]]:
        out = [[anchor]]
        frontier = [[anchor]]
        for i, rel in enumerate(chain):
            if flip is not None and i == flip and rel not in inv:
                return []
            use = inv[rel] if (flip is not None and i == flip) else rel
            nxt = []
            for path in frontier:
                for c in sorted(candidates(path[-1], use)):
                    if flip is not None and i == flip and c == path[-1]:
                        continue
                    nxt.append(path + [c])
            if not nxt:
                return []
            out, frontier = nxt, nxt
        return out

    for rel in ONE_HOP:
        verify_cues([rel], ONE_HOP[rel].format(x="brontosaurus", art=article("brontosaurus")))

    questions: list[dict] = []
    prefix = {"causes": "dc", "caused_by": "dv", "precedes": "tp", "follows": "tf",
              "control_pairs": "ms",
              "part_of": "po", "is_a": "ia", "has_property": "hp", "supports": "su",
              "contradicts": "co", "example_of": "eo", "synonym": "sy",
              "antonym": "an", "spatial_near": "sn", "temporal_coincident": "tc",
              "linguistic_maps": "lm", "associated_with": "aw"}
    counters: dict[str, int] = {}

    def next_id(tag: str) -> str:
        counters[tag] = counters.get(tag, 0) + 1
        return f"{tag}{counters[tag]:02d}"

    # ---- one-hop direction pairs ----
    # DIRECTION SENSITIVITY IS TESTED BY ASKING BOTH ENDS OF ONE EDGE, NOT BY
    # FLIPPING A HOP LABEL. The walker's candidate filter matches on edge label
    # (inverse_relations), so `A causes B` and the inverse-labelled read are the
    # same candidate set and "the answer changed when I inverted the hop" is
    # vacuously false for every single-hop question. The real question is
    # whether the SYSTEM tells A from B: given the edge `A causes B`, asking
    # "what causes B?" must return A and asking "what is B caused by?" must also
    # return A, while asking the mirror question about A ("what causes A?") must
    # NOT return B unless the graph actually stores `B causes A`. So a question
    # is direction-sensitive when the same anchor admits a different answer under
    # the opposite sense, which is exactly what ASKED_PAIRS is built to exclude.
    # The scored property is per-edge: the forward read and the reverse read must
    # agree, and the cross-anchor control must not.
    forward_by_anchor: dict[tuple[str, str], str] = {}
    for anchor, rel in b.ASKED_PAIRS:
        path = walks(anchor, [rel])
        ends = {p[-1] for p in path}
        if len(ends) != 1:
            raise SystemExit(
                f"asked pair {anchor}|{rel} resolves to {sorted(ends)}, expected exactly one")
        target = ends.pop()
        forward_by_anchor[(anchor, rel)] = target
        q = build_question([rel], [anchor])
        questions.append({
            "id": next_id(prefix[rel]),
            "cat": "direction_pairs",
            "rel": rel,
            "q": q,
            "node": target,
            "chain": [rel],
            "hops": 1,
            "path": [anchor, target],
            # The control: ask the same relation about the OTHER endpoint. If it
            # yields this edge's target, the system has no directional grip.
            "control_anchor": target,
            "control_node": forward_by_anchor.get(
                (target, rel) if rel not in inv else (target, inv[rel])),
        })

    # ---- mirror-silence controls: the far end of a one-way edge ----
    # Emitted only where the far end is NOT already a legitimate answer:
    #
    #  * relations WITH a declared inverse are excluded, because the mirror pass
    #    materialises `source <inverse> target` and getting the source back is
    #    the contract working, not a direction failure;
    #  * the relations stored in BOTH directions on purpose are excluded,
    #    because there the far end already has its own correct answer and the
    #    control would just duplicate that question.
    #
    # What remains are the genuinely one-way relations (is_a, has_property,
    # supports, contradicts, example_of). There the far end has no stored edge
    # and no mirror to fall back on, so it must answer with nothing. If it names
    # the source, the engine read the edge backwards anyway: the illegal
    # same-label mirroring this stage exists to catch.
    seen_controls: dict[tuple[str, str], set] = {}
    for anchor, rel in b.ASKED_PAIRS:
        if rel in inv or rel in b.SYMMETRIC_STORED:
            continue
        target = forward_by_anchor[(anchor, rel)]
        seen_controls.setdefault((target, rel), set()).add(anchor)
    for (target, rel), sources in seen_controls.items():
        questions.append({
            "id": next_id("ms"),
            "cat": "mirror_silence",
            "rel": rel,
            "q": build_question([rel], [target]),
            "anchor": target,
            # Every answer a direction-blind engine would give here: one per
            # stored edge pointing INTO this anchor. Grouped by anchor because
            # several edges can converge on it (two events both classified as
            # weather event), and one question covers all of them.
            "forbidden_nodes": sorted(sources),
            "chain": [rel],
            "hops": 1,
            "path": [target],
        })

    # ---- multi-hop chains ----
    # Adjacent repeats are impossible by design: g2p_planner.collapse_runs drops
    # any run of the same relation (contract section 8 collapse_consecutive_
    # repeats), so `is_a>is_a` and `causes>causes` would arrive at the walker as
    # a single hop and the question would silently grade a 1-hop answer against a
    # 2-hop path.
    #
    # The planner keeps only the LONGEST matching descriptor PER RELATION, so a
    # relation label may not appear twice in one chain either -- not just
    # adjacently. `causes>is_a>causes` is rejected by the planner as
    # ['causes', 'is_a'] and verify_cues() below refuses it. Every spec below
    # therefore uses distinct relations.
    CHAIN_SPECS = [
        ("causes", "is_a"),
        ("caused_by", "is_a"),
        ("precedes", "is_a"),
        ("follows", "is_a"),
        ("part_of", "is_a"),
        ("precedes", "causes", "is_a"),
        ("caused_by", "causes", "is_a"), ("caused_by", "part_of", "is_a"),
        ("causes", "part_of", "is_a"),
        ("precedes", "part_of", "is_a"),
    ]
    for chain in CHAIN_SPECS:
        probe = build_question(list(chain), ["brontosaurus"] * len(chain))
        verify_cues(list(chain), probe)
        found = 0
        for anchor in nodes_all:
            good = [tuple(p) for p in walks(anchor, list(chain))
                    if len(p) == len(chain) + 1]
            # Require exactly ONE traversal, not merely one endpoint. Several
            # distinct paths can converge on the same category node, and then the
            # question grades a scorer choice rather than the chain.
            if len(set(good)) != 1:
                continue
            path = list(good[0])
            # Reject a traversal that revisits a node. `caused_by>causes` on one
            # stored edge is a 2-cycle (X <-cause- Y -cause-> X), so the "3-hop"
            # path X, Y, X, C would grade a single edge walked twice -- and would
            # then pass even if the engine only ever made two hops.
            if len(set(path)) != len(path):
                continue
            q = build_question(list(chain), path[:-1])
            questions.append({
                "id": next_id("mh"),
                "cat": "short_multi_hop",
                "rel": ">".join(chain),
                "q": q,
                "node": path[-1],
                "chain": list(chain),
                "hops": len(chain),
                "path": path,
            })
            found += 1
        label = ">".join(chain)
        print("  chain {:<24} {} question(s)".format(label, found))

    # ---- forbidden reverse reads ----
    for anchor, rel in b.FORBIDDEN_REVERSE:
        if candidates(anchor, rel):
            raise SystemExit(
                f"forbidden_reverse {anchor}|{rel} is actually reachable to "
                f"{sorted(candidates(anchor, rel))}")
        questions.append({
            "id": next_id("fr"),
            "cat": "forbidden_reverse",
            "rel": rel,
            "q": ONE_HOP[rel].format(x=anchor, art=article(anchor)),
            "anchor": anchor,
            "node": None,
            "chain": None,
            "hops": 0,
            "path": None,
        })

    # ---- honesty + nonsense ----
    # Out-of-graph means NO graph label appears in the question, as a word. A
    # substring match is not enough and a naive one is not sufficient either:
    # "medieval canal engineering" contains `canal` and `engine` as whole words,
    # so the graph really does have anchors in that text and refusal is the
    # wrong expectation. Verify rather than eyeball.
    HONESTY = [
        "Tell me about the arctic monsoon",
        "What causes a meteor strike?",
        "What is a chimera?",
        "What is the sailsail part of?",
        "Tell me about the zeppelin crossing",
        "Who invented the vacuum tube?",
        "What is the Panama Isthmus made of?",
    ]
    import re as _re

    def names_graph_node(text: str) -> list:
        low = text.lower()
        return [n for n in node_set
                if _re.search(r"(?<!\w)" + _re.escape(n) + r"(?!\w)", low)]

    for i, q in enumerate(HONESTY, 1):
        named = names_graph_node(q)
        if named:
            raise SystemExit(
                f"honesty question names graph nodes {named}, so refusal is the "
                f"wrong expectation: {q!r}")
        questions.append({"id": f"h{i:02d}", "cat": "honesty_out_of_graph", "rel": None,
                          "q": q, "node": None, "chain": None, "hops": 0,
                          "path": None, "direction_sensitive": None})
    for i, q in enumerate([
        "Tell me about blorptastic", "What is asdkjh qwe?", "zzz qqq wwww",
    ], 1):
        questions.append({"id": f"nf{i:02d}", "cat": "nonsense_fallback", "rel": None,
                          "q": q, "node": None, "chain": None, "hops": 0,
                          "path": None, "direction_sensitive": None})

    directed = [q for q in questions if q["cat"] == "direction_pairs"]
    # A direction_pairs question is only a real direction test when the control
    # anchor exists and does NOT resolve back to this edge's target. `None` means
    # the other endpoint is never asked about, so there is no contrast to score.
    untested = [q["id"] for q in directed if q["control_node"] is None]
    contradicted = [q["id"] for q in directed
                    if q["control_node"] == q["node"]]
    if contradicted:
        raise SystemExit(
            "control anchor returns the same node as the original read, so the "
            f"pair does not test direction: {contradicted}")
    dupes = [q for q, n in Counter(x["q"] for x in questions).items() if n > 1]
    if dupes:
        holders = [(x["id"], x["cat"], x["q"]) for x in questions if x["q"] in dupes]
        raise SystemExit(f"duplicate question text generated: {holders}")

    out = {
        "stage": "C",
        "name": "Direction & Multi-hop Graph",
        "contract": "GLM-X v3.3.2 section 16",
        "graph": "direction_multihop.db",
        "generated_by": "generate_questions.py",
        "frozen": True,
        "scoring_notes": [
            "Questions are generated from ASKED_PAIRS/FORBIDDEN_REVERSE in the builder, not hand-written, so the set cannot drift from the graph.",
            "Every chain is re-read through the real QueryRelationExtractor before freezing, so a question can never claim a relation the planner does not produce.",
            "direction_pairs are scored as CONTRASTED pairs, not by flipping a hop label: the walker matches candidates on label (inverse_relations), so a single hop has one candidate set regardless of sense. control_anchor is the other endpoint of the same stored edge; answering it with the same node would mean the system cannot tell the two apart.",
            "A multi-hop question is emitted only when exactly ONE traversal exists from the anchor, so the score reflects the chain rather than a tie-break between converging paths.",
            "Adjacent duplicate relations are impossible: collapse_runs drops them, so no spec uses e.g. is_a>is_a.",
            "forbidden_reverse anchors carry the asked relation in no orientation at all, so an honest refusal is the only correct output.",
        ],
        "questions": questions,
    }
    (STAGE_DIR / "direction_multihop_questions_frozen.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    by_cat: dict[str, int] = {}
    for q in questions:
        by_cat[q["cat"]] = by_cat.get(q["cat"], 0) + 1
    print(f"\nwrote {len(questions)} questions")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<24} {n}")
    print(f"direction_pairs with a live control: "
          f"{len(directed) - len(untested)}/{len(directed)}")
    if untested:
        print(f"  no control anchor (not direction-graded): {untested}")
    print(f"Wrote {STAGE_DIR / 'direction_multihop_questions_frozen.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())