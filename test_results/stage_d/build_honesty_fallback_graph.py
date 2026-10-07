"""Stage D -- Honesty & Fallback graph.

Contract: GLM-X v3.3.2 sections 6, 8 and 16.

WHAT THIS STAGE IS FOR
    Stages A-C asked "does the machine walk correctly?". Stage D asks the
    opposite question: "when there is nothing to walk, does it say so, or does
    it make something up?". A graph engine that answers every question is
    indistinguishable from a hallucinating one, so this graph is built to be
    FULL OF GAPS on purpose.

    It is small and hand-auditable so every gap can be verified by reading the
    edge list below, and it carries all 16 canonical relations so the
    no-illegal-mirroring guarantee can be asserted against the full vocabulary
    rather than a convenient subset.

DESIGN: FOUR KINDS OF GAP
    1. control              -- a stored edge exists and is unambiguous. The
                               system must answer normally. Without these, a
                               system that refuses everything would score 100%.
    2. missing_relation     -- the anchor IS in the graph but has NO edge of
                               the asked relation in either orientation. The
                               system must refuse rather than substitute a
                               relation it does have (`abacus` is_a tool does
                               not license "the abacus is part of the
                               toolbox").
    3. inverse_direction    -- the asked relation EXISTS but only in the
                               opposite stored direction, on a relation with
                               NO declared inverse. Reading it backwards is a
                               different, false claim. This is the trap that
                               illegal mirroring would fall into, so it is the
                               sharpest test in the stage.
    4. out_of_graph         -- the anchor is absent from the graph entirely.

MIRRORING
    Only the four declared pairs may be mirrored, and the mirrored copy always
    carries the inverse label. `is_a` is NOT mirrored: its declared inverse
    label is `is_a` itself, so a mirrored copy would be indistinguishable by
    label from a real forward edge (scripts/glmx_ask.py
    MIRROR_SELF_INVERSE_IS_A = False). This builder therefore treats is_a as
    forward-only, matching the orchestrator, and its guards are written against
    that same model so the audit and the runtime cannot silently disagree.

    Symmetric relations (synonym, antonym, spatial_near, temporal_coincident)
    have no declared inverse label, so the mirror pass may NOT synthesise their
    reverse. Their symmetry is stored explicitly in both directions instead.
    That asymmetry is deliberate and is the whole point of category 3.

Usage:
    python test_results/stage_d/build_honesty_fallback_graph.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore  # noqa: E402

DB_PATH = ROOT / "test_results" / "stage_d" / "honesty_fallback.db"

CANONICAL_RELATIONS = [
    "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
    "contradicts", "supports", "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident", "spatial_near",
    "linguistic_maps",
]

# Runtime inverse labels, mirroring scripts/glmx_ask.py:INVERSE_RELATION_LABELS
# with is_a removed (MIRROR_SELF_INVERSE_IS_A = False).
INVERSE = {
    "causes": "caused_by", "caused_by": "causes",
    "precedes": "follows", "follows": "precedes",
    "part_of": "has_part", "has_part": "part_of",
}
MIRRORED = frozenset(INVERSE)  # the ONLY relations the mirror pass may copy

NO_DECLARED_INVERSE = (
    "is_a", "has_property", "contradicts", "supports", "associated_with",
    "example_of", "synonym", "antonym", "temporal_coincident",
    "spatial_near", "linguistic_maps",
)

# Stored in BOTH directions by hand because the mirror pass must not do it.
SYMMETRIC_STORED = (
    "synonym", "antonym", "spatial_near", "temporal_coincident",
    "linguistic_maps",
)

# --------------------------------------------------------------------------
CONCEPTS = [
    # taxonomy spine
    "animal", "mammal", "bird", "reptile", "insect", "fish", "plant", "tree",
    "flower", "metal", "tool", "instrument", "toy", "vehicle",
    "country", "city", "sport", "natural event", "device", "place",
    # mammals
    "cat", "dog", "horse",
    # birds / reptile / insects / fish
    "eagle", "sparrow", "snake", "butterfly", "bee", "trout",
    # plants
    "oak", "pine", "maple", "fern", "moss", "rose", "tulip",
    # materials / tools / instruments / toys / vehicles
    "iron", "copper", "hammer", "wrench", "screwdriver", "chisel", "saw",
    "abacus", "harmonica", "kite", "marble", "canoe", "toolbox",
    # geography
    "france", "germany", "spain", "paris", "berlin", "madrid", "rhine",
    "versailles",
    # sport
    "tennis", "chess", "football", "racket", "goal", "knight",
    # events
    "earthquake", "volcano", "storm", "flood", "eruption", "landslide",
    "building collapse", "road closure",
    # devices
    "microscope", "compass",
    # timeline (precedes/follows only)
    "spore", "sprout", "plantlet",
    # evidence / claims
    "fossil record", "evolution theory", "young earth claim",
    # lexicon
    "perro", "gato", "spanner", "songbird", "physician", "doctor",
    "cabinet", "cupboard",
    # seasons
    "christmas", "new year", "autumn", "harvest",
    # properties
    "furry", "red", "magnetic", "tall",
    # opposites
    "hot", "cold", "wet", "dry", "north", "south", "big", "small",
]

EDGES: list[tuple[str, str, str, float, float]] = []


def _fwd(src: str, rel: str, tgt: str, s: float = 0.95, c: float = 0.95) -> None:
    EDGES.append((src, rel, tgt, s, c))


def _pair(a: str, b: str, left: str, right: str,
          s: float = 0.95, c: float = 0.95) -> None:
    """Store a declared inverse pair in both directions.

    Both members are stored rather than trusting only the runtime mirror so
    that direction is readable straight out of the database.
    """
    _fwd(a, left, b, s, c)
    _fwd(b, right, a, s, c)


def _both_symmetric(a: str, b: str, rel: str,
                    s: float = 0.95, c: float = 0.95) -> None:
    _fwd(a, rel, b, s, c)
    _fwd(b, rel, a, s, c)


# ===================== is_a : taxonomy, forward only =======================
for _leaf, _parent in (
    ("cat", "mammal"), ("dog", "mammal"), ("horse", "mammal"),
    ("mammal", "animal"),
    ("eagle", "bird"), ("sparrow", "bird"), ("bird", "animal"),
    ("snake", "reptile"), ("reptile", "animal"),
    ("butterfly", "insect"), ("bee", "insect"), ("insect", "animal"),
    ("trout", "fish"), ("fish", "animal"),
    ("oak", "tree"), ("pine", "tree"), ("maple", "tree"), ("tree", "plant"),
    ("fern", "plant"), ("moss", "plant"),
    ("rose", "flower"), ("tulip", "flower"), ("flower", "plant"),
    ("iron", "metal"), ("copper", "metal"),
    ("hammer", "tool"), ("wrench", "tool"), ("screwdriver", "tool"),
    ("chisel", "tool"), ("saw", "tool"), ("abacus", "tool"),
    ("harmonica", "instrument"),
    ("kite", "toy"), ("marble", "toy"),
    ("canoe", "vehicle"),
    ("paris", "city"), ("berlin", "city"), ("madrid", "city"), ("city", "place"),
    ("france", "country"), ("germany", "country"), ("spain", "country"),
    ("country", "place"),
    ("tennis", "sport"), ("chess", "sport"), ("football", "sport"),
    ("earthquake", "natural event"), ("volcano", "natural event"),
    ("storm", "natural event"), ("flood", "natural event"),
    ("eruption", "natural event"), ("landslide", "natural event"),
    ("microscope", "device"), ("compass", "device"),
):
    _fwd(_leaf, "is_a", _parent, 0.97, 0.97)

# ===================== part_of : forward only, concrete wholes ===============
for _part, _whole in (
    ("hammer", "toolbox"), ("wrench", "toolbox"), ("screwdriver", "toolbox"),
    ("chisel", "toolbox"), ("saw", "toolbox"),
    ("paris", "france"), ("berlin", "germany"), ("madrid", "spain"),
    ("rhine", "germany"),
):
    _fwd(_part, "part_of", _whole, 0.96, 0.96)

# ===================== causes / caused_by : stored both ways =================
_pair("earthquake", "building collapse", "causes", "caused_by", 0.96, 0.96)
_pair("building collapse", "road closure", "causes", "caused_by", 0.94, 0.94)
_pair("volcano", "eruption", "causes", "caused_by", 0.96, 0.96)
_pair("storm", "flood", "causes", "caused_by", 0.96, 0.96)

# ===================== precedes / follows : strictly linear ==================
_pair("spore", "sprout", "precedes", "follows", 0.95, 0.95)
_pair("sprout", "plantlet", "precedes", "follows", 0.95, 0.95)

# ===================== synonym / antonym : symmetric, stored both ways ======
_both_symmetric("sparrow", "songbird", "synonym", 0.96, 0.96)
_both_symmetric("wrench", "spanner", "synonym", 0.96, 0.96)
_both_symmetric("physician", "doctor", "synonym", 0.96, 0.96)
_both_symmetric("cabinet", "cupboard", "synonym", 0.95, 0.95)

_both_symmetric("hot", "cold", "antonym", 0.96, 0.96)
_both_symmetric("wet", "dry", "antonym", 0.96, 0.96)
_both_symmetric("north", "south", "antonym", 0.96, 0.96)
_both_symmetric("big", "small", "antonym", 0.95, 0.95)

# ===================== spatial_near / temporal_coincident : symmetric ========
_both_symmetric("cat", "dog", "spatial_near", 0.92, 0.92)
_both_symmetric("oak", "pine", "spatial_near", 0.92, 0.92)
_both_symmetric("paris", "versailles", "spatial_near", 0.93, 0.93)
_both_symmetric("hammer", "wrench", "spatial_near", 0.93, 0.93)

_both_symmetric("christmas", "new year", "temporal_coincident", 0.95, 0.95)
_both_symmetric("autumn", "harvest", "temporal_coincident", 0.95, 0.95)

# ===================== one-way relations, forward only ======================
# These are the relations that give category 3 its teeth: each exists in ONE
# stored direction, and none of them has a declared inverse, so the reverse read
# is a different and false claim.
_fwd("sparrow", "example_of", "bird", 0.96, 0.96)
_fwd("trout", "example_of", "fish", 0.96, 0.96)
_fwd("tulip", "example_of", "flower", 0.96, 0.96)
_fwd("butterfly", "example_of", "insect", 0.96, 0.96)
_fwd("hammer", "example_of", "tool", 0.96, 0.96)

_fwd("tennis", "associated_with", "racket", 0.96, 0.96)
_fwd("chess", "associated_with", "knight", 0.96, 0.96)
_fwd("football", "associated_with", "goal", 0.96, 0.96)

_fwd("cat", "has_property", "furry", 0.96, 0.96)
_fwd("rose", "has_property", "red", 0.96, 0.96)
_fwd("iron", "has_property", "magnetic", 0.96, 0.96)
_fwd("oak", "has_property", "tall", 0.96, 0.96)

_fwd("fossil record", "supports", "evolution theory", 0.96, 0.96)
_fwd("fossil record", "contradicts", "young earth claim", 0.96, 0.96)

_both_symmetric("dog", "perro", "linguistic_maps", 0.95, 0.95)
_both_symmetric("cat", "gato", "linguistic_maps", 0.95, 0.95)

# --------------------------------------------------------------------------
# Anchors whose edges are deliberately SPARSE, so "what else does this connect
# to" has no answer. These are the category-2 anchors.
SPARSE_ANCHORS = {
    "abacus": {"is_a"},          # a tool, and that is all it knows
    "microscope": {"is_a"},
    "marble": {"is_a"},
    "canoe": {"is_a"},
    "kite": {"is_a"},
    "harmonica": {"is_a"},
    "oak": {"is_a", "spatial_near", "has_property"},
    "rose": {"is_a", "has_property"},
    "butterfly": {"is_a", "example_of"},
    "trout": {"is_a", "example_of"},
    "chess": {"is_a", "associated_with"},
    "spore": {"precedes", "follows"},
}

# (anchor, relation) pairs the system is expected to REFUSE.
#   missing_relation  -- the anchor exists but has no such relation at all
#   inverse_direction -- the relation exists, but only the other way round, on
#                        a relation with no declared inverse
#
# PLANTLET_PRECEDES_NOTE -- a pair that was WRONGLY listed here and has been
# moved to CONTROL_PAIRS. `("plantlet", "precedes")` was asserted to be an
# unbridgeable gap. It is not. The graph stores `plantlet follows sprout`, and
# follows is the declared inverse of precedes, so the walker accepts that edge
# for an asked `precedes` and "What comes after the plantlet?" is correctly
# answered with `sprout`. The old validator missed this because it matched the
# asked relation label exactly instead of over the declared inverse pair. A
# must-refuse entry that the system is actually right to answer is worse than no
# entry at all: it encodes a false belief about the graph.
MUST_REFUSE: dict[str, list[tuple[str, str]]] = {
    "missing_relation": [
        ("abacus", "part_of"),
        ("abacus", "causes"),
        ("microscope", "spatial_near"),
        ("marble", "synonym"),
        ("canoe", "antonym"),
        ("kite", "has_property"),
        ("harmonica", "temporal_coincident"),
        ("oak", "synonym"),
        ("rose", "supports"),
        ("chess", "example_of"),
        ("butterfly", "spatial_near"),
    ],
    "inverse_direction": [
        ("evolution theory", "supports"),
        ("red", "has_property"),
        ("insect", "example_of"),
        ("racket", "associated_with"),
        ("young earth claim", "contradicts"),
        ("bird", "example_of"),
        ("tall", "has_property"),
    ],
}

# Control anchors: a stored forward edge exists and must be the unique answer.
#
# ("plantlet", "precedes") joins this list from MUST_REFUSE; see the note above.
#
# ("building collapse", "caused_by") was REMOVED as ambiguous. The graph stores
# BOTH `building collapse caused_by earthquake` AND `building collapse causes road
# closure`, and `causes` is the declared inverse of `caused_by`, so both targets
# are one hop away under the walker's own candidate filter. The old validator
# called it unique because it matched the asked label exactly; it only ever
# answered `earthquake` because `GraphWalker._select_index` prefers an exact
# label match over a higher-scoring inverse. That is a tie-break, not a
# guarantee, so it must not be what a control is built on. Replaced by
# ("earthquake", "causes"), which is genuinely 1-hop unique: `earthquake` has no
# `caused_by` edge, so nothing competes for the `causes` slot.
#
# ("fossil record", "supports") is added so `supports` has a POSITIVE control
# beside the negative one in MUST_REFUSE["inverse_direction"]. Without it the
# stage would only ever prove the system refuses `supports` backwards, never
# that it walks it forwards.
CONTROL_PAIRS: list[tuple[str, str]] = [
    ("cat", "is_a"),
    ("hammer", "is_a"),
    ("wrench", "part_of"),
    ("screwdriver", "part_of"),
    ("storm", "causes"),
    ("flood", "caused_by"),
    ("volcano", "causes"),
    ("earthquake", "causes"),
    ("spore", "precedes"),
    ("plantlet", "follows"),
    ("plantlet", "precedes"),
    ("hot", "antonym"),
    ("sparrow", "synonym"),
    ("oak", "spatial_near"),
    ("cat", "has_property"),
    ("dog", "linguistic_maps"),
    ("sparrow", "example_of"),
    ("tennis", "associated_with"),
    ("paris", "part_of"),
    ("fossil record", "contradicts"),
    ("fossil record", "supports"),
    ("trout", "example_of"),
    ("autumn", "temporal_coincident"),
    ("microscope", "is_a"),
]


def _mirror_edges() -> set[tuple[str, str, str]]:
    """Reverse edges the runtime mirror pass is allowed to synthesise."""
    out = set()
    for src, rel, tgt, _, _ in EDGES:
        if rel in MIRRORED:
            out.add((tgt, INVERSE[rel], src))
    return out


def asked_labels(rel: str) -> set[str]:
    """Relation labels the walker accepts for an asked relation.

    Mirrors `walker/graph_walker.py:inverse_relations`. The walker treats a
    relation and its DECLARED INVERSE as two labels for the SAME directed edge,
    so both must be counted when deciding what a question can reach. This is the
    correction that the previous version of this function was missing: it
    filtered on `r == rel` only, so for every mirrorable relation it
    under-counted reachability and would have certified an unanswerable gap as
    a "must refuse" pair. See PLANTLET_PRECEDES_NOTE below.
    """
    labels = {rel}
    inverse = INVERSE.get(rel)
    if inverse:
        labels.add(inverse)
    return labels


def reach_from(anchor: str, rel: str, hops: int | None = 1) -> set[str]:
    """Targets the WALKER can actually reach from `anchor` for an asked `rel`.

    Source-oriented (the walker only traverses edges leaving the current node)
    and label-permissive over the declared inverse pair, over the same edge pool
    the runtime assembles: stored edges plus the legal mirror copies.

    `hops=1` measures what a ONE-HOP question can grade. That distinction is
    load-bearing: `cat is_a mammal is_a animal` means a one-hop "what is a cat?"
    is unambiguous (mammal) even though a transitive search also reaches animal.
    Checking control uniqueness transitively would reject every taxonomy edge,
    because every taxonomy has a spine.

    `hops=None` searches to a fixpoint. That is the right test for a must-refuse
    pair: the anchor must be unable to reach anything under that relation by ANY
    route the walker could take, not merely by one step.
    """
    fwd = {(s, r, t) for s, r, t, _, _ in EDGES}
    pool = fwd | _mirror_edges()
    asked = asked_labels(rel)
    best: dict[str, int] = {anchor: 0}
    frontier = [anchor]
    while frontier:
        cur = frontier.pop()
        depth = best[cur]
        if hops is not None and depth >= hops:
            continue  # budget exhausted: cur's neighbours are out of range
        for s, r, t in pool:
            if s != cur or r not in asked:
                continue
            if t not in best or best[t] > depth + 1:
                best[t] = depth + 1
                frontier.append(t)
    return set(best) - {anchor}


def validate() -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []

    # every edge endpoint must be a declared concept
    declared = set(CONCEPTS)
    for src, _, tgt, _, _ in EDGES:
        for end in (src, tgt):
            if end not in declared:
                problems.append(("edge-endpoint-not-declared", end))
    unused = declared - {x for e in EDGES for x in (e[0], e[2])}
    if unused:
        # Isolated nodes are legal but defeat the point of a hand-audited graph:
        # an anchor nobody can reach will fail for embedding reasons and mask
        # whether the HONESTY GATE or the WALKER was responsible.
        problems.append(("declared-but-unreachable", ",".join(sorted(unused))))

    # Control pairs must be answerable by a ONE-HOP walk to exactly one target.
    #
    # "Exactly one stored edge" is judged over the ACCEPTED label set, not the
    # asked label alone, because the walker traverses the declared inverse of
    # the asked relation as the same edge (`("plantlet", "precedes")` is reached
    # through the stored `plantlet follows sprout`, and never through a stored
    # `plantlet precedes ...`). Demanding a forward edge under the asked label
    # would reject that control while accepting the genuinely ambiguous
    # `("building collapse", "caused_by")`, whose two one-hop candidates are
    # `caused_by earthquake` and `causes road closure`. If a control is not
    # 1-hop unique it cannot be graded.
    fwd = {(s, r, t) for s, r, t, _, _ in EDGES}
    for anchor, rel in CONTROL_PAIRS:
        accepted = asked_labels(rel)
        stored = sorted(t for s, r, t in fwd if s == anchor and r in accepted)
        if len(stored) != 1:
            problems.append(("control-pair-not-unique",
                             f"{anchor}|{rel}={stored} (accepted labels {sorted(accepted)})"))
        reach = reach_from(anchor, rel, hops=1)
        if len(reach) != 1:
            problems.append(("control-pair-ambiguous", f"{anchor}|{rel}->{sorted(reach)}"))
        if reach and sorted(reach) != stored:
            problems.append(("control-pair-mirror-drift",
                             f"{anchor}|{rel}->{sorted(reach)} vs stored {stored}"))

    # must-refuse pairs must be unreachable in EVERY sense the walker has
    for kind, pairs in MUST_REFUSE.items():
        for anchor, rel in pairs:
            if anchor not in declared:
                problems.append((f"{kind}-anchor-missing", anchor))
                continue
            # An `inverse_direction` entry is only meaningful when the relation
            # has NO declared inverse. If it did, reading the edge backwards
            # would be the contract working, not a violation, and the pair would
            # be silently testing nothing. (This is the same guard Stage C
            # applies to its mirror_silence controls.)
            if kind == "inverse_direction" and rel in INVERSE:
                problems.append((f"{kind}-relation-has-inverse", f"{anchor}|{rel}"))
            # fixpoint search: the anchor must be unable to reach anything
            # under that relation by any route, not merely in one step
            reach = reach_from(anchor, rel, hops=None)
            if reach:
                problems.append((f"{kind}-actually-reachable",
                                 f"{anchor}|{rel}->{sorted(reach)}"))
            # and specifically no direct edge of the asked relation either,
            # which would mean the gap is only transitive
            direct = [t for s, r, t, _, _ in EDGES if s == anchor and r == rel]
            if direct:
                problems.append((f"{kind}-has-direct-edge",
                                 f"{anchor}|{rel}->{direct}"))

    # no undeclared same-label reverse edge anywhere
    seen_edges = {(s, r, t) for s, r, t, _, _ in EDGES}
    for src, rel, tgt, _, _ in EDGES:
        if rel in SYMMETRIC_STORED:
            continue
        if rel not in INVERSE and (tgt, rel, src) in seen_edges:
            problems.append(("undeclared-same-label-mirror",
                             f"{tgt} -{rel}-> {src}"))

    # the mirror pass must not be able to invert a no-inverse relation
    for rel in NO_DECLARED_INVERSE:
        if rel in MIRRORED:
            problems.append(("no-inverse-relation-is-mirrorable", rel))

    # An (anchor, relation) pair cannot be both answerable and unanswerable.
    # This is not a defensive nicety: `("plantlet", "precedes")` sat in both
    # lists at once until the reachability model was corrected, and the
    # contradiction was invisible because each list was checked in isolation.
    control_set = set(CONTROL_PAIRS)
    for kind, pairs in MUST_REFUSE.items():
        for pair in pairs:
            if pair in control_set:
                problems.append((f"{kind}-pair-is-also-a-control", f"{pair[0]}|{pair[1]}"))

    return problems


def build() -> int:
    problems = validate()
    if problems:
        print("REFUSING TO BUILD:")
        for kind, detail in problems:
            print(f"  {kind}: {detail}")
        return 1

    labels = list(dict.fromkeys(CONCEPTS))
    used = sorted({r for _, r, _, _, _ in EDGES})
    print(f"Building honesty & fallback graph: {len(labels)} nodes, {len(EDGES)} edges")
    print(f"  relations used ({len(used)}/16): {used}")

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    enc = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {lab: np.asarray(enc[i], dtype=np.float32) for i, lab in enumerate(labels)}

    for stale in (DB_PATH, DB_PATH.with_suffix(".db-wal"), DB_PATH.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    concepts = {label: i + 1 for i, label in enumerate(labels)}
    store = SQLiteGraphStore(db_path=str(DB_PATH))
    store.add_dataset(
        concepts=concepts,
        edges=[{"source": concepts[a], "target": concepts[b], "relation": r,
                "strength": s, "confidence": c} for a, r, b, s, c in EDGES],
        id_to_label={i: l for l, i in concepts.items()},
        embeddings=embeddings,
        protected_labels=list(labels),
    )
    store.set_metadata("dataset_name", "honesty_fallback_stage_d")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.set_metadata("frozen", "true")
    store.set_metadata("contract", "GLM-X v3.3.2 sections 6/8/16")
    store.set_metadata("version", "1.0.0")
    store.save_state(str(DB_PATH))

    journal = json.loads(store.get_metadata("merge_journal") or "[]")
    if journal:
        raise SystemExit(f"REFUSING TO FREEZE: add_dataset merged {journal}")
    if store.get_node_count() != len(labels):
        raise SystemExit("REFUSING TO FREEZE: node count mismatch")

    stored = sorted(store.get_all_relations())
    missing = [r for r in CANONICAL_RELATIONS if r not in stored]
    if missing:
        raise SystemExit(f"REFUSING TO FREEZE: relations absent from db: {missing}")
    if "has_part" in stored:
        raise SystemExit("REFUSING TO FREEZE: has_part is a runtime mirror label")

    print(f"Wrote {DB_PATH}")
    print(f"  nodes={store.get_node_count()} edges={store.get_edge_count()}")
    print(f"  stored relations ({len(stored)}/16): {stored}")
    print(f"  control pairs: {len(CONTROL_PAIRS)}")
    for kind, pairs in MUST_REFUSE.items():
        print(f"  {kind} pairs verified unreachable: {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())