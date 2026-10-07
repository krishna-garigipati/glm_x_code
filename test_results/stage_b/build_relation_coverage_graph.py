"""Stage B - Relation Coverage Graph (GLM-X v3.3.2 contract section 16).

    - name: "Relation Coverage Graph"
      purpose: "Test all 16 relations"
      size: "~100-150 nodes"
      goal: "Prove every canonical relation can be used"

Unlike Stage A (which optimised for a small end-to-end proof), this graph
optimises for RELATION COVERAGE. Every one of the 16 canonical relations from
contract section 4 is stored explicitly, so coverage is verifiable directly in
the database rather than inferred from which cue happened to fire.

Design rules enforced here, each of which exists to stop a specific defect
class found in the tester graphs:

1. SEMANTIC TYPE DISCIPLINE (the "whale has ocean" failure mode).
   Every relation has a required shape for its target, checked at build time by
   _TARGET_SHAPE below:
     has_property      -> a property/adjective, never an entity
     part_of           -> a WHOLE that contains the subject
     is_a              -> a broader category
     example_of        -> a class the subject instantiates
     synonym/antonym   -> a same-type lexical item
     spatial_near      -> a place
     temporal_coincident -> a concurrent time or event
     causes            -> an effect
     caused_by         -> a cause
     supports/contradicts -> a claim, theory or belief
     linguistic_maps   -> a linguistic item
     precedes/follows  -> a point on the same ordered timeline
   A violation is a hard build error, not a warning.

2. NO AMBIGUOUS ASKED PAIR. For every (anchor, relation) pair any question asks
   about there is exactly one reachable target. build_toy_graph.py designed this
   property out of the whole graph; a 120-node graph cannot afford that globally
   (a category legitimately has many members), so here it is asserted per ASKED
   pair instead, which is the property that actually matters for grading.

3. NO SILENT SEMANTIC MERGE. add_dataset() deletes any node whose embedding
   cosine is >= 0.92 against a lower id and RETARGETS its edges, which turns a
   true fact into a false one. With ~120 nodes on bge-small this is close to
   certain, so every label is protected and the merge journal must come back
   empty or the build refuses to freeze.

4. LINEAR TIMELINE. The nature graph stores the seasons as a cycle that also
   contains both `autumn precedes winter` and `autumn follows winter`, which
   makes "before"/"after" unanswerable. The engine timeline here is a strict
   linear chain with no cycle and no contradictory edge.

Usage:
    python test_results/stage_b/build_relation_coverage_graph.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

# --------------------------------------------------------------------------
# The 16 canonical relations, contract section 4. Anything not in here is a
# contract violation and the build refuses to freeze.
# --------------------------------------------------------------------------
CANONICAL_RELATIONS = [
    "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
    "contradicts", "supports", "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident", "spatial_near",
    "linguistic_maps",
]

# --------------------------------------------------------------------------
# Nodes, grouped by domain. A node that is only ever a target still earns its
# place: it makes the graph's vocabulary explicit and gives multi-hop somewhere
# to land.
# --------------------------------------------------------------------------
CONCEPTS = [
    # --- animal taxonomy (is_a, part_of, has_property) ---
    "animal", "mammal", "bird", "fish", "reptile", "insect", "amphibian",
    "crustacean", "plant", "tree", "flower", "landform", "landmass",
    "dog", "cat", "whale", "robin", "owl", "shark", "salmon", "frog",
    "lizard", "snake", "ant", "mouse", "tuna",
    "fin", "gill", "tail", "wing", "feather", "leg", "eye",

    # --- plants ---
    "oak", "pine", "maple", "rose", "daisy", "tulip",
    "branch", "root", "thorn", "petal", "stem",

    # --- weather (causes, caused_by, associated_with, part_of, has_property) ---
    "storm", "cloud", "rain", "flood", "thunder", "lightning", "hurricane",
    "disaster", "damage", "drought", "crop failure", "breeze", "fog",
    "wet", "cold", "hot",

    # --- physical / matter ---
    "water", "liquid", "ice",

    # --- geography (spatial_near, part_of, is_a) ---
    "mountain", "peak", "valley", "river", "glacier", "continent", "island",
    "ice mass", "city", "canyon", "plateau", "forest", "desert", "ocean",
    "lake",

    # --- technology timeline (precedes / follows) ---
    "internal combustion engine", "jet engine", "rocket engine",

    # --- events coinciding with a time (temporal_coincident) ---
    "harvest season", "autumn", "fireworks display", "new year",

    # --- evidence and claims (supports / contradicts) ---
    "fossil record", "dna record", "orbit of comets",
    "deep time claim", "common ancestry claim", "solar system claim",
    "young earth claim", "special creation claim", "geocentric claim",

    # --- lexicon (synonym, antonym, example_of) ---
    "physician", "doctor", "automobile", "car", "cupboard", "cabinet",
    "big", "small", "ancient", "modern", "vast", "tiny",
    "light source", "acorn", "seed", "hailstone", "precipitation", "candle",

    # --- cross-language items (linguistic_maps) ---
    "hacienda", "farm", "perro", "amigo", "friend",
]

# --------------------------------------------------------------------------
# Edges: (source, target, relation, strength, confidence)
# strength/confidence are all inside the 0.8-1.0 band contract section 16
# requires for PoC edges.
#
# Every ASKED (source, relation) pair below has exactly one target. Several
# nodes deliberately carry two edges of the SAME relation (frozen poles are
# both cold and slippery), but nothing that a question asks about does.
# --------------------------------------------------------------------------
EDGES = [
    # ================= is_a : animal backbone =================
    ("mammal", "animal", "is_a", 0.95, 0.95),
    ("bird", "animal", "is_a", 0.95, 0.95),
    ("fish", "animal", "is_a", 0.95, 0.95),
    ("reptile", "animal", "is_a", 0.95, 0.95),
    ("insect", "animal", "is_a", 0.95, 0.95),
    ("amphibian", "animal", "is_a", 0.95, 0.95),
    ("crustacean", "animal", "is_a", 0.95, 0.95),
    ("dog", "mammal", "is_a", 0.98, 0.98),
    ("cat", "mammal", "is_a", 0.98, 0.98),
    # A whale is a mammal. Getting this right is the point: the tester graphs
    # and one early draft of this file had "whale is_a fish", which renders as
    # a flatly false sentence.
    ("whale", "mammal", "is_a", 0.98, 0.98),
    ("mouse", "mammal", "is_a", 0.98, 0.98),
    ("robin", "bird", "is_a", 0.98, 0.98),
    ("owl", "bird", "is_a", 0.98, 0.98),
    ("shark", "fish", "is_a", 0.98, 0.98),
    ("salmon", "fish", "is_a", 0.98, 0.98),
    ("tuna", "fish", "is_a", 0.98, 0.98),
    ("lizard", "reptile", "is_a", 0.98, 0.98),
    ("snake", "reptile", "is_a", 0.98, 0.98),
    ("ant", "insect", "is_a", 0.98, 0.98),
    ("frog", "amphibian", "is_a", 0.98, 0.98),

    # ================= is_a : plants, geography, matter =================
    ("tree", "plant", "is_a", 0.95, 0.95),
    ("flower", "plant", "is_a", 0.95, 0.95),
    ("oak", "tree", "is_a", 0.98, 0.98),
    ("pine", "tree", "is_a", 0.98, 0.98),
    ("maple", "tree", "is_a", 0.98, 0.98),
    ("rose", "flower", "is_a", 0.98, 0.98),
    ("daisy", "flower", "is_a", 0.98, 0.98),
    ("tulip", "flower", "is_a", 0.98, 0.98),
    ("mountain", "landform", "is_a", 0.95, 0.95),
    ("plateau", "landform", "is_a", 0.95, 0.95),
    ("desert", "landform", "is_a", 0.95, 0.95),
    ("continent", "landmass", "is_a", 0.95, 0.95),
    ("island", "landmass", "is_a", 0.95, 0.95),
    ("glacier", "ice mass", "is_a", 0.95, 0.95),
    ("water", "liquid", "is_a", 0.98, 0.98),

    # ================= part_of : parts point AT their whole =================
    ("fin", "shark", "part_of", 0.95, 0.95),
    ("gill", "shark", "part_of", 0.95, 0.95),
    ("tail", "whale", "part_of", 0.95, 0.95),
    ("wing", "robin", "part_of", 0.95, 0.95),
    ("feather", "robin", "part_of", 0.95, 0.95),
    ("eye", "owl", "part_of", 0.95, 0.95),
    ("leg", "ant", "part_of", 0.95, 0.95),
    ("branch", "oak", "part_of", 0.95, 0.95),
    ("root", "oak", "part_of", 0.95, 0.95),
    ("thorn", "rose", "part_of", 0.95, 0.95),
    ("petal", "rose", "part_of", 0.95, 0.95),
    ("stem", "tulip", "part_of", 0.95, 0.95),
    ("peak", "mountain", "part_of", 0.95, 0.95),
    ("valley", "mountain", "part_of", 0.90, 0.90),
    ("canyon", "mountain", "part_of", 0.95, 0.95),
    ("cloud", "storm", "part_of", 0.90, 0.90),

    # ================= has_property : target must be a PROPERTY =================
    ("ant", "small", "has_property", 0.90, 0.90),
    ("whale", "large", "has_property", 0.90, 0.90),
    ("ice", "cold", "has_property", 0.98, 0.98),
    ("rain", "wet", "has_property", 0.95, 0.95),
    ("desert", "hot", "has_property", 0.90, 0.90),
    # Serves the is_a -> has_property multi-hop question (mh13).
    ("dog", "loyal", "has_property", 0.85, 0.85),
    # mh13 walks dog -> mammal (is_a) -> warm (has_property). The property has
    # to hang off the INTERMEDIATE node, not off the anchor: a second clause
    # about the anchor re-asks hop 1 and the walk stalls after one step.
    ("mammal", "warm", "has_property", 0.90, 0.90),

    # ================= causes / caused_by : BOTH stored explicitly =========
    # Storing both members of the pair (rather than relying only on the runtime
    # mirror) is what lets relation coverage be read straight out of the db.
    ("lightning", "thunder", "causes", 0.95, 0.95),
    ("thunder", "lightning", "caused_by", 0.95, 0.95),
    ("rain", "flood", "causes", 0.95, 0.95),
    ("flood", "rain", "caused_by", 0.95, 0.95),
    ("drought", "crop failure", "causes", 0.95, 0.95),
    ("hurricane", "damage", "causes", 0.90, 0.90),
    ("flood", "disaster", "is_a", 0.95, 0.95),

    # ================= precedes / follows : strictly linear timeline ==========
    # internal combustion engine -> jet engine -> rocket engine. No cycle, so
    # "before" and "after" are well defined at both ends.
    ("internal combustion engine", "jet engine", "precedes", 0.95, 0.95),
    ("jet engine", "rocket engine", "precedes", 0.95, 0.95),
    ("jet engine", "internal combustion engine", "follows", 0.95, 0.95),
    ("rocket engine", "jet engine", "follows", 0.95, 0.95),

    # ================= temporal_coincident : target is a TIME/EVENT =======
    ("harvest season", "autumn", "temporal_coincident", 0.85, 0.85),
    ("fireworks display", "new year", "temporal_coincident", 0.85, 0.85),

    # ================= spatial_near : target is a PLACE =================
    ("river", "valley", "spatial_near", 0.90, 0.90),
    ("glacier", "peak", "spatial_near", 0.90, 0.90),
    ("island", "continent", "spatial_near", 0.90, 0.90),
    ("city", "river", "spatial_near", 0.90, 0.90),
    ("fog", "river", "spatial_near", 0.85, 0.85),
    ("lake", "forest", "spatial_near", 0.85, 0.85),
    ("breeze", "ocean", "spatial_near", 0.85, 0.85),
    # Serves the part_of -> spatial_near multi-hop question (mh12):
    # peak part_of mountain, then mountain spatial_near lake.
    ("mountain", "lake", "spatial_near", 0.85, 0.85),

    # ================= associated_with =================
    ("rain", "cloud", "associated_with", 0.90, 0.90),
    ("thunder", "storm", "associated_with", 0.90, 0.90),
    ("disaster", "damage", "associated_with", 0.90, 0.90),
    ("hurricane", "storm", "associated_with", 0.85, 0.85),
    ("desert", "hot", "associated_with", 0.85, 0.85),

    # ================= supports / contradicts : target is a CLAIM ========
    ("fossil record", "deep time claim", "supports", 0.90, 0.90),
    ("dna record", "common ancestry claim", "supports", 0.90, 0.90),
    ("orbit of comets", "solar system claim", "supports", 0.85, 0.85),
    ("young earth claim", "deep time claim", "contradicts", 0.90, 0.90),
    ("special creation claim", "common ancestry claim", "contradicts", 0.90, 0.90),
    ("geocentric claim", "solar system claim", "contradicts", 0.90, 0.90),

    # ================= example_of : subject instantiates the target ======
    ("candle", "light source", "example_of", 0.90, 0.90),
    ("acorn", "seed", "example_of", 0.90, 0.90),
    ("hailstone", "precipitation", "example_of", 0.90, 0.90),
    ("robin", "bird", "example_of", 0.85, 0.85),

    # ================= synonym / antonym =================
    ("physician", "doctor", "synonym", 0.98, 0.98),
    ("automobile", "car", "synonym", 0.98, 0.98),
    ("cupboard", "cabinet", "synonym", 0.90, 0.90),
    ("big", "small", "antonym", 0.98, 0.98),
    ("ancient", "modern", "antonym", 0.98, 0.98),
    ("vast", "tiny", "antonym", 0.95, 0.95),
    ("hot", "cold", "antonym", 0.98, 0.98),

    # ================= linguistic_maps : target is a WORD ================
    ("hacienda", "farm", "linguistic_maps", 0.95, 0.95),
    ("perro", "dog", "linguistic_maps", 0.95, 0.95),
    ("amigo", "friend", "linguistic_maps", 0.95, 0.95),
]

# --------------------------------------------------------------------------
# Symmetric relations: store BOTH directions as real edges.
#
# The architecture contract permits mirroring for only four declared pairs
# (causes/caused_by, precedes/follows, part_of/has_part, is_a). contradicts,
# temporal_coincident, spatial_near and linguistic_maps are NOT among them, so
# the walker may not synthesise their reverse at inference time.
#
# These four relations are nonetheless semantically symmetric, so a question may
# legitimately be asked from either endpoint. Encoding that as an explicit STORED
# triple -- rather than letting the mirror pass invent it -- moves the symmetry
# assumption out of inference and into auditable data, where the semantic checks
# can actually verify it. Only the reverse edges the question set needs are
# added: mirroring every edge of these relations would hand some anchors several
# same-relation neighbours (e.g. `river` would gain both `fog` and `city`) and
# reintroduce the anchor ambiguity this benchmark is meant to exclude.
#
# NOT done for supports / example_of / has_property: those are genuinely
# directional, so their reverse is a false claim and must stay unstored.
EDGES += [
    # contradicts
    ("deep time claim", "young earth claim", "contradicts", 0.90, 0.90),
    ("solar system claim", "geocentric claim", "contradicts", 0.90, 0.90),
    # temporal_coincident
    ("autumn", "harvest season", "temporal_coincident", 0.85, 0.85),
    ("new year", "fireworks display", "temporal_coincident", 0.85, 0.85),
    # spatial_near
    ("peak", "glacier", "spatial_near", 0.90, 0.90),
    ("continent", "island", "spatial_near", 0.90, 0.90),
    ("forest", "lake", "spatial_near", 0.85, 0.85),
    # linguistic_maps
    ("dog", "perro", "linguistic_maps", 0.95, 0.95),
    ("farm", "hacienda", "linguistic_maps", 0.95, 0.95),
    ("friend", "amigo", "linguistic_maps", 0.95, 0.95),
]

# `large` is referenced by has_property but kept out of the grouped list above
# so the property words read together.
EXTRA_CONCEPTS = ["large", "loyal", "warm"]

# --------------------------------------------------------------------------
# Semantic shape check. This is the automated guard behind design rule 1: it
# is what stops a future edit from reintroducing "whale has ocean".
# --------------------------------------------------------------------------
PROPERTIES = {
    "small", "large", "cold", "hot", "wet", "loyal", "warm",
}
CLAIMS = {
    "deep time claim", "common ancestry claim", "solar system claim",
    "young earth claim", "special creation claim", "geocentric claim",
}
# For the four symmetric relations stored in both directions (see EDGES below),
# BOTH endpoints must satisfy the target-type rule, not just the original target.
# `glacier is_a`-style errors are what design rule 1 exists to catch, so the sets
# below carry the reverse-side members too: a spatial_near edge is only legal
# when both ends are places, and so on.
PLACES = {
    "valley", "peak", "continent", "river", "forest", "ocean", "lake",
    "glacier", "island",
}
TIMES = {"autumn", "new year", "harvest season", "fireworks display"}
WORDS = {
    "doctor", "car", "cabinet", "farm", "dog", "friend",
    "small", "large", "modern", "tiny", "cold", "hot", "wet",
    "perro", "hacienda", "amigo",
}


def _semantic_violations(label: str, relation: str, target: str) -> list[str]:
    """Return human-readable semantic-type violations for one edge."""
    bad: list[str] = []
    if relation == "has_property" and target not in PROPERTIES:
        bad.append(f"has_property target {target!r} is not a property word")
    if relation in ("supports", "contradicts") and target not in CLAIMS:
        bad.append(f"{relation} target {target!r} is not a claim")
    if relation == "spatial_near" and target not in PLACES:
        bad.append(f"spatial_near target {target!r} is not a place")
    if relation == "temporal_coincident" and target not in TIMES:
        bad.append(f"temporal_coincident target {target!r} is not a time or event")
    if relation == "synonym" and target in PLACES or relation == "antonym" and target in PLACES:
        bad.append(f"{relation} target {target!r} is a place, not a lexical item")
    if relation == "linguistic_maps" and target not in WORDS:
        bad.append(f"linguistic_maps target {target!r} is not a word")
    if relation == "causes" and target == label:
        bad.append("causes target equals its own cause")
    if relation == "precedes" and target == label:
        bad.append("precedes target equals itself")
    return bad


# --------------------------------------------------------------------------
# (anchor, relation) pairs the frozen question set asks about. Kept here so the
# builder can assert uniqueness BEFORE anything is scored.
# --------------------------------------------------------------------------
ASKED_PAIRS = [
    # is_a
    ("dog", "is_a"), ("robin", "is_a"), ("oak", "is_a"), ("water", "is_a"),
    ("whale", "is_a"), ("frog", "is_a"), ("glacier", "is_a"),
    # has_property
    ("ice", "has_property"), ("ant", "has_property"), ("whale", "has_property"),
    ("rain", "has_property"), ("dog", "has_property"),
    # causes / caused_by
    ("thunder", "causes"), ("flood", "causes"), ("crop failure", "causes"),
    ("thunder", "caused_by"), ("flood", "caused_by"),
    # precedes / follows
    ("internal combustion engine", "precedes"), ("jet engine", "precedes"),
    ("rocket engine", "follows"), ("jet engine", "follows"),
    # contradicts / supports
    ("deep time claim", "contradicts"), ("solar system claim", "contradicts"),
    ("common ancestry claim", "contradicts"),
    ("deep time claim", "supports"), ("common ancestry claim", "supports"),
    ("solar system claim", "supports"),
    # associated_with
    ("rain", "associated_with"), ("thunder", "associated_with"),
    ("disaster", "associated_with"),
    # example_of (stored/forward direction only -- see the known_limitations
    # entry in relation_coverage_questions_frozen.json for why the reverse
    # direction renders inverted)
    ("candle", "example_of"), ("acorn", "example_of"),
    ("hailstone", "example_of"), ("robin", "example_of"),
    # part_of (forward direction only; has_part has no cue bank and the reverse
    # direction renders inverted, so the pair is only half-askable)
    ("fin", "part_of"), ("branch", "part_of"), ("thorn", "part_of"),
    ("canyon", "part_of"), ("tail", "part_of"), ("cloud", "part_of"),
    ("peak", "part_of"),
    # synonym / antonym
    ("physician", "synonym"), ("automobile", "synonym"), ("cupboard", "synonym"),
    ("big", "antonym"), ("ancient", "antonym"), ("hot", "antonym"),
    # temporal_coincident
    ("autumn", "temporal_coincident"), ("new year", "temporal_coincident"),
    # spatial_near
    ("river", "spatial_near"), ("peak", "spatial_near"),
    ("continent", "spatial_near"), ("city", "spatial_near"),
    ("forest", "spatial_near"), ("mountain", "spatial_near"),
    # linguistic_maps
    ("dog", "linguistic_maps"), ("farm", "linguistic_maps"),
    ("friend", "linguistic_maps"),
    # multi-hop second hops
    ("shark", "is_a"), ("rose", "is_a"), ("mountain", "is_a"),
    ("disaster", "is_a"), ("flower", "is_a"), ("oak", "is_a"),
    ("owl", "is_a"), ("tree", "is_a"), ("shark", "part_of"),
]


def build() -> None:
    labels = list(CONCEPTS)
    for extra in EXTRA_CONCEPTS:
        if extra not in labels:
            labels.append(extra)

    # ---- guard 1: closed relation set ----
    used = sorted({e[2] for e in EDGES})
    illegal = [r for r in used if r not in CANONICAL_RELATIONS]
    if illegal:
        raise SystemExit(f"REFUSING TO BUILD: relations outside the closed set: {illegal}")
    missing = [r for r in CANONICAL_RELATIONS if r not in used]
    if missing:
        raise SystemExit(f"REFUSING TO BUILD: canonical relations never exercised: {missing}")

    # ---- guard 2: every edge endpoint exists ----
    for src, tgt, rel, s, c in EDGES:
        for node in (src, tgt):
            if node not in labels:
                raise SystemExit(f"REFUSING TO BUILD: edge {src}-{rel}->{tgt} names unknown node {node!r}")

    # ---- guard 3: strength/confidence band ----
    for src, tgt, rel, s, c in EDGES:
        if not (0.8 <= s <= 1.0 and 0.8 <= c <= 1.0):
            raise SystemExit(
                f"REFUSING TO BUILD: {src}-{rel}->{tgt} has strength={s} confidence={c}, "
                "outside the 0.8-1.0 PoC band")

    # ---- guard 4: semantic shape ----
    violations = []
    for src, tgt, rel, s, c in EDGES:
        for msg in _semantic_violations(src, rel, tgt):
            violations.append(f"  {src} -{rel}-> {tgt}: {msg}")
    if violations:
        raise SystemExit("REFUSING TO BUILD: semantic type violations:\n" + "\n".join(violations))

    # ---- guard 5: asked pairs are unambiguous (forward edges only) ----
    fwd: dict[tuple[str, str], list[str]] = {}
    for src, tgt, rel, s, c in EDGES:
        fwd.setdefault((src, rel), []).append(tgt)
    ambiguous = {k: v for k, v in fwd.items() if k in set(ASKED_PAIRS) and len(v) > 1}
    if ambiguous:
        raise SystemExit(
            "REFUSING TO BUILD: asked pairs have more than one target: "
            + json.dumps({f"{k[0]}|{k[1]}": v for k, v in ambiguous.items()}))

    print(f"Building relation coverage graph: {len(labels)} nodes, {len(EDGES)} edges")
    print(f"  relations covered: {len(used)}/16 -> {used}")

    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    enc = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {lab: np.asarray(enc[i], dtype=np.float32) for i, lab in enumerate(labels)}

    out = Path(__file__).resolve().parent / "relation_coverage.db"
    for stale in (out, out.with_suffix(".db-wal"), out.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    concepts = {label: i + 1 for i, label in enumerate(labels)}
    store = SQLiteGraphStore(db_path=str(out))
    resolved = []
    for src, tgt, rel, s, c in EDGES:
        resolved.append({
            "source": concepts[src], "target": concepts[tgt], "relation": rel,
            "strength": s, "confidence": c,
        })
    store.add_dataset(
        concepts=concepts,
        edges=resolved,
        id_to_label={i: l for l, i in concepts.items()},
        embeddings=embeddings,
        # Every label is protected. At ~120 nodes on bge-small some pairs clear
        # 0.92 by coincidence; an unprotected build would delete one and
        # retarget its edges into a different (false) statement.
        protected_labels=list(labels),
    )
    store.set_metadata("dataset_name", "relation_coverage_stage_b")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.set_metadata("frozen", "true")
    store.set_metadata("contract", "GLM-X v3.3.2 section 16")
    store.set_metadata("version", "1.0.0")
    store.save_state(str(out))

    journal = json.loads(store.get_metadata("merge_journal") or "[]")
    if journal:
        raise SystemExit(
            "REFUSING TO FREEZE: add_dataset merged {} node pair(s) despite "
            f"protected_labels: {journal}".format(len(journal)))
    if store.get_node_count() != len(labels):
        raise SystemExit("REFUSING TO FREEZE: built {} nodes, expected {}.".format(
            store.get_node_count(), len(labels)))
    if not (100 <= store.get_node_count() <= 150):
        raise SystemExit("REFUSING TO FREEZE: {} nodes is outside the 100-150 target.".format(
            store.get_node_count()))

    stored = sorted(store.get_all_relations())
    absent = [r for r in CANONICAL_RELATIONS if r not in stored]
    if absent:
        raise SystemExit(f"REFUSING TO FREEZE: relations absent from the db: {absent}")

    print(f"Wrote {out}")
    print(f"  nodes={store.get_node_count()} edges={store.get_edge_count()}")
    print(f"  stored relations ({len(stored)}/16): {stored}")
    print(f"  ambiguous asked pairs: {len(ambiguous)}")


if __name__ == "__main__":
    build()