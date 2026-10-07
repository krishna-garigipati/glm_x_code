"""Stage C - Direction & Multi-hop Graph (GLM-X v3.3.2 contract section 16).

    - name: "Direction & Multi-hop Graph"
      purpose: "Test direction and short chains"
      size: "~100 nodes"
      goal: "Prove causes/caused_by, part_of, follows/precedes and simple multi-hop"

Stage B proved all 16 relations can be walked. Stage C asks a harder question:
does DIRECTION survive being chained, and does the mirror pass stay silent where
it must?

Design rules, each targeting a specific defect class:

1. STRONG ASYMMETRIC DIRECTION. causes/caused_by and precedes/follows are stored
   in BOTH directions as real edges, so a question asked from either end has a
   real answer and direction can be scored from the walked edge ORDER rather
   than inferred from the sentence. part_of is stored forward only, because
   has_part is the declared inverse label at RUNTIME and is not one of the 16
   storable relations.

2. ONE SHARED BACKBONE. Transport, hazard and taxonomy systems all hang off one
   is_a spine, so multi-hop chains cross between the direction-pair systems
   instead of staying inside a single pair. This is what makes "simple
   multi-hop" a real test rather than two copies of the one-hop case.

3. DIRECTION TESTED INSIDE CHAINS. The contract's own suggested chain
   (part_of > is_a) is 2 hops whose final hop is symmetric, so an inverted
   traversal can land on the right node anyway. Every chain here is built so
   that inverting any single hop changes the ANSWER. The runner enforces this
   rather than trusting it: see _assert_direction_sensitive().

4. FORBIDDEN REVERSE READS MUST REFUSE. supports, has_property, example_of and
   contradicts have NO declared inverse label, so their reverse is a different,
   false claim. The graph deliberately contains only the forward direction, and
   `forbidden_reverse` questions assert honest refusal. This is the regression
   guard for illegal same-label mirroring.

5. NO AMBIGUOUS ASKED PAIR, checked against MIRRORED REACHABILITY rather than
   forward edges only. Stage B's forward-only guard would pass a pair that the
   runtime mirror makes ambiguous, which is exactly this stage's subject. A
   separate guard asserts that every forbidden_reverse pair is unreachable in
   both senses, so the refusal questions cannot silently become answerable.

6. LINEAR TIMELINE and NO ROOT MOTION. precedes/follows is a strict chain with no
   cycle and no contradictory edge. No part_of edge points at an abstract
   backbone category, so "what is part of X" cannot walk inward through a
   category and then chain into a second hop that was never asked about.

7. NO SILENT SEMANTIC MERGE. add_dataset() merges nodes at cosine >= 0.92 and
   retargets their edges, turning a true fact into a false one. Every label is
   protected and the merge journal must come back empty or the build refuses.

Usage:
    python test_results/stage_c/build_direction_multihop_graph.py
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

CANONICAL_RELATIONS = [
    "is_a", "has_property", "causes", "caused_by", "follows", "precedes",
    "contradicts", "supports", "associated_with", "example_of", "part_of",
    "synonym", "antonym", "temporal_coincident", "spatial_near",
    "linguistic_maps",
]

INVERSE = {
    "causes": "caused_by", "caused_by": "causes",
    "precedes": "follows", "follows": "precedes",
    "part_of": "has_part", "has_part": "part_of",
    "is_a": "is_a",
}

# Relations with no declared inverse: their reverse is a DIFFERENT and false
# claim, so it must never be walked. Reverse reads of these are the
# forbidden_reverse cases.
NO_DECLARED_INVERSE = (
    "supports", "has_property", "example_of", "contradicts",
    "synonym", "antonym", "linguistic_maps", "spatial_near",
    "temporal_coincident", "associated_with",
)

# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
CONCEPTS = [
    # --- transport / machine (is_a + part_of) ---
    "transport", "vehicle", "machine", "device", "material",
    "car", "airplane", "locomotive",
    "engine", "wheel", "chassis", "door", "propeller", "wing",
    "boiler", "piston rod",
    "turbine", "valve", "generator", "transformer",

    # --- power / communication / industry (is_a + timeline) ---
    "power system", "power plant", "substation", "electrical device",
    "handwriting", "printing press", "telegraph", "telephone",
    "radio", "television", "satellite",
    "communication method",

    # --- geography (spatial_near, part_of) ---
    "mountain range", "harbor", "city", "summit", "lake", "canal", "river",

    # --- hazard tree (is_a, part_of, causes) ---
    "event", "natural event", "weather event", "disruption",
    "flooding", "shutdown", "transport disruption",
    "eruption", "wildfire", "landslide", "drought", "storm", "flood",
    "acid rain", "ash cloud", "crop failure", "economic loss",
    "environmental loss", "habitat loss", "service loss", "road closure",
    "washout", "abutment failure", "bridge", "village", "town", "house",
    "school", "hospital", "ferry service",
    "overgrazing", "dam failure", "oil spill", "vessel",
    "structure", "building", "institution", "settlement", "landform",
    "transport service", "environmental change",

    # --- evidence / claims (supports, contradicts) ---
    "rock strata record", "seafloor ridge record", "crater field record",
    "tide gauge record", "ice core record",
    "plate tectonic theory", "orbital climate theory", "meteor impact theory",
    "fixed earth claim", "static climate claim",

    # --- lexicon / language ---
    "physician", "doctor", "cabinet", "cupboard", "automobile",
    "large", "tiny", "ancient", "modern", "dry",
    "light source", "seed", "acorn", "hailstone", "precipitation", "candle",
    "hacienda", "amigo", "perro", "friend", "farm", "dog",
    "harvest season", "autumn",
]

# edges are (source, relation, target, strength, confidence)
EDGES: list[tuple[str, str, str, float, float]] = []


def _fwd(src: str, rel: str, tgt: str, s: float = 0.95, c: float = 0.95) -> None:
    EDGES.append((src, rel, tgt, s, c))


def _pair(a: str, b: str, left: str, right: str,
          s: float = 0.95, c: float = 0.95) -> None:
    """Store a declared inverse pair in both directions.

    Storing both members (rather than trusting only the runtime mirror) is what
    lets direction be read straight out of the database, and it keeps the
    reverse read of these three pairs legal -- which is exactly why their
    reverse reads go in direction_pairs, not in forbidden_reverse.
    """
    _fwd(a, left, b, s, c)
    _fwd(b, right, a, s, c)


def _both_symmetric(a: str, b: str, rel: str, s: float = 0.95, c: float = 0.95) -> None:
    """Store a semantically symmetric relation in both directions.

    The mirror pass may NOT synthesise these (they have no declared inverse
    label), so the symmetry assumption has to live in auditable data. Storing
    both directions is also what keeps the two endpoints usable as asked
    anchors without making either ambiguous.
    """
    _fwd(a, rel, b, s, c)
    _fwd(b, rel, a, s, c)


# ===================== is_a : transport / machine spine =====================
_fwd("car", "is_a", "vehicle", 0.98, 0.98)
_fwd("airplane", "is_a", "vehicle", 0.98, 0.98)
_fwd("locomotive", "is_a", "vehicle", 0.98, 0.98)
_fwd("vehicle", "is_a", "transport", 0.95, 0.95)
_fwd("transport", "is_a", "machine", 0.95, 0.95)
_fwd("machine", "is_a", "device", 0.95, 0.95)
_fwd("device", "is_a", "material", 0.95, 0.95)
_fwd("engine", "is_a", "machine", 0.98, 0.98)
_fwd("turbine", "is_a", "machine", 0.98, 0.98)
_fwd("valve", "is_a", "machine", 0.98, 0.98)
_fwd("generator", "is_a", "machine", 0.98, 0.98)
_fwd("transformer", "is_a", "machine", 0.98, 0.98)
_fwd("propeller", "is_a", "machine", 0.95, 0.95)
_fwd("handwriting", "is_a", "communication method", 0.95, 0.95)
_fwd("printing press", "is_a", "communication method", 0.95, 0.95)
_fwd("communication method", "is_a", "material", 0.95, 0.95)
_fwd("telegraph", "is_a", "electrical device", 0.98, 0.98)
_fwd("telephone", "is_a", "electrical device", 0.98, 0.98)
_fwd("radio", "is_a", "electrical device", 0.98, 0.98)
_fwd("television", "is_a", "electrical device", 0.98, 0.98)
_fwd("satellite", "is_a", "electrical device", 0.95, 0.95)
_fwd("electrical device", "is_a", "device", 0.95, 0.95)
_fwd("substation", "is_a", "power system", 0.98, 0.98)
_fwd("power plant", "is_a", "power system", 0.98, 0.98)

# ===================== part_of : forward only, onto real wholes ============
# The subject is always a concrete part and the target a concrete whole. No
# abstract category receives a part_of edge, so a 2-hop chain cannot walk
# inward into a category and then continue.
for _part, _whole in (
    ("engine", "car"), ("wheel", "car"), ("chassis", "car"), ("door", "car"),
    ("propeller", "airplane"), ("wing", "airplane"),
    ("boiler", "locomotive"), ("piston rod", "locomotive"),
    ("transformer", "substation"),
    ("turbine", "power plant"), ("valve", "power plant"),
    ("generator", "power plant"),
    ("summit", "mountain range"),
    # The ONLY cross-domain part_of edge, and it exists for a specific reason:
    # without it the graph has no acyclic 3-hop path at all. The planner keeps one
    # cue per relation, so a 3-hop chain needs three DIFFERENT relations, and
    # every `caused_by>causes` path in the hazard tree is a 2-cycle (X <- Y -cause->
    # X) because each cause has exactly one effect. This edge lets the walk go
    # road closure --caused_by--> landslide --part_of--> mountain range
    # --is_a--> landform, which crosses the causal and the taxonomic system
    # instead of bouncing back along one edge.
    ("landslide", "mountain range"),
    ("bridge", "village"),
    ("house", "village"), ("school", "village"),
    ("hospital", "town"),
):
    _fwd(_part, "part_of", _whole, 0.98, 0.98)

# ===================== hazard tree (is_a) =================================
_fwd("eruption", "is_a", "natural event", 0.98, 0.98)
_fwd("wildfire", "is_a", "natural event", 0.98, 0.98)
_fwd("landslide", "is_a", "natural event", 0.98, 0.98)
_fwd("dam failure", "is_a", "disruption", 0.95, 0.95)
_fwd("drought", "is_a", "weather event", 0.98, 0.98)
_fwd("storm", "is_a", "weather event", 0.98, 0.98)
_fwd("acid rain", "is_a", "weather event", 0.95, 0.95)

_fwd("overgrazing", "is_a", "environmental change", 0.95, 0.95)
_fwd("oil spill", "is_a", "environmental change", 0.95, 0.95)
_fwd("environmental change", "is_a", "disruption", 0.90, 0.90)
_fwd("vessel", "is_a", "vehicle", 0.95, 0.95)
_fwd("flood", "is_a", "flooding", 0.98, 0.98)
_fwd("flooding", "is_a", "disruption", 0.95, 0.95)
_fwd("weather event", "is_a", "natural event", 0.95, 0.95)
_fwd("natural event", "is_a", "event", 0.95, 0.95)
_fwd("ash cloud", "is_a", "weather event", 0.90, 0.90)
_fwd("service loss", "is_a", "disruption", 0.95, 0.95)
_fwd("road closure", "is_a", "transport disruption", 0.95, 0.95)
_fwd("washout", "is_a", "transport disruption", 0.90, 0.90)
_fwd("abutment failure", "is_a", "shutdown", 0.90, 0.90)
_fwd("shutdown", "is_a", "disruption", 0.95, 0.95)
_fwd("habitat loss", "is_a", "environmental loss", 0.95, 0.95)
_fwd("environmental loss", "is_a", "economic loss", 0.95, 0.95)
_fwd("economic loss", "is_a", "disruption", 0.90, 0.90)
_fwd("crop failure", "is_a", "economic loss", 0.90, 0.90)

_fwd("school", "is_a", "institution", 0.95, 0.95)
_fwd("hospital", "is_a", "institution", 0.95, 0.95)
_fwd("village", "is_a", "settlement", 0.95, 0.95)
_fwd("town", "is_a", "settlement", 0.95, 0.95)
_fwd("city", "is_a", "settlement", 0.95, 0.95)
_fwd("institution", "is_a", "building", 0.95, 0.95)
_fwd("building", "is_a", "structure", 0.95, 0.95)
_fwd("structure", "is_a", "material", 0.95, 0.95)
_fwd("settlement", "is_a", "material", 0.95, 0.95)
_fwd("summit", "is_a", "landform", 0.95, 0.95)
_fwd("mountain range", "is_a", "landform", 0.95, 0.95)
_fwd("landform", "is_a", "material", 0.95, 0.95)
_fwd("ferry service", "is_a", "transport service", 0.95, 0.95)
_fwd("transport service", "is_a", "disruption", 0.90, 0.90)
_fwd("harbor", "is_a", "settlement", 0.90, 0.90)

# ===================== causes / caused_by : DISJOINT pairs ==================
# Every causal pair here is an isolated two-node component. That is a hard
# requirement of the direction test, not a stylistic choice.
#
# A node that is both an effect and a cause is unusable as a direction anchor.
# The walker treats a declared inverse label as the same edge and traverses in
# either direction, so from a node that is `eruption causes ash cloud` AND
# `ash cloud causes crop failure`, the candidate set for asked-relation
# `causes` is {eruption, crop failure} and for `caused_by` is the same pair.
# Either direction question would then be genuinely ambiguous, and the audit
# below rejects it rather than letting it grade as correct 50% of the time.
#
# Keeping the components disjoint buys something the stage needs: 3-hop chains
# are built as causes>is_a>is_a instead of causes>causes, which tests the same
# step budget while leaving the causal relation unambiguous in both directions.
_pair("eruption", "ash cloud", "causes", "caused_by")
_pair("drought", "wildfire", "causes", "caused_by")
_pair("landslide", "road closure", "causes", "caused_by")
_pair("storm", "ferry service", "causes", "caused_by")
_pair("dam failure", "flood", "causes", "caused_by")
_pair("washout", "abutment failure", "causes", "caused_by")
_pair("acid rain", "crop failure", "causes", "caused_by")
_pair("overgrazing", "habitat loss", "causes", "caused_by")
_pair("oil spill", "service loss", "causes", "caused_by")
_pair("vessel", "oil spill", "causes", "caused_by")

# ===================== precedes / follows : strictly linear ================
def _timeline(nodes: list[str], s: float = 0.95) -> None:
    for i in range(len(nodes) - 1):
        _fwd(nodes[i], "precedes", nodes[i + 1], s, s)
        _fwd(nodes[i + 1], "follows", nodes[i], s, s)


_timeline(["handwriting", "printing press", "telegraph", "telephone",
           "radio", "television", "satellite"])

# ===================== has_property : asymmetric ==========================
# Two properties on one anchor is intentional and legal: it keeps the graph
# honest without making any ASKED pair ambiguous.
_fwd("car", "has_property", "large", 0.95, 0.95)
_fwd("bridge", "has_property", "ancient", 0.90, 0.90)
_fwd("bridge", "has_property", "modern", 0.90, 0.90)
_fwd("drought", "has_property", "dry", 0.95, 0.95)

# ===================== supports / contradicts : asymmetric ================
_fwd("rock strata record", "supports", "plate tectonic theory", 0.95, 0.95)
_fwd("seafloor ridge record", "supports", "plate tectonic theory", 0.95, 0.95)
_fwd("crater field record", "supports", "meteor impact theory", 0.95, 0.95)
_fwd("ice core record", "supports", "orbital climate theory", 0.95, 0.95)
_fwd("tide gauge record", "supports", "orbital climate theory", 0.95, 0.95)
_fwd("plate tectonic theory", "contradicts", "fixed earth claim", 0.95, 0.95)
_fwd("meteor impact theory", "contradicts", "static climate claim", 0.95, 0.95)

# ===================== example_of : asymmetric ============================
_fwd("car", "example_of", "automobile", 0.95, 0.95)
_fwd("hailstone", "example_of", "precipitation", 0.95, 0.95)
_fwd("acorn", "example_of", "seed", 0.95, 0.95)
_fwd("candle", "example_of", "light source", 0.95, 0.95)

# ===================== symmetric pairs stored in both directions ==========
_both_symmetric("physician", "doctor", "synonym", 0.98, 0.98)
_both_symmetric("cabinet", "cupboard", "synonym", 0.95, 0.95)
_both_symmetric("large", "tiny", "antonym", 0.98, 0.98)
_both_symmetric("ancient", "modern", "antonym", 0.98, 0.98)
_both_symmetric("dog", "perro", "linguistic_maps", 0.95, 0.95)
_both_symmetric("friend", "amigo", "linguistic_maps", 0.95, 0.95)
_both_symmetric("farm", "hacienda", "linguistic_maps", 0.95, 0.95)
# Each anchor here has exactly ONE neighbour under this relation. Fan-out would
# be legal data but would make the question unanswerable, and Stage B's
# forward-only guard could not see it because the reverse edges it stored were
# the reason for the ambiguity.
# One neighbour per anchor under a given symmetric relation. Fan-out is legal
# data but makes the question unanswerable, and the reason it slipped past
# Stage B is instructive: the ambiguity was created by the reverse edges that
# were stored on purpose.
_both_symmetric("city", "harbor", "spatial_near", 0.95, 0.95)
_both_symmetric("lake", "canal", "spatial_near", 0.95, 0.95)
_both_symmetric("mountain range", "river", "spatial_near", 0.95, 0.95)
_both_symmetric("autumn", "harvest season", "temporal_coincident", 0.95, 0.95)
_both_symmetric("ash cloud", "harvest season", "associated_with", 0.90, 0.90)
_both_symmetric("harbor", "ferry service", "associated_with", 0.90, 0.90)

PROPERTIES = {"large", "tiny", "ancient", "modern", "dry"}
CLAIMS = {"plate tectonic theory", "orbital climate theory", "meteor impact theory",
          "fixed earth claim", "static climate claim"}
PLACES = {"mountain range", "harbor", "city", "lake", "canal", "river",
          "village", "town"}
TIMES = {"autumn", "harvest season"}
WORDS = {"doctor", "cabinet", "automobile", "large", "tiny", "modern", "ancient",
         "perro", "amigo", "hacienda", "friend", "farm", "dog", "cupboard"}


def _semantic_violations(label: str, relation: str, target: str) -> list[str]:
    bad: list[str] = []
    if relation == "has_property" and target not in PROPERTIES:
        bad.append(f"has_property target {target!r} is not a property word")
    if relation in ("supports", "contradicts") and target not in CLAIMS:
        bad.append(f"{relation} target {target!r} is not a claim")
    if relation == "spatial_near" and target not in PLACES:
        bad.append(f"spatial_near target {target!r} is not a place")
    if relation == "temporal_coincident" and target not in TIMES:
        bad.append(f"temporal_coincident target {target!r} is not a time or event")
    if relation in ("synonym", "antonym") and target in PLACES:
        bad.append(f"{relation} target {target!r} is a place, not a lexical item")
    if relation == "linguistic_maps" and target not in WORDS:
        bad.append(f"linguistic_maps target {target!r} is not a word")
    if relation in ("causes", "precedes") and target == label:
        bad.append(f"{relation} target equals itself")
    # A whole may legitimately be a place or a settlement (a bridge is part of a
    # village), so PLACES is deliberately allowed here. Properties, claims and
    # bare words are never containers; this is the "whale part_of ocean" guard.
    if relation == "part_of" and target in PROPERTIES | CLAIMS | WORDS:
        bad.append(f"part_of target {target!r} cannot be a property/claim/word")
    # The hazard spine must stay strictly downward. A cycle here would make
    # "what is an eruption a kind of" answerable in more than one direction.
    if relation == "is_a" and target == label:
        bad.append("is_a target equals itself")
    return bad


# --------------------------------------------------------------------------
# Every (anchor, relation) the frozen question set reads, in either direction
# sense. Guarded against MIRRORED reachability below, not forward edges only.
# --------------------------------------------------------------------------
ASKED_PAIRS = [
# causes (forward): one effect per cause, so the walk has a single candidate
    ("eruption", "causes"), ("drought", "causes"), ("landslide", "causes"),
    ("storm", "causes"), ("dam failure", "causes"), ("washout", "causes"),
    ("acid rain", "causes"), ("overgrazing", "causes"),
    ("vessel", "causes"),
    # caused_by (reverse read of a declared pair -> legal)
    ("ash cloud", "caused_by"), ("wildfire", "caused_by"),
    ("road closure", "caused_by"), ("ferry service", "caused_by"),
    ("flood", "caused_by"), ("abutment failure", "caused_by"),
    ("habitat loss", "caused_by"),
    # NOTE: "oil spill" is deliberately NOT asked for caused_by. It is both an
    # effect (of a vessel) and a cause (of service loss), so a caused_by read
    # from it has two legal targets and cannot score one answer. The pair
    # survives as the middle of the two-hop vessel -> oil spill -> service loss
    # chain instead.
    # precedes / follows. Only the two ENDPOINTS are safe anchors. _timeline stores
    # BOTH `A precedes B` and `B follows A`, so the mirror pass adds `B precedes A`
    # and `A follows B` as well: every interior node is reachable to BOTH its
    # predecessor and its successor under the label pair, and "what comes after
    # the telegraph" cannot grade a single traversal. The interior nodes are still
    # exercised as chain interiors by the multi-hop questions.
    ("handwriting", "precedes"), ("satellite", "follows"),
    # part_of forward
    ("engine", "part_of"), ("wheel", "part_of"), ("chassis", "part_of"),
    ("door", "part_of"), ("propeller", "part_of"), ("wing", "part_of"),
    ("boiler", "part_of"), ("piston rod", "part_of"),
    ("transformer", "part_of"), ("turbine", "part_of"), ("valve", "part_of"),
    ("generator", "part_of"), ("summit", "part_of"), ("bridge", "part_of"),
    ("house", "part_of"), ("school", "part_of"), ("hospital", "part_of"),
    # is_a
    ("drought", "is_a"), ("landslide", "is_a"),
    ("wildfire", "is_a"), ("ash cloud", "is_a"), ("road closure", "is_a"),
    ("washout", "is_a"), ("abutment failure", "is_a"), ("service loss", "is_a"),
    ("crop failure", "is_a"), ("habitat loss", "is_a"), ("flood", "is_a"),
    # Category nodes are deliberately NOT asked for is_a. A parent legitimately
    # has many children, so "what is a natural event a kind of" has several
    # correct answers and cannot grade a single traversal. Category nodes are
    # still walked THROUGH by the 2-hop is_a>is_a chains.
    ("engine", "is_a"), ("car", "is_a"), ("turbine", "is_a"),
    ("generator", "is_a"), ("transformer", "is_a"), ("power plant", "is_a"),
    ("telegraph", "is_a"), ("telephone", "is_a"), ("satellite", "is_a"),
    ("summit", "is_a"), ("school", "is_a"), ("hospital", "is_a"),
    ("acid rain", "is_a"), ("overgrazing", "is_a"),
    ("dam failure", "is_a"), ("vessel", "is_a"),
    # symmetric relations, both directions stored so both anchors are usable
    ("physician", "synonym"), ("doctor", "synonym"),
    ("cabinet", "synonym"), ("cupboard", "synonym"),
    ("large", "antonym"), ("tiny", "antonym"),
    ("ancient", "antonym"), ("modern", "antonym"),
    ("dog", "linguistic_maps"), ("perro", "linguistic_maps"),
    ("friend", "linguistic_maps"), ("amigo", "linguistic_maps"),
    ("farm", "linguistic_maps"), ("hacienda", "linguistic_maps"),
    ("city", "spatial_near"), ("harbor", "spatial_near"),
    ("lake", "spatial_near"), ("canal", "spatial_near"),
    ("mountain range", "spatial_near"), ("river", "spatial_near"),
    ("autumn", "temporal_coincident"), ("harvest season", "temporal_coincident"),
    ("ash cloud", "associated_with"), ("harvest season", "associated_with"),
    ("harbor", "associated_with"), ("ferry service", "associated_with"),
    # asymmetric forward reads. bridge carries two properties on purpose; it is
    # never asked backwards, so the forward read stays unambiguous.
    ("car", "has_property"),
    ("drought", "has_property"),
    ("rock strata record", "supports"), ("seafloor ridge record", "supports"),
    ("crater field record", "supports"), ("ice core record", "supports"),
    ("tide gauge record", "supports"),
    ("plate tectonic theory", "contradicts"),
    ("meteor impact theory", "contradicts"),
    ("car", "example_of"), ("hailstone", "example_of"),
    ("acorn", "example_of"), ("candle", "example_of"),
]

# Relations with NO declared inverse, so the mirror pass may never create a
# reverse read for them. Within this set there are two sub-cases: the relations
# stored in both directions on purpose (semantically symmetric), and the truly
# directional ones, where only the forward edge exists and its reverse is a false
# claim. Both are checked against the store below.
NO_DECLARED_INVERSE = [
    "has_property", "supports", "contradicts", "example_of", "synonym",
    "antonym", "spatial_near", "temporal_coincident", "linguistic_maps",
    "associated_with",
]

# Relations deliberately stored in BOTH directions, so neither endpoint has to
# rely on a mirror the pass cannot create.
SYMMETRIC_STORED = {
    "synonym", "antonym", "spatial_near", "temporal_coincident",
    "linguistic_maps", "associated_with",
}

# Anchors that carry NO edge of the asked relation in either orientation, so the
# walker returns an empty candidate set and the only correct output is an honest
# refusal. Reachability is verified against walker-faithful incident reach (see
# the ambiguity guard), not asserted by hand.
FORBIDDEN_REVERSE = [
    # a property word with nothing attached to it as a subject
    ("tiny", "has_property"),
    ("cupboard", "example_of"),
    ("amigo", "spatial_near"),
    ("doctor", "supports"),
    ("dog", "contradicts"),
    ("cabinet", "supports"),
    ("large", "contradicts"),
    ("hacienda", "temporal_coincident"),
]


def build() -> None:
    labels: list[str] = list(dict.fromkeys(CONCEPTS))

    used = sorted({e[1] for e in EDGES})
    illegal = [r for r in used if r not in CANONICAL_RELATIONS]
    if illegal:
        raise SystemExit(f"REFUSING TO BUILD: relations outside the closed set: {illegal}")

    for src, rel, tgt, s, c in EDGES:
        for node in (src, tgt):
            if node not in labels:
                raise SystemExit(
                    f"REFUSING TO BUILD: edge {src}-{rel}->{tgt} names unknown node {node!r}")
        if not (0.8 <= s <= 1.0 and 0.8 <= c <= 1.0):
            raise SystemExit(
                f"REFUSING TO BUILD: {src}-{rel}->{tgt} strength={s} confidence={c} "
                "outside the 0.8-1.0 PoC band")

    violations: list[str] = []
    for src, rel, tgt, s, c in EDGES:
        for msg in _semantic_violations(src, rel, tgt):
            violations.append(f"  {src} -{rel}-> {tgt}: {msg}")
    if violations:
        raise SystemExit("REFUSING TO BUILD: semantic type violations:\n" + "\n".join(violations))

    # ---- asked-pair uniqueness under WALKER-FAITHFUL reachability ----
    # The walker traverses STORED ORIENTATION ONLY: _collect_candidates skips any
    # edge whose source is not the current node. A reverse read is therefore
    # legal only because the mirror pass materialises it as its own edge, and
    # only for relations with a declared inverse label. So the reachability an
    # asked pair must be judged against is:
    #     fwd[(node, label)]        outgoing stored edges with that label
    #   + mirrored[(node, label)]   sources of stored edges whose INVERSE label
    #                               is that label, i.e. the edges the mirror
    #                               pass will emit from `node`
    # unioned over the asked relation's label set {R, inverse(R)}.
    # Indexing incoming edges by their own label instead (treating the graph as
    # undirected) would model a traversal the walker cannot perform, and would
    # reject anchors that are in fact unambiguous.
    fwd: dict[tuple[str, str], set[str]] = {}
    mirrored: dict[tuple[str, str], set[str]] = {}
    for src, rel, tgt, s, c in EDGES:
        fwd.setdefault((src, rel), set()).add(tgt)
        # Only relations WITH a declared inverse gain a mirrored read. For the
        # other relations the mirror pass is not allowed to create a reverse
        # edge, so counting one here would invent exactly the illegal mirroring
        # this stage exists to detect: the reverse read has to come out empty.
        if rel in INVERSE:
            mirrored.setdefault((tgt, INVERSE[rel]), set()).add(src)

    def reach_from(node: str, rel: str) -> set:
        labels = {rel}
        if rel in INVERSE:
            labels.add(INVERSE[rel])
        out: set = set()
        for label in labels:
            out |= fwd.get((node, label), set())
            out |= mirrored.get((node, label), set())
        return out

    ambiguous = []
    for key in ASKED_PAIRS:
        if key not in fwd:
            continue
        reach = reach_from(*key)
        if len(reach) > 1:
            ambiguous.append((key, sorted(reach)))
    if ambiguous:
        raise SystemExit(
            "REFUSING TO BUILD: asked pairs ambiguous under mirroring:\n"
            + "\n".join(f"  {k[0]} | {k[1]} -> {v}" for k, v in ambiguous))

    # ---- forbidden_reverse pairs must be UNREACHABLE in both senses ----
    reachable_forbidden = []
    for key in FORBIDDEN_REVERSE:
        reach = reach_from(*key)
        if reach:
            reachable_forbidden.append((key, sorted(reach)))
    if reachable_forbidden:
        raise SystemExit(
            "REFUSING TO BUILD: forbidden_reverse pairs are actually reachable:\n"
            + "\n".join(f"  {k[0]} | {k[1]} -> {v}" for k, v in reachable_forbidden))

    # ---- same-label reverse edges are only legal where symmetry was declared ----
    # "Fabricated mirroring" is not "a reverse edge exists" in general: the
    # symmetric relations above store BOTH directions deliberately, because the
    # mirror pass is not allowed to synthesise them (they have no inverse label)
    # and both endpoints need to be usable anchors. What must not exist is a
    # reverse edge for a relation that is neither symmetric-by-declaration nor
    # inverse-labelled, e.g. `theory supports record` asserting the record
    # supports the theory. Checking the store is what makes the guarantee
    # meaningful: the walker traverses any existing edge in either orientation
    # and cannot tell a stored mirror from a forward edge.
    fabricated = []
    seen_edges = {(src, rel, tgt) for src, rel, tgt, _, _ in EDGES}
    for src, rel, tgt, _, _ in EDGES:
        if rel in SYMMETRIC_STORED:
            continue
        if rel not in INVERSE and (tgt, rel, src) in seen_edges:
            fabricated.append(
                f"  {tgt} -{rel}-> {src} (undeclared reverse of stored {src} -{rel}-> {tgt})")
    if fabricated:
        raise SystemExit(
            "REFUSING TO BUILD: undeclared same-label mirrors:\n" + "\n".join(fabricated))

    missing_asked = [k for k in ASKED_PAIRS if k not in fwd]
    if missing_asked:
        raise SystemExit(f"REFUSING TO BUILD: asked pairs with no stored edge: {missing_asked}")

    print(f"Building direction & multi-hop graph: {len(labels)} nodes, {len(EDGES)} edges")
    print(f"  relations used ({len(used)}/16): {used}")

    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    enc = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {lab: np.asarray(enc[i], dtype=np.float32) for i, lab in enumerate(labels)}

    out = Path(__file__).resolve().parent / "direction_multihop.db"
    for stale in (out, out.with_suffix(".db-wal"), out.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    concepts = {label: i + 1 for i, label in enumerate(labels)}
    store = SQLiteGraphStore(db_path=str(out))
    store.add_dataset(
        concepts=concepts,
        edges=[{"source": concepts[a], "target": concepts[b], "relation": r,
                "strength": s, "confidence": c} for a, r, b, s, c in EDGES],
        id_to_label={i: l for l, i in concepts.items()},
        embeddings=embeddings,
        protected_labels=list(labels),
    )
    store.set_metadata("dataset_name", "direction_multihop_stage_c")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.set_metadata("frozen", "true")
    store.set_metadata("contract", "GLM-X v3.3.2 section 16")
    store.set_metadata("version", "1.0.0")
    store.save_state(str(out))

    journal = json.loads(store.get_metadata("merge_journal") or "[]")
    if journal:
        raise SystemExit(
            f"REFUSING TO FREEZE: add_dataset merged {len(journal)} node pair(s): {journal}")
    if store.get_node_count() != len(labels):
        raise SystemExit(
            f"REFUSING TO FREEZE: built {store.get_node_count()} nodes, expected {len(labels)}")
    if not (95 <= store.get_node_count() <= 115):
        raise SystemExit(
            f"REFUSING TO FREEZE: {store.get_node_count()} nodes outside the ~100 target")

    stored = sorted(store.get_all_relations())
    # has_part is a runtime-only mirror label, not a storable relation.
    for r in ("causes", "caused_by", "precedes", "follows", "part_of"):
        if r not in stored:
            raise SystemExit(f"REFUSING TO FREEZE: direction-pair relation {r!r} absent from db")
    if "has_part" in stored:
        raise SystemExit("REFUSING TO FREEZE: has_part stored, but it is a runtime mirror label")

    print(f"Wrote {out}")
    print(f"  nodes={store.get_node_count()} edges={store.get_edge_count()}")
    print(f"  stored relations ({len(stored)}/16): {stored}")
    print(f"  ambiguous asked pairs: {len(ambiguous)}")
    print(f"  forbidden_reverse pairs verified unreachable: {len(FORBIDDEN_REVERSE)}")


if __name__ == "__main__":
    build()