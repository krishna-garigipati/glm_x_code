"""Stage E: build the Frozen PoC Evaluation Graph (GLM-X v3.3.2 section 16).

    - name: "Frozen PoC Evaluation Set"
      purpose: "Final scoring"
      goal: "Reach >= 90% accuracy"

Contract section 16 requires the final PoC score to be reported on a frozen
evaluation set, and forbids testing the PoC "only on large noisy graphs without
a clean controlled set". So this is a purpose-built clean graph: entirely
hand-authored, high strength and confidence, unambiguous facts, and every one of
the 16 canonical relations present.

It is deliberately NOT one of the Stage A-D graphs. Those were built to find
pipeline bugs, incrementally, with questions co-designed against failures already
seen. Re-running one of them would re-measure a graph the system has already been
debugged against, which is not evidence about a system meeting a PoC bar. This
graph is authored from the contract's own rules, and the question set is then
derived MECHANICALLY from the finished graph by generate_questions.py. Nothing in
the question set is chosen because the engine gets it right.

Design rules held here, all traceable to the contract:

  * section 16 `graph_building_rules`: only the 16 canonical relations; clear
    unambiguous facts; strength and confidence high (0.8-1.0) for PoC edges;
    "add direction pairs deliberately"; and "do not expect the system to invent
    missing edges" -- which is why every edge below is an explicit, checkable
    triple rather than something left to be inferred.
  * section 16 `node_creation_guidelines`: clear names, no ambiguous labels,
    specific entities alongside broader categories so multi-hop paths exist.
  * section 16 `practical_workflow` step 1: start very small and grow. The
    contract sizes its own direction/multi-hop graph at "~100 nodes" (line 439).
  * section 9: only causes/caused_by, precedes/follows and part_of/has_part may
    be mirrored at runtime. causes/caused_by and precedes/follows are STORED IN
    BOTH DIRECTIONS here (Stage D's proven pattern) so direction is readable
    straight out of the database. part_of is stored forward-only and its reverse
    comes from the runtime mirror, matching Stage D exactly and keeping has_part
    out of the stored relation set.
  * section 9: `is_a` is forward-only by prior agreement, so no `is_a` edge is
    ever written backwards.
  * section 17: no new relations, ever.

The five symmetric relations (synonym, antonym, spatial_near,
temporal_coincident, linguistic_maps) are stored in BOTH directions by hand,
because the mirror pass is forbidden to produce them and a one-way store would
make half the vocabulary unanswerable.

TWO LESSONS THIS BUILDER ENCODES, both found the hard way while writing it:

  1. A direction pair is only auto-gradable when BOTH endpoints have exactly one
     incident edge under {relation, declared inverse}. Any node with a second
     part_of edge (violin has string, bow and orchestra) or a second causes edge
     (solar flare has sun and radio blackout) is a hub, and from a hub both
     readings are legal, so neither end can be graded against one gold node.
     Hub nodes are kept -- deleting content to make a score look better is
     exactly the wrong move -- but they yield no direction-pair question, and
     generate_questions.py reports how many were skipped for this reason.

  2. A multi-hop chain can never repeat a MIRRORABLE relation. Walking
     `lens -part_of-> telescope -part_of-> observatory`, the second hop accepts
     both part_of and has_part, so telescope offers observatory (out) AND lens
     (in, mirrored) and the walk can diverge. Stage C never hit this because
     every chain there ends `... -is_a->`, and is_a has no inverse, so inbound
     edges are filtered out cleanly. Chains here follow that rule.

Usage:
    python test_results/stage_e/build_poc_graph.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore  # noqa: E402

STAGE_DIR = ROOT / "test_results" / "stage_e"
DB_PATH = STAGE_DIR / "poc_eval.db"

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

# Subjects genuinely absent from the graph, each carrying a literal relation cue
# so the refusal has to come from the entity gate and not from cue absence.
OUT_OF_GRAPH_SUBJECTS: list[tuple[str, str]] = [
    ("unicorn", "What is a unicorn?"),
    ("phoenix", "What is a phoenix?"),
    ("kraken", "What is a kraken?"),
    ("basilisk", "What is a basilisk?"),
    ("telepathy", "What causes telepathy?"),
    ("hovercraft", "What is the hovercraft part of?"),
    ("quark", "What is another word for quark?"),
    ("dragon", "What causes a dragon?"),
    ("trololo", "What does the trololo cause?"),
]

# Cue-free questions. The contract section 8 fallback triggers on cue ABSENCE, so
# every one of these must set heuristic_fallback_used. The generator's preflight
# re-checks that mechanically against the descriptor bank, so a question that
# quietly acquires a cue cannot slip in.
NONSENSE_CUE_FREE: list[str] = [
    "What colour is the smell of Tuesday?",
    "How many buttons in a whisper?",
    "Where does silence taste bitter?",
    # NOT "What weight is a memory?" -- the literal is_a descriptor "is a" sits
    # inside "...is a memory", so that phrasing resolves to chain ['is_a'] and
    # never reaches the cue-absence fallback at all. Measured, not assumed.
    "How much does a memory weigh?",
    "How round is the number seven?",
]

# Cue-free, but naming a real anchor that HAS a has_property edge, so the
# fallback fires, a real answer is produced from a real edge, and the disclosure
# branch is actually reached. Without this, five refusals would prove the
# fallback fired but never that a guess was disclosed as a guess.
# (anchor, question)
# `pyramid` rather than `grand piano`: "grand piano" is anchored to "piano",
# which has no has_property edge, so the question refuses instead of disclosing.
DISCLOSED_GUESS: tuple[str, str] = ("pyramid", "How wooden is the pyramid?")

# --------------------------------------------------------------------------
CONCEPTS = [
    # --- astronomy, deliberately built as a causes spine and a part_of chain
    "sun", "solar flare", "solar phenomenon", "natural phenomenon",
    "radio blackout", "disruption", "space weather",
    "mars", "jupiter", "planet", "star", "satellite", "celestial body",
    "moon", "spacecraft",
    "gas giant", "terrestrial planet",
    "telescope", "lens", "observatory", "instrument", "device",
    "probe", "mission", "activity",
    # --- isolated cause/effect pairs, each with its own short taxonomy tail so
    #     it also yields a 3-hop chain ending in is_a
    "smoke", "fire alarm", "alarm", "safety device",
    "short circuit", "power outage",
    "ice", "slippery road", "road condition",
    "sunlight", "photosynthesis", "biological process", "process",
    "rust", "stiff hinge", "frost", "cracked pavement",
    "gravity", "tidal bulge", "lack of rain", "dry riverbed",
    "overcooking", "tough meat",
    # --- isolated part_of pairs; each parent has exactly one child, which is
    #     what makes both directions of the pair auto-gradable
    "wheel", "bicycle", "vehicle", "transport", "fuel",
    "sail", "yacht", "rudder", "ship",
    "engine", "car", "pedal", "rickshaw",
    # --- geography: a part_of CHAIN, so it yields multi-hop and hub skips
    "pyramid", "giza", "cairo", "egypt", "africa", "nile",
    "monument", "structure", "city", "country", "place",
    # --- music: part_of sink, properties, association
    "violin", "cello", "orchestra", "string instrument", "musical group", "group",
    "piano", "grand piano", "drum", "concert", "musician",
    "loud", "soft", "fast", "slow", "heavy", "wooden",
    # --- evidence: the supports / contradicts pair
    "sunspot record", "solar cycle theory", "constant solar output",
    "moon crater record", "impact theory", "smooth sea floor claim",
    # --- time: precedes spine plus two coincident seasons
    "planting season", "spring", "harvest festival", "autumn",
    "seed", "seedling", "mature plant", "forest",
    "christmas", "new year",
    # --- everyday causality chain
    "heavy rain", "flooding", "crop loss", "food price rise",
    # --- lexicon
    "house", "home", "casa", "water", "agua", "bread", "pan",
]

EDGES: list[tuple[str, str, str, float, float]] = []


def _fwd(src: str, rel: str, tgt: str, s: float = 0.95, c: float = 0.95) -> None:
    EDGES.append((src, rel, tgt, s, c))


def _pair(a: str, b: str, left: str, right: str,
          s: float = 0.95, c: float = 0.95) -> None:
    """Store a declared inverse pair in both directions.

    Both members are stored rather than trusting only the runtime mirror, so
    direction is readable straight out of the database and the store does not
    depend on the mirror pass to be correct. Same pattern as Stage D.
    """
    _fwd(a, left, b, s, c)
    _fwd(b, right, a, s, c)


def _both_symmetric(a: str, b: str, rel: str,
                    s: float = 0.95, c: float = 0.95) -> None:
    _fwd(a, rel, b, s, c)
    _fwd(b, rel, a, s, c)


# ============================ is_a : taxonomy, forward only ==================
for _leaf, _parent in (
    ("mars", "terrestrial planet"),
    ("terrestrial planet", "planet"),
    ("jupiter", "gas giant"),
    ("gas giant", "planet"),
    ("planet", "celestial body"),
    ("sun", "star"),
    ("star", "celestial body"),
    ("moon", "satellite"),
    ("satellite", "spacecraft"),
    ("spacecraft", "vehicle"),
    ("solar flare", "solar phenomenon"),
    ("solar phenomenon", "natural phenomenon"),
    ("radio blackout", "disruption"),
    ("power outage", "disruption"),
    ("fire alarm", "alarm"),
    ("alarm", "safety device"),
    ("slippery road", "road condition"),
    ("photosynthesis", "biological process"),
    ("biological process", "process"),
    ("bicycle", "vehicle"),
    ("vehicle", "transport"),
    ("mission", "activity"),
    ("telescope", "instrument"),
    ("instrument", "device"),
    ("violin", "string instrument"),
    ("cello", "string instrument"),
    ("string instrument", "instrument"),
    ("orchestra", "musical group"),
    ("musical group", "group"),
    ("pyramid", "monument"),
    ("monument", "structure"),
    ("giza", "city"),
    ("cairo", "city"),
    ("city", "place"),
    ("egypt", "country"),
    ("country", "place"),
    ("mature plant", "forest"),
    ("forest", "place"),
):
    _fwd(_leaf, "is_a", _parent, 0.97, 0.97)

# ============================ part_of : forward only =========================
# Reverse direction comes from the runtime mirror, exactly as in Stage D, so
# has_part never enters the stored relation set.
for _part, _whole in (
    ("lens", "telescope"),
    ("telescope", "observatory"),
    ("probe", "mission"),
    # deliberately a CHAIN: giza and egypt are hubs, which the validator and the
    # generator both account for rather than quietly hiding
    ("pyramid", "giza"),
    ("giza", "egypt"),
    ("cairo", "egypt"),
    ("egypt", "africa"),
    ("violin", "orchestra"),
    ("cello", "orchestra"),
    # isolated pairs: exactly one incident edge each end, so both directions
    # are auto-gradable
    ("wheel", "bicycle"),
    ("sail", "yacht"),
    ("rudder", "ship"),
    ("engine", "car"),
    ("pedal", "rickshaw"),
):
    _fwd(_part, "part_of", _whole, 0.96, 0.96)

# ==================== causes / caused_by : stored both ways ==================
_pair("sun", "solar flare", "causes", "caused_by", 0.96, 0.96)
_pair("solar flare", "radio blackout", "causes", "caused_by", 0.96, 0.96)
_pair("heavy rain", "flooding", "causes", "caused_by", 0.96, 0.96)
_pair("flooding", "crop loss", "causes", "caused_by", 0.96, 0.96)
_pair("crop loss", "food price rise", "causes", "caused_by", 0.96, 0.96)
_pair("smoke", "fire alarm", "causes", "caused_by", 0.96, 0.96)
_pair("short circuit", "power outage", "causes", "caused_by", 0.96, 0.96)
_pair("ice", "slippery road", "causes", "caused_by", 0.95, 0.95)
_pair("sunlight", "photosynthesis", "causes", "caused_by", 0.96, 0.96)

# Isolated cause/effect pairs, added purely to give the `direction_pair`
# category margin above the contract minimum of 10 questions. Only pairs whose
# BOTH endpoints have exactly one incident causes/caused_by edge can be graded in
# both directions, and in a causal CHAIN no interior node can: from `flooding`
# under `causes` the walker can legally reach both `heavy rain` (via the
# mirrored caused_by) and `crop loss` (forward), so neither direction has a
# single gold answer. Each pair below stands alone for that reason.
_pair("rust", "stiff hinge", "causes", "caused_by", 0.95, 0.95)
_pair("frost", "cracked pavement", "causes", "caused_by", 0.95, 0.95)
_pair("gravity", "tidal bulge", "causes", "caused_by", 0.95, 0.95)
_pair("lack of rain", "dry riverbed", "causes", "caused_by", 0.95, 0.95)
_pair("overcooking", "tough meat", "causes", "caused_by", 0.95, 0.95)

# ==================== precedes / follows : strictly linear ==================
_pair("seed", "seedling", "precedes", "follows", 0.95, 0.95)
_pair("seedling", "mature plant", "precedes", "follows", 0.95, 0.95)
_pair("christmas", "new year", "precedes", "follows", 0.95, 0.95)

# ==================== synonym : symmetric, stored both ways ==================
_both_symmetric("piano", "grand piano", "synonym", 0.96, 0.96)
_both_symmetric("house", "home", "synonym", 0.96, 0.96)

# ==================== antonym : symmetric, stored both ways ===================
_both_symmetric("loud", "soft", "antonym", 0.96, 0.96)
_both_symmetric("fast", "slow", "antonym", 0.96, 0.96)

# ==================== spatial_near : symmetric, stored both ways =============
_both_symmetric("cairo", "giza", "spatial_near", 0.94, 0.94)
_both_symmetric("nile", "cairo", "spatial_near", 0.94, 0.94)

# ==================== temporal_coincident : stored both ways =================
_both_symmetric("planting season", "spring", "temporal_coincident", 0.95, 0.95)
_both_symmetric("harvest festival", "autumn", "temporal_coincident", 0.95, 0.95)

# ==================== linguistic_maps : stored both ways =====================
_both_symmetric("house", "casa", "linguistic_maps", 0.96, 0.96)
_both_symmetric("water", "agua", "linguistic_maps", 0.96, 0.96)
_both_symmetric("bread", "pan", "linguistic_maps", 0.96, 0.96)

# ==================== has_property : forward only ============================
for _subj, _prop in (
    ("drum", "loud"),
    ("grand piano", "heavy"),
    ("violin", "wooden"),
    ("pyramid", "wooden"),
):
    _fwd(_subj, "has_property", _prop, 0.95, 0.95)

# ==================== example_of : forward only ==============================
for _item, _cls in (
    ("jupiter", "gas giant"),
    ("violin", "string instrument"),
    ("cello", "string instrument"),
):
    _fwd(_item, "example_of", _cls, 0.96, 0.96)

# ==================== associated_with : forward only ========================
# Each of these extra edges exists so a 3-hop chain can be built with THREE
# DISTINCT relations. They are not decoration: `collapse_consecutive_repeats`
# is enabled in configs/config_g2p.yaml, so any chain like [part_of, is_a, is_a]
# is contracted to [part_of, is_a] before it is ever walked and is therefore
# unanswerable as declared. Every 3-hop chain here is [R1, R2, R3] with the
# three relations pairwise different.
for _item, _thing in (
    ("concert", "orchestra"),
    ("musician", "orchestra"),
    ("mission", "observatory"),
    ("activity", "observatory"),
    ("fuel", "vehicle"),
    ("space weather", "solar phenomenon"),
    ("city", "country"),
):
    _fwd(_item, "associated_with", _thing, 0.95, 0.95)

# ==================== supports / contradicts : forward only =================
_fwd("sunspot record", "supports", "solar cycle theory", 0.96, 0.96)
_fwd("moon crater record", "supports", "impact theory", 0.96, 0.96)
_fwd("sunspot record", "contradicts", "constant solar output", 0.96, 0.96)
_fwd("moon crater record", "contradicts", "smooth sea floor claim", 0.96, 0.96)


# ==========================================================================
# Reachability
# ==========================================================================
def _triples() -> set[tuple[str, str, str]]:
    return {(s, r, t) for s, r, t, _, _ in EDGES}


def _mirror_edges() -> set[tuple[str, str, str]]:
    """Reverse edges the runtime mirror pass is allowed to synthesise."""
    return {(t, INVERSE[r], s) for s, r, t, _, _ in EDGES if r in MIRRORED}


def asked_labels(rel: str) -> set[str]:
    """Relation labels the walker accepts for an asked relation.

    A relation and its DECLARED INVERSE are two labels for the same directed
    edge, so both count when deciding what a question can reach. Judging on the
    asked label alone under-counts reachability for every mirrorable relation,
    which is how ("plantlet", "precedes") came to be filed as an unanswerable
    gap in Stage D when the graph stores `plantlet follows sprout`.
    """
    return {rel, INVERSE[rel]} if rel in INVERSE else {rel}


def reach_from(anchor: str, rel: str, hops: int | None = 1) -> set[str]:
    """What the walker can reach from `anchor` under `rel`.

    `hops=1` is the right question for a graded one-hop question: uniqueness must
    hold for the single hop actually being graded. `cat is_a mammal is_a animal`
    means "What is a cat?" is unambiguous even though a transitive search also
    reaches animal.
    """
    labels = asked_labels(rel)
    pool = _triples() | _mirror_edges()
    best: dict[str, int] = {anchor: 0}
    frontier = [anchor]
    while frontier:
        cur = frontier.pop()
        if hops is not None and best[cur] >= hops:
            continue
        for s, r, t in pool:
            if s == cur and r in labels and (t not in best or best[t] > best[cur] + 1):
                best[t] = best[cur] + 1
                frontier.append(t)
    return set(best) - {anchor}


# ==========================================================================
# Mechanical derivation of what is auto-gradable
# ==========================================================================
def unique_one_hop() -> list[tuple[str, str, str]]:
    """Every stored edge (src, rel, tgt) that is the ONLY legal answer.

    Deterministic order: sorted, so the frozen set never depends on dict or set
    iteration order.
    """
    out = []
    for s, r, t in sorted(_triples()):
        if reach_from(s, r, 1) == {t}:
            out.append((s, r, t))
    return out


def ambiguous_one_hop() -> list[tuple[str, str, list[str]]]:
    """Stored edges that cannot be auto-graded, with the competing answers.

    Reported rather than hidden. If this list is long, the evaluation set covers
    less of the graph than the edge count suggests, and the reader is entitled to
    know that.
    """
    out = []
    for s, r, t in sorted(_triples()):
        reach = reach_from(s, r, 1)
        if reach != {t}:
            out.append((s, r, sorted(reach)))
    return out


def direction_pairs() -> list[tuple[str, str, str]]:
    """Declared-inverse pairs gradeable in BOTH directions.

    Requires each endpoint to have exactly one incident edge under
    {relation, declared inverse}; see lesson 1 in the module docstring.

    Deduplicated, and this mattered: because causes/caused_by and
    precedes/follows are stored in both directions, iterating the raw triple set
    yields BOTH (a, causes, b) and (b, caused_by, a), and both satisfy every
    condition. Returning each pair twice produced two identical questions per
    direction, which is exactly the kind of duplicate that inflates a score.
    """
    seen: dict[tuple, tuple[str, str, str]] = {}
    for s, r, t in sorted(_triples()):
        if r not in MIRRORED:
            continue
        inv = INVERSE[r]
        other = (t, inv, s)
        if other not in _triples():
            continue
        if reach_from(s, r, 1) != {t}:
            continue
        if reach_from(t, inv, 1) != {s}:
            continue
        key = tuple(sorted(((s, r, t), other)))
        seen.setdefault(key, min((s, r, t), other))
    return [seen[k] for k in sorted(seen)]


def one_way_direction_pairs() -> list[tuple[str, str, str]]:
    """Every gradeable single read of a mirrorable pair, deduplicated.

    A pair proven in BOTH directions is much stronger evidence of direction
    handling than one proven once, so `direction_pairs()` is the headline set and
    this is the superset that feeds the one-hop count.
    """
    out = set()
    for s, r, t in sorted(_triples()):
        if r not in MIRRORED or (t, INVERSE[r], s) not in _triples():
            continue
        if reach_from(s, r, 1) == {t}:
            out.add((s, r, t))
        if reach_from(t, INVERSE[r], 1) == {s}:
            out.add((t, INVERSE[r], s))
    return sorted(out)


def _would_collapse(chain: list[str]) -> bool:
    """True if `chain` cannot be expressed as a question at all.

    Two independent mechanisms in the pipeline make a repeated relation
    inexpressible, and both were found by this generator refusing to freeze:

      1. `configs/config_g2p.yaml` sets collapse_consecutive_repeats: true, so an
         ADJACENT repeat like [is_a, is_a] is contracted to [is_a] before the
         walk starts.
      2. `QueryRelationExtractor._literal_cue_relations` keeps only the longest
         matching descriptor PER RELATION, so a relation cued twice contributes
         one entry. "What type of thing is X, and what is that associated with,
         and what type of thing is that?" yields chain ['is_a',
         'associated_with'] -- the second is_a cue is swallowed, even though the
         two is_a clauses are not adjacent.

    So the rule is pairwise-distinct relations, which is strictly stronger than
    the adjacent-repeat rule and covers both. Such a chain is not a hard
    question, it is an IMPOSSIBLE one: no answer can ever match the declared
    chain. They are excluded from the frozen set and counted separately so the
    exclusion is visible rather than silent.
    """
    return len(set(chain)) != len(chain)


def walkable_edges() -> dict[str, set[tuple[str, str]]]:
    """Adjacency the walker can actually traverse, as {source: {(rel, target)}}.

    Two corrections over naively indexing the stored triples:

      * A step may use a MIRROR-derived edge. `telescope part_of observatory` is
        stored forward only, but the walker can legitimately go
        observatory --part_of--> telescope through the section 9 mirror, so that
        step is real and must be available to chain construction. Both
        orientations are labelled with the FORWARD relation (`part_of`), which is
        the one that has a cue phrase; the mirrored label `has_part` has no
        descriptor in configs/config_g2p.yaml and so cannot be asked for.

      * Every step must actually CONNECT. An earlier version iterated stored
        triples and tested only `reach_from(b, rel, 1) == {c}`, discarding the
        connecting edge's source. Because reach_from consults the mirror, that
        test passes even when no edge starts at `b` at all, and the derivation
        happily built chains across disconnected parts of the graph.
    """
    adj: dict[str, set[tuple[str, str]]] = {}
    for s, r, t in _triples():
        adj.setdefault(s, set()).add((r, t))
        if r in MIRRORED:
            adj.setdefault(t, set()).add((r, s))
    return adj


def multi_hop_chains() -> list[tuple[list[str], list[str]]]:
    """Every 2-hop and 3-hop path whose each step is individually unambiguous.

    Derived, not hand-listed. A step (a, rel) -> b is kept only when
    reach_from(a, rel, 1) == {b}, which is what rules out the repeated-mirrorable
    -relation chains described in lesson 2 of the module docstring. Paths may not
    revisit a node, since a walk that does is a cycle rather than a short chain.
    Chains the engine could not express are dropped (see `_would_collapse`).
    Deduplicated, because symmetric relations are stored in both directions and
    would otherwise yield the same path twice.
    """
    adj = walkable_edges()
    out: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for a in sorted(adj):
        for r0, b in sorted(adj[a]):
            if b == a or reach_from(a, r0, 1) != {b}:
                continue
            for r1, c in sorted(adj.get(b, ())):
                if c in (a, b) or reach_from(b, r1, 1) != {c}:
                    continue
                if not _would_collapse([r0, r1]):
                    out.add(((a, b, c), (r0, r1)))
                for r2, d in sorted(adj.get(c, ())):
                    if d in (a, b, c) or reach_from(c, r2, 1) != {d}:
                        continue
                    if not _would_collapse([r0, r1, r2]):
                        out.add(((a, b, c, d), (r0, r1, r2)))
    return [(list(p), list(ch)) for p, ch in sorted(out)]


def chains_by_hops() -> dict[int, list[tuple[list[str], list[str]]]]:
    """Multi-hop chains grouped by hop count. A path of N+1 nodes is an N-hop walk."""
    out: dict[int, list[tuple[list[str], list[str]]]] = {}
    for path, chain in multi_hop_chains():
        out.setdefault(len(chain), []).append((path, chain))
    return out


def mirror_controls() -> list[tuple[str, str, list[str]]]:
    """Backward reads that MUST refuse, as (anchor, relation, forbidden_far_ends).

    For a no-inverse relation stored one way, `a rel b`, asking `rel` from `b`
    must not name `a`: the mirror pass is forbidden from copying these relations,
    so answering would be inventing knowledge. This is the strongest available
    evidence against illegal same-label mirroring, and it is derived from the
    graph rather than chosen.

    Grouped by (anchor, relation) because several edges can share a target.
    `violin has_property wooden` and `pyramid has_property wooden` both make
    "What is known for the wooden?" a single question whose forbidden set is
    {violin, pyramid}. Emitting it twice would duplicate a question and, worse,
    each copy would forbid only one of the two nodes it must not name. Ambiguity
    does not spoil a must-REFUSE row: the expectation is that no far end is
    named at all.
    """
    grouped: dict[tuple[str, str], set[str]] = {}
    for s, r, t in sorted(_triples()):
        if r in NO_DECLARED_INVERSE and (t, r, s) not in _triples():
            if reach_from(t, r, 1):  # some other edge would legitimately answer
                continue
            grouped.setdefault((t, r), set()).add(s)
    return [(a, r, sorted(fars)) for (a, r), fars in sorted(grouped.items())]


# ==========================================================================
# Validation
# ==========================================================================
def validate() -> list[tuple[str, str]]:
    """Everything the frozen question set's conclusions will rest on.

    The question set is derived from this graph, so a mistake here would be baked
    into the "frozen" evaluation set and would silently decide the PoC verdict.
    """
    problems: list[tuple[str, str]] = []
    declared = set(CONCEPTS)
    trip = _triples()
    used_labels = {x for e in EDGES for x in (e[0], e[2])}

    # A label listed twice is silently deduplicated by `dict.fromkeys` below, so
    # the graph would still build and still look right -- but the declared count
    # would no longer match the node count, and any later check that compares the
    # two would be comparing different things.
    dupes = sorted({c for c in CONCEPTS if CONCEPTS.count(c) > 1})
    if dupes:
        problems.append(("concept-declared-twice", ", ".join(dupes)))

    for src, rel, tgt, _, _ in EDGES:
        for end in (src, tgt):
            if end not in declared:
                problems.append(("edge-endpoint-not-declared", end))
        if rel not in CANONICAL_RELATIONS:
            problems.append(("relation-not-canonical", rel))

    unused = declared - used_labels
    if unused:
        # An isolated node is legal but defeats the point of a hand-audited
        # graph: an anchor nobody can reach fails for embedding reasons and masks
        # whether the planner, the walker or the decoder is at fault.
        problems.append(("declared-but-unused-concepts", ", ".join(sorted(unused))))
    if len(EDGES) != len(trip):
        problems.append(("duplicate-edge", f"{len(EDGES)} rows, {len(trip)} distinct"))

    used_rels = {r for _, r, _, _, _ in EDGES}
    for rel in CANONICAL_RELATIONS:
        if rel not in used_rels:
            problems.append(("relation-unused", rel))
    if "has_part" in used_rels:
        problems.append(("relation-not-canonical", "has_part is a runtime mirror label"))
    for rel in NO_DECLARED_INVERSE:
        if rel in MIRRORED:
            problems.append(("no-inverse-relation-is-mirrorable", rel))
    for rel in SYMMETRIC_STORED:
        for s, r, t in trip:
            if r == rel and (t, r, s) not in trip:
                problems.append(("symmetric-relation-not-stored-both-ways",
                                 f"{s} {r} {t}"))

    # --- honesty sets --------------------------------------------------
    # The out-of-graph subjects carry the refusal burden, so both halves must
    # hold: the subject must really be absent from the graph, and the question
    # must really name it. A subject that silently failed to appear in its own
    # question would degrade into a generic cue-less question and prove only that
    # the fallback fires.
    for subject, question in OUT_OF_GRAPH_SUBJECTS:
        if subject.lower() in {c.lower() for c in CONCEPTS}:
            problems.append(("out-of-graph-subject-is-a-concept", subject))
        if subject.lower() in {u.lower() for u in used_labels}:
            problems.append(("out-of-graph-subject-is-in-the-graph", subject))
        if subject.lower() not in question.lower():
            problems.append(("out-of-graph-question-does-not-name-subject",
                             f"{subject} :: {question}"))
    if DISCLOSED_GUESS[0] not in declared:
        problems.append(("disclosed-guess-anchor-not-a-concept", DISCLOSED_GUESS[0]))
    if reach_from(DISCLOSED_GUESS[0], "has_property", 1) != {"wooden"}:
        problems.append(("disclosed-guess-anchor-has-no-ambiguous-free-has_property",
                         DISCLOSED_GUESS[0]))

    # --- derived sets must be non-trivial, or the matrix minima cannot be met
    by_hops = chains_by_hops()
    hops2 = by_hops.get(2, [])
    hops3 = by_hops.get(3, [])
    if len(direction_pairs()) < 5:
        problems.append(("too-few-bidirectional-direction-pairs",
                         f"{len(direction_pairs())} pairs < 5"))
    if len(hops2) < 4:
        problems.append(("too-few-2-hop-chains", f"{len(hops2)} < 4"))
    if len(hops3) < 2:
        problems.append(("too-few-3-hop-chains", f"{len(hops3)} < 2"))
    if len(unique_one_hop()) < 30:
        problems.append(("too-few-unique-one-hop-edges", f"{len(unique_one_hop())} < 30"))
    if len(mirror_controls()) < 8:
        problems.append(("too-few-mirror-controls", f"{len(mirror_controls())} < 8"))
    return problems


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize_and_hash(store) -> str:
    """Checkpoint the WAL, close the store, then hash the file.

    SQLiteGraphStore opens in WAL mode and exposes no close(), so hashing the
    .db file straight after save_state() produced a digest of a file that was
    still missing whatever the WAL had not yet flushed -- the same graph hashed
    differently on a later run. The frozen hash has to describe the file that
    will actually be loaded, so the WAL is checkpointed and the connection
    closed first. `_conn` is private, but there is no public equivalent and an
    unstable freeze hash is worse than reaching in.
    """
    store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store._conn.close()
    for side in (DB_PATH.with_suffix(".db-wal"), DB_PATH.with_suffix(".db-shm")):
        if side.exists():
            side.unlink()
    return sha256(DB_PATH)


def build() -> int:
    problems = validate()
    if problems:
        print("REFUSING TO BUILD:")
        for kind, detail in problems:
            print(f"  {kind}: {detail}")
        return 1

    labels = list(dict.fromkeys(CONCEPTS))
    used = sorted({r for _, r, _, _, _ in EDGES})
    by_hops = chains_by_hops()
    print(f"Building Frozen PoC Evaluation Graph: {len(labels)} nodes, {len(EDGES)} edges")
    print(f"  relations used ({len(used)}/16): {used}")
    print(f"  unique one-hop edges   : {len(unique_one_hop())}")
    print(f"  ambiguous one-hop edges: {len(ambiguous_one_hop())}")
    print(f"  bidirectional pairs    : {len(direction_pairs())}")
    print(f"  gradeable pair reads   : {len(one_way_direction_pairs())}")
    print(f"  2-hop chains           : {len(by_hops.get(2, []))}")
    print(f"  3-hop chains           : {len(by_hops.get(3, []))}")
    print(f"  mirror controls        : {len(mirror_controls())}")

    import numpy as np
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
    store.set_metadata("dataset_name", "poc_eval_stage_e")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.set_metadata("frozen", "true")
    store.set_metadata("contract", "GLM-X v3.3.2 sections 8/9/15/16")
    store.set_metadata("version", "1.0.0")
    store.save_state(str(DB_PATH))

    # --- freeze assertions ---------------------------------------------
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

    digest = finalize_and_hash(store)

    print(f"\nWrote {DB_PATH}")
    print(f"  nodes={len(labels)} edges={len(EDGES)}")
    print(f"  stored relations ({len(stored)}/16): {stored}")
    print(f"  sha256(db) = {digest}")
    return 0


if __name__ == "__main__":
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    raise SystemExit(build())
