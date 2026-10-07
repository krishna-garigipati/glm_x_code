"""Stage F: build the HELD-OUT Evaluation Graph (GLM-X v3.3.2 sections 8/9/15/16).

    Contract section 16: "Final PoC score must be reported on a frozen evaluation
    set", and stage E already reported one. This is the second and final
    frozen set, built AFTER the Stage E result was known, to answer the only
    question a second evaluation can answer that a re-run cannot:

        does GLM-X still meet the PoC bar on data it was NOT tuned against?

WHY THIS IS NOT A RE-RUN
    A re-run of Stage E measures the same graph the anchor defect was diagnosed
    on. Its 208/208 therefore also measures the fix on its own test data. That
    is a real result but it is not evidence of generalisation. So the domain is
    entirely new (food, cooking, nutrition -- no overlap with the toy,
    relation-coverage, direction/multi-hop, honesty/fallback or mixed Stage A-E
    domains), the graph is authored from the contract's own rules, and the
    question set is then derived MECHANICALLY by generate_questions.py. No
    question is chosen because the engine gets it right, and nothing is edited
    after scoring.

THE NO-LOOK COMMITMENT
    Everything below was authored BEFORE any Stage F question was scored. The
    graph is frozen first, the question set is derived and frozen second, and
    the scored run happens once. `generate_questions.py` reuses Stage E's
    templates verbatim so the question generator is not tuned to this graph
    either. If the score comes back low, that is the held-out number and it
    gets reported as-is.

Design rules held here, all traceable to the contract:

  * section 5 `graph_building.process`: choose a simple clear domain, create
    unambiguous nodes, use only the 16 canonical relations, strength and
    confidence in 0.8-1.0, support direction pairs and short multi-hop paths,
    version and freeze.
  * section 5 `size_strategy`: a ~100-150 node clean controlled graph is the
    contract's own "Relation Coverage Graph" size; the clean graph is the point.
  * section 5 `node_rules`: clear unambiguous labels, no vague or overloaded
    names, specific entities alongside broader categories so multi-hop exists.
  * section 9: only causes/caused_by, precedes/follows and part_of/has_part may
    be mirrored at runtime. causes/caused_by and precedes/follows are STORED IN
    BOTH DIRECTIONS here (Stage D's proven pattern). part_of is stored
    forward-only and its reverse comes from the runtime mirror, keeping
    has_part out of the stored relation set.
  * section 9: `is_a` is forward-only by prior agreement.
  * section 17: no new relations, ever.

FOUR LESSONS CARRIED OVER FROM STAGE E, all found the hard way there:

  1. A direction pair is only auto-gradable when BOTH endpoints have exactly
     one incident edge under {relation, declared inverse}. Any node with a
     second `causes` edge is a hub, and from a hub both readings are legal, so
     neither end can be graded against one gold node. Hub nodes are kept --
     deleting content to make a score look better is the wrong move -- but they
     yield no direction-pair question and are counted separately.

  2. A multi-hop chain can never repeat a MIRRORABLE relation. Walking
     `lens -part_of-> telescope -part_of-> observatory`, the second hop accepts
     both part_of and has_part, so telescope offers observatory (out) AND lens
     (in, mirrored) and the walk can diverge.

  3. A node label must not CONTAIN a descriptor phrase from
     `configs/config_g2p.yaml`, or the anchor inside a question silently cues a
     relation the gold never claimed. Stage E's own builder missed this. The
     matcher uses `(?<!\\w)...(?!\\w)` word-boundary anchors, so single letters
     such as the `is_a` cue "a" cannot match inside "water" -- but multi-word
     descriptors can still match a label verbatim. `LABEL_CUE_TRAPS` below
     refuses to build if any label collides.

  4. A node label must not be a whole token that `_QUESTION_FILLER` drops. The
     filler bank contains "part", "cause", "before", "after", "near",
     "example", "related", "word", "type" and others, so a node called "part"
     or "saltwater foam" containing the token "before" could not be anchored.
     `LABEL_FILLER_TRAPS` refuses that too.

Usage:
    python test_results/stage_f/build_holdout_graph.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore  # noqa: E402

STAGE_DIR = ROOT / "test_results" / "stage_f"
DB_PATH = STAGE_DIR / "holdout_eval.db"

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

# Lesson 3: descriptor phrases that must not appear inside any node label.
# Multi-word descriptors only -- the matcher's word boundaries already make a
# single-letter cue like "a" unable to match inside "water".
LABEL_CUE_TRAPS = (
    "at the same time as", "what is associated with", "what is another word for",
    "what is the opposite of", "what is the term for", "what do you call",
    "what comes before", "what comes after", "what type of", "what kind of",
    "is classified as", "is caused by", "is supported by", "is an example",
    "is an instance of", "part of", "made up of", "contained in",
    "component of", "related to", "linked to", "example of", "instance of",
    "opposite of", "prior to", "close to", "adjacent to", "backed by",
    "evidence for", "for instance", "same as", "also called", "cause",
    "against", "succeeds", "precedes", "refutes", "possesses",
)

# Lesson 4: whole tokens GLMXPipeline._QUESTION_FILLER drops during anchoring.
LABEL_FILLER_TRAPS = frozenset("""
    property part parts word another opposite related associated example
    examples cause causes caused follows follow precedes comes after before
    near based on kind kinds type types thing things name belong belongs
    which what how why when where because since
""".split())

# Subjects genuinely absent from the graph, each carrying a literal relation cue
# so the refusal has to come from the entity gate and not from cue absence.
# Checked mechanically in validate(): absent from CONCEPTS AND named in its own
# question. All food-domain words that are deliberately NOT in this graph.
OUT_OF_GRAPH_SUBJECTS: list[tuple[str, str]] = [
    ("persimmon", "What type of thing is the persimmon?"),
    ("tamarind", "What type of thing is the tamarind?"),
    ("nougat", "What is a nougat?"),
    ("fondue", "What does the fondue cause?"),
    ("watercress", "What causes the watercress?"),
    ("quenelle", "What is another word for the quenelle?"),
    ("marzipan", "What was the marzipan caused by?"),
    ("tagliatelle", "What is the tagliatelle part of?"),
    ("gremolata", "What is near the gremolata?"),
]

# Cue-free questions. Contract section 8 fallback triggers on cue ABSENCE, so
# every one of these must set heuristic_fallback_used. The generator's preflight
# re-checks that mechanically against the real descriptor bank, so a question
# that quietly acquires a cue cannot slip in.
NONSENSE_CUE_FREE: list[str] = [
    "How many legs does the letter seven have?",
    "What colour is the sound of a doorbell?",
    "How heavy is the taste of Tuesday?",
    "What is the temperature of the word apple?",
    "How bitter is the shape of a circle?",
    "What is the length of the number nine?",
]

# A real in-graph anchor whose has_property read is unique, asked in a way that
# carries NO relation cue. The system must fall back, and section 8 requires it
# to DISCLOSE that it is guessing. Tests the disclosure path, not refusal.
DISCLOSED_GUESS: tuple[str, str] = ("cutting board", "How wooden is the cutting board?")

CONCEPTS: list[str] = []
EDGES: list[tuple[str, str, str, float, float]] = []


def _fwd(src: str, rel: str, tgt: str, s: float = 0.95, c: float = 0.95) -> None:
    EDGES.append((src, rel, tgt, s, c))


def _pair(a: str, b: str, left: str, right: str,
          s: float = 0.95, c: float = 0.95) -> None:
    """Store a declared inverse pair in both directions.

    Both members are stored rather than trusting only the runtime mirror, so
    direction is readable straight out of the database and the store does not
    depend on the mirror pass to be correct. Same pattern as Stages D and E.
    """
    _fwd(a, left, b, s, c)
    _fwd(b, right, a, s, c)


def _both_symmetric(a: str, b: str, rel: str,
                    s: float = 0.95, c: float = 0.95) -> None:
    _fwd(a, rel, b, s, c)
    _fwd(b, rel, a, s, c)


def _pct(num: int, den: int) -> float:
    """Strength/confidence inside the contract's 0.8-1.0 band, deterministically
    spread over 0.86-0.98 so a reader can see the band was respected rather than
    assumed. The division by 11 is load-bearing: without it the spread ran to
    2.18, and validate() caught it."""
    return 0.86 + 0.12 * (((num * 7 + den * 3) % 12) / 11.0)


def _add_edge(src: str, rel: str, tgt: str, i: int) -> None:
    _fwd(src, rel, tgt, _pct(i, 1), _pct(i, 2))


# ============================ is_a : taxonomy, forward only ==================
_i = 0


def _isa(leaf: str, parent: str) -> None:
    global _i
    _i += 1
    _add_edge(leaf, "is_a", parent, _i)


# Internals carry the taxonomy so `part_of -> is_a` and `is_a -> is_a` chains
# exist; leaves are specific entities alongside them, as section 5 node_rules
# asks for.
for _leaf, _parent in (
    # plant branch
    ("plant food", "food"), ("vegetable", "plant food"), ("fruit", "plant food"),
    ("grain", "plant food"), ("herb", "plant food"), ("legume", "plant food"),
    ("vegetable oil", "food"), ("staple food", "food"), ("sweetener", "food"),
    ("root vegetable", "vegetable"), ("leafy vegetable", "vegetable"),
    ("citrus fruit", "fruit"), ("berry", "fruit"),
    ("carrot", "root vegetable"), ("cabbage", "leafy vegetable"),
    ("spinach", "leafy vegetable"), ("lettuce", "leafy vegetable"),
    ("orange", "citrus fruit"), ("lemon", "citrus fruit"),
    ("wheat", "grain"), ("rice", "grain"),
    ("sugar", "sweetener"), ("honey", "sweetener"),
    ("olive oil", "vegetable oil"), ("bread", "staple food"),
    # animal branch
    ("animal food", "food"), ("meat", "animal food"), ("dairy", "animal food"),
    ("fish", "animal food"), ("poultry", "meat"), ("beef", "meat"),
    ("salmon", "fish"), ("milk", "dairy"), ("butter", "dairy"),
    ("cheese", "dairy"), ("yogurt", "dairy"), ("egg", "animal food"),
    # kitchen branch
    ("equipment", "kitchen"), ("appliance", "equipment"), ("utensil", "equipment"),
    ("knife", "utensil"), ("whisk", "utensil"), ("oven", "appliance"),
):
    _isa(_leaf, _parent)

# ====================== has_property : forward only ===========================
def _prop(anchor: str, value: str) -> None:
    global _i
    _i += 1
    _add_edge(anchor, "has_property", value, _i)


for _anchor, _value in (
    ("carrot", "orange"), ("spinach", "green"),
    ("bread", "crusty"), ("flour", "powdery"), ("milk", "liquid"),
    ("cheese", "aged"), ("salmon", "oily"), ("egg", "perishable"),
    ("cutting board", "wooden"),
):
    _prop(_anchor, _value)

# ====================== causes / caused_by : stored both ways =================
# Every endpoint below carries exactly ONE causal edge, so each is a legal
# bidirectional direction pair (lesson 1).
def _cause(a: str, b: str, i: int) -> None:
    _pair(a, b, "causes", "caused_by", _pct(i, 1), _pct(i, 2))


for _n, (_a, _b) in enumerate((
    ("heat", "melting"), ("yeast", "rising"), ("rust", "stain"),
    ("mold", "spoilage"), ("bacteria", "illness"), ("smoke", "coughing"),
    ("salt", "hardening"), ("cold", "numbness"),
), 1):
    _cause(_a, _b, _n)

# ====================== precedes / follows : stored both ways =================
# Same rule: exactly one temporal edge per endpoint, so each is gradeable in
# both directions. The first eight are one real preparation order, which is also
# why eight direction pairs cost only nine nodes.
def _precedes(a: str, b: str, i: int) -> None:
    _pair(a, b, "precedes", "follows", _pct(i, 1), _pct(i, 2))


for _n, (_a, _b) in enumerate((
    ("harvest", "milling"), ("milling", "sifting"), ("sifting", "mixing"),
    ("mixing", "kneading"), ("kneading", "proofing"), ("proofing", "baking"),
    ("baking", "serving"), ("serving", "eating"),
    ("boiling", "simmering"),
), 1):
    _precedes(_a, _b, _n)

# ====================== part_of : stored forward only =========================
# section 9: the reverse `has_part` is synthesised by the runtime mirror, so
# `has_part` must never appear in the stored relation set.
def _part(child: str, whole: str) -> None:
    global _i
    _i += 1
    _add_edge(child, "part_of", whole, _i)


for _child, _whole in (
    ("crust", "bread"), ("crumb", "bread"), ("bran", "flour"),
    ("blade", "knife"), ("egg white", "egg"), ("egg yolk", "egg"),
    ("core", "apple"), ("peel", "orange"), ("seed", "pumpkin"),
):
    _part(_child, _whole)

# ====================== example_of : forward only ============================
def _example(item: str, category: str) -> None:
    global _i
    _i += 1
    _add_edge(item, "example_of", category, _i)


for _item, _cat in (
    ("trout", "fish"), ("cheddar", "cheese"), ("basmati", "rice"),
    ("romaine", "lettuce"), ("sirloin", "beef"), ("sourdough", "bread"),
):
    _example(_item, _cat)

# ====================== associated_with : forward only ======================
def _assoc(a: str, b: str) -> None:
    global _i
    _i += 1
    _add_edge(a, "associated_with", b, _i)


for _a, _b in (
    ("knife", "chopping"), ("whisk", "beating"), ("apron", "cooking"),
    ("sieve", "sifting"), ("timer", "baking"), ("peeler", "peeling"),
):
    _assoc(_a, _b)

# ====================== supports / contradicts : forward only ================
# Evidence nodes point at claims. Direction is meaningful and the contract lists
# supports among the relations for which a reversed label would assert a
# directed triple the graph never stored.
def _evidence(name: str, supports: str, contradicts: str) -> None:
    global _i
    _i += 1
    _add_edge(name, "supports", supports, _i)
    _i += 1
    _add_edge(name, "contradicts", contradicts, _i)


for _evidence_name, _supported, _refuted in (
    ("diet survey", "fibre benefit claim", "low fat claim"),
    ("cell study", "antioxidant claim", "sugar illness claim"),
    ("cooking study", "browning benefit claim", "raw food claim"),
):
    _evidence(_evidence_name, _supported, _refuted)

# ====================== symmetric relations, stored both ways ================
# Section 9 forbids the mirror pass from producing these five, so both
# orientations are written by hand: a one-way store would make half the
# vocabulary unanswerable.
for _n, (_left, _right) in enumerate((
    # synonym -- regional and equivalent-term pairs
    ("aubergine", "eggplant"), ("zucchini", "courgette"),
    ("arugula", "rocket"), ("chickpea", "garbanzo"),
    ("passata", "tomato puree"),
), 1):
    _both_symmetric(_left, _right, "synonym", _pct(_n, 1), _pct(_n, 2))

for _n, (_left, _right) in enumerate((
    # antonym -- each endpoint has exactly one antonym edge
    ("ripe", "unripe"), ("wet", "dry"), ("raw", "cooked"),
    ("soft", "hard"),
), 1):
    _both_symmetric(_left, _right, "antonym", _pct(_n, 1), _pct(_n, 2))

for _n, (_left, _right) in enumerate((
    # spatial_near -- four of these six pairs reuse nodes that already exist
    ("knife", "cutting board"), ("salt", "stove"), ("bread", "butter"),
    ("lemon", "lime"), ("whisk", "mixing bowl"), ("oven", "sink"),
), 1):
    _both_symmetric(_left, _right, "spatial_near", _pct(_n, 1), _pct(_n, 2))

for _n, (_left, _right) in enumerate((
    # temporal_coincident -- distinct from precedes/follows: coincidence, not order
    ("kneading", "mixing"), ("proofing", "fermentation"),
    ("serving", "plating"),
), 1):
    _both_symmetric(_left, _right, "temporal_coincident",
                    _pct(_n, 1), _pct(_n, 2))

# linguistic_maps -- English <-> Spanish. The English word is the anchor and the
# Spanish word the answer, so only answers carry non-ASCII characters.
for _n, (_left, _right) in enumerate((
    ("tomato", "tomate"), ("milk", "leche"), ("bread", "pan"),
    ("cheese", "queso"), ("sugar", "azúcar"), ("egg", "huevo"),
), 1):
    _both_symmetric(_left, _right, "linguistic_maps",
                    _pct(_n, 1), _pct(_n, 2))

# The node list is DERIVED from the edges, so an edge endpoint typo becomes a
# node rather than a validation error. That trade is deliberate and is why the
# label checks below are strict: every label must be lowercase, free of cues and
# anchor filler, and either ASCII or one of the declared Spanish answers. A typo
# that survives all of those is caught by the freeze digest rather than here.
SPANISH_ANSWERS = frozenset({
    "tomate", "leche", "pan", "queso", "azúcar", "huevo", "agua",
})

CONCEPTS.extend(sorted({x for e in EDGES for x in (e[0], e[2])}))


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
    asked label alone under-counts reachability for every mirrorable relation.
    """
    return {rel, INVERSE[rel]} if rel in INVERSE else {rel}


def reach_from(anchor: str, rel: str, hops: int | None = 1) -> set[str]:
    """What the walker can reach from `anchor` under `rel`.

    `hops=1` is the right question for a graded one-hop question: uniqueness must
    hold for the single hop actually being graded. `is_a` chains mean "What is a
    carrot?" is unambiguous even though a transitive search also reaches food.
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

    Deterministic order: sorted, so the frozen set never depends on set or dict
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
    less of the graph than the edge count suggests, and the reader is entitled
    to know that.
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

    Deduplicated: because causes/caused_by and precedes/follows are stored in
    both directions, iterating the raw triple set yields BOTH (a, causes, b) and
    (b, caused_by, a) and both satisfy every condition. Returning each pair
    twice would produce two identical questions per direction, which is exactly
    the kind of duplicate that inflates a score.
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
    """Every gradeable single read of a mirrorable pair, deduplicated."""
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

    Two independent mechanisms make a repeated relation inexpressible:

      1. `configs/config_g2p.yaml` sets collapse_consecutive_repeats: true, so an
         ADJACENT repeat like [is_a, is_a] is contracted to [is_a] before the
         walk starts.
      2. `QueryRelationExtractor._literal_cue_relations` keeps only the longest
         matching descriptor PER RELATION, so a relation cued twice contributes
         one entry.

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

      * A step may use a MIRROR-derived edge. `egg yolk part_of egg` is stored
        forward only, but the walker can legitimately go
        egg --part_of--> egg yolk through the section 9 mirror, so that step is
        real and must be available to chain construction. Both orientations are
        labelled with the FORWARD relation (`part_of`), which is the one that
        has a cue phrase; the mirrored label `has_part` has no descriptor in
        configs/config_g2p.yaml and so cannot be asked for.

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
    reach_from(a, rel, 1) == {b}, which rules out the repeated-mirrorable-
    relation chains in lesson 2. Paths may not revisit a node, since a walk
    that does is a cycle rather than a short chain. Chains the engine could not
    express are dropped (see `_would_collapse`). Deduplicated, because symmetric
    relations are stored in both directions and would otherwise repeat.
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
    must not name `a`: the mirror pass is forbidden from copying these
    relations, so answering would be inventing knowledge. This is the strongest
    available evidence against illegal same-label mirroring, and it is derived
    from the graph rather than chosen.

    Grouped by (anchor, relation) because several edges can share a target.
    Emitting one question per edge would duplicate it and, worse, each copy
    would forbid only one of the nodes it must not name. Ambiguity does not
    spoil a must-REFUSE row: the expectation is that no far end is named at all.
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

    The question set is derived from this graph, so a mistake here would be
    baked into the "held-out" evaluation set and would silently decide the PoC
    verdict.
    """
    problems: list[tuple[str, str]] = []
    declared = set(CONCEPTS)
    trip = _triples()
    used_labels = {x for e in EDGES for x in (e[0], e[2])}

    # A label listed twice is silently deduplicated by `dict.fromkeys` below, so
    # the graph would still build and still look right -- but the declared count
    # would no longer match the node count.
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

    # --- lessons 3 and 4: labels must be anchorable and cue-neutral ----------
    for label in sorted(declared):
        low = label.lower()
        for trap in LABEL_CUE_TRAPS:
            if trap in low:
                problems.append(("label-contains-a-relation-cue",
                                 f"{label!r} contains {trap!r}"))
        for tok in low.split():
            if tok.strip(".,!?") in LABEL_FILLER_TRAPS:
                problems.append(("label-token-is-anchor-filler",
                                 f"{label!r} token {tok!r}"))
        if low != label:
            problems.append(("label-not-lowercase", label))
        if not label.isascii() and label not in SPANISH_ANSWERS:
            problems.append(("label-not-ascii-and-not-a-declared-spanish-answer",
                             repr(label)))
        if any(ch.isdigit() for ch in label):
            problems.append(("label-contains-a-digit", repr(label)))

    # --- honesty sets ------------------------------------------------------
    # The out-of-graph subjects carry the refusal burden, so both halves must
    # hold: the subject must really be absent from the graph, and the question
    # must really name it.
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
        problems.append(("disclosed-guess-anchor-has-no-unique-has_property",
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

    # --- the contract sizes this graph at 100-150 nodes ---------------------
    n_nodes = len(declared)
    if not 100 <= n_nodes <= 150:
        problems.append(("node-count-outside-contract-size",
                         f"{n_nodes} nodes, contract section 5 says ~100-150"))
    out_of_band = [f"{s} {r} {t} {s_}/{c_}" for s, r, t, s_, c_ in EDGES
                   if not (0.8 <= s_ <= 1.0 and 0.8 <= c_ <= 1.0)]
    if out_of_band:
        problems.append(("edge-strength-or-confidence-outside-0.8-1.0",
                         "; ".join(out_of_band[:5])))
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
    .db file straight after save_state() produced a digest of a file still
    missing whatever the WAL had not yet flushed -- the same graph hashed
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
        print(f"\n({len(problems)} problems)")
        return 1

    labels = list(dict.fromkeys(CONCEPTS))
    used = sorted({r for _, r, _, _, _ in EDGES})
    by_hops = chains_by_hops()
    print("Building HELD-OUT Evaluation Graph: "
          f"{len(labels)} nodes, {len(EDGES)} edges")
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
    store.set_metadata("dataset_name", "holdout_eval_stage_f")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.set_metadata("frozen", "true")
    store.set_metadata("contract", "GLM-X v3.3.2 sections 8/9/15/16")
    store.set_metadata("version", "1.0.0")
    store.save_state(str(DB_PATH))

    # --- freeze assertions -------------------------------------------------
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