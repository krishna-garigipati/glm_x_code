"""UNSEEN-DATA END-TO-END TEST: build the frozen evaluation graph.

This is a single independent stress test of GLM-X v3.3.2 on data the system has
never been tuned on. It is not a continuation of any earlier process stage.

DOMAIN: WEAVING AND TEXTILE CRAFT.

Chosen by auditing every domain present in this repository first, because a
domain that already exists cannot test anything. In use elsewhere: toy,
nature/weather/climate, geography glossary, science evidence, food and
food-bio, astronomy, computing, history, human body, music
(tester-c/datasets/build_music_small.py), aviation
(test_results/final_validation/), the Stage A-E graphs, and the lead
zoo/mountain material. Weaving appears in none of them.

WHAT IS REUSED AND WHY
    One thing is imported: the graph algebra in
    `test_results/stage_f/build_holdout_graph.py` (reach_from, unique_one_hop,
    direction_pairs, multi_hop_chains, mirror_controls). That is data-independent
    reasoning about which stored edges are uniquely readable, and it encodes real
    engine semantics -- notably which relation the mirror synthesises for
    `part_of`. Reimplementing it would risk a silent divergence, and then a
    failure would mean my reimplementation was wrong rather than the pipeline.
    It reads whatever CONCEPTS/EDGES it is given, so no graph data comes from it.

    Everything else here is new: the domain, every node, every edge, the
    validation, and (in generate_questions.py) every question template.

WHAT IS NEW IN THE VALIDATION, AND WHY
    Two checks exist here that did not exist before, because the aviation set
    showed they were needed:

      * `label-token-is-another-label`. The identity gate and the anchorer both
        match on tokens, so a multi-word label containing a token that is itself
        a label is unanswerable-by-construction: "instrument landing system"
        lost its anchor to the label "landing" when the question inflected the
        subject. Rejected at build time now.
      * `out-of-graph-subject-grounds-on-a-label`. "hang glider" contains the
        token "glider", so the identity gate grounded an out-of-graph subject on
        an in-graph node and answered confidently. An honesty question whose
        subject token matches a label is not an honesty question, so it is
        rejected at build time now.

    `label-prefix-collision` and `too-few-out-of-graph-subjects` are carried over
    from the aviation set for the same reasons they were added there.

Usage:
    python test_results/unseen_e2e/build_graph.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

# --- the one import: data-independent graph algebra, not data ---------------
_SRC = ROOT / "test_results" / "stage_f" / "build_holdout_graph.py"
_spec = importlib.util.spec_from_file_location("_algebra", _SRC)
B = importlib.util.module_from_spec(_spec)
sys.modules["_algebra"] = B
_spec.loader.exec_module(B)

DB_PATH = HERE / "unseen_eval.db"

# ---------------------------------------------------------------------------
# Out-of-graph subjects: genuinely absent, each carrying a LITERAL cue phrase
# from configs/config_g2p.yaml so a refusal is attributable to entity identity
# rather than to cue absence. generate_questions.py re-verifies the cue count
# against the live planner before freezing.
# ---------------------------------------------------------------------------
OUT_OF_GRAPH_SUBJECTS: list[tuple[str, str]] = [
    ("felt", "What type of thing is the felt?"),
    ("burlap", "What type of thing is the burlap?"),
    ("macrame", "What type of thing is the macrame?"),
    ("chenille", "What type of thing is the chenille?"),
    ("canvas", "What type of thing is the canvas?"),
    ("muslin", "What type of thing is the muslin?"),
    ("corduroy", "What type of thing is the corduroy?"),
    ("ticking", "What type of thing is the ticking?"),
    ("batik", "What causes the batik?"),
    ("retting", "What was the retting caused by?"),
    ("stenter", "What leads to the stenter?"),
    ("dobby", "What results in the dobby?"),
    ("sizing", "What is the sizing part of?"),
    ("lease rod", "What is the lease rod part of?"),
    ("tuck stitch", "What is the tuck stitch an example of?"),
    ("creel", "What is the creel associated with?"),
    ("roller", "What is known for the roller?"),
    ("drum", "What is near the drum?"),
    ("counterbalance", "What is the counterbalance part of?"),
    ("footman", "What is the footman part of?"),
    ("quill", "What is the quill part of?"),
    ("lapet", "What is the lapet part of?"),
    ("beam press", "What is the beam press part of?"),
]

# Cue-free questions. Section 8 fallback triggers on cue ABSENCE, so every one
# of these must set heuristic_fallback_used; preflight re-checks that against the
# live descriptor bank.
NONSENSE_CUE_FREE: list[str] = [
    "How many colours does the number three have?",
    "What is the temperature of a shadow?",
    "How loud is the shape of a circle?",
    "What is the weight of a musical note?",
    "How bitter is the smell of Sunday?",
    "What is the length of the letter z?",
]

# A real in-graph anchor with a unique has_property read, asked with NO relation
# cue. The system must fall back AND disclose that it is guessing (section 8).
# Handlooms really are wooden frames, so the property is "wooden" honestly --
# which also means this satisfies the inherited disclosed-guess fixture check
# natively instead of needing it filtered out.
DISCLOSED_GUESS: tuple[str, str] = ("loom", "How wooden is the loom?")

# Non-ASCII answers the inherited label check must tolerate (French terms).
SPANISH_ANSWERS = frozenset({"fil", "metier", "tissage", "aiguille", "bobine"})
FRENCH_TERMS = SPANISH_ANSWERS

CONCEPTS: list[str] = []
EDGES: list[tuple[str, str, str, float, float]] = []

B.CONCEPTS = CONCEPTS
B.EDGES = EDGES
B.OUT_OF_GRAPH_SUBJECTS = OUT_OF_GRAPH_SUBJECTS
B.DISCLOSED_GUESS = DISCLOSED_GUESS
B.SPANISH_ANSWERS = SPANISH_ANSWERS
B.DB_PATH = DB_PATH

# ============================ is_a : taxonomy, forward only ==================
for _leaf, _parent in (
    ("equipment", "object"),
    ("textile machine", "equipment"),
    ("machine component", "equipment"),
    ("loom", "textile machine"),
    ("handloom", "loom"),
    ("drawloom", "loom"),
    ("great wheel", "textile machine"),
    ("sewing machine", "textile machine"),
    ("warp beam", "machine component"),
    ("cloth beam", "machine component"),
    ("reed", "machine component"),
    ("heddle", "machine component"),
    ("shuttle", "machine component"),
    ("bobbin", "machine component"),
    ("treadle", "machine component"),
    ("frame", "machine component"),
    ("pit", "machine component"),
    ("jacquard head", "machine component"),
    ("spindle", "machine component"),
    ("needle", "machine component"),
    # materials
    ("substance", "object"),
    ("textile material", "substance"),
    ("dressing agent", "textile material"),
    ("textile fibre", "textile material"),
    ("yarn", "textile material"),
    ("fabric", "textile material"),
    ("dye", "dressing agent"),
    ("mordant", "dressing agent"),
    ("indigo", "dye"),
    ("madder", "dye"),
    ("warp thread", "yarn"),
    ("weft thread", "yarn"),
    ("linen", "textile fibre"),
    ("wool", "textile fibre"),
    ("cotton", "textile fibre"),
    ("hemp", "textile fibre"),
    ("silk", "textile fibre"),
    # structures
    ("textile pattern", "object"),
    ("weave structure", "textile pattern"),
    ("plain weave", "weave structure"),
    ("twill", "weave structure"),
    ("satin", "weave structure"),
    ("tapestry", "weave structure"),
    ("cloth feature", "fabric"),
    ("selvedge", "cloth feature"),
    ("fringe", "cloth feature"),
    ("surface pile", "cloth feature"),
    # processes
    ("manufacturing process", "activity"),
    ("textile process", "manufacturing process"),
    ("carding", "textile process"),
    ("spinning", "textile process"),
    ("roving", "textile process"),
    ("winding", "textile process"),
    ("warping", "textile process"),
    ("denting", "textile process"),
    ("weaving", "textile process"),
    ("beating", "textile process"),
    ("shedding", "textile process"),
    ("scouring", "textile process"),
    ("bleaching", "textile process"),
    ("dyeing", "textile process"),
    ("drying", "textile process"),
    ("finishing", "textile process"),
    ("grading", "textile process"),
    ("packing", "textile process"),
    # people
    ("handicraft", "activity"),
    ("textile craft", "handicraft"),
    ("textile worker", "person"),
    ("weaver", "textile worker"),
    ("spinner", "textile worker"),
    ("dyer", "textile worker"),
):
    B._isa(_leaf, _parent)

# ====================== has_property : exactly one per anchor ================
for _anchor, _value in (
    ("loom", "wooden"),
    ("shuttle", "pointed tip"),
    ("bobbin", "conical core"),
    ("heddle", "split eye"),
    ("reed", "comb teeth"),
    ("warp beam", "uniform wrap"),
    ("linen", "long fibre"),
    ("wool", "natural crimp"),
    ("cotton", "short staple"),
    ("silk", "long filament"),
    ("hemp", "bast fibre"),
    ("plain weave", "alternating tabby"),
    ("twill", "diagonal rib"),
    ("satin", "long float"),
    ("tapestry", "slit weave"),
    ("weaver", "shed control"),
    ("dyer", "immersion control"),
    ("indigo", "deep blue colour"),
    ("madder", "red colour"),
    ("mordant", "metal salt"),
    ("selvedge", "woven edge"),
):
    B._prop(_anchor, _value)

# ====================== causes / caused_by : stored both ways =================
# Disjoint pairs: a shared endpoint is a hub, and from a hub both readings are
# legal, so neither end is auto-gradable against one gold node.
for _n, (_a, _b) in enumerate((
    ("friction", "snapped warp thread"),
    ("dyeing", "colour bleeding"),
    ("mordant", "fast colour"),
    ("broken shaft", "missed thread"),
    ("uneven warp tension", "skewed cloth"),
    ("humidity", "slub formation"),
    ("overdyeing", "residue deposit"),
    ("tension overload", "broken end"),
), 1):
    B._cause(_a, _b, _n)

# ====================== precedes / follows : stored both ways ================
# Deliberately NOT the carding->spinning->warping->weaving->finishing chain: in
# a chain every interior node carries two temporal edges, making it a hub and
# throwing away both of its direction reads. These pairs are disjoint instead.
for _n, (_a, _b) in enumerate((
    ("warping", "denting"),
    ("carding", "roving"),
    ("scouring", "bleaching"),
    ("drying", "finishing"),
    ("grading", "packing"),
    ("spinning", "winding"),
), 1):
    B._precedes(_a, _b, _n)

# ====================== part_of : stored forward only =========================
# Section 9: the reverse `has_part` is synthesised by the runtime mirror, so
# `has_part` must never appear in the stored relation set.
for _child, _whole in (
    ("warp beam", "handloom"),
    ("cloth beam", "handloom"),
    ("reed", "handloom"),
    ("heddle", "handloom"),
    ("shuttle", "handloom"),
    ("bobbin", "shuttle"),
    ("frame", "handloom"),
    ("treadle", "handloom"),
    ("pit", "handloom"),
    ("jacquard head", "drawloom"),
    ("spindle", "great wheel"),
    ("needle", "sewing machine"),
):
    B._part(_child, _whole)

# ====================== example_of : forward only ============================
for _item, _cat in (
    ("kente cloth", "narrow woven band"),
    ("gabardine", "tight weave cloth"),
    ("brocade", "supplementary weave cloth"),
    ("damask", "figured weave cloth"),
    ("hand towel", "plain weave"),
):
    B._example(_item, _cat)

# ====================== associated_with : forward only =======================
for _a, _b in (
    ("reed", "thread density"),
    ("shuttle", "pick"),
    ("weaver", "shedding"),
    ("dyer", "vat"),
    ("scouring", "impurity removal"),
    ("twill", "warp float"),
):
    B._assoc(_a, _b)

# ====================== supports / contradicts : forward only ================
def _evidence(name: str, supports: str, contradicts: str) -> None:
    B._add_edge(name, "supports", supports, B._i + 1)
    B._i += 1
    B._add_edge(name, "contradicts", contradicts, B._i + 1)
    B._i += 1


for _name, _supported, _refuted in (
    ("inspection note", "stable thread count claim", "uneven thread count claim"),
    ("colour log", "even uptake claim", "patchy uptake claim"),
    ("alignment check", "aligned warp claim", "skewed warp claim"),
):
    _evidence(_name, _supported, _refuted)

# ====================== symmetric relations, stored both ways ================
# Section 9 forbids the mirror from producing these, so both orientations are
# written by hand.
for _n, (_l, _r) in enumerate((
    ("selvedge", "selvage"),
    ("twill", "diagonal weave"),
    ("mordant", "fixative"),
    ("bobbin", "spool"),
    ("fringe", "tassel"),
), 1):
    B._both_symmetric(_l, _r, "synonym", B._pct(_n, 1), B._pct(_n, 2))

for _n, (_l, _r) in enumerate((
    ("warp faced", "weft faced"),
    ("dense thread count", "sparse thread count"),
    ("positive tension", "negative tension"),
), 1):
    B._both_symmetric(_l, _r, "antonym", B._pct(_n, 1), B._pct(_n, 2))

for _n, (_l, _r) in enumerate((
    ("warp beam", "cloth beam"),
    ("heddle", "reed"),
    ("shuttle", "bobbin"),
    ("frame", "pit"),
), 1):
    B._both_symmetric(_l, _r, "spatial_near", B._pct(_n, 1), B._pct(_n, 2))

for _n, (_l, _r) in enumerate((
    ("shedding", "weft insertion"),
    ("beating", "weft insertion"),
    ("warping", "denting"),
), 1):
    B._both_symmetric(_l, _r, "temporal_coincident", B._pct(_n, 1), B._pct(_n, 2))

# linguistic_maps -- English <-> French. The question template says "translates
# to" and never names a language, so no question can assert a language the stored
# mapping does not use. (The aviation set hardcoded "in Spanish" for an Italian
# mapping and 40 of its questions were therefore false; this bank cannot repeat
# that, and the generator refuses a language-naming template whose source side
# is not English.)
for _n, (_l, _r) in enumerate((
    ("yarn", "fil"),
    ("loom", "metier"),
    ("weaving", "tissage"),
    ("needle", "aiguille"),
    ("bobbin", "bobine"),
), 1):
    B._both_symmetric(_l, _r, "linguistic_maps", B._pct(_n, 1), B._pct(_n, 2))

CONCEPTS.extend(sorted({x for e in EDGES for x in (e[0], e[2])}))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
MIN_OUT_OF_GRAPH_SUBJECTS = 20
MIN_PREFIX_LEN = 4


def _tokens(label: str) -> list[str]:
    return [t for t in label.lower().replace("-", " ").split() if t]


def added_problems() -> list[tuple[str, str]]:
    """Checks the inherited validate() cannot know to make."""
    problems: list[tuple[str, str]] = []
    labels = sorted(set(CONCEPTS))

    # The identity gate carries the honesty category. Too few subjects and a
    # single refusal flip decides the verdict.
    n = len(OUT_OF_GRAPH_SUBJECTS)
    if n < MIN_OUT_OF_GRAPH_SUBJECTS:
        problems.append(("too-few-out-of-graph-subjects",
                         f"{n} < {MIN_OUT_OF_GRAPH_SUBJECTS}"))

    # NEW. A label whose token is itself another label is unanswerable by
    # construction: the anchorer matches on tokens, so the shorter label wins.
    singles = {l for l in labels if " " not in l}
    for l in labels:
        for t in _tokens(l):
            if t != l.lower() and t in singles:
                problems.append(("label-token-is-another-label", f"{l!r} -> {t!r}"))

    # NEW. An out-of-graph subject that shares a token with a label will be
    # grounded on that label by the identity gate, so the question stops being
    # an honesty question and becomes a confidently-wrong-answer test.
    for subject, _q in OUT_OF_GRAPH_SUBJECTS:
        for t in _tokens(subject):
            if t in singles:
                problems.append(("out-of-graph-subject-grounds-on-a-label",
                                 f"{subject!r} -> {t!r}"))
            for l in labels:
                if len(t) >= MIN_PREFIX_LEN and l.lower().startswith(t):
                    problems.append(("out-of-graph-subject-prefixes-a-label",
                                     f"{subject!r} -> {l!r}"))

    # Carried over: a 4+ character label prefix lets "wings" ground "wing".
    for short in labels:
        if len(short) < MIN_PREFIX_LEN:
            continue
        clashes = sorted(o for o in labels if o != short and o.startswith(short))
        if clashes:
            problems.append(("label-prefix-collision", f"{short!r} -> {clashes}"))

    # The disclosed-guess anchor must exist or the fallback-disclosure path is
    # never exercised. The inherited validate() already enforces a unique
    # has_property read from it.
    anchor, _q = DISCLOSED_GUESS
    if anchor not in set(CONCEPTS):
        problems.append(("disclosed-guess-anchor-not-a-concept", anchor))

    return problems


def validate() -> list[tuple[str, str]]:
    return B.validate() + added_problems()


# ---------------------------------------------------------------------------
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
    by_hops = B.chains_by_hops()
    print("Building UNSEEN-DATA evaluation graph (weaving domain): "
          f"{len(labels)} nodes, {len(EDGES)} edges")
    print(f"  relations used ({len(used)}/16): {used}")
    print(f"  unique one-hop edges   : {len(B.unique_one_hop())}")
    print(f"  ambiguous one-hop edges: {len(B.ambiguous_one_hop())}")
    print(f"  bidirectional pairs    : {len(B.direction_pairs())}")
    print(f"  gradeable pair reads   : {len(B.one_way_direction_pairs())}")
    print(f"  2-hop chains           : {len(by_hops.get(2, []))}")
    print(f"  3-hop chains           : {len(by_hops.get(3, []))}")
    print(f"  mirror controls        : {len(B.mirror_controls())}")
    print(f"  out-of-graph subjects  : {len(OUT_OF_GRAPH_SUBJECTS)}")
    print(f"  nonsense cue-free      : {len(NONSENSE_CUE_FREE)}")

    import numpy as np
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    enc = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {l: np.asarray(enc[i], dtype=np.float32) for i, l in enumerate(labels)}

    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

    for stale in (DB_PATH, DB_PATH.with_suffix(".db-wal"), DB_PATH.with_suffix(".db-shm")):
        if stale.exists():
            stale.unlink()

    concepts = {l: i + 1 for i, l in enumerate(labels)}
    store = SQLiteGraphStore(db_path=str(DB_PATH))
    store.add_dataset(
        concepts=concepts,
        edges=[{"source": concepts[a], "target": concepts[b], "relation": r,
                "strength": s, "confidence": c} for a, r, b, s, c in EDGES],
        id_to_label={i: l for l, i in concepts.items()},
        embeddings=embeddings,
        protected_labels=list(labels),
    )
    store.set_metadata("dataset_name", "unseen_e2e_weaving")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.set_metadata("frozen", "true")
    store.set_metadata("contract", "GLM-X v3.3.2 sections 8/9/15/16")
    store.set_metadata("version", "1.0.0")
    store.save_state(str(DB_PATH))

    digest = B.finalize_and_hash(store)
    print(f"\nFROZEN: {DB_PATH}")
    print(f"sha256: {digest}")
    (HERE / "frozen_graph_digest.txt").write_text(
        f"{digest}  unseen_eval.db\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())