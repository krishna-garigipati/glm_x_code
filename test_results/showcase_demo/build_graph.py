"""SHOWCASE DEMO: build the frozen evaluation graph for the final PoC demo.

DOMAIN: POTTERY AND CERAMICS.

Chosen by auditing every domain already present in this repository, because a
domain the system has already been scored on cannot demonstrate anything new.
In use elsewhere: toy, nature/weather/climate, geography glossary, science
evidence, food and food-bio, astronomy, computing, history, human body, music,
aviation, weaving, and the Stage A-F graphs. Pottery appears in none of them.

This is the demo graph referenced by Part 4 of POC_SHOWCASE_REPORT.md. It is a
third independent domain, built and frozen by the same rules as the aviation
(official PoC) and weaving (unseen stress test) sets.

WHAT IS REUSED, AND WHY
    One import only: the graph algebra in `test_results/stage_f/
    build_holdout_graph.py` (reach_from, unique_one_hop, direction_pairs,
    multi_hop_chains, mirror_controls, validate, finalize_and_hash). That is
    data-independent reasoning about which stored edges are uniquely readable,
    and it encodes real engine semantics -- notably which relation the section 9
    mirror synthesises for `part_of`. Reimplementing it would risk a silent
    divergence, and then a failure would mean my reimplementation was wrong
    rather than the pipeline. It reads whatever CONCEPTS/EDGES it is given, so
    no graph data comes from it.

    Everything else is new: the domain, every node, every edge, the validation
    additions, and every question template.

WHAT THE VALIDATION ENFORCES, AND WHY IT MATTERS TO THE CLAIM
    Inherited: no duplicate/unused/dangling nodes, all 16 relations exercised,
    `has_part` never stored, symmetric relations stored both ways, labels must
    be lowercase ASCII with no digits and contain no descriptor phrase
    (`LABEL_CUE_TRAPS`) or anchoring filler token (`LABEL_FILLER_TRAPS`), edges
    within strength/confidence 0.8-1.0, node count 100-150, >=5 direction pairs,
    >=4 two-hop, >=2 three-hop, >=30 unique one-hop, >=8 mirror controls, every
    out-of-graph subject genuinely absent AND named in its own question, and the
    disclosed-guess anchor carrying a unique `wooden` property.

    Added here (all four carried from the unseen weaving set, because the
    aviation set proved they were needed):

      * `label-token-is-another-label`. The anchorer matches on tokens, so a
        multi-word label containing a token that is itself a label is
        unanswerable by construction -- "instrument landing system" lost its
        anchor to the label "landing".
      * `out-of-graph-subject-grounds-on-a-label` / `...-prefixes-a-label`. An
        honesty subject sharing a token or a 4-char prefix with a label gets
        grounded on that label by the identity gate, so it stops being an
        honesty question and becomes a confidently-wrong-answer test.
      * `label-prefix-collision`. A 4+ character label that prefixes another
        label lets a question ground on the wrong one.
      * `too-few-out-of-graph-subjects`. One refusal flip must not decide a
        verdict, so >=20 subjects.

Usage:
    python test_results/showcase_demo/build_graph.py
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
_spec = importlib.util.spec_from_file_location("_showcase_algebra", _SRC)
B = importlib.util.module_from_spec(_spec)
sys.modules["_showcase_algebra"] = B
_spec.loader.exec_module(B)

DB_PATH = HERE / "showcase_eval.db"

# ---------------------------------------------------------------------------
# Out-of-graph subjects: pottery/ceramics terms genuinely absent from this
# graph. Each carries a LITERAL cue phrase from configs/config_g2p.yaml so a
# refusal is attributable to entity identity rather than to cue absence.
# generate_questions.py re-verifies the cue count against the live planner.
# ---------------------------------------------------------------------------
OUT_OF_GRAPH_SUBJECTS: list[tuple[str, str]] = [
    ("raku", "What type of thing is the raku?"),
    ("saggar", "What type of thing is the saggar?"),
    ("anagama", "What type of thing is the anagama?"),
    ("jiggering", "What type of thing is the jiggering?"),
    ("jollying", "What type of thing is the jollying?"),
    ("slipware", "What type of thing is the slipware?"),
    ("saltglaze", "What type of thing is the saltglaze?"),
    ("mochaware", "What type of thing is the mochaware?"),
    ("jasperware", "What type of thing is the jasperware?"),
    ("basaltware", "What type of thing is the basaltware?"),
    ("kintsugi", "What type of thing is the kintsugi?"),
    ("mishima", "What type of thing is the mishima?"),
    ("sgraffito", "What type of thing is the sgraffito?"),
    ("neriage", "What type of thing is the neriage?"),
    ("nerikomi", "What type of thing is the nerikomi?"),
    ("faience", "What type of thing is the faience?"),
    ("azulejo", "What type of thing is the azulejo?"),
    ("delft", "What is known for the delft?"),
    ("tureen", "What is the tureen part of?"),
    ("cachepot", "What is the cachepot part of?"),
    ("bail", "What is the bail part of?"),
    ("noggin", "What is the noggin an example of?"),
    ("bead roll", "What is the bead roll an example of?"),
]

# Cue-free questions. Section 8 fallback triggers on cue ABSENCE, so every one
# of these must set heuristic_fallback_used; preflight re-checks that against
# the live descriptor bank rather than trusting this comment.
NONSENSE_CUE_FREE: list[str] = [
    "How many edges does the number four have?",
    "What is the temperature of a whisper?",
    "How loud is the shape of a triangle?",
    "What is the weight of a colour?",
    "How bitter is the smell of Wednesday?",
    "What is the length of a thought?",
]

# A real in-graph anchor with a unique has_property read, asked with NO
# relation cue. The system must fall back AND disclose that it is guessing
# (section 8). A potter's workbench genuinely is wooden, so the property is
# honest -- which is also what satisfies the inherited unique-`wooden` check.
DISCLOSED_GUESS: tuple[str, str] = ("workbench", "How wooden is the workbench?")

# Non-ASCII answers the inherited label check must tolerate. French terms used
# by linguistic_maps below. (ASCII spellings: the validator rejects digits and
# requires lowercase, and keeping them ASCII means they are still tokenisable.)
SPANISH_ANSWERS = frozenset({"argile", "four", "tour", "potier", "gres"})
FRENCH_TERMS = SPANISH_ANSWERS

CONCEPTS: list[str] = []
EDGES: list[tuple[str, str, str, float, float]] = []

B.CONCEPTS = CONCEPTS
B.EDGES = EDGES
B.OUT_OF_GRAPH_SUBJECTS = OUT_OF_GRAPH_SUBJECTS
# Rebinding NONSENSE_CUE_FREE is what makes the fallback category test THIS
# domain. Stage F's default list is the same six questions the aviation PoC and
# the weaving set already scored, so omitting it would quietly re-score old data
# while claiming a new domain. Found by provenance check after run 1; both runs
# are disclosed in POC_SHOWCASE_REPORT.md.
B.NONSENSE_CUE_FREE = NONSENSE_CUE_FREE
B.DISCLOSED_GUESS = DISCLOSED_GUESS
B.SPANISH_ANSWERS = SPANISH_ANSWERS
B.DB_PATH = DB_PATH

# ==================== is_a : taxonomy, forward only ==========================
# Naming discipline: no multi-word label may contain a token that is itself a
# single-token label, and no label may be a 4+ character prefix of another.
# Hence "ceramic craft" not "pottery craft" (would collide with "potter"),
# "enamel" the material alongside "glazing" the process (so that "glaze" is not
# a label at all), and no compound containing "kiln", "clay", "wheel" or
# "slip".
for _leaf, _parent in (
    # --- equipment and its parts ---
    ("equipment", "object"),
    ("kiln", "equipment"),
    ("wheel", "equipment"),
    ("ceramic tool", "equipment"),
    ("ribbon tool", "ceramic tool"),
    ("wire cutter", "ceramic tool"),
    ("sponge", "ceramic tool"),
    ("caliper", "ceramic tool"),
    ("component", "object"),
    ("workbench", "component"),
    ("shelf", "component"),
    ("burner", "component"),
    ("door", "component"),
    ("flue", "component"),
    ("bat", "component"),
    ("shaft", "component"),
    ("treadle", "component"),
    ("handle", "component"),
    ("spout", "component"),
    ("foot", "component"),
    ("lid", "component"),
    ("rim", "component"),
    # --- materials ---
    ("substance", "object"),
    ("ceramic material", "substance"),
    ("clay", "ceramic material"),
    ("kaolin", "clay"),
    ("grog", "ceramic material"),
    ("feldspar", "ceramic material"),
    ("earthenware", "ceramic material"),
    ("terracotta", "earthenware"),
    ("stoneware", "ceramic material"),
    ("porcelain", "ceramic material"),
    # --- surface treatment ---
    ("surface finish", "object"),
    ("enamel", "surface finish"),
    ("engobe", "surface finish"),
    ("lustre", "surface finish"),
    # --- ware ---
    ("ceramic ware", "object"),
    ("vessel", "ceramic ware"),
    ("mug", "vessel"),
    ("jug", "vessel"),
    ("bowl", "vessel"),
    ("plate", "vessel"),
    ("vase", "vessel"),
    ("jar", "vessel"),
    ("garden ware", "ceramic ware"),
    ("architectural ceramic", "ceramic ware"),
    ("decorative ware", "ceramic ware"),
    ("tableware", "ceramic ware"),
    ("storage ware", "ceramic ware"),
    # --- processes ---
    ("activity", "object"),
    ("handicraft", "activity"),
    ("ceramic craft", "handicraft"),
    ("manufacturing process", "activity"),
    ("ceramic process", "manufacturing process"),
    ("throwing", "ceramic process"),
    ("trimming", "ceramic process"),
    ("wedging", "ceramic process"),
    ("drying", "ceramic process"),
    ("firing", "ceramic process"),
    ("glazing", "ceramic process"),
    ("casting", "ceramic process"),
    ("decorating", "ceramic process"),
    ("candling", "ceramic process"),
    ("grinding", "ceramic process"),
    ("polishing", "ceramic process"),
    ("mixing", "ceramic process"),
    ("screening", "ceramic process"),
    ("stamping", "ceramic process"),
    ("burnishing", "ceramic process"),
    # --- people ---
    ("person", "object"),
    ("craft worker", "person"),
    ("potter", "craft worker"),
    ("ceramist", "craft worker"),
    ("decorator", "craft worker"),
    ("stoker", "craft worker"),
):
    B._isa(_leaf, _parent)

# ==================== has_property : exactly one per anchor ==================
for _anchor, _value in (
    ("workbench", "wooden"),          # the disclosed-guess anchor: MUST be wooden
    ("kiln", "high heat"),
    ("wheel", "rotating platform"),
    ("bat", "round disc"),
    ("shaft", "central axis"),
    ("treadle", "push rod"),
    ("shelf", "flat rack"),
    ("burner", "gas flame"),
    ("door", "hinged panel"),
    ("flue", "smoke duct"),
    ("clay", "plastic body"),
    ("kaolin", "white powder"),
    ("grog", "coarse grit"),
    ("feldspar", "flux mineral"),
    ("earthenware", "low fired"),
    ("stoneware", "vitreous body"),
    ("porcelain", "translucent"),
    ("terracotta", "orange red"),
    ("enamel", "glossy coat"),
    ("engobe", "matte layer"),
    ("lustre", "metallic sheen"),
):
    B._prop(_anchor, _value)

# ==================== causes / caused_by : stored both ways ==================
# Disjoint pairs: a shared endpoint is a hub, and from a hub both readings are
# legal, so neither end is auto-gradable against one gold node.
for _n, (_a, _b) in enumerate((
    ("overheating", "warping"),
    ("air bubble", "blowout"),
    ("slow cooling", "crazing"),
    ("uneven glaze", "drip mark"),
    ("moisture", "explosion"),
    ("misalignment", "lopsided body"),
    ("excessive heat", "shell crack"),
    ("pressure", "hairline fracture"),
), 1):
    B._cause(_a, _b, _n)

# ==================== precedes / follows : stored both ways ==================
# Deliberately NOT a process chain. In a chain every interior node carries two
# temporal edges, making it a hub and throwing away both of its direction
# reads. All 12 nodes below appear exactly once in the temporal family.
for _n, (_a, _b) in enumerate((
    ("wedging", "throwing"),
    ("mixing", "screening"),
    ("casting", "candling"),
    ("decorating", "glazing"),
    ("grinding", "polishing"),
    ("stamping", "burnishing"),
), 1):
    B._precedes(_a, _b, _n)

# ==================== part_of : stored forward only ==========================
# Section 9: the reverse `has_part` is synthesised by the runtime mirror, so
# `has_part` must never appear in the stored relation set.
for _child, _whole in (
    ("bat", "wheel"),
    ("shaft", "wheel"),
    ("treadle", "wheel"),
    ("shelf", "kiln"),
    ("burner", "kiln"),
    ("door", "kiln"),
    ("flue", "kiln"),
    ("handle", "mug"),
    ("spout", "jug"),
    ("foot", "plate"),
    ("lid", "jar"),
    ("rim", "bowl"),
):
    B._part(_child, _whole)

# ==================== example_of : forward only =============================
for _item, _cat in (
    ("planter", "garden ware"),
    ("tile", "architectural ceramic"),
    ("figurine", "decorative ware"),
    ("teapot", "tableware"),
    ("crock", "storage ware"),
):
    B._example(_item, _cat)

# ==================== associated_with : forward only =========================
for _a, _b in (
    ("potter", "throwing"),
    ("ceramist", "drying"),
    ("enamel", "brushwork"),
    ("stoneware", "high temperature"),
    ("porcelain", "translucency"),
    ("grog", "shrinkage control"),
):
    B._assoc(_a, _b)


# ==================== supports / contradicts : forward only ==================
def _evidence(name: str, supports: str, contradicts: str) -> None:
    B._add_edge(name, "supports", supports, B._i + 1)
    B._i += 1
    B._add_edge(name, "contradicts", contradicts, B._i + 1)
    B._i += 1


for _name, _supported, _refuted in (
    ("inspection note", "even soak claim", "uneven soak claim"),
    ("colour record", "even tint claim", "patchy tint claim"),
    ("shape check", "true form claim", "warped form claim"),
):
    _evidence(_name, _supported, _refuted)

# ==================== symmetric relations, stored both ways ==================
# Section 9 forbids the mirror from producing these, so both orientations are
# written by hand.
for _n, (_l, _r) in enumerate((
    ("kiln", "furnace"),
    ("terracotta", "baked earth"),
    ("mug", "cup"),
    ("wedging", "kneading"),
    ("porcelain", "china"),
), 1):
    B._both_symmetric(_l, _r, "synonym", B._pct(_n, 1), B._pct(_n, 2))

for _n, (_l, _r) in enumerate((
    ("matte finish", "gloss finish"),
    ("dense body", "porous body"),
    ("thick wall", "thin wall"),
), 1):
    B._both_symmetric(_l, _r, "antonym", B._pct(_n, 1), B._pct(_n, 2))

for _n, (_l, _r) in enumerate((
    ("bat", "shelf"),
    ("handle", "spout"),
    ("flue", "door"),
    ("planter", "tile"),
), 1):
    B._both_symmetric(_l, _r, "spatial_near", B._pct(_n, 1), B._pct(_n, 2))

for _n, (_l, _r) in enumerate((
    ("mixing", "wedging"),
    ("drying", "candling"),
    ("decorating", "glazing"),
), 1):
    B._both_symmetric(_l, _r, "temporal_coincident", B._pct(_n, 1), B._pct(_n, 2))

# linguistic_maps -- English <-> French. The question template says "translates
# to" and never names a language, so no question can assert a language the
# stored mapping does not use.
for _n, (_l, _r) in enumerate((
    ("clay", "argile"),
    ("kiln", "four"),
    ("wheel", "tour"),
    ("potter", "potier"),
    ("stoneware", "gres"),
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

    # A label whose token is itself another label is unanswerable by
    # construction: the anchorer matches on tokens, so the shorter label wins.
    singles = {l for l in labels if " " not in l}
    for l in labels:
        for t in _tokens(l):
            if t != l.lower() and t in singles:
                problems.append(("label-token-is-another-label", f"{l!r} -> {t!r}"))

    # An out-of-graph subject that shares a token with a label will be grounded
    # on that label by the identity gate, so the question stops being an
    # honesty question and becomes a confidently-wrong-answer test.
    for subject, _q in OUT_OF_GRAPH_SUBJECTS:
        for t in _tokens(subject):
            if t in singles:
                problems.append(("out-of-graph-subject-grounds-on-a-label",
                                 f"{subject!r} -> {t!r}"))
            for l in labels:
                if len(t) >= MIN_PREFIX_LEN and l.lower().startswith(t):
                    problems.append(("out-of-graph-subject-prefixes-a-label",
                                     f"{subject!r} -> {l!r}"))

    # A 4+ character label prefix lets one question ground on the wrong node.
    for short in labels:
        if len(short) < MIN_PREFIX_LEN:
            continue
        clashes = sorted(o for o in labels if o != short and o.startswith(short))
        if clashes:
            problems.append(("label-prefix-collision", f"{short!r} -> {clashes}"))

    # The disclosed-guess anchor must exist or the fallback-disclosure path is
    # never exercised. The inherited validate() already enforces it carries a
    # unique `wooden` property.
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
    print("Building POC SHOWCASE demo graph (pottery & ceramics): "
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
    store.set_metadata("dataset_name", "showcase_pottery")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.set_metadata("frozen", "true")
    store.set_metadata("contract", "GLM-X v3.3.2 sections 8/9/15/16")
    store.set_metadata("version", "1.0.0")
    store.save_state(str(DB_PATH))

    digest = B.finalize_and_hash(store)
    print(f"\nFROZEN: {DB_PATH}")
    print(f"sha256: {digest}")
    (HERE / "frozen_graph_digest.txt").write_text(
        f"{digest}  showcase_eval.db\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
