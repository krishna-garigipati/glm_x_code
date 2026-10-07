"""FINAL PoC validation set: build the frozen evaluation graph (GLM-X v3.3.2).

This is the sign-off artefact, not another stage. It exists because the two
defects found by the previous held-out evaluation were fixed in code AFTER that
evaluation was scored:

  1. an entity-identity gap -- the walk anchored on an entity the question never
     named and answered confidently about it;
  2. resonance pruning of real stored edges -- the tier1 node budget silently
     deleted true edges incident to a budget-dropped node.

Re-scoring the old held-out set would not be a proof of anything: those
questions have already been seen, and the fixes were designed while looking at
them. So the graph below is a NEW domain and the question set is derived
mechanically from it, mechanically preflighted, frozen, and then scored ONCE.

DOMAIN: AVIATION. Chosen by auditing every domain that already exists in this
repository, not by taste. In use elsewhere: toy, nature/weather/climate,
geography glossary, science evidence, food and food-bio, astronomy, computing,
history, human body, music (tester-c/datasets/build_music_small.py, which also
already carries guitar->chitarra / violin->violino linguistic_maps), the Stage
A-E graphs and the lead zoo/mountain material. Aviation appears in none of
them. An earlier draft of this file used music and was discarded when that
overlap was found -- reusing a probed domain would have quietly voided the
no-look commitment this whole artefact exists to honour.

WHY THIS REUSES STAGE F's BUILDER MODULE
    Everything except the node and edge tables is domain-independent machinery:
    reach_from / unique_one_hop / direction_pairs / multi_hop_chains /
    mirror_controls / validate / finalize_and_hash. Rather than reimplement
    ~800 lines of proven derivation (and risk a silent divergence that quietly
    changed what counts as gradable), this module LOADS that module by path and
    REBINDS only the data globals: CONCEPTS, EDGES, OUT_OF_GRAPH_SUBJECTS,
    DISCLOSED_GUESS, the non-ASCII answer allowlist and DB_PATH. This is the
    same binding pattern stage_f_runner already uses to reuse the shared grader.

    Two checks are ADDED on top of the inherited validate(), both forced by the
    fixes themselves rather than by taste:

      * `too-few-out-of-graph-subjects` -- the identity gate now carries the
        honesty category, and at 9 questions one question is worth 11 points,
        so a 90% target cannot be resolved. 20+ is the minimum that makes the
        category meaningful.
      * `label-prefix-collision` -- the identity gate grounds an anchor when a
        label shares a 4+ character prefix with a question token, so `wing`
        would ground the question "wings". Any label that is a 4+ character
        prefix of another label is now a build refusal.

    Nothing inherited is filtered, suppressed or weakened. Every inherited check
    that fires is a real problem, including the disclosed-guess fixture check
    that hardcodes the literal `{"wooden"}` -- which is why the glider's
    has_property value below IS "wooden" rather than this module filtering that
    check out as "belonging to another stage".

NO-LOOK COMMITMENT
    Authored before any question in this set was scored. Build, validate,
    derive, preflight, freeze, then score exactly once. If the score comes back
    low it is reported as-is.

Usage:
    python test_results/final_validation/build_graph.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Load the proven builder machinery and rebind ONLY the data globals.
# ---------------------------------------------------------------------------
_SRC = ROOT / "test_results" / "stage_f" / "build_holdout_graph.py"
_spec = importlib.util.spec_from_file_location("_stage_f_builder", _SRC)
B = importlib.util.module_from_spec(_spec)
sys.modules["_stage_f_builder"] = B
_spec.loader.exec_module(B)

DB_PATH = HERE / "final_eval.db"

# ---------------------------------------------------------------------------
# Subjects genuinely absent from the graph. Each carries a LITERAL cue phrase
# from configs/config_g2p.yaml `extraction.relation_variants`, so the refusal
# has to come from the entity/identity gate and not from cue absence -- otherwise
# the honesty category would be measuring the wrong thing. generate_questions
# re-verifies that mechanically against the live planner before freezing.
#
# Two of these are deliberately near-misses against real labels, because that is
# the exact shape of the defect that was fixed: `stall recovery` against the
# `stall` node, and `windshear detector` against the `wind shear` node. A system
# that answers either confidently is answering about an entity the question did
# not name.
# ---------------------------------------------------------------------------
OUT_OF_GRAPH_SUBJECTS: list[tuple[str, str]] = [
    ("biplane", "What type of thing is the biplane?"),
    ("microlight", "What type of thing is the microlight?"),
    ("hydrofoil", "What type of thing is the hydrofoil?"),
    ("drone", "What type of thing is the drone?"),
    ("autogyro", "What type of thing is the autogyro?"),
    ("gyroplane", "What type of thing is the gyroplane?"),
    ("flying boat", "What causes the flying boat?"),
    ("air taxi", "What was the air taxi caused by?"),
    ("crop duster", "What leads to the crop duster?"),
    ("skyranch", "What results in the skyranch?"),
    ("zeppelin", "What is the zeppelin part of?"),
    ("gyrocopter", "What is the gyrocopter part of?"),
    ("hang glider", "What is the hang glider part of?"),
    ("air ambulance", "What is the air ambulance part of?"),
    ("parachute", "What is the parachute part of?"),
    ("skywriter", "What is the skywriter an example of?"),
    ("tanker aircraft", "What is the tanker aircraft associated with?"),
    ("black box", "What is known for the black box?"),
    ("stall recovery", "What is the stall recovery an example of?"),
    ("windshear detector", "What is the windshear detector associated with?"),
    ("pressure altimeter", "What is known for the pressure altimeter?"),
    ("supersonic wind tunnel", "What is the supersonic wind tunnel an example of?"),
]

# Cue-free questions. Contract section 8 fallback triggers on cue ABSENCE, so
# every one of these must set heuristic_fallback_used. The generator's preflight
# re-checks that mechanically against the live descriptor bank.
NONSENSE_CUE_FREE: list[str] = [
    "How many engines does the letter seven have?",
    "What colour is the sound of a doorbell?",
    "How heavy is the smell of Tuesday?",
    "What is the temperature of the word apple?",
    "How bitter is the shape of a triangle?",
    "What is the length of the number nine?",
]

# A real in-graph anchor whose has_property read is unique, asked with NO
# relation cue. The system must fall back AND disclose that it is guessing.
# Section 8 requires the disclosure; this exercises that path, not refusal.
#
# The guess being disclosed is literally "wooden", so the glider's stored
# has_property value is "wooden". Vintage gliders genuinely are wooden, so the
# data is not contrived to satisfy the inherited fixture check -- but that check
# hardcodes this literal, and satisfying it natively is why no inherited check
# has to be filtered out below.
DISCLOSED_GUESS: tuple[str, str] = ("glider", "How wooden is the glider?")

# Non-ASCII answers allowed by the inherited label check. Italian terms only.
ITALIAN_ANSWERS = frozenset({"ala", "ingegnere", "motore", "timone", "pista"})

# ---------------------------------------------------------------------------
# THE GRAPH. Nodes are DERIVED from the edge endpoints, so a typo becomes a node
# rather than an error -- which is why the inherited label checks are strict.
# ---------------------------------------------------------------------------
CONCEPTS: list[str] = []
EDGES: list[tuple[str, str, str, float, float]] = []

# ---- rebind the builder module onto THIS data before calling its helpers ----
B.CONCEPTS = CONCEPTS
B.EDGES = EDGES
B.OUT_OF_GRAPH_SUBJECTS = OUT_OF_GRAPH_SUBJECTS
B.DISCLOSED_GUESS = DISCLOSED_GUESS
B.SPANISH_ANSWERS = ITALIAN_ANSWERS
B.DB_PATH = DB_PATH

# ============================ is_a : taxonomy, forward only ==================
# The top label is "aircraft class", not "aircraft" or "aircraft type":
# "aircraft" would be a 4+ character prefix of "aircraft component", and "type"
# is a token GLMXPipeline._QUESTION_FILLER drops during anchoring, so a label
# carrying it would be silently unmatchable by its own question.
for _leaf, _parent in (
    ("aircraft class", "vehicle"),
    ("crew member", "aviation role"),
    ("flight phase", "aviation process"),
    ("airport facility", "infrastructure"),
    ("flight document", "record"),
    ("navigation aid", "infrastructure"),
    ("weather hazard", "hazard"),
    ("flight hazard", "hazard"),
    # families
    ("fixed wing aircraft", "aircraft class"),
    ("rotary wing aircraft", "aircraft class"),
    ("airship", "aircraft class"),
    ("commercial aircraft", "aircraft class"),
    ("general aviation aircraft", "aircraft class"),
    ("cargo aircraft", "aircraft class"),
    ("twin engine aircraft", "aircraft class"),
    ("single engine aircraft", "aircraft class"),
    ("airliner", "commercial aircraft"),
    ("business jet", "commercial aircraft"),
    ("freighter", "cargo aircraft"),
    ("seaplane", "aircraft class"),
    ("glider", "aircraft class"),
    ("rotorcraft", "rotary wing aircraft"),
    # people
    ("pilot", "crew member"),
    ("flight engineer", "crew member"),
    ("ground controller", "crew member"),
    # facilities
    ("runway", "airport facility"),
    ("control tower", "airport facility"),
    ("hangar", "airport facility"),
    ("terminal building", "airport facility"),
    ("windsock", "airport facility"),
    ("instrument landing system", "navigation aid"),
    ("radar beacon", "navigation aid"),
    # records
    ("flight plan", "flight document"),
    ("maintenance log", "record"),
    # weather
    ("icing", "weather hazard"),
    ("turbulence", "weather hazard"),
    ("thunderstorm", "weather hazard"),
    ("crosswind", "weather hazard"),
    # flight phases and hazards
    ("taxiing", "flight phase"),
    ("takeoff", "flight phase"),
    ("climb", "flight phase"),
    ("cruise", "flight phase"),
    ("descent", "flight phase"),
    ("landing", "flight phase"),
    ("rotation", "flight phase"),
    ("stall", "flight hazard"),
    # parts
    ("aircraft component", "component"),
    ("wing", "aircraft component"),
    ("fuselage", "aircraft component"),
    ("empennage", "aircraft component"),
    ("jet engine", "aircraft component"),
    ("nose wheel", "aircraft component"),
    ("radome", "aircraft component"),
    ("aileron", "aircraft component"),
    ("elevator", "aircraft component"),
    ("rudder", "aircraft component"),
):
    B._isa(_leaf, _parent)

# ====================== has_property : exactly one per anchor ================
for _anchor, _value in (
    ("airliner", "pressurised cabin"),
    ("freighter", "side loading door"),
    ("seaplane", "water hull"),
    ("glider", "wooden"),
    ("rotorcraft", "collective pitch control"),
    ("pilot", "instrument rating"),
    ("ground controller", "clearance phraseology"),
    ("runway", "marked threshold"),
    ("hangar", "sliding door"),
    ("instrument landing system", "localizer beam"),
    ("flight plan", "filed route"),
    ("icing", "clear ice"),
    ("turbulence", "rolling motion"),
    ("thunderstorm", "cumulonimbus cloud"),
    ("crosswind", "lateral drift"),
    ("wing", "lifting surface"),
    ("fuselage", "pressure shell"),
    ("empennage", "stability surface"),
    ("jet engine", "thrust output"),
):
    B._prop(_anchor, _value)

# ====================== causes / caused_by : stored both ways =================
# Disjoint pairs on purpose. A shared endpoint is a hub: from a hub both
# readings are legal, so neither end is auto-gradable against one gold node.
for _n, (_a, _b) in enumerate((
    ("crosswind", "weathervaning"),
    ("thunderstorm", "wind shear"),
    ("fatigue", "misread instrument"),
    ("icing", "lift reduction"),
    ("poor maintenance", "hydraulic failure"),
    ("headwind", "fuel burn"),
    ("miscommunication", "level bust"),
    ("bird strike", "engine shutdown"),
), 1):
    B._cause(_a, _b, _n)

# ====================== precedes / follows : stored both ways ================
# Also disjoint, for the same hub reason. Note these are deliberately NOT the
# taxiing->takeoff->climb->cruise->descent->landing chain: in a chain every
# interior node has two temporal edges, which makes it a hub and throws away
# both of its direction reads.
for _n, (_a, _b) in enumerate((
    ("rotation", "initial climb"),
    ("fuel planning", "weight calculation"),
    ("deicing", "departure clearance"),
    ("slot allocation", "gate assignment"),
    ("briefing", "sign off"),
    ("taxi clearance", "engine start"),
), 1):
    B._precedes(_a, _b, _n)

# ====================== part_of : stored forward only =========================
# Section 9: the reverse `has_part` is synthesised by the runtime mirror, so
# `has_part` must never appear in the stored relation set.
for _child, _whole in (
    ("wing", "airliner"),
    ("fuselage", "airliner"),
    ("empennage", "airliner"),
    ("jet engine", "airliner"),
    ("radome", "business jet"),
    ("nose wheel", "rotorcraft"),
    ("rotor blade", "rotorcraft"),
    ("tail fin", "glider"),
    ("aileron", "wing"),
    ("elevator", "empennage"),
    ("rudder", "empennage"),
):
    B._part(_child, _whole)

# ====================== example_of : forward only ============================
for _item, _cat in (
    ("concorde", "supersonic airliner"),
    ("quadcopter", "unmanned rotorcraft"),
    ("hot air balloon", "lighter than air craft"),
    ("pilot", "flight crew"),
    ("flight engineer", "flight crew"),
):
    B._example(_item, _cat)

# ====================== associated_with : forward only =======================
for _a, _b in (
    ("runway", "ascent path"),
    ("pilot", "flight duty"),
    ("control tower", "approach sequence"),
    ("icing", "cold temperature"),
    ("glider", "ridge lift"),
    ("navigation aid", "final approach"),
):
    B._assoc(_a, _b)

# ====================== supports / contradicts : forward only ================
def _evidence(name: str, supports: str, contradicts: str) -> None:
    B._add_edge(name, "supports", supports, B._i + 1)
    B._i += 1
    B._add_edge(name, "contradicts", contradicts, B._i + 1)
    B._i += 1


# The claim labels deliberately avoid being extensions of a real node label
# ("pilot error claim" would extend `pilot`; "weather hazard claim" would extend
# `weather hazard`), which would trip the prefix check.
for _name, _supported, _refuted in (
    ("safety review", "stable airframe claim", "aggressive loading claim"),
    ("accident report", "crew mistake claim", "maintenance fault claim"),
    ("flight recorder", "adverse weather claim", "system failure claim"),
):
    _evidence(_name, _supported, _refuted)

# ====================== symmetric relations, stored both ways ================
# Section 9 forbids the mirror pass from producing these five, so both
# orientations are written by hand.
for _n, (_l, _r) in enumerate((
    ("aeroplane", "airplane"),
    ("rotorcraft", "helicopter"),
    ("runway", "airstrip"),
    ("windsock", "wind cone"),
    ("empennage", "tail assembly"),
), 1):
    B._both_symmetric(_l, _r, "synonym", B._pct(_n, 1), B._pct(_n, 2))

for _n, (_l, _r) in enumerate((
    ("ascending", "descending"),
    ("inbound", "outbound"),
    ("left bank", "right bank"),
), 1):
    B._both_symmetric(_l, _r, "antonym", B._pct(_n, 1), B._pct(_n, 2))

for _n, (_l, _r) in enumerate((
    ("aileron", "elevator"),
    ("rudder", "elevator"),
    ("pilot", "flight engineer"),
    ("runway", "control tower"),
), 1):
    B._both_symmetric(_l, _r, "spatial_near", B._pct(_n, 1), B._pct(_n, 2))

for _n, (_l, _r) in enumerate((
    ("takeoff", "rotation"),
    ("landing", "touchdown"),
    ("cruise", "level flight"),
), 1):
    B._both_symmetric(_l, _r, "temporal_coincident", B._pct(_n, 1), B._pct(_n, 2))

# linguistic_maps -- English <-> Italian. The literal cue for this relation is
# "what do you call" / "how do you say", NOT the language name, so the question
# template can honestly say "in Italian" while still cuing the relation.
for _n, (_l, _r) in enumerate((
    ("wing", "ala"),
    ("flight engineer", "ingegnere"),
    ("jet engine", "motore"),
    ("rudder", "timone"),
    ("runway", "pista"),
), 1):
    B._both_symmetric(_l, _r, "linguistic_maps", B._pct(_n, 1), B._pct(_n, 2))

CONCEPTS.extend(sorted({x for e in EDGES for x in (e[0], e[2])}))


# ---------------------------------------------------------------------------
# Added validation
# ---------------------------------------------------------------------------
MIN_OUT_OF_GRAPH_SUBJECTS = 20
MIN_PREFIX_LEN = 4


def added_problems() -> list[tuple[str, str]]:
    """Checks the inherited validate() cannot know to make.

    Everything domain-independent is already covered upstream. These two exist
    because the two fixes changed what the evaluation set has to be able to
    measure.
    """
    problems: list[tuple[str, str]] = []
    labels = sorted(set(CONCEPTS))

    # The identity gate now carries honesty_out_of_graph. With 9 questions one
    # question is worth 11 points, so the category cannot resolve a 90% target
    # and a single refusal flip would decide the PoC verdict.
    n = len(OUT_OF_GRAPH_SUBJECTS)
    if n < MIN_OUT_OF_GRAPH_SUBJECTS:
        problems.append(("too-few-out-of-graph-subjects",
                         f"{n} < {MIN_OUT_OF_GRAPH_SUBJECTS}"))

    # _anchor_identity_grounded treats a shared 4+ character prefix as identity
    # evidence, so `wing` would ground the question "wings". Any label that is a
    # prefix of another label makes that inference unsound.
    for short in labels:
        if len(short) < MIN_PREFIX_LEN:
            continue
        clashes = sorted(o for o in labels
                         if o != short and o.startswith(short))
        if clashes:
            problems.append(("label-prefix-collision",
                             f"{short!r} is a prefix of {clashes}"))

    # The disclosed-guess anchor must exist, or the fallback-disclosure path is
    # not exercised at all. The inherited validate() already enforces a unique
    # has_property read from it.
    anchor, _q = DISCLOSED_GUESS
    if anchor not in set(CONCEPTS):
        problems.append(("disclosed-guess-anchor-not-a-concept", anchor))

    return problems


def validate() -> list[tuple[str, str]]:
    return B.validate() + added_problems()


# ---------------------------------------------------------------------------
# Build + freeze
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
    print("Building FINAL PoC Evaluation Graph (aviation domain): "
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

    import numpy as np
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer("BAAI/bge-small-en-v1.5")
    enc = sbert.encode(labels, normalize_embeddings=True, show_progress_bar=False)
    embeddings = {lab: np.asarray(enc[i], dtype=np.float32) for i, lab in enumerate(labels)}

    from graph.graph_component_implementation.sqlite_graph_store import SQLiteGraphStore

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
    store.set_metadata("dataset_name", "final_poc_eval_aviation")
    store.set_metadata("embed_model", "BAAI/bge-small-en-v1.5")
    store.set_metadata("frozen", "true")
    store.set_metadata("contract", "GLM-X v3.3.2 sections 8/9/15/16")
    store.set_metadata("version", "1.0.0")
    store.save_state(str(DB_PATH))

    digest = B.finalize_and_hash(store)
    print(f"\nFROZEN: {DB_PATH}")
    print(f"sha256: {digest}")
    (HERE / "frozen_graph_digest.txt").write_text(
        f"{digest}  final_eval.db\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
