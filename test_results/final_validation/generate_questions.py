"""FINAL PoC validation set: derive and freeze the question set (v3.3.2 §16).

Contract section 16: "Freeze the question set before final evaluation
(pre-registered)". This derives every question MECHANICALLY from the frozen
aviation graph. No question is hand-picked and no gold answer is adjusted to suit
a phrasing.

THE NO-LOOK COMMITMENT, AND HOW IT IS ENFORCED HERE
    This is the sign-off evaluation set. It was built AFTER both fixes were
    written and AFTER the previous held-out set had already been scored and
    found wanting, and it is in a domain (aviation) that appears nowhere else in
    this repository. Everything it contains was derived before a single one of
    its questions was scored.

    The strongest available guard against having tuned the generator to this
    graph is that the templates ARE Stage E's templates: `ONE_HOP_TEMPLATE` and
    `CONT_CLAUSE` are IMPORTED from `test_results/stage_e/generate_questions.py`
    rather than copied and edited, and the construction logic is Stage F's, run
    unchanged. If the templates were edited to flatter this graph, the import
    would change and the diff here would show it.

WHAT IS NEW HERE, AND WHY IT IS STILL DERIVED NOT CHOSEN
    Every previous set asked its one-hop questions with exactly ONE phrasing per
    relation -- the Stage E template. So the measured `simple_one_hop` rate has
    always been "can the walk find a uniquely-stored edge from a known phrasing",
    which flatters the system: nothing ever tested whether it survives the same
    question said differently.

    So a deterministic slice of the unique one-hop edges is re-asked using
    cue-preserving rewordings, tagged `variant_kind`:
      * "paraphrase" -- different framing, anchor literal, SAME literal cue.
      * "plural"     -- anchor morphologically inflected ("wing" -> "wings").

    The constraint that makes this fair rather than cruel: `_literal_cue_relations`
    is a pure regex over the config descriptor bank, so a rewording that loses
    its cue stops being a paraphrase and becomes an unanswerable question. Every
    variant template below is therefore verified against the LIVE planner in
    `preflight()`, exactly like the Stage E templates are, and a variant that
    does not emit its declared relation refuses the freeze.

    They are tagged rather than given their own category on purpose: an unknown
    category is reported `unscored_category` by the shared grader, and editing the
    shared grader's CATEGORY_TARGETS would retroactively change what the frozen
    Stage E number means. These ARE one-hop reads of a uniquely-stored edge, so
    they are scored as one-hop reads and broken out in the report by tag.

CATEGORIES ARE DISJOINT
    A one-hop read that is one end of a proven direction pair is counted as
    `direction_pair`, not also as `simple_one_hop`. The paraphrase slice is then
    taken FROM the simple_one_hop pool and removed from it, so no edge is graded
    twice under two phrasings and the pooled score never double-counts.

Usage:
    python test_results/final_validation/generate_questions.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parents[1]
sys.path.insert(0, str(ROOT))


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The builder, rebound onto the aviation data. Its `B` is the Stage F builder
# module with CONCEPTS/EDGES/OUT_OF_GRAPH_SUBJECTS/DISCLOSED_GUESS/DB_PATH
# pointing at aviation.
FG = _load_by_path("_final_build", STAGE_DIR / "build_graph.py")
B = FG.B

# Stage F's generator does `import build_holdout_graph as B`, so registering the
# rebound builder under that name is what makes the shared derivation run against
# aviation rather than silently re-deriving the food graph.
sys.modules["build_holdout_graph"] = B
G = _load_by_path("_stage_f_genq", ROOT / "test_results" / "stage_f" / "generate_questions.py")

ONE_HOP_TEMPLATE = G.ONE_HOP_TEMPLATE
CONT_CLAUSE = G.CONT_CLAUSE

FROZEN_PATH = STAGE_DIR / "final_questions_frozen.json"
VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Cue-preserving rewordings, one entry per (relation, variant_kind).
#
# Every template must contain a literal descriptor from
# configs/config_g2p.yaml `extraction.relation_variants` for its OWN relation and
# for no other relation. The cue is quoted beside each template so a future edit
# that breaks it is obvious. These are asserted mechanically, not by inspection:
# `preflight()` runs the live planner over every generated question.
#
# Note the two traps this had to respect:
#   * `causes` cues on the bare word "cause" but the regex is `(?<!\w)cause(?!\w)`,
#     so "caused" does NOT match it -- that is what keeps "is caused by" from
#     emitting both causes and caused_by.
#   * `part_of` has a reverse-sounding pair ("what makes up", "made up of") that
#     asks the OPPOSITE direction. None of the variants below use it.
# ---------------------------------------------------------------------------
PARAPHRASE_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "is_a": [
        ("paraphrase", "What kind of thing is the {a}?"),              # "what kind of"
        ("paraphrase", "The {a} belongs to the class of what?"),      # "belongs to the class"
        ("plural", "What type of thing is the {plural}?"),             # "what type of"
    ],
    "has_property": [
        ("paraphrase", "The {a} has the characteristic of what?"),     # "has the characteristic"
        ("paraphrase", "The {a} possesses what property?"),            # "possesses"
        ("plural", "What is known for the {plural}?"),                 # "is known for"
    ],
    "causes": [
        ("paraphrase", "What leads to the {a}?"),                      # "what leads to"
        ("paraphrase", "What results in the {a}?"),                    # "what results in"
        ("plural", "What causes the {plural}?"),                       # "cause"
    ],
    "caused_by": [
        ("paraphrase", "The {a} is caused by what?"),                  # "is caused by"
        ("paraphrase", "What is responsible for the {a}?"),            # "what is responsible for"
        ("plural", "The {plural} are caused by what?"),                # "are caused by"
    ],
    "precedes": [
        ("paraphrase", "What happens after the {a}?"),                 # "happens after"
        ("plural", "What comes after the {plural}?"),                  # "comes after"
    ],
    "follows": [
        ("paraphrase", "What happened before the {a}?"),               # "happened before"
        ("plural", "What comes before the {plural}?"),                 # "comes before"
    ],
    "part_of": [
        ("paraphrase", "The {a} is contained in what?"),               # "contained in"
        # Plural form uses the passive, because "What is the wings part of?"
        # is ungrammatical.
        ("plural", "The {plural} are contained in what?"),             # "contained in"
    ],
    "synonym": [
        ("paraphrase", "What is another term for the {a}?"),           # "another term for"
        ("paraphrase", "The {a} is also called what?"),                # "also called"
        ("plural", "What is another word for the {plural}?"),          # "what is another word for"
    ],
    "antonym": [
        ("paraphrase", "What is the antonym of the {a}?"),             # "antonym of"
        ("plural", "What is the opposite of the {plural}?"),           # "opposite of"
    ],
    "example_of": [
        # "The {a} is an example of what?" would ALSO cue is_a, because "is an"
        # is an is_a descriptor and _literal_cue_relations returns every relation
        # that matches anywhere in the question. "stands as an example of" keeps
        # the example_of cue and contains no is_a descriptor.
        ("paraphrase", "The {a} stands as an example of what?"),       # "example of"
        ("plural", "What is the {plural} an example of?"),             # "example of"
    ],
    "associated_with": [
        ("paraphrase", "The {a} is linked to what?"),                  # "linked to"
        ("plural", "What is the {plural} associated with?"),           # "associated with"
    ],
    "supports": [
        ("paraphrase", "What substantiates the {a}?"),                 # "substantiates"
        ("plural", "What evidence supports the {plural}?"),             # "evidence for"
    ],
    "contradicts": [
        ("paraphrase", "What refutes the {a}?"),                       # "refutes"
        ("plural", "What contradicts the {plural}?"),                  # "contradicts"
    ],
    "temporal_coincident": [
        ("paraphrase", "What coincides with the {a}?"),                # "coincides with"
        ("plural", "What happens at the same time as the {plural}?"),  # "happens at the same time as"
    ],
    "spatial_near": [
        ("paraphrase", "What is close to the {a}?"),                   # "what is close to"
        ("plural", "What is near the {plural}?"),                      # "what is near"
    ],
    "linguistic_maps": [
        ("paraphrase", "How do you say the {a} in Italian?"),          # "how do you say"
        ("plural", "What do you call the {plural} in Italian?"),       # "what do you call"
    ],
}

# Deterministic slice of the simple_one_hop pool to re-ask with a variant.
# Every STRIDE-th, capped, so the slice spans the sorted edge order rather than
# clustering on whatever sorts first.
PARAPHRASE_STRIDE = 4
PARAPHRASE_MAX = 36


def pluralize(label: str) -> str:
    """Plain English pluralisation of the LAST word of a label.

    Deliberately naive and rule-based rather than clever: an irregular table
    would be one more hand-authored thing that could flatter this graph, and
    every aviation label here pluralises regularly.
    """
    head, _, last = label.rpartition(" ")
    if re.search(r"(s|x|z|ch|sh)$", last):
        plural = last + "es"
    elif re.search(r"[^aeiou]y$", last):
        plural = last[:-1] + "ies"
    else:
        plural = last + "s"
    return f"{head} {plural}" if head else plural


# --------------------------------------------------------------------------
def build_questions() -> tuple[list[dict], dict]:
    questions, population = G.build_questions()

    # Re-ask a deterministic slice of the one-hop pool with a variant phrasing.
    one_hop_idx = [i for i, q in enumerate(questions)
                   if q["category"] == "simple_one_hop"]
    chosen = one_hop_idx[::PARAPHRASE_STRIDE][:PARAPHRASE_MAX]

    variants: list[dict] = []
    for i in chosen:
        q = questions[i]
        rel = q["rel"]
        bank = PARAPHRASE_VARIANTS.get(rel)
        if not bank:
            continue
        # Rotate the variant by position in the chosen slice so every relation
        # contributes all of its rewordings across the set instead of one.
        kind, template = bank[len(variants) % len(bank)]
        anchor = q["anchor"]
        text = template.format(a=anchor, plural=pluralize(anchor))
        v = dict(q)
        v["qid"] = f"pv{len(variants) + 1:03d}"
        v["question"] = text
        v["variant_kind"] = kind
        v["variant_of"] = q["qid"]
        variants.append(v)
        questions[i]["variant_kind"] = "template"

    # Disjointness: the variant rows leave the simple_one_hop pool entirely.
    chosen_set = set(chosen)
    kept = [q for i, q in enumerate(questions)
            if not (q["category"] == "simple_one_hop" and i in chosen_set)]
    population["simple_one_hop"] = sum(
        1 for q in kept if q["category"] == "simple_one_hop")
    population["paraphrase_one_hop"] = len(variants)
    population["paraphrase_by_kind"] = {
        kind: sum(1 for v in variants if v["variant_kind"] == kind)
        for kind in ("paraphrase", "plural")
    }
    population["paraphrase_stride"] = PARAPHRASE_STRIDE
    population["paraphrase_max"] = PARAPHRASE_MAX
    return kept + variants, population


# --------------------------------------------------------------------------
def preflight(questions: list[dict], population: dict) -> list[tuple[str, str]]:
    """Refuse to freeze a set that cannot be scored or cannot be trusted.

    The inherited preflight() covers identity, category minima, graded-edge
    uniqueness, chain expressibility against the LIVE planner, mirror-control
    direction, and out-of-graph absence. Three things are added, all of them
    consequences of what is new in this set.
    """
    problems: list[tuple[str, str]] = list(G.preflight(questions, population))
    labels = set(B.CONCEPTS)

    # --- the planner, loaded once, for the added cue checks ------------------
    try:
        sys.path.insert(0, str(ROOT / "test_results" / "stage_a"))
        sys.path.insert(0, str(ROOT / "test_results" / "stage_b"))
        sys.path.insert(0, str(ROOT / "test_results" / "stage_c"))
        sys.path.insert(0, str(ROOT / "test_results" / "stage_d"))
        from stage_d_runner import load_planner
        planner = load_planner()
    except Exception as exc:  # noqa: BLE001
        problems.append(("could-not-load-planner-for-added-checks", repr(exc)))
        planner = None

    # --- every relation has at least one rewording --------------------------
    for rel in B.CANONICAL_RELATIONS:
        if rel not in PARAPHRASE_VARIANTS:
            problems.append(("relation-has-no-paraphrase-variant", rel))
    for rel in PARAPHRASE_VARIANTS:
        if rel not in B.CANONICAL_RELATIONS:
            problems.append(("paraphrase-variant-for-unknown-relation", rel))

    # --- every declared variant template must emit its own relation ----------
    # Checked on the templates directly as well as on the generated rows, so a
    # broken template is reported even if the slice happened not to use it.
    if planner is not None:
        for rel, bank in sorted(PARAPHRASE_VARIANTS.items()):
            for kind, template in bank:
                probe = template.format(a="wing", plural="wings")
                got = planner._literal_cue_relations(probe)
                if got != [rel]:
                    problems.append((
                        "paraphrase-template-does-not-emit-declared-relation",
                        f"{rel}/{kind}: declared=[{rel}] got={got} :: {probe}"))

        # --- honesty rows must carry a REAL cue -----------------------------
        # Otherwise a refusal proves nothing: the engine could be refusing
        # because it found no cue at all, which is the fallback path, not the
        # entity-identity gate this category is supposed to measure. Exactly one
        # cue is required -- more than one would be a multi-hop question.
        for q in questions:
            if q["category"] != "honesty_out_of_graph":
                continue
            got = planner._literal_cue_relations(q["question"])
            if len(got) != 1:
                problems.append((
                    "out-of-graph-question-does-not-carry-exactly-one-cue",
                    f"{q['qid']} cues={got} :: {q['question']}"))

    # --- generated variant rows are well-formed -----------------------------
    variants = [q for q in questions if q.get("variant_kind") in ("paraphrase", "plural")]
    if len(variants) < 20:
        problems.append(("too-few-paraphrase-questions", f"{len(variants)} < 20"))
    if not any(q["variant_kind"] == "plural" for q in variants):
        problems.append(("no-plural-variants-generated", ""))
    if not any(q["variant_kind"] == "paraphrase" for q in variants):
        problems.append(("no-paraphrase-variants-generated", ""))

    qids = [q["qid"] for q in questions]
    if len(set(qids)) != len(qids):
        problems.append(("duplicate-qid-after-variant-split", ""))

    for v in variants:
        if v["anchor"] not in labels:
            problems.append(("variant-anchor-not-a-node", v["qid"]))
        if v["expected_node"] not in labels:
            problems.append(("variant-expected-node-not-a-node", v["qid"]))
        # The rewording must not have drifted onto a different edge.
        if B.reach_from(v["anchor"], v["rel"], 1) != {v["expected_node"]}:
            problems.append(("variant-gold-edge-not-unique", v["qid"]))
        if v["rel"] not in PARAPHRASE_VARIANTS:
            problems.append(("variant-relation-has-no-bank", v["qid"]))
        # A "plural" row must actually contain an inflected anchor, and must NOT
        # still contain the singular as a standalone word. This has to be a
        # WORD-boundary test: "wing" is a substring of "wings", so a plain
        # substring check reports every correctly-inflected variant as uninflected.
        if v["variant_kind"] == "plural":
            if pluralize(v["anchor"]) not in v["question"]:
                problems.append(("plural-variant-does-not-inflect-anchor", v["qid"]))
            head, _, last = v["anchor"].rpartition(" ")
            if re.search(r"(?<!\w)" + re.escape(last) + r"(?!\w)", v["question"]):
                problems.append(("plural-variant-still-contains-singular-anchor", v["qid"]))

    # --- the split must not have double-graded an edge ----------------------
    graded: dict[tuple, str] = {}
    for q in questions:
        if q["category"] not in ("simple_one_hop", "direction_pair"):
            continue
        key = (q["anchor"], q["rel"], q["expected_node"])
        if key in graded:
            problems.append(("edge-graded-twice",
                             f"{key} via {graded[key]} and {q['qid']}"))
        graded[key] = q["qid"]
    return problems


def main() -> int:
    questions, population = build_questions()
    problems = preflight(questions, population)
    if problems:
        print("REFUSING TO FREEZE:")
        for kind, detail in problems:
            print(f"  {kind}: {detail}")
        print(f"\n({len(problems)} problems)")
        return 1

    spec = {
        "version": VERSION,
        "contract": "GLM-X v3.3.2 section 16 -- FINAL PoC Evaluation Set",
        "generated_by": (
            "generate_questions.py (Stage F derivation run unchanged against "
            "build_graph.py; ONE_HOP_TEMPLATE and CONT_CLAUSE imported verbatim "
            "from test_results/stage_e/generate_questions.py)"),
        "held_out": (
            "derived and frozen BEFORE any question in this set was scored; the "
            "graph is the aviation domain, which appears in no other dataset, "
            "stage or probe in this repository"),
        "graph": {
            "path": str(FG.DB_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": B.sha256(FG.DB_PATH),
            "nodes": len(set(B.CONCEPTS)),
            "edges": len(B._triples()),
            "relations": sorted(B.CANONICAL_RELATIONS),
        },
        "population": population,
        "counts": {c: sum(1 for q in questions if q["category"] == c)
                   for c in sorted({q["category"] for q in questions})},
        "questions": questions,
    }
    payload = json.dumps(spec, indent=2, ensure_ascii=False).encode("utf-8")
    FROZEN_PATH.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (STAGE_DIR / "frozen_questions_digest.txt").write_text(
        f"{digest}  final_questions_frozen.json\n", encoding="utf-8")

    print(f"Froze {len(questions)} questions -> {FROZEN_PATH}")
    print(f"  version {VERSION}")
    print(f"  graph sha256 {spec['graph']['sha256']}")
    print(f"  set   sha256 {digest}")
    for cat, n in spec["counts"].items():
        print(f"    {cat:24} {n}")
    print(f"  variant rows: {population['paraphrase_by_kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
