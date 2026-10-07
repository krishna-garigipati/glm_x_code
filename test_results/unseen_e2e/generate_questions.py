"""UNSEEN-DATA END-TO-END TEST: derive and freeze the question set.

Contract section 16 requires the question set to be frozen before final
evaluation. This derives every question MECHANICALLY from the frozen weaving
graph. No question is hand-picked, no gold answer is adjusted to suit a
phrasing, and nothing here was chosen after seeing a score.

NO-LOOK COMMITMENT
    The domain (weaving), the graph, the out-of-graph subjects, the nonsense
    questions, the disclosed guess and the templates below were all authored
    before a single question in this set was scored, and none of them exists in
    any other dataset, stage or probe in this repository.

WHAT IS REUSED, AND WHAT IS NEW
    Reused: the derivation in `test_results/stage_f/generate_questions.py`
    (`build_questions`, `preflight`) and the shared grader. That derivation is
    data-independent graph algebra -- which uniquely-readable edges become
    questions, how direction pairs get sibling ids, how mirror controls get
    their forbidden far ends. It is imported by path and its data globals
    (CONCEPTS/EDGES/OUT_OF_GRAPH_SUBJECTS/NONSENSE_CUE_FREE/DISCLOSED_GUESS)
    are rebound to the weaving graph, so none of its data comes from it.

    New: every template in this file.

WHY THE TEMPLATES ARE NEW RATHER THAN IMPORTED
    Stage E's `ONE_HOP_TEMPLATE["linguistic_maps"]` hard-codes the phrase "in
    Spanish". The aviation set stored English<->Italian pairs, so 40 of its 355
    questions asked for a Spanish translation the graph did not contain: 11.3%
    of that sign-off set were semantically false. Nothing mechanical caught it,
    because a cue-preservation check cannot tell whether a question's CLAIM is
    true of the data.

    Two things change here:
      * the base template for `linguistic_maps` is "What does the {a} translate
        to?", which names no language at all and is therefore true in whichever
        direction the stored pair runs;
      * the only template that DOES name a language is the `linguistic_maps`
        paraphrase, it is restricted to English-side anchors, and
        `preflight()` proves the restriction holds on every generated row via
        `question-names-a-language-the-graph-does-not-use`.

    So the guard against template edits is mechanical rather than
    provenance-based: every template, base and variant, is asserted against the
    LIVE planner, and any template that stops emitting its own declared relation
    refuses the freeze. Editing a template to flatter this graph would have to
    edit the check too.

THE VARIANT SLICE
    Every earlier set asked its one-hop questions with exactly one phrasing per
    relation, so the measured one-hop rate has always been "can the walk find a
    uniquely-stored edge from a known phrasing" -- which flatters the system.
    Nothing ever tested the same edge asked differently.

    So a deterministic slice of the one-hop pool is re-asked with cue-preserving
    rewordings, tagged `variant_kind`:
      * "paraphrase" -- different framing, anchor literal, same literal cue.
      * "plural"     -- anchor morphologically inflected ("beam" -> "beams").

    They are TAGGED rather than given their own category on purpose: an unknown
    category is reported `unscored_category` by the shared grader, and editing
    the shared grader's CATEGORY_TARGETS would retroactively change what the
    frozen Stage E number means. They are one-hop reads of uniquely-stored
    edges, so they are scored as one-hop reads and broken out by tag. The slice
    is removed from the `simple_one_hop` pool so no edge is graded twice.

Usage:
    python test_results/unseen_e2e/generate_questions.py
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


FG = _load_by_path("_unseen_build", STAGE_DIR / "build_graph.py")
B = FG.B

# Stage F's generator does `import build_holdout_graph as B` at import time, so
# registering the rebound builder under that name is what makes the shared
# derivation run against weaving rather than re-deriving the food graph.
sys.modules["build_holdout_graph"] = B
G = _load_by_path("_stage_f_genq", ROOT / "test_results" / "stage_f" / "generate_questions.py")

FROZEN_PATH = STAGE_DIR / "unseen_questions_frozen.json"
VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Template bank, authored for this domain. One entry per canonical relation.
#
# Every template must contain a literal descriptor from
# configs/config_g2p.yaml `extraction.relation_variants` for its OWN relation and
# for no other relation; `_literal_cue_relations` returns every relation that
# matches anywhere in the question, so a stray second descriptor would make the
# question two-hop. Nothing here is verified by inspection -- `preflight()`
# runs the live planner over the templates AND over every generated row.
# ---------------------------------------------------------------------------
ONE_HOP_TEMPLATE: dict[str, str] = {
    "is_a":                "What type of thing is the {a}?",            # "what type of"
    "has_property":        "What is known for the {a}?",                 # "is known for"
    "causes":              "What does the {a} cause?",                   # "cause"
    "caused_by":           "What was the {a} caused by?",               # "was caused by"
    "precedes":            "What comes after the {a}?",                  # "comes after"
    "follows":             "What comes before the {a}?",                 # "comes before"
    "part_of":             "What is the {a} part of?",                   # "part of"
    "synonym":             "What is another word for the {a}?",          # "what is another word for"
    "antonym":             "What is the opposite of the {a}?",           # "opposite of"
    "example_of":          "What is the {a} an example of?",             # "example of"
    "associated_with":     "What is the {a} associated with?",           # "associated with"
    "supports":            "What evidence supports the {a}?",            # "evidence for"
    "contradicts":         "What contradicts the {a}?",                  # "contradicts"
    "temporal_coincident": "What happens at the same time as the {a}?",   # "same time as"
    "spatial_near":        "What is near the {a}?",                      # "what is near"
    # The cue is the literal phrase "translates to", so the verb must appear
    # inflected -- "What does the {a} translate to?" cues NOTHING. And naming
    # no language means it cannot assert one the stored pair lacks, which is the
    # defect the aviation set shipped 40 false questions of.
    "linguistic_maps":     "The {a} translates to what?",                 # "translates to"
}

# Continuation clauses, appended as ", and <clause>" for hop 2 onwards. Each must
# carry its relation's cue, and must not carry any other relation's.
CONT_CLAUSE: dict[str, str] = {
    "is_a":                "what type of thing that is",
    "has_property":        "what is known for that",
    "causes":              "what does that cause",
    "caused_by":           "what was that caused by",
    "precedes":            "what comes after that",
    "follows":             "what comes before that",
    "part_of":             "what that is part of",
    # The synonym cues are all full wh-phrases except "also called", and none of
    # them survives being made into a continuation clause: "what is another word
    # for that" reads as a second standalone question, and "what else is another
    # word for that" contains no literal cue at all. "also called" is the one
    # short cue in the bank.
    "synonym":             "what that is also called",
    "antonym":             "what the opposite of that is",
    # "what that is an example of" would ALSO cue is_a, because "is an" is an
    # is_a descriptor. "one example of" keeps the cue and adds no is_a phrase.
    "example_of":          "what that is one example of",
    "associated_with":     "what that is associated with",
    "supports":            "what evidence supports that",
    "contradicts":         "what contradicts that",
    "temporal_coincident": "what happens at the same time as that",
    "spatial_near":        "what is near that",
    "linguistic_maps":     "what that translates to",
}

# Bind the new bank into the shared derivation's namespace. `build_questions`
# reads ONE_HOP_TEMPLATE and CONT_CLAUSE as ITS module globals, so assigning
# here is what makes the proven construction emit these phrasings.
G.ONE_HOP_TEMPLATE = ONE_HOP_TEMPLATE
G.CONT_CLAUSE = CONT_CLAUSE

# ---------------------------------------------------------------------------
# Cue-preserving rewordings, one bank per relation. Verified mechanically.
#
# Two traps this had to respect:
#   * `causes` cues on the bare word "cause" with the regex
#     `(?<!\w)cause(?!\w)`, so "caused" does NOT match it. That asymmetry is
#     what keeps "is caused by" from also emitting `causes`.
#   * `part_of` carries a reverse-sounding descriptor pair ("what makes up",
#     "made up of") that asks the OPPOSITE direction. No variant below uses it.
# ---------------------------------------------------------------------------
PARAPHRASE_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "is_a": [
        ("paraphrase", "What kind of thing is the {a}?"),              # "what kind of"
        ("plural", "What type of thing is the {plural}?"),              # "what type of"
    ],
    "has_property": [
        ("paraphrase", "The {a} possesses what property?"),             # "possesses"
        ("plural", "What is known for the {plural}?"),                  # "is known for"
    ],
    "causes": [
        ("paraphrase", "What leads to the {a}?"),                       # "what leads to"
        ("plural", "What does the {plural} cause?"),                     # "cause"
    ],
    "caused_by": [
        ("paraphrase", "The {a} is caused by what?"),                   # "is caused by"
        ("plural", "The {plural} are caused by what?"),                 # "are caused by"
    ],
    "precedes": [
        ("paraphrase", "What happens after the {a}?"),                  # "happens after"
        ("plural", "What comes after the {plural}?"),                   # "comes after"
    ],
    "follows": [
        ("paraphrase", "What happened before the {a}?"),                # "happened before"
        ("plural", "What comes before the {plural}?"),                  # "comes before"
    ],
    "part_of": [
        ("paraphrase", "The {a} is contained in what?"),                # "contained in"
        # Plural uses the passive: "What is the beams part of?" is wrong.
        ("plural", "The {plural} are contained in what?"),              # "contained in"
    ],
    "synonym": [
        ("paraphrase", "The {a} is also called what?"),                 # "also called"
        ("plural", "What is another word for the {plural}?"),           # "another word for"
    ],
    "antonym": [
        ("paraphrase", "What is the antonym of the {a}?"),              # "antonym of"
        ("plural", "What is the opposite of the {plural}?"),            # "opposite of"
    ],
    "example_of": [
        # "The {a} is an example of what?" would also cue is_a ("is an").
        ("paraphrase", "The {a} stands as an example of what?"),        # "example of"
        ("plural", "What is the {plural} an example of?"),              # "example of"
    ],
    "associated_with": [
        ("paraphrase", "The {a} is linked to what?"),                   # "linked to"
        ("plural", "What is the {plural} associated with?"),            # "associated with"
    ],
    "supports": [
        ("paraphrase", "What substantiates the {a}?"),                  # "substantiates"
        ("plural", "What evidence supports the {plural}?"),              # "evidence for"
    ],
    "contradicts": [
        ("paraphrase", "What refutes the {a}?"),                        # "refutes"
        ("plural", "What contradicts the {plural}?"),                   # "contradicts"
    ],
    "temporal_coincident": [
        ("paraphrase", "What coincides with the {a}?"),                 # "coincides with"
        ("plural", "What happens at the same time as the {plural}?"),   # "same time as"
    ],
    "spatial_near": [
        ("paraphrase", "What is close to the {a}?"),                    # "what is close to"
        ("plural", "What is near the {plural}?"),                       # "what is near"
    ],
    "linguistic_maps": [
        # The ONLY language-naming template in this file. Restricted to
        # English-side anchors: "how do you say X in French" presupposes X is
        # English, so it would be false of "metier". preflight() enforces that.
        ("paraphrase", "How do you say the {a} in French?"),            # "how do you say"
        ("plural", "The {plural} translates to what?"),                  # "translates to"
    ],
}

# A language-naming template may only name the language the stored pairs
# actually use. The graph stores English<->French, so "French" is the only
# permitted name and the target-language terms are the only anchors it may not
# take. Checked on every generated row.
STORED_LANGUAGE = "French"
TARGET_LANGUAGE_TERMS = frozenset(FG.FRENCH_TERMS)
LANGUAGE_NAMES = ("Spanish", "Italian", "French", "German", "Latin", "English")

# Deterministic slice of the simple_one_hop pool to re-ask with a variant.
# Every STRIDE-th, capped, so the slice spans the sorted edge order rather than
# clustering on whatever sorts first.
PARAPHRASE_STRIDE = 4
PARAPHRASE_MAX = 36


def pluralize(label: str) -> str:
    """Plain English pluralisation of the LAST word of a label.

    Deliberately naive and rule-based: an irregular table would be one more
    hand-authored thing that could flatter this graph.
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
        bank = PARAPHRASE_VARIANTS.get(q["rel"])
        if not bank:
            continue
        anchor = q["anchor"]
        # A language-naming template is only truthful for a source-language
        # anchor. Skipping is deterministic, so the slice stays mechanical.
        if "in French" in bank[0][1] and anchor in TARGET_LANGUAGE_TERMS:
            continue
        # Rotate the variant by position in the chosen slice so every relation
        # contributes all of its rewordings across the set instead of one.
        kind, template = bank[len(variants) % len(bank)]
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
    population["variant_one_hop"] = len(variants)
    population["variant_by_kind"] = {
        kind: sum(1 for v in variants if v["variant_kind"] == kind)
        for kind in ("paraphrase", "plural")
    }
    population["variant_stride"] = PARAPHRASE_STRIDE
    population["variant_max"] = PARAPHRASE_MAX
    population["template_bank"] = "authored for test_results/unseen_e2e"
    return kept + variants, population


# --------------------------------------------------------------------------
def preflight(questions: list[dict], population: dict) -> list[tuple[str, str]]:
    """Refuse to freeze a set that cannot be scored, or cannot be trusted.

    The inherited preflight() already covers identity, category minima,
    graded-edge uniqueness, chain expressibility against the LIVE planner
    (including every 1-hop and mirror row, since they carry a chain),
    mirror-control direction, and out-of-graph absence. What is added here is
    everything that is new in this set.
    """
    problems: list[tuple[str, str]] = list(G.preflight(questions, population))
    labels = set(B.CONCEPTS)

    try:
        for stage in ("stage_a", "stage_b", "stage_c", "stage_d"):
            sys.path.insert(0, str(ROOT / "test_results" / stage))
        from stage_d_runner import load_planner
        planner = load_planner()
    except Exception as exc:  # noqa: BLE001
        problems.append(("could-not-load-planner-for-added-checks", repr(exc)))
        planner = None

    # --- the template bank covers exactly the canonical relations ------------
    for rel in B.CANONICAL_RELATIONS:
        if rel not in ONE_HOP_TEMPLATE:
            problems.append(("relation-has-no-template", rel))
        if rel not in CONT_CLAUSE:
            problems.append(("relation-has-no-continuation-clause", rel))
        if rel not in PARAPHRASE_VARIANTS:
            problems.append(("relation-has-no-paraphrase-variant", rel))
    for rel in set(ONE_HOP_TEMPLATE) | set(CONT_CLAUSE) | set(PARAPHRASE_VARIANTS):
        if rel not in B.CANONICAL_RELATIONS:
            problems.append(("template-for-unknown-relation", rel))

    if planner is None:
        return problems

    # --- every template emits EXACTLY its own relation ----------------------
    # Probed on the templates directly as well as on the generated rows, so a
    # broken template is reported even if the slice happened not to use it. The
    # probe subject is a real label, so any drift onto another relation caused
    # by a label rather than a template would show up too.
    probe = "beam"
    for rel, template in sorted(ONE_HOP_TEMPLATE.items()):
        got = planner._literal_cue_relations(template.format(a=probe))
        if got != [rel]:
            problems.append(("template-does-not-emit-declared-relation",
                             f"{rel}: declared=[{rel}] got={got} :: {template}"))
    # Probed as the SECOND hop behind a `causes` first clause, which is how
    # `multi_hop_question` actually composes these. Probing with `is_a` would be
    # wrong twice over: `_literal_cue_relations` dedupes per relation, so an
    # `is_a` clause yields ["is_a"] and not a doubled pair.
    first = "causes"
    for rel, clause in sorted(CONT_CLAUSE.items()):
        text = f"{ONE_HOP_TEMPLATE[first].format(a=probe).rstrip('?')}, and {clause}"
        want = [first] if rel == first else [first, rel]
        got = planner._literal_cue_relations(text)
        if got != want:
            problems.append(("continuation-clause-does-not-emit-declared-relation",
                             f"{rel}: declared={want} got={got} :: {text}"))
    for rel, bank in sorted(PARAPHRASE_VARIANTS.items()):
        for kind, template in bank:
            got = planner._literal_cue_relations(
                template.format(a=probe, plural=pluralize(probe)))
            if got != [rel]:
                problems.append((
                    "paraphrase-template-does-not-emit-declared-relation",
                    f"{rel}/{kind}: declared=[{rel}] got={got} :: {template}"))

    # --- a question may not name a language the graph does not use ----------
    # This is the check the aviation set did not have and needed. It cannot be
    # caught by cue preservation: "in Spanish" cues `linguistic_maps` perfectly
    # whether or not the stored pairs are Spanish.
    for q in questions:
        named = [name for name in LANGUAGE_NAMES
                 if re.search(r"(?<!\w)" + name + r"(?!\w)", q["question"])]
        if not named:
            continue
        for name in named:
            if name != STORED_LANGUAGE:
                problems.append((
                    "question-names-a-language-the-graph-does-not-use",
                    f"{q['qid']} names {name}, graph stores {STORED_LANGUAGE} :: "
                    f"{q['question']}"))
        anchor = q.get("anchor") or ""
        if anchor in TARGET_LANGUAGE_TERMS:
            problems.append((
                "question-asks-target-language-term-to-translate",
                f"{q['qid']} anchor {anchor!r} is a {STORED_LANGUAGE} term :: "
                f"{q['question']}"))

    # --- honesty rows must carry a REAL cue -------------------------------
    # Otherwise a refusal proves nothing: the engine could be refusing because
    # it found no cue at all, which is the fallback path, not the entity-identity
    # gate this category measures. Exactly one cue -- more would be multi-hop.
    for q in questions:
        if q["category"] != "honesty_out_of_graph":
            continue
        got = planner._literal_cue_relations(q["question"])
        if len(got) != 1:
            problems.append((
                "out-of-graph-question-does-not-carry-exactly-one-cue",
                f"{q['qid']} cues={got} :: {q['question']}"))

    # --- nonsense rows must be cue-FREE ------------------------------------
    # Section 8 fallback triggers on cue ABSENCE, so a nonsense row that did
    # cue a relation would test the walk instead of the fallback.
    for q in questions:
        if q["category"] != "nonsense_fallback" or q.get("expect_guess_disclosed"):
            continue
        got = planner._literal_cue_relations(q["question"])
        if got:
            problems.append(("nonsense-question-is-not-cue-free",
                             f"{q['qid']} cues={got} :: {q['question']}"))

    # --- the disclosed guess must have exactly ONE has_property read -------
    # Otherwise "How wooden is the loom?" is answerable in more than one way and
    # the fallback-disclosure row is not testing what it claims to.
    anchor, text = B.DISCLOSED_GUESS
    if B.reach_from(anchor, "has_property", 1) != {"wooden"}:
        problems.append(("disclosed-guess-has-no-unique-has-property-read",
                         f"{anchor!r} -> {sorted(B.reach_from(anchor, 'has_property', 1))}"))

    # --- generated variant rows are well-formed ---------------------------
    variants = [q for q in questions if q.get("variant_kind") in ("paraphrase", "plural")]
    if len(variants) < 20:
        problems.append(("too-few-variant-questions", f"{len(variants)} < 20"))
    for kind in ("paraphrase", "plural"):
        if not any(q["variant_kind"] == kind for q in variants):
            problems.append((f"no-{kind}-variants-generated", ""))

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
        if v.get("variant_of") and v["category"] != "simple_one_hop":
            problems.append(("variant-row-left-its-pool", v["qid"]))
        # A "plural" row must actually inflect the anchor, and must NOT still
        # contain the singular as a standalone word. WORD-boundary, because
        # "beam" is a substring of "beams" and a plain substring test reports
        # every correctly-inflected variant as uninflected.
        if v["variant_kind"] == "plural":
            if pluralize(v["anchor"]) not in v["question"]:
                problems.append(("plural-variant-does-not-inflect-anchor", v["qid"]))
            head, _, last = v["anchor"].rpartition(" ")
            if re.search(r"(?<!\w)" + re.escape(last) + r"(?!\w)", v["question"]):
                problems.append(("plural-variant-still-contains-singular-anchor", v["qid"]))

    # --- the split must not have double-graded an edge --------------------
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
        "contract": "GLM-X v3.3.2 section 16 -- unseen-data end-to-end evaluation set",
        "generated_by": (
            "test_results/unseen_e2e/generate_questions.py: Stage F derivation run "
            "unchanged against build_graph.py, with an independently authored "
            "ONE_HOP_TEMPLATE / CONT_CLAUSE / PARAPHRASE_VARIANTS bank, every "
            "template asserted against the live planner at freeze time"),
        "held_out": (
            "derived and frozen BEFORE any question in this set was scored; the "
            "graph is the weaving domain, which appears in no other dataset, "
            "stage or probe in this repository"),
        "pass_criteria": {
            "_note": (
                "Pre-registered here, inside the frozen artefact, before scoring. "
                "overall/semantic targets are the shared grader's own "
                "CATEGORY_TARGETS + MECHANISM_TARGETS from "
                "test_results/stage_e/stage_e_runner.py lines 146-170; the "
                "additional hard conditions are listed with their rationale."),
            "overall_min": 0.90,
            "simple_one_hop_min": 0.95,
            "direction_pair_min": 0.90,
            "short_multi_hop_min": 0.75,
            "honesty_out_of_graph_min": 0.90,
            "nonsense_fallback_min": 0.90,
            "mirror_silence_min": 0.90,
            "central_invariant_min": 0.90,
            "fallback_must_trigger_on_every_nonsense_row": True,
            "guess_disclosure_required_on_disclosed_guess_row": True,
            "illegal_mirrors_must_equal": 0,
            "fabricated_nodes_or_relations_must_equal": 0,
            "unaccounted_entities_must_equal": 0,
            "determinism_differing_fields_must_equal": 0,
            "_source": "contract section 16 recommended_poc_test_matrix",
        },
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
        f"{digest}  unseen_questions_frozen.json\n", encoding="utf-8")

    print(f"Froze {len(questions)} questions -> {FROZEN_PATH}")
    print(f"  version {VERSION}")
    print(f"  graph sha256 {spec['graph']['sha256']}")
    print(f"  set   sha256 {digest}")
    for cat, n in spec["counts"].items():
        print(f"    {cat:24} {n}")
    print(f"  variant rows: {population['variant_by_kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())