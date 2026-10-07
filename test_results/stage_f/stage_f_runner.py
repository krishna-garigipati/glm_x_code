"""Stage F: score the HELD-OUT PoC Evaluation Set under the strict contract rules.

GLM-X v3.3.2 section 16 calls the final PoC score a number to be reported on a
frozen evaluation set. Stage E reported one. This scores a SECOND frozen set,
built after Stage E's result was known, in a domain the system has never been
scored on.

WHY THIS FILE IS A BINDING RATHER THAN A COPY
    The grader is imported wholesale from `test_results/stage_e/stage_e_runner.py`
    and only its data globals are rebound. Nothing is reimplemented.

    That matters because the obvious alternative -- copying the runner and
    adapting it -- is how two "independent" evaluations quietly stop measuring
    the same thing. Stage E's runner already wraps the Stage A-D grading helpers
    (central invariant, hop classification, refusal semantics, fabricated-entity
    detection). Stage F runs exactly those, so a Stage F verdict means what a
    Stage E verdict means, and the two sets are directly comparable.

    The Stage E runner's only stage-specific literals are its display name and
    its default output filename, and both are now DERIVED from its own directory
    name. So rebinding `STAGE_DIR` is enough to make its banner and its results
    path say "STAGE F" without editing a line of grading logic.

THE PINNED HASH IS A PROMISE, NOT A LOCK
    `FROZEN_SET_SHA256` below is Stage F's own set, read once at freeze time and
    never edited since. Stage E's pin is untouched and still reads
    `8454dc23...`. Both pins are enforced by the same `preflight()`. If someone
    edits a Stage F question to raise the score, this run hard-fails on hash
    drift instead of quietly reporting a better number.

WHAT "HELD OUT" DOES AND DOES NOT MEAN HERE
    The Stage F graph is a new domain (food, cooking, nutrition) and its
    questions were derived mechanically from it by a generator that imports
    Stage E's template bank unchanged. So no question wording, no gold answer and
    no threshold was chosen with Stage F's score in view.

    It does NOT make the result a general accuracy claim, and the report says so.
    A graph authored by the same person who authored the engine's evaluation
    harness is still a controlled set. What Stage F adds is evidence that the
    Stage E result was not an artefact of having seen that one graph.

Usage:
    python test_results/stage_f/stage_f_runner.py
    python test_results/stage_f/stage_f_runner.py --only simple_one_hop
    python test_results/stage_f/stage_f_runner.py --out .../run1.json
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parents[1]
sys.path.insert(0, str(ROOT))


def _load_stage_e_runner():
    """Load the shared grader by explicit file path.

    Loading by location rather than by name means this keeps working no matter
    what an earlier stage runner has already put in sys.modules.
    """
    src = ROOT / "test_results" / "stage_e" / "stage_e_runner.py"
    spec = importlib.util.spec_from_file_location("_stage_e_runner", src)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the shared grader from {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_stage_e_runner()

# --- rebind ONLY the data locations and display name ------------------------
R.STAGE_DIR = STAGE_DIR
R.ROOT = ROOT
R.DB_PATH = STAGE_DIR / "holdout_eval.db"
R.QUESTIONS_PATH = STAGE_DIR / "holdout_questions_frozen.json"
R.RESULTS_NAME = "stage_f_results.json"
# The shared runner derives these from its own directory at import time, which
# was still stage_e when it was loaded. Recompute them against stage_f so the
# banner and the results JSON name themselves correctly.
R.STAGE_KEY = "F"
R.STAGE = "STAGE F"

# Pinned at freeze time, read once, never updated to make a run pass.
R.FROZEN_SET_SHA256 = (
    "0e0caab131c8894f9514a5691029297f048552ec23b411de33c6219e301a7e43")

if __name__ == "__main__":
    raise SystemExit(R.main())