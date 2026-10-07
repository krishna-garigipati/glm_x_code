"""POC SHOWCASE DEMO: score the frozen pottery set.

This is the one and only scored run for the showcase demo. Every question
exercises the full pipeline: Encode -> Spread/Resonance -> Plan -> Walk -> Decode.

WHY THIS IS A BINDING RATHER THAN A COPY
    The grader is imported wholesale from `test_results/stage_e/stage_e_runner.py`
    and only its data globals are rebound. Nothing is reimplemented.

    That matters because copying the runner is how two "independent" evaluations
    quietly stop measuring the same thing. The Stage E runner wraps the Stage A-D
    grading helpers (central invariant, hop classification, refusal semantics,
    fabricated-entity detection), so this run exercises exactly those, and the
    numbers mean what Stage E numbers mean.

    The one shared-grader change made for any variant row is the `variant_kind`
    field in the per-row projection, which is additive: it reads an optional key
    and adds a column. It cannot change any score.

ARCHITECTURE CONSTRAINTS THIS RUN MAY NOT VIOLATE
    Version 3.3.2 strict. No cosine in the main Walker scoring. `relation_chain`
    remains the operative signal. Inference deterministic and frozen by default.
    No invented relations. The grader's refusal, hop-provenance and
    fabricated-entity checks are what measure those, and they are unmodified.

TWO INDEPENDENT PINS, BOTH WRITTEN BEFORE ANY QUESTION WAS SCORED
    `FROZEN_SET_SHA256` is handed to the shared grader, which compares it against
    the question file's own digest. `FROZEN_GRAPH_SHA256` is checked here, against
    the database, before the grader is even entered.

    The second pin exists because the graph hash also lives INSIDE the question
    file. If only the set were pinned, an edited set could carry a matching inner
    graph hash and verify itself. Pinning the graph here, in a file the freeze did
    not produce, closes that loop.

    Both constants were read once at freeze time. Changing either to make a run
    pass would defeat the purpose of having them.

Usage:
    python test_results/showcase_demo/run_eval.py
    python test_results/showcase_demo/run_eval.py --out .../rerun.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
ROOT = STAGE_DIR.parents[1]
sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_shared_runner():
    """Load the shared grader by explicit file path.

    Loading by location rather than by name means this keeps working no matter
    what an earlier runner has already put in sys.modules.
    """
    src = ROOT / "test_results" / "stage_e" / "stage_e_runner.py"
    spec = importlib.util.spec_from_file_location("_stage_e_runner", src)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the shared grader from {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_shared_runner()

# --- rebind ONLY the data locations and display name ------------------------
R.STAGE_DIR = STAGE_DIR
R.ROOT = ROOT
R.DB_PATH = STAGE_DIR / "showcase_eval.db"
R.QUESTIONS_PATH = STAGE_DIR / "showcase_questions_frozen.json"
R.RESULTS_NAME = "showcase_results.json"
# The shared runner derives these from its own directory at import time, which was
# still stage_e when it was loaded. Recompute them so the banner and the results
# JSON name themselves correctly.
R.STAGE_KEY = "SHOWCASE"
R.STAGE = "POC SHOWCASE DEMO -- POTTERY AND CERAMICS"

# Read once at freeze time. Never updated to make a run pass.
# RUN 2: run 1 scored with a stale nonsense list (an omitted
# B.NONSENSE_CUE_FREE binding in build_graph.py), so its fallback category
# re-used the stage_f/aviation/weaving questions instead of this domain's. Both
# runs and both digests are disclosed in POC_SHOWCASE_REPORT.md.
R.FROZEN_SET_SHA256 = (
    "5ff31283180093d5ee956d8d5855009de3e0fc153f2d621463e3dc42f96d08da")
FROZEN_GRAPH_SHA256 = (
    "e465e0707ae9b0f78dff617f365cd03ad15dbd274af7ee5c9af1d86cc7378d2e")


def _verify_pins() -> list[str]:
    problems = []
    for path, expected, what in (
        (R.QUESTIONS_PATH, R.FROZEN_SET_SHA256, "question set"),
        (R.DB_PATH, FROZEN_GRAPH_SHA256, "graph"),
    ):
        if not path.exists():
            problems.append(f"{what} missing: {path}")
            continue
        got = _sha256(path)
        if got != expected:
            problems.append(
                f"{what} hash drift: on disk {got}, frozen {expected}")
    return problems


if __name__ == "__main__":
    drift = _verify_pins()
    if drift:
        print("REFUSING TO SCORE -- frozen artefact drift:")
        for line in drift:
            print(f"  {line}")
        raise SystemExit(1)
    raise SystemExit(R.main())
