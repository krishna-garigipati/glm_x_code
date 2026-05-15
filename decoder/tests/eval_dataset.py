"""
GLM-X Decoder: Toy Dataset Evaluation
Reads toy_dataset.yaml, runs each scenario through MicroDecoder
in all specified modes, outputs JSON results.
"""
import json
import os
import sys
import time
import traceback
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DECODER import MicroDecoder, WalkResult, Plan, Answer
from DECODER.errors import DecoderError
from DECODER.config_loader import load_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "DECODER", "config_decoder.yaml")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "toy_dataset.yaml")

config = load_config(CONFIG_PATH)
validation_cfg = config["validation"]


def walkresult_from_dict(d: dict) -> WalkResult:
    return WalkResult(
        path=d["path"],
        path_edges=d.get("path_edges", []),
        path_labels=d.get("path_labels", []),
        path_activations=d.get("path_activations", [1.0] * len(d["path"])),
        path_confidences=d.get("path_confidences", [1.0] * max(len(d["path"]) - 1, 0)),
        walk_confidence=d.get("walk_confidence", 1.0),
        final_activation=d.get("final_activation", 0.5),
        steps_taken=d.get("steps_taken", len(d["path"]) - 1),
        timestamp=d.get("timestamp", time.time()),
        intent_sequence_used=d.get("intent_sequence_used", []),
    )


def plan_from_dict(d: dict) -> Plan:
    return Plan(
        intent_sequence=d["intent_sequence"],
        plan_confidence=d.get("plan_confidence", 1.0),
        heuristic_fallback_used=d.get("heuristic_fallback_used", False),
        intent_names=d.get("intent_names", []),
    )


# Load dataset
with open(DATASET_PATH, "r") as f:
    dataset = yaml.safe_load(f)

# Init decoders once
md_template = MicroDecoder(CONFIG_PATH)
md_template._mode = "template"

md_t5 = MicroDecoder(CONFIG_PATH)
md_t5._mode = "t5"

md_hybrid = MicroDecoder(CONFIG_PATH)
md_hybrid._mode = "hybrid"

decoders = {
    "template": md_template,
    "t5": md_t5,
    "hybrid": md_hybrid,
}

# Mode name mapping for output
MODE_LABELS = {
    "template": "template",
    "t5": "t5",
    "hybrid": "hybrid",
    "fallback_only": "template",
}

results = []

for scenario in dataset:
    sid = scenario["scenario_id"]
    desc = scenario["description"]
    modes_to_test = scenario.get("modes", ["template"])

    try:
        walk_obj = walkresult_from_dict(scenario["walk"])
        plan_obj = plan_from_dict(scenario["plan"])
    except Exception as e:
        results.append({
            "scenario_id": sid,
            "description": desc,
            "error": f"Failed to build dataclasses: {e}",
            "mode_results": [],
        })
        continue

    mode_results = []
    for mode in modes_to_test:
        decoder_key = MODE_LABELS.get(mode, mode)
        decoder = decoders.get(decoder_key)
        if decoder is None:
            mode_results.append({
                "mode": mode,
                "status": "SKIP",
                "error": f"Unknown mode: {mode}",
            })
            continue

        start = time.time()
        status = "PASS"
        answer_text = None
        confidence = None
        generation_method = None
        nodes_mentioned = None
        error_msg = None
        wall_clock_ms = None

        try:
            if mode == "fallback_only":
                from DECODER.template_decoder import TemplateDecoder
                _config = load_config(CONFIG_PATH)
                _td = TemplateDecoder(
                    _config["templates"]["definitions"],
                    _config["templates"]["relation_phrases"],
                    _config["templates"]["sentence_starters"],
                    _config["fallback"],
                    _config["validation"],
                )
                text, ok = _td.decode(
                    scenario["walk"]["path_labels"],
                    scenario["walk"]["path_edges"],
                    scenario["plan"]["intent_sequence"],
                )
                if ok:
                    answer_text = text
                    generation_method = "template"
                    confidence = 1.0
                else:
                    answer_text = _td.fallback(
                        scenario["walk"]["path_labels"],
                        scenario["walk"]["path_edges"],
                        scenario["plan"]["intent_sequence"],
                    )
                    generation_method = "fallback"
                    confidence = 0.0
                nodes_mentioned = []
            else:
                answer: Answer = decoder.decode(walk_obj, plan_obj)
                answer_text = answer.text
                confidence = answer.confidence
                generation_method = answer.generation_method
                nodes_mentioned = answer.nodes_mentioned

            elapsed_ms = round((time.time() - start) * 1000, 1)

        except DecoderError as e:
            status = "FAIL"
            error_msg = f"{type(e).__name__}: {e}"
            elapsed_ms = round((time.time() - start) * 1000, 1)
        except Exception as e:
            status = "FAIL"
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            elapsed_ms = round((time.time() - start) * 1000, 1)

        mode_result = {
            "mode": mode,
            "status": status,
            "duration_ms": elapsed_ms,
            "generated_text": answer_text,
            "confidence": confidence,
            "generation_method": generation_method,
            "nodes_mentioned": nodes_mentioned,
        }
        if error_msg:
            mode_result["error"] = error_msg

        mode_results.append(mode_result)

    results.append({
        "scenario_id": sid,
        "description": desc,
        "walk": {
            "path": scenario["walk"]["path"],
            "path_labels": scenario["walk"]["path_labels"],
            "path_edges": scenario["walk"]["path_edges"],
        },
        "plan": {
            "intent_sequence": scenario["plan"]["intent_sequence"],
        },
        "mode_results": mode_results,
    })

# Build report
total_scenarios = len(results)
total_runs = sum(len(r["mode_results"]) for r in results)
total_passes = sum(
    1 for r in results for m in r["mode_results"] if m["status"] == "PASS"
)
total_fails = total_runs - total_passes

report = {
    "evaluation": "GLM-X Decoder Toy Dataset Evaluation",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "config_file": "config_decoder.yaml",
    "validation_params": {
        "min_output_length": validation_cfg["min_output_length"],
        "max_output_length": validation_cfg["max_output_length"],
        "require_node_mention": validation_cfg["require_node_mention"],
        "max_repetitive_ngrams": validation_cfg["max_repetitive_ngrams"],
    },
    "summary": {
        "total_scenarios": total_scenarios,
        "total_runs": total_runs,
        "passed": total_passes,
        "failed": total_fails,
        "pass_rate": round(total_passes / total_runs * 100, 1) if total_runs > 0 else 0,
    },
    "breakdown_by_mode": {},
    "scenarios": results,
}

# Mode breakdown
mode_counts = {}
for r in results:
    for m in r["mode_results"]:
        mode_counts.setdefault(m["mode"], {"total": 0, "passed": 0, "failed": 0})
        mode_counts[m["mode"]]["total"] += 1
        if m["status"] == "PASS":
            mode_counts[m["mode"]]["passed"] += 1
        else:
            mode_counts[m["mode"]]["failed"] += 1

for mode, counts in sorted(mode_counts.items()):
    counts["pass_rate"] = round(counts["passed"] / counts["total"] * 100, 1)
    report["breakdown_by_mode"][mode] = counts

output_path = os.path.join(os.path.dirname(__file__), "dataset_eval_results.json")
with open(output_path, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"Scenarios: {total_scenarios}")
print(f"Runs: {total_runs}")
print(f"Passed: {total_passes}")
print(f"Failed: {total_fails}")
print(f"Rate: {report['summary']['pass_rate']}%")
for mode, counts in sorted(mode_counts.items()):
    print(f"  {mode}: {counts['passed']}/{counts['total']} ({counts['pass_rate']}%)")
print(f"\nResults: {output_path}")
