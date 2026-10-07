"""Print the Stage A mechanism matrix and every failure with full trace."""
import json
import sys
from pathlib import Path

p = Path(__file__).resolve().parent / "stage_a_results.json"
d = json.loads(p.read_text(encoding="utf-8"))

print("rows", d["n_questions"])
print("overall", d["overall"])
inv = {k: v for k, v in d["central_invariant"].items() if k != "checks"}
print("invariant", inv)
print()
for c, m in d["mechanism_matrix"].items():
    flag = "OK  " if m["meets_target"] else "MISS"
    print("{:22} {:>3}/{:<3} {:>7.1%}  target={} min={} -> {}".format(
        c, m["passed"], m["n"], m["rate"], m["target"], m.get("minimum"), flag))

fails = [r for r in d["rows"] if not r["passed"]]
print("\nFAILURES:", len(fails))
for r in fails:
    print("\n{} [{}] {}".format(r["id"], r["cat"], r["question"]))
    print("   ans:    ", r["answer"])
    print("   anchor: ", r["selected_anchor"], "| topsim", r["entity_top_sim"],
          "| notfound", r["entity_not_found"])
    print("   chain:  ", r["relation_chain"], "-> walk", r["path_edges"])
    print("   labels: ", r["path_labels"])
    print("   heur:   ", r["heuristic_used"], "disclosed", r["heuristic_disclosed"],
          "| honest", r["honest_no_relation"])
    print("   inv:    ", r["invariant_ok"], r["invariant_checks"])
    print("   why:    ", r["fail_reason"])