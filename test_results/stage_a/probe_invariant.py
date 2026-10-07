import json
from pathlib import Path

d = json.loads((Path(__file__).resolve().parent / "stage_a_results.json").read_text(encoding="utf-8"))

print("rows with a failed invariant check:\n")
for r in d["rows"]:
    if not r["invariant_ok"]:
        print("{} [{}] {}".format(r["id"], r["cat"], r["question"]))
        print("   answer        :", r["answer"])
        print("   chain         :", r["relation_chain"])
        print("   path_edges    :", r["path_edges"])
        print("   path_labels   :", r["path_labels"])
        print("   anchor        :", r["selected_anchor"])
        print("   honest        :", r["honest_no_relation"],
              "| heur", r["heuristic_used"], "disclosed", r["heuristic_disclosed"])
        print("   checks        :", r["invariant_checks"])
        print("   leaked        :", r.get("leaked_labels"))
        print("   category pass :", r["passed"])
        print()

inv = sum(1 for r in d["rows"] if r["invariant_ok"])
print("invariant: {}/{}".format(inv, len(d["rows"])))

from collections import Counter
per = Counter()
for r in d["rows"]:
    for k, v in r["invariant_checks"].items():
        if not v:
            per[k] += 1
print("failed check tally:", dict(per) or "none")