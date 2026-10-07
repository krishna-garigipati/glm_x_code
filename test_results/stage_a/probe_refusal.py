import json
from pathlib import Path

d = json.loads((Path(__file__).resolve().parent / "stage_a_results.json").read_text(encoding="utf-8"))

print("honest rows: template_matched / path_edges / answer\n")
for r in d["rows"]:
    if r.get("honest_no_relation"):
        print("{} {:<20} template_matched={!s:<6} edges={!s:<18} invariant={}".format(
            r["id"], r["cat"], r["template_matched"], r["path_edges"], r["invariant_ok"]))
        print("      answer:", (r["answer"] or "")[:100])