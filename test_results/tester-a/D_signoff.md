# Tester A - D Sign-off: Walker & Decoder

**Dataset:** nature_weather_small.db

| Checklist (Section 6.3) | Status | Evidence |
|-------------------------|--------|----------|
| D1 No "Therefore," prefix | ✅ | Output never starts with "Therefore," or LLM-style text |
| D2 Template coverage | ✅ | template_matched=True for all 10 in-graph golden questions |
| D3 Node mention | ✅ | Answer mentions at least one node from walk path |
| D4 Honest fallback | ⏳ | Not yet tested (missing relation edge case) |
| D5 Reject patterns | ✅ | None of "intent=", "->", "generate:" appear in output |
| D6 Length bounds | ✅ | Answer length 5-500 chars |
| D7 Path quality | ✅ | Walk path uses chain relations; n_walk_steps ≥ 1 |
| D8 Determinism sanity | ✅ | Same question + seeded graph: stable chain across runs |

**Signed:** tester-a
**Date:** 2026-09-18