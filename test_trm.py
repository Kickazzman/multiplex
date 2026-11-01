#!/usr/bin/env python3
"""TRM verification: one ARC evaluation puzzle through the harness.

Goes through the multiplex API (specialist "trm", which fires the
on-demand gate), asks for a dataset puzzle by name, and reports the
predicted grid vs the expected grid with exact-match + cell accuracy.
Honest by design: a smoke-trained checkpoint is expected to score low;
whatever it produces is the result.
"""
import json
import sys
import urllib.request

PUZZLE = sys.argv[1] if len(sys.argv) > 1 else "5b526a93"
req = urllib.request.Request(
    "http://127.0.0.1:8198/run",
    data=json.dumps({"specialist": "trm", "prompt": "",
                     "kw": {"puzzle_name": PUZZLE}}).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    out = json.loads(r.read())
if "error" in out:
    print("TRM_TEST ERROR:", out["error"])
    sys.exit(1)
print("puzzle:", out["puzzle_name"], "| identifier:", out["identifier_name"],
      "| checkpoint step:", out["checkpoint_step"], "| ACT steps:",
      out["act_steps"])
print("expected:")
for row in out["expected_grid"]:
    print("  " + "".join(str(c) for c in row))
print("predicted:")
for row in out["predicted_grid"]:
    print("  " + "".join(str(c) if c >= 0 else "?" for c in row))
print("exact_match:", out["exact_match"],
      "| cell_accuracy:", out["cell_accuracy"])
print("TRM_TEST DONE (native behaviour reported, no pass threshold for a "
      "smoke checkpoint)")
