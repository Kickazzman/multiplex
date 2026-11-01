#!/usr/bin/env python3
"""Whimsy trial, 2026-10-28: 20 attempts, pre-set kill criterion.

Problem (real, unsolved from the campaign): obtain a designable
landmark in the PS5 heap read — a heap object we can place and
find reliably, not just observe.
Each attempt: POST :8212/whimsy, then novelty_score(response, corpus).
Survivor (for this trial, stated up front): novelty > 0.8 AND the
response contains a kill test AND a checkable step. Final human
judgment (Muse) then checks survivors for actual checkability.
Kill criterion: 0 survivors => drop the whimsy seat.
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/joshua/llm/voice-loop/multiplex")
from novelty import novelty_score  # noqa: E402

PROBLEM = ("How can we obtain a designable landmark in the PS5 heap "
           "read — a heap object we can place and find reliably, "
           "not just observe?")
CORPUS = [
    "census P680 strip holds a complete BMP header at offset 484",
    "differential C and D agree under a 32 byte shift",
    "second copy of our own BMP sits 2012 bytes past the first copy",
    "P170 family is 5 fields at offsets 6 14 22 46 166",
    "planted marker found 0 of 12 strips",
    "mechanism per-load duplicate is intermittent 1 of 5",
    "TreeWalker is a cycle target visited 22 times",
    "post-splice survival window under 700 ms 16 of 16",
]


def attempt(i):
    req = urllib.request.Request(
        "http://127.0.0.1:8212/whimsy",
        data=json.dumps({"problem": PROBLEM}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            res = json.loads(r.read() or b"{}")
        text = res.get("text", "") if isinstance(res, dict) else str(res)
    except Exception as e:  # noqa: BLE001
        return {"i": i, "error": f"{type(e).__name__}: {e}"}
    low = text.lower()
    return {"i": i, "text": text,
            "novelty": novelty_score(text, CORPUS),
            "has_kill": "kill" in low,
            "has_check": "check" in low,
            "chars": len(text)}


def main():
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(attempt, range(1, 21)))
    results.sort(key=lambda r: r["i"])
    for r in results:
        if "error" in r:
            print(f"attempt {r['i']}: ERROR {r['error']}")
        else:
            print(f"attempt {r['i']}: novelty={r['novelty']} "
                  f"kill={r['has_kill']} check={r['has_check']} "
                  f"chars={r['chars']}")
    with open("/home/joshua/llm/voice-loop/multiplex/out/"
               "whimsy-trial-2026-10-28.json", "w") as f:
        json.dump({"problem": PROBLEM, "corpus": CORPUS,
                   "results": results}, f, indent=1)
    print("saved to out/whimsy-trial-2026-10-28.json")


if __name__ == "__main__":
    main()
