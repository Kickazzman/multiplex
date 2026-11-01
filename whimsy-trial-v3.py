#!/usr/bin/env python3
"""Whimsy trial v3, 2026-10-28: grounded analogy mapper re-test.

Fix vs v2: the analogy mapper may use ONLY the recorded terms in
novelty.RECORDED_TERMS, with their definitions as given. A
deterministic guard rejects any mapping citing external docs, SDKs,
allocators, or invented tools. Same 20 angles and same problem as v2.
Survivor (stated up front): unique whimsy + analogy NOT "NO MAPPING"
+ guard passed + analogy names a recorded project step (strip/census/
differential/marker) with a measurement + novelty of the translated
idea > 0.8 vs corpus. Muse then judges survivors for real checkability.
Kill criterion: 0 survivors => drop the whimsy seat.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/joshua/llm/voice-loop/multiplex")
from novelty import (WHIMSY_PROMPT, analogy, analogy_guard,  # noqa: E402
                     novelty_score, run_specialist)

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
ANGLES = [
    ("invert the process", "music"),
    ("exaggerate it a thousandfold", "gardening"),
    ("shrink it to a single atom", "cooking"),
    ("run it backwards in time", "astronomy"),
    ("make it a game with rules", "fishing"),
    ("make it a living creature", "weather"),
    ("make it a city with streets", "beekeeping"),
    ("make it a river with currents", "knitting"),
    ("make it a library with a librarian", "sailing"),
    ("make it a market with traders", "blacksmithing"),
    ("make it a dance", "cartography"),
    ("make it a war with spies", "baking"),
    ("make it a garden that grows", "lighthouse keeping"),
    ("make it a machine with gears", "pottery"),
    ("make it a dream", "mountaineering"),
    ("make it a courtroom trial", "glassblowing"),
    ("make it a train network", "perfumery"),
    ("make it a coral reef", "watchmaking"),
    ("make it a school for children", "falconry"),
    ("make it a volcano", "calligraphy"),
]
PROJECT_WORDS = ["strip", "census", "differential", "marker"]


def attempt(args):
    i, (move, domain) = args
    prompt = (WHIMSY_PROMPT.format(problem=PROBLEM)
              + f"\nWhimsy move for this attempt: {move}. "
              f"Combine with: {domain}.")
    try:
        w = run_specialist("quicklook", prompt)
        whimsy_text = w.get("text", "") if isinstance(w, dict) else str(w)
        a = analogy(whimsy_text)
        analogy_text = a.get("text", "") if isinstance(a, dict) else str(a)
    except Exception as e:  # noqa: BLE001
        return {"i": i, "move": move, "domain": domain,
                "error": f"{type(e).__name__}: {e}"}
    low = analogy_text.lower()
    passed, hits = analogy_guard(analogy_text)
    return {"i": i, "move": move, "domain": domain,
            "whimsy": whimsy_text, "analogy": analogy_text,
            "no_mapping": "no mapping" in low,
            "guard_passed": passed, "guard_hits": hits,
            "has_project_step": any(w in low for w in PROJECT_WORDS),
            "novelty": novelty_score(analogy_text, CORPUS)}


def main():
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(attempt, list(enumerate(ANGLES, 1))))
    results.sort(key=lambda r: r["i"])
    for r in results:
        if "error" in r:
            print(f"attempt {r['i']} ({r['domain']}): ERROR {r['error']}")
        else:
            print(f"attempt {r['i']} ({r['domain']}): "
                  f"no_mapping={r['no_mapping']} "
                  f"guard={r['guard_passed']} hits={r['guard_hits']} "
                  f"project_step={r['has_project_step']} "
                  f"novelty={r['novelty']}")
    with open("/home/joshua/llm/voice-loop/multiplex/out/"
               "whimsy-trial-v3-2026-10-28.json", "w") as f:
        json.dump({"problem": PROBLEM, "corpus": CORPUS,
                   "results": results}, f, indent=1)
    print("saved to out/whimsy-trial-v3-2026-10-28.json")


if __name__ == "__main__":
    main()
