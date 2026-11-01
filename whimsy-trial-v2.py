#!/usr/bin/env python3
"""Whimsy trial v2, 2026-10-28: varied sampling + analogy translation.

Fixes v1's flaw (deterministic quicklook repeated one answer 19 times):
varied sampling = 20 DISTINCT angle prompts (move + unrelated domain),
so even a deterministic model must produce distinct outputs.
New step per attempt: analogy mapper — translate the whimsical idea
onto the actual project process (strips, BMP, displacement, census,
differential, planted marker, P170, TreeWalker). If no faithful
mapping exists, the mapper must say NO MAPPING.
Problem: designable landmark in the PS5 heap read.
Survivor (stated up front): unique whimsy + analogy NOT "NO MAPPING"
+ analogy names a project step (strip/BMP/displacement/census/
differential/marker) with a measurement + novelty of the translated
idea > 0.8 vs corpus. Muse then judges survivors for real checkability.
Kill criterion: 0 survivors => drop the whimsy seat.
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/joshua/llm/voice-loop/multiplex")
from novelty import WHIMSY_PROMPT, novelty_score, run_specialist  # noqa: E402

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
ANALOGY_PROMPT = (
    "You are an analogy mapper. Take the whimsical idea below and map\n"
    "each of its elements onto the actual PS5 heap project process:\n"
    "510-byte strips, BMP header at offset 484, displacement, census,\n"
    "differential comparison, planted marker, P170 family, TreeWalker.\n"
    "Output a translated idea in project terms, with a checkable\n"
    "measurement step and a kill test. If NO faithful mapping exists,\n"
    "say exactly NO MAPPING and stop.\n"
    "Whimsical idea: {whimsy}"
)
PROJECT_WORDS = ["strip", "bmp", "displacement", "census",
                 "differential", "marker", "p170", "treewalker"]


def attempt(args):
    i, (move, domain) = args
    prompt = (WHIMSY_PROMPT.format(problem=PROBLEM)
              + f"\nWhimsy move for this attempt: {move}. "
              f"Combine with: {domain}.")
    try:
        w = run_specialist("quicklook", prompt)
        whimsy_text = w.get("text", "") if isinstance(w, dict) else str(w)
        a = run_specialist("quicklook",
                           ANALOGY_PROMPT.format(whimsy=whimsy_text[:3000]))
        analogy_text = a.get("text", "") if isinstance(a, dict) else str(a)
    except Exception as e:  # noqa: BLE001
        return {"i": i, "move": move, "domain": domain,
                "error": f"{type(e).__name__}: {e}"}
    low = analogy_text.lower()
    has_project = any(w in low for w in PROJECT_WORDS)
    return {"i": i, "move": move, "domain": domain,
            "whimsy": whimsy_text, "analogy": analogy_text,
            "no_mapping": "no mapping" in low,
            "has_project_step": has_project,
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
                  f"project_step={r['has_project_step']} "
                  f"novelty={r['novelty']}")
    with open("/home/joshua/llm/voice-loop/multiplex/out/"
               "whimsy-trial-v2-2026-10-28.json", "w") as f:
        json.dump({"problem": PROBLEM, "corpus": CORPUS,
                   "results": results}, f, indent=1)
    print("saved to out/whimsy-trial-v2-2026-10-28.json")


if __name__ == "__main__":
    main()
