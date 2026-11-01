#!/usr/bin/env python3
"""Whimsy trial v3b: re-run ONLY the analogy step from v3, with no_think.

v3 flaw found on inspection: 10 of 20 grounded analogy calls returned
EMPTY text — not a mapping refusal, but the known quicklook quirk
(Qwen3 thinking eats the whole budget on evidence-heavy prompts).
Whimsy ideas are reused verbatim from v3; only the analogy step is
re-run, now with no_think=True (fixed in novelty.analogy).
Survivor rule unchanged: NOT no_mapping + guard passed + project
step + novelty > 0.8.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/joshua/llm/voice-loop/multiplex")
from novelty import analogy, analogy_guard, novelty_score  # noqa: E402

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
PROJECT_WORDS = ["strip", "census", "differential", "marker"]


def redo(r):
    try:
        a = analogy(r["whimsy"])
        text = a.get("text", "") if isinstance(a, dict) else str(a)
    except Exception as e:  # noqa: BLE001
        return {**r, "analogy_v3b": "", "error": f"{type(e).__name__}: {e}"}
    low = text.lower()
    passed, hits = analogy_guard(text)
    return {**r, "analogy_v3b": text,
            "no_mapping_v3b": "no mapping" in low,
            "guard_passed_v3b": passed, "guard_hits_v3b": hits,
            "has_project_step_v3b": any(w in low for w in PROJECT_WORDS),
            "novelty_v3b": novelty_score(text, CORPUS)}


def main():
    d = json.load(open("/home/joshua/llm/voice-loop/multiplex/out/"
                        "whimsy-trial-v3-2026-10-28.json"))
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(redo, d["results"]))
    results.sort(key=lambda r: r["i"])
    for r in results:
        print(f"attempt {r['i']} ({r['domain']}): "
              f"no_mapping={r.get('no_mapping_v3b')} "
              f"guard={r.get('guard_passed_v3b')} "
              f"hits={r.get('guard_hits_v3b')} "
              f"project={r.get('has_project_step_v3b')} "
              f"novelty={r.get('novelty_v3b')} "
              f"chars={len(r.get('analogy_v3b', ''))}")
    with open("/home/joshua/llm/voice-loop/multiplex/out/"
               "whimsy-trial-v3b-2026-10-28.json", "w") as f:
        json.dump({"problem": d["problem"], "corpus": CORPUS,
                   "results": results}, f, indent=1)
    print("saved to out/whimsy-trial-v3b-2026-10-28.json")


if __name__ == "__main__":
    main()
