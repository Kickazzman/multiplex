#!/usr/bin/env python3
"""Kyutai STT verification: transcribe the F5 GLaDOS wav from the v2 tests.

The wav lives on mrfloppy; this test runs wherever invoked, pulls the file
over the tailnet via the multiplex host if needed, uploads it base64 to the
STT service on frank (:8209), and scores a rough word error rate against
the known text (from test_easy2.py of the v2 suite).
"""
import base64
import json
import sys
import urllib.request

TRUTH = ("Hello. It has been a long time. I have been testing while you "
         "were gone, and the results are... interesting.")
WAV_LOCAL = "/home/joshua/llm/voice-loop/multiplex/out/f5-20261020-075151.wav"


def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    # strip punctuation for a fair rough WER
    import re
    r = [re.sub(r"[^a-z0-9']", "", w) for w in r]
    h = [re.sub(r"[^a-z0-9']", "", w) for w in h]
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(len(r), 1)


def main():
    with open(WAV_LOCAL, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    req = urllib.request.Request(
        "http://[REDACTED-IP]:8209/transcribe",
        data=json.dumps({"audio_b64": b64}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    hyp = out["text"]
    print("TRUTH:", TRUTH)
    print("HYP  :", hyp)
    w = wer(TRUTH, hyp)
    print(f"rough WER: {w:.3f}  (service took {out.get('seconds')}s)")
    print("KYUTAI_TEST", "PASS" if w <= 0.25 else "FAIL")
    sys.exit(0 if w <= 0.25 else 1)


if __name__ == "__main__":
    main()
