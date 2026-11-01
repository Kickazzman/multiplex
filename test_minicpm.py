#!/usr/bin/env python3
"""MiniCPM-V verification: transcribe the ground-truth test image."""
import base64
import json
import sys
import urllib.request

IMG = "/home/joshua/llm/voice-loop/multiplex/out/vision-test.png"
TRUTH = [
    "GLADOS VISION CHECK",
    "The cake is a lie.",
    "Aperture Science Test 4217",
    "Columbus Cleveland Cincinnati",
]
# keep the gate's idle stamp fresh
import subprocess
subprocess.run(["touch", "/home/joshua/llm/voice-loop/multiplex/.vision-stamp"])

with open(IMG, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
payload = {
    "model": "minicpm-v",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text",
             "text": "Transcribe all of the text in this image exactly, line by line. Output only the text."},
        ],
    }],
    "max_tokens": 300,
    "temperature": 0.0,
}
req = urllib.request.Request(
    "http://127.0.0.1:8204/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.loads(r.read())
text = out["choices"][0]["message"]["content"].strip()
print("MODEL OUTPUT:")
print(text)
print("---")
hits = {line: (line.lower() in text.lower()) for line in TRUTH}
for line, ok in hits.items():
    print(("MATCH " if ok else "MISS  ") + line)
n = sum(hits.values())
print(f"VISION_TEST {n}/4 lines matched ->", "PASS" if n == 4 else ("PARTIAL" if n >= 3 else "FAIL"))
sys.exit(0 if n >= 3 else 1)
