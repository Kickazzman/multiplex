#!/usr/bin/env python3
"""Verifier server for the GLaDOS specialist multiplexer (duplex checker).

Holds Skywork-Reward-V2 (Llama-3.2-1B) resident on the GPU and scores
(prompt, response) pairs. This is the trained judge in perg's duplex:
workers produce, this model scores, low score = reject and redo.

Run with the multiplex env (CUDA torch):
    ~/llm/multiplex-env/bin/python verifier_server.py

API (stdlib HTTP, port 8199, bind 0.0.0.0 - tailnet only in practice):
    GET  /health                -> {"ok": true, "model": ..., "device": ...}
    POST /score {"pairs": [{"prompt": "...", "response": "..."}, ...]}
                                -> {"scores": [float, ...]}
    POST /score {"prompt": "...", "response": "..."}  (single pair)
                                -> {"score": float}

Scores are raw reward logits (higher = better). Unbounded; calibrate
against known good/bad pairs, do not assume a 0..1 range.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = os.environ.get("VERIFIER_MODEL", "Skywork/Skywork-Reward-V2-Llama-3.2-1B")
PORT = int(os.environ.get("VERIFIER_PORT", "8199"))
MAX_LEN = 3072

_state = {}


def load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="cuda")
    mdl.eval()
    _state.update(tok=tok, mdl=mdl, torch=torch,
                  load_seconds=round(time.time() - t0, 1))
    print(f"[verifier] {MODEL_ID} loaded on cuda in {_state['load_seconds']}s",
          flush=True)


def score_pairs(pairs):
    tok, mdl, torch = _state["tok"], _state["mdl"], _state["torch"]
    scores = []
    for pair in pairs:
        msgs = [{"role": "user", "content": pair["prompt"]},
                {"role": "assistant", "content": pair["response"]}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = pair["prompt"] + "\n" + pair["response"]
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=MAX_LEN).to("cuda")
        with torch.no_grad():
            out = mdl(**enc)
        val = out.logits.reshape(-1)[0].item()
        scores.append(round(float(val), 4))
    return scores


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": "mdl" in _state, "model": MODEL_ID,
                             "device": "cuda",
                             "load_seconds": _state.get("load_seconds")})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if "pairs" in req:
                self._send(200, {"scores": score_pairs(req["pairs"])})
            else:
                s = score_pairs([{"prompt": req["prompt"],
                                  "response": req["response"]}])
                self._send(200, {"score": s[0]})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    load()
    print(f"[verifier] serving on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
