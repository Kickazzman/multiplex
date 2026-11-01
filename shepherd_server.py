#!/usr/bin/env python3
"""Math-Shepherd step-verifier server for the GLaDOS specialist multiplexer.

Holds Math-Shepherd (Renmin University PRM, Mistral-7B, arXiv 2312.08935)
resident on the GPU in 4-bit and scores each reasoning step of a solution.
Complements the Skywork verifier (:8199), which scores whole outputs:
Shepherd scores step by step, so a duplex gate can see WHERE a chain broke.

Scoring follows the official model-card recipe exactly:
  input  = f"{question} {output}" where output steps are separated by the
           step tag 'ки', each step ending with ' ки'
  score(step i) = softmax over candidate tokens {'+', '−'} of the logit at
           the step-tag position -> P('+') in 0..1 (higher = step looks right)

Run with the shepherd env:
    ~/llm/shepherd-env/bin/python shepherd_server.py

API (stdlib HTTP, port 8206, bind 0.0.0.0 - tailnet only in practice):
    GET  /health -> {"ok": ..., "model": ..., "device": "cuda", ...}
    POST /score_steps {"question": "...", "steps": ["...", "..."]}
                      -> {"step_scores": [float...], "min_step": int,
                          "min_score": float, "mean_score": float}
    POST /score_steps {"question": "...", "output": "Step 1: ... ки\n..."}
                      (preformatted output, passed through verbatim)
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = os.environ.get("SHEPHERD_MODEL", "peiyi9979/math-shepherd-mistral-7b-prm")
PORT = int(os.environ.get("SHEPHERD_PORT", "8206"))
STEP_TAG = "ки"
GOOD_TOKEN = "+"
BAD_TOKEN = "-"  # ASCII U+002D: the card code is redacted in rendering, but its published candidate ids [648, 387] pin it - 387 is the ASCII hyphen piece; U+2212 is id 8798 and scores ~never, pinning every step to 1.0

_state = {}


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    candidate_tokens = tok.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:]
    step_tag_id = tok.encode(f"{STEP_TAG}")[-1]
    print(f"[shepherd] candidate_tokens={candidate_tokens} step_tag_id={step_tag_id}",
          flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="cuda")
    mdl.eval()
    _state.update(tok=tok, mdl=mdl, torch=torch,
                  candidate_tokens=candidate_tokens, step_tag_id=step_tag_id,
                  load_seconds=round(time.time() - t0, 1))
    print(f"[shepherd] {MODEL_ID} loaded on cuda (4-bit) in "
          f"{_state['load_seconds']}s", flush=True)


def format_output(steps):
    lines = []
    for i, s in enumerate(steps, 1):
        s = s.strip()
        if s.lower().startswith("step ") and ":" in s[:10]:
            body = s
        else:
            body = f"Step {i}: {s}"
        if not body.endswith(STEP_TAG):
            body = body + f" {STEP_TAG}"
        lines.append(body)
    return "\n".join(lines)


def score_steps(question, steps=None, output=None):
    tok, mdl, torch = _state["tok"], _state["mdl"], _state["torch"]
    if output is None:
        output = format_output(steps or [])
    input_text = f"{question} {output}"
    input_id = tok.encode(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = mdl(input_id).logits[:, :, _state["candidate_tokens"]]
    scores = logits.softmax(dim=-1)[:, :, 0]
    step_scores = scores[input_id == _state["step_tag_id"]]
    vals = [round(float(v), 4) for v in step_scores.tolist()]
    if not vals:
        raise ValueError("no step tags found in input")
    mn = min(vals)
    return {"step_scores": vals, "min_step": vals.index(mn) + 1,
            "min_score": mn,
            "mean_score": round(sum(vals) / len(vals), 4)}


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
                             "device": "cuda", "quant": "4bit-nf4",
                             "load_seconds": _state.get("load_seconds")})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score_steps":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            out = score_steps(req["question"], req.get("steps"),
                              req.get("output"))
            self._send(200, out)
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    load()
    print(f"[shepherd] serving on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
