#!/usr/bin/env python3
"""TRM server for the GLaDOS specialist multiplexer (mrfloppy).

Serves the Tiny Recursive Model (Samsung SAIL Montreal) trained by the
smoke run in ~/llm/trm (see train-smoke.log). TRM is NOT a chat model:
it consumes a tokenized ARC grid + a puzzle identifier and emits a
tokenized output grid through recursive refinement with ACT halting.

HONEST CONTRACT (also documented in multiplex.py):
  POST /predict {"puzzle_name": "<base name, e.g. 5b526a93>"}
      Uses that puzzle's entry from the built dataset's test split
      (its trained identifier - TRM is transductive by design: the
      identifier embedding is learned per puzzle variant). Returns the
      predicted grid plus exact_match / cell_accuracy vs the known
      test output.
  POST /predict {"puzzle": {"train": [{"input": g, "output": g}, ...],
                            "test_input": g}}
      Encodes the raw test input (no translation) with the BLANK
      identifier (id 0) - the model's unseen-puzzle mode. Expect weak
      results; train pairs are accepted for contract completeness but
      the model itself only conditions on the identifier + test input.
  GET /health -> {"ok": ..., "checkpoint": ..., "step": ...}

Checkpoint: newest step_* file under
checkpoints/Arc1concept-aug-100-small-ACT-torch/*/ (env TRM_CKPT to pin).
The arch config is read from the same run dir's all_config.yaml.
Port 8210 (env TRM_PORT). Runs in ~/llm/trm-env; started on demand by
multiplex's trm_gate.sh, reaped after 180s idle.
"""
import glob
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("DISABLE_COMPILE", "1")
TRM_DIR = "/home/joshua/llm/trm"
sys.path.insert(0, TRM_DIR)
os.chdir(TRM_DIR)

DATA_DIR = os.path.join(TRM_DIR, "data/arc1concept-aug-100")
PORT = int(os.environ.get("TRM_PORT", "8210"))
MAX_GRID = 30

_state = {}


def find_checkpoint():
    if os.environ.get("TRM_CKPT"):
        return os.environ["TRM_CKPT"]
    runs = sorted(glob.glob(os.path.join(
        TRM_DIR, "checkpoints", "Arc1concept-aug-100-small-ACT-torch", "*")),
        key=os.path.getmtime)
    best = None
    for run in runs:
        for f in glob.glob(os.path.join(run, "step_*")):
            try:
                step = int(f.rsplit("step_", 1)[1])
            except ValueError:
                continue
            if best is None or step > best[1]:
                best = (f, step, run)
    if best is None:
        raise RuntimeError("no TRM checkpoint found")
    return best  # (path, step, run_dir)


def load():
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from utils.functions import load_model_class

    t0 = time.time()
    ckpt_path, step, run_dir = find_checkpoint()
    cfg = OmegaConf.load(os.path.join(run_dir, "all_config.yaml"))
    arch = OmegaConf.to_container(cfg.arch, resolve=True)
    meta = json.load(open(os.path.join(DATA_DIR, "test", "dataset.json")))

    model_cfg = {k: v for k, v in arch.items() if k not in ("name", "loss")}
    model_cfg.update(
        batch_size=32,
        vocab_size=meta["vocab_size"],
        seq_len=meta["seq_len"],
        num_puzzle_identifiers=meta["num_puzzle_identifiers"],
        causal=False)
    model_cls = load_model_class(arch["name"])
    loss_cls = load_model_class(arch["loss"]["name"])
    loss_kw = {k: v for k, v in arch["loss"].items() if k != "name"}
    with torch.device("cuda"):
        inner = model_cls(model_cfg)
        model = loss_cls(inner, **loss_kw)
    sd = torch.load(ckpt_path, map_location="cuda")
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items()}
    model.load_state_dict(sd, assign=True)
    # Buffers that are not in the checkpoint (persistent=False, e.g.
    # H_init) can stay on CPU despite the cuda device context at build
    # time; normalize every tensor onto the GPU before serving.
    model = model.cuda()
    model.eval()

    test = {n: np.load(os.path.join(DATA_DIR, "test", f"all__{n}.npy"))
            for n in ("inputs", "labels", "puzzle_identifiers",
                      "puzzle_indices")}
    identifiers = json.load(open(os.path.join(DATA_DIR, "identifiers.json")))
    _state.update(model=model, torch=torch, np=np, step=step,
                  ckpt=ckpt_path, test=test, identifiers=identifiers,
                  blank_id=meta["blank_identifier_id"],
                  load_seconds=round(time.time() - t0, 1))
    print(f"[trm] checkpoint step {step} loaded in "
          f"{_state['load_seconds']}s", flush=True)


def encode_grid(grid):
    """Raw ARC grid -> 900 tokens (PAD 0, EOS 1, colors +2), no translation."""
    np = _state["np"]
    g = np.array(grid, dtype=np.int64)
    h, w = g.shape
    out = np.zeros((MAX_GRID, MAX_GRID), dtype=np.int64)
    out[:h, :w] = g + 2
    if h < MAX_GRID:
        out[h, :w] = 1
    if w < MAX_GRID:
        out[:h, w] = 1
    return out.flatten()


def decode_grid(tokens, box=None):
    """900 tokens -> color grid. box=(r0, c0, h, w) crops the content
    area; without a box, the extent is derived from the tokens' own
    nonzero frame (EOS row/col included in the frame, then stripped)."""
    np = _state["np"]
    g = np.array(tokens, dtype=np.int64).reshape(MAX_GRID, MAX_GRID)
    if box is None:
        nz = np.argwhere(g != 0)
        if len(nz) == 0:
            return []
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        h, w = int(r1 - r0), int(c1 - c0)
        box = (int(r0), int(c0), h, w)
    r0, c0, h, w = box
    content = g[r0:r0 + h, c0:c0 + w]
    colors = np.where(content >= 2, content - 2, -1)
    return colors.tolist()


def label_box(label_tokens):
    np = _state["np"]
    g = np.array(label_tokens, dtype=np.int64).reshape(MAX_GRID, MAX_GRID)
    nz = np.argwhere(g != 0)
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return int(r0), int(c0), int(r1 - r0), int(c1 - c0)


def run_batch(inputs, labels, puzzle_ids):
    """ACT loop, mirroring pretrain.evaluate(). inputs/labels: lists of
    900-token arrays. Pads to the model's batch size of 32."""
    torch = _state["torch"]
    np = _state["np"]
    model = _state["model"]
    n = len(inputs)
    pad = 32 - n
    if pad > 0:
        inputs = list(inputs) + [inputs[0]] * pad
        labels = list(labels) + [labels[0]] * pad
        puzzle_ids = list(puzzle_ids) + [puzzle_ids[0]] * pad
    batch = {
        "inputs": torch.tensor(np.stack(inputs), dtype=torch.int32,
                               device="cuda"),
        "labels": torch.tensor(np.stack(labels), dtype=torch.int32,
                               device="cuda"),
        "puzzle_identifiers": torch.tensor(puzzle_ids, dtype=torch.int32,
                                            device="cuda"),
    }
    # pretrain.evaluate() wraps carry creation in a cuda device context:
    # the model's empty_carry() builds tensors with torch.empty() and no
    # explicit device, so the ambient default device decides. Mirror it.
    with torch.inference_mode(), torch.device("cuda"):
        carry = model.initial_carry(batch)
        steps = 0
        while True:
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch, return_keys={"preds"})
            steps += 1
            if all_finish:
                break
    return preds["preds"][:n].cpu().numpy(), steps


def predict_dataset(puzzle_name):
    test, identifiers = _state["test"], _state["identifiers"]
    # find a variant identifier for this base puzzle (prefer the bare one)
    vid = None
    for i, name in enumerate(identifiers):
        if name == puzzle_name:
            vid = i
            break
    if vid is None:
        for i, name in enumerate(identifiers):
            if name.split("|||")[0] == puzzle_name:
                vid = i
                break
    if vid is None:
        return {"error": f"puzzle {puzzle_name} not in dataset identifiers"}
    # puzzle_indices is indexed by the test split's variant ORDINAL p;
    # puzzle_identifiers[p] is that variant's global identifier id.
    np = _state["np"]
    p = int(np.where(test["puzzle_identifiers"] == vid)[0][0]) \
        if (test["puzzle_identifiers"] == vid).any() else None
    if p is None:
        return {"error": f"puzzle {puzzle_name} has no test-split variant"}
    pidx = test["puzzle_indices"]
    rows = list(range(int(pidx[p]), int(pidx[p + 1])))
    if not rows:
        return {"error": f"no test examples for puzzle {puzzle_name}"}
    r = rows[0]
    inputs = test["inputs"][r]
    labels = test["labels"][r]
    preds, steps = run_batch([inputs], [labels], [vid])
    box = label_box(labels)
    pred_grid = decode_grid(preds[0], box)
    true_grid = decode_grid(labels, box)
    total = sum(len(row) for row in true_grid)
    match = sum(1 for y in range(len(true_grid))
                for x in range(len(true_grid[y]))
                if pred_grid[y][x] == true_grid[y][x])
    return {"puzzle_name": puzzle_name, "puzzle_id_used": vid,
            "identifier_name": identifiers[vid],
            "predicted_grid": pred_grid, "expected_grid": true_grid,
            "exact_match": pred_grid == true_grid,
            "cell_accuracy": round(match / max(total, 1), 4),
            "act_steps": steps, "checkpoint_step": _state["step"]}


def predict_raw(puzzle):
    test_input = puzzle["test_input"]
    enc_in = encode_grid(test_input)
    # dummy well-formed labels (same shape, all color 0) - labels feed
    # only the loss head, never the model's input.
    enc_lab = encode_grid([[0] * len(test_input[0])
                           for _ in range(len(test_input))])
    preds, steps = run_batch([enc_in], [enc_lab], [_state["blank_id"]])
    return {"puzzle_id_used": _state["blank_id"],
            "identifier_name": "<blank> (unseen puzzle)",
            "predicted_grid": decode_grid(preds[0]),
            "act_steps": steps, "checkpoint_step": _state["step"]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
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
            self._send(200, {"ok": "model" in _state,
                             "checkpoint": _state.get("ckpt"),
                             "step": _state.get("step"),
                             "load_seconds": _state.get("load_seconds")})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if "puzzle_name" in req:
                out = predict_dataset(req["puzzle_name"])
            elif "puzzle" in req:
                out = predict_raw(req["puzzle"])
            else:
                self._send(400, {"error": "need puzzle_name or puzzle"})
                return
            self._send(200 if "error" not in out else 404, out)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    load()
    print(f"[trm] serving on :{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
