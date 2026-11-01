#!/usr/bin/env python3
"""hal_dispatch — combined-multiplexer trial router (mrfloppy side).

Deterministic routing, per the federation design:
  - task_type "deep"  -> hal / strata via the blackboard gateway, ONLY if
                         the gateway reports the lane live (his toggle).
  - anything else     -> local multiplex API :8198 (never leaves the bench).
  - gateway down / lane off -> deep tasks QUEUE (reported, never silently
                         rerouted to a weaker local seat).

Secrets discipline:
  - The gateway bearer token is read from ~/llm/blackboard/gateway-client.env.
  - His Strata API key is NEVER stored here. It is presented per call as a
    header through the gateway (the gateway stores no lane secrets). It is
    taken from env HAL_STRATA_KEY if set, else fetched in-memory at runtime
    from hal's own running Strata process arguments over ssh (the brothers'
    reciprocal access), held only in this process, never written to disk
    or to the log.
Logging: every cross-bench call appends one JSON line to the blackboard
derived trial log (timestamp, lane, task_type, status, elapsed, prompt
sha256, response chars). The gateway logs its side on hal.
Correctness: callers get the raw text back; any verdict must be re-derived
on this bench (a final gate is a re-derivation, not a judgment).
"""
import argparse, hashlib, json, os, subprocess, sys, time
import urllib.request, urllib.error

GATEWAY = "http://[REDACTED-IP]:8299"
HAL_SSH = "cobrajet@[REDACTED-IP]"
ENV_FILE = os.path.expanduser("~/llm/blackboard/gateway-client.env")
LOG_FILE = os.path.expanduser("~/llm/blackboard/derived/trial-log-2026-10-22.jsonl")
LOCAL_API = "http://127.0.0.1:8198"

def gateway_token():
    for line in open(ENV_FILE):
        line = line.strip()
        if line.startswith("GATEWAY_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("GATEWAY_TOKEN not found in gateway-client.env")

def strata_key():
    k = os.environ.get("HAL_STRATA_KEY", "").strip()
    if k:
        return k
    # In-memory fetch from hal's running Strata process args; never persisted.
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=8", HAL_SSH,
         "ps aux | grep strata | grep server.py | grep -v grep"],
        capture_output=True, text=True, timeout=30).stdout
    for tok in out.split():
        pass
    parts = out.split("--api-key")
    if len(parts) < 2:
        raise SystemExit("could not obtain strata key at runtime (no --api-key in process args)")
    return parts[1].split()[0].strip()

def gw(path, payload=None, timeout=190):
    tok = gateway_token()
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(GATEWAY + path, data=data,
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
        method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def capabilities():
    return gw("/capabilities")

def lane_live(name):
    try:
        caps = capabilities()
    except Exception as e:
        return False, f"gateway unreachable: {e}"
    for lane in caps.get("lanes", []):
        if lane.get("lane") == name:
            return bool(lane.get("live")), str(lane)
    return False, "lane not advertised"

def log_call(task_type, lane, status, elapsed, prompt, resp_chars):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "bench_from": "mrfloppy",
           "lane": lane, "task_type": task_type, "status": status,
           "elapsed_s": round(elapsed, 2),
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
           "response_chars": resp_chars}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

def dispatch_deep(prompt, max_tokens=1024, temperature=0.0):
    live, why = lane_live("strata")
    if not live:
        return {"routed": "queued", "reason": f"hal/strata not live ({why})"}
    key = strata_key()
    body = {"model": "qwen3.8-flash-next-iq3_s",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens,
            "enable_thinking": False}
    t0 = time.time()
    out = gw("/call", {"lane": "strata", "path": "/v1/chat/completions",
                       "method": "POST", "body": body,
                       "headers": {"Authorization": f"Bearer {key}"}})
    elapsed = time.time() - t0
    status = out.get("status")
    raw = out.get("body", "")
    text = ""
    try:
        parsed = json.loads(raw)
        msg = parsed["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or ""
    except Exception:
        text = raw
    log_call("deep", "hal/strata", status, elapsed, prompt, len(text))
    return {"routed": "hal/strata", "status": status, "elapsed_s": round(elapsed, 2),
            "text": text}

def dispatch_local(specialist, prompt):
    data = json.dumps({"specialist": specialist, "prompt": prompt}).encode()
    req = urllib.request.Request(LOCAL_API + "/run", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    out["routed"] = f"local/{specialist}"
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out

def route(task_type, prompt, specialist="quicklook", max_tokens=1024):
    """The deterministic rule: deep -> hal strata (if his toggle is on)."""
    if task_type == "deep":
        return dispatch_deep(prompt, max_tokens=max_tokens)
    return dispatch_local(specialist, prompt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-type", default="deep")
    ap.add_argument("--specialist", default="quicklook")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--prompt", required=True)
    a = ap.parse_args()
    print(json.dumps(route(a.task_type, a.prompt, a.specialist, a.max_tokens), indent=1)[:8000])
