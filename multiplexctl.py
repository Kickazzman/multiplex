#!/usr/bin/env python3
"""multiplexctl - CLI for the multiplex profile manager.

Talks to the API on :8198 (override with MULTIPLEX_API_URL), so all
the API-side rules apply (budget checks, run-in-flight refusals).

    multiplexctl.py profiles            casts + VRAM math + active
    multiplexctl.py bench               every model + cost + state
    multiplexctl.py stage <profile>     stage a cast
    multiplexctl.py start <model>       individual switch on
    multiplexctl.py stop <model>        individual switch off
    multiplexctl.py status              health + active profile
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("MULTIPLEX_API_URL", "http://127.0.0.1:8198")


def call(method, path, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type":
                                          "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": str(e)}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "profiles":
        code, out = call("GET", "/profiles")
    elif cmd == "bench":
        code, out = call("GET", "/bench")
    elif cmd == "status":
        code, out = call("GET", "/health")
    elif cmd == "stage" and len(argv) == 3:
        code, out = call("POST", "/profile", {"name": argv[2]})
    elif cmd in ("start", "stop") and len(argv) == 3:
        code, out = call("POST", f"/bench/{argv[2]}/{cmd}")
    else:
        print(__doc__)
        return 2
    print(json.dumps(out, indent=1, default=str))
    return 0 if code == 200 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
