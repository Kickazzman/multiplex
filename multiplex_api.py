#!/usr/bin/env python3
"""Multiplex API server - how frank's specialists join the duplex.

Perg's design: the giant shared context (blackboard) lives in mrfloppy's
RAM; frank's group duplexes with mrfloppy's group through this API.
Coarse-grained calls only (network latency is real): run a specialist,
run a verified task, score something, read/write the blackboard.

Profiles (2026-10-20, profiles.py): the bench of specialists is
interchangeable per task. POST /profile stages a named cast (VRAM
budget-checked, refused while runs are in flight); GET /profiles and
GET /bench expose every cast/model WITH its VRAM cost and live state -
the on/off switches a control panel (or GLaDOS via vision) toggles.
Every bench model is individually switchable via
POST /bench/<name>/start|stop.

Run on mrfloppy (system python3 is fine - stdlib + local modules):
    nohup python3 multiplex_api.py > api.log 2>&1 &

Endpoints (port 8198, bind 0.0.0.0):
    GET  /health
    GET  /blackboard                 full snapshot
    GET  /blackboard/<section>       one section
    POST /blackboard {section, key, value, mode: write|append}
    POST /run {specialist, prompt, kw}
    POST /verify {prompt, response, criteria}
    POST /run_verified {specialist, prompt, criteria, threshold,
                        max_retries, kw}
    POST /run_re {binary, threshold, max_retries, max_tool_rounds,
                  max_tokens}
    GET  /profiles                   casts + VRAM math + active profile
    GET  /bench                      every model + cost + live state
    POST /profile {name, cast?}      stage a profile (cast = ad-hoc
                                     {mrfloppy: [...], frank: [...]})
    POST /bench/<name>/start|stop    individual model switch
    GET  /mode                       mode-hook status (toggle + index)
    GET  /mode/<task_kind>           resolve one task kind to its mode
                                     (null when unknown or hook off)

Mode-switch dispatch hook (2026-10-30, mode_hook.py):
    POST /run, /run_verified, /run_re accept an optional "task_kind".
    When the hook is ON (mode_hook.enabled exists) and the kind is a
    known index key, the result gains a "mode" block (seat plan,
    escalation rule, tests, avoid list, cost posture). Unknown,
    missing, or malformed kinds resolve to NO mode and behavior is
    exactly current: no defaulting, no inference of kind. When the
    hook is OFF the behavior is exactly current. The hook never
    blocks, rejects, reroutes, or changes seat state; the existing
    generation serialization is preserved exactly (annotation happens
    after fn() returns, outside the generation lock). Toggle needs
    no restart. Disable in one command:
        rm ~/llm/voice-loop/multiplex/mode_hook.enabled

Enforcement (2026-10-30, mode_enforce.py): when the hook toggle AND
the separate enforce toggle are both ON and the kind resolves to a
mode whose seat plan names a registered serial seat, the generation
runs inside a trap-restore window (park residents, load serial seat,
serve, stop it, restore residents) inside the generation lock, and
the annotation gains a "mode"."enforcement" window record. Resident,
rule-layer, and unbuilt/none seat plans are served normally and
annotated (unbuilt/none carry an explicit not-enforceable marker).
Enforcement executes seat plans ONLY -- escalation/decider logic
stays annotation-only. Disable enforcement in one command:
        rm ~/llm/voice-loop/multiplex/mode_enforce.enabled

Decider (2026-10-30, mode_decide.py): for CODE kinds only
    (code-patch-correct, code-which-better), when the separate
    decider toggle is ON, the served fast result is parsed for
    the JEV contract and the lean rule is applied (escalate on
    null OR confidence < 0.8; full-insurance toggle adds
    disagreement escalation). Escalation runs the test/oracle
    calculator from the request's optional "decider" payload;
    the fast result text is never rewritten and every failure
    degrades to a decider-skipped annotation. The decider step
    runs inside the generation lock, after the generation is
    served; its record attaches under result["mode"]["decider"].
    Disable the decider in one command:
        rm ~/llm/voice-loop/multiplex/mode_decide.enabled
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.pop("MULTIPLEX_API", None)  # server must run locally, not proxy

import blackboard  # noqa: E402
import multiplex  # noqa: E402
import profiles  # noqa: E402

try:
    import mode_hook  # noqa: E402  (dispatch hook; must never break the API)
except Exception:
    mode_hook = None

try:
    import mode_enforce  # noqa: E402  (enforcement; must never break API)
except Exception:
    mode_enforce = None

try:
    import mode_decide  # noqa: E402  (JEV decider; must never break API)
except Exception:
    mode_decide = None

PORT = int(os.environ.get("MULTIPLEX_API_PORT", "8198"))


# Generation serialization (2026-10-29, contention measurement):
# concurrent decodes share a fixed ~245 tok/s budget, so one
# decode at a time finishes every job sooner (mean latency
# 1.19s vs 2.04s for five 100-token jobs). This lock queues
# generations; it is held only for the generation call itself,
# never for network I/O, and is bounded by downstream timeouts.
# NOT put on /route: /route reaches generations via loopback
# /run calls, so locking /route itself would deadlock. The
# loopback /run handler acquires this lock, so routed
# generations serialize through it without deadlock.
# Known gaps (not silently covered): /verify stays unlocked
# (scoring path, verifier paused); code-execution gates are
# CPU, not GPU decode, and stay unlocked.
_GENERATION_LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _run_guard(self, kind, specialist=None):
        """Cast check + in-flight tracking for run endpoints.

        Returns an error dict (caller sends it) or None when the run
        may proceed - in which case the caller MUST wrap the dispatch
        in profiles.run_begin()/run_end() via _guarded()."""
        problems = profiles.check_run(kind, specialist)
        if problems:
            return {"error": "cast not staged: " + "; ".join(problems),
                    "kind": "cast", "problems": problems}
        return None

    def _dispatch(self, fn, mode_kind):
        """Enforced dispatch: seat-plan execution when the hook and
        enforce toggles are both ON and the kind resolves; plain fn()
        otherwise. Enforcement never raises for toggle/index problems
        (mode_enforce degrades internally); window failures restore
        the baseline and fall back to a normal serve with a record."""
        if mode_enforce is None:
            return fn(), None
        return mode_enforce.dispatch(fn, mode_kind)

    def _decide(self, result, mode_kind, decider_input):
        """JEV decider step (code kinds only). Runs inside the
        generation lock, after the generation is served. Never
        raises, never mutates the result: mode_decide.apply()
        returns (result, record); the record is attached under
        result["mode"]["decider"] by _guarded when a mode block
        exists. Any failure degrades to no decider record."""
        if mode_decide is None or mode_kind is None:
            return None
        try:
            _result, record = mode_decide.apply(
                result, mode_kind, decider_input)
            return record
        except Exception:
            return None

    def _guarded(self, kind, specialist, fn, serialize=True,
                 mode_kind=None, decider_input=None):
        err = self._run_guard(kind, specialist)
        if err:
            self._send(503, err)
            return
        try:
            profiles.run_begin()
        except profiles.BusyError as e:
            self._send(503, {"error": str(e), "kind": "busy"})
            return
        try:
            if serialize:
                with _GENERATION_LOCK:
                    result, _enf = self._dispatch(fn, mode_kind)
                    _dec = self._decide(result, mode_kind,
                                        decider_input)
            else:
                result, _enf = self._dispatch(fn, mode_kind)
                _dec = None
            # Dispatch hook: annotate AFTER fn() returns, outside the
            # generation lock, so serialization is preserved exactly.
            # No-op when the hook is off, the kind is unknown/missing,
            # or the result is not a dict. Never blocks or reroutes.
            if mode_hook is not None:
                result = mode_hook.annotate(result, mode_kind)
            # Enforcement record: attached only when a mode resolved
            # and enforce was ON; the annotation itself is unchanged.
            if _enf is not None and isinstance(result, dict) \
                    and isinstance(result.get("mode"), dict):
                result["mode"]["enforcement"] = _enf
            if _dec is not None and isinstance(result, dict) \
                    and isinstance(result.get("mode"), dict):
                result["mode"]["decider"] = _dec
            self._send(200, result)
        finally:
            profiles.run_end()

    def do_GET(self):
        try:
            if self.path == "/health":
                health = {"ok": True,
                          "specialists": list(multiplex._SPECIALISTS),
                          **profiles.health_state()}
                # Additive only: existing keys untouched.
                health["mode_hook"] = {
                    "enabled": mode_hook.enabled()
                    if mode_hook is not None else False}
                health["mode_enforce"] = {
                    "enabled": mode_enforce.enabled()
                    if mode_enforce is not None else False}
                _dec_st = mode_decide.status() \
                    if mode_decide is not None else None
                health["mode_decide"] = {
                    "enabled": _dec_st["enabled"]
                    if _dec_st is not None else False,
                    "full_enabled": _dec_st["full_enabled"]
                    if _dec_st is not None else False}
                self._send(200, health)
            elif self.path == "/mode":
                if mode_hook is None:
                    self._send(200, {"enabled": False,
                                     "error": "mode_hook unavailable"})
                else:
                    st = mode_hook.status()
                    if mode_enforce is not None:
                        st["enforce"] = mode_enforce.status()
                    if mode_decide is not None:
                        st["decide"] = mode_decide.status()
                    self._send(200, st)
            elif self.path.startswith("/mode/"):
                kind = unquote(self.path.split("/", 2)[2])
                mode = (mode_hook.resolve(kind)
                        if mode_hook is not None else None)
                self._send(200, {
                    "task_kind": kind,
                    "mode": mode,
                    "enabled": mode_hook.enabled()
                    if mode_hook is not None else False})
            elif self.path == "/blackboard":
                self._send(200, blackboard.snapshot())
            elif self.path.startswith("/blackboard/"):
                section = self.path.split("/", 2)[2]
                self._send(200, blackboard.read(section))
            elif self.path == "/profiles":
                self._send(200, profiles.list_profiles())
            elif self.path == "/bench":
                self._send(200, profiles.list_bench())
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        try:
            req = self._body()
            if self.path == "/blackboard":
                mode = req.get("mode", "write")
                if mode == "append":
                    blackboard.append(req["section"], req["key"],
                                      req["value"])
                else:
                    blackboard.write(req["section"], req["key"],
                                     req["value"])
                self._send(200, {"ok": True})
            elif self.path == "/route":
                import job_router
                self._send(200, job_router.route(
                    req["task_type"], req["prompt"], req.get("kw"),
                    req.get("criteria", ""), req.get("max_tokens")))
            elif self.path == "/run":
                self._guarded("run", req["specialist"], lambda: (
                    multiplex.run_specialist(req["specialist"],
                                             req["prompt"],
                                             **req.get("kw", {}))),
                    mode_kind=req.get("task_kind"),
                    decider_input=req.get("decider"))
            elif self.path == "/verify":
                self._guarded("verify", None, lambda: (
                    {"score": multiplex.verify(req["prompt"],
                                               req["response"],
                                               req.get("criteria", ""))}), serialize=False)
            elif self.path == "/run_verified":
                self._guarded("run_verified", req["specialist"], lambda: (
                    multiplex.run_verified(
                        req["specialist"], req["prompt"],
                        criteria=req.get("criteria", ""),
                        threshold=req.get("threshold"),
                        max_retries=req.get("max_retries", 2),
                        **req.get("kw", {}))),
                    mode_kind=req.get("task_kind"),
                    decider_input=req.get("decider"))
            elif self.path == "/run_re":
                self._guarded("run_re", None, lambda: (
                    multiplex.run_re(
                        req["binary"],
                        threshold=req.get("threshold"),
                        max_retries=req.get("max_retries", 2),
                        max_tool_rounds=req.get("max_tool_rounds", 14),
                        max_tokens=req.get("max_tokens", 2400))),
                    mode_kind=req.get("task_kind"),
                    decider_input=req.get("decider"))
            elif self.path == "/profile":
                res = profiles.stage_profile(req["name"],
                                             req.get("cast"))
                code = 200 if res.get("ok") else {
                    "budget": 400, "unknown": 404,
                    "runs_in_flight": 409, "busy": 409}.get(
                        res.get("kind"), 500)
                self._send(code, res)
            elif self.path.startswith("/bench/"):
                parts = self.path.strip("/").split("/")
                if len(parts) == 3 and parts[2] in ("start", "stop"):
                    res = profiles.bench_action(parts[1], parts[2])
                    code = 200 if res.get("ok") else {
                        "unknown": 404, "runs_in_flight": 409,
                        "busy": 409}.get(res.get("kind"), 500)
                    self._send(code, res)
                else:
                    self._send(404, {"error": "not found"})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(f"[multiplex-api] serving on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
