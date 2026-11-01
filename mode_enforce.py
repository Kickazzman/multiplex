#!/usr/bin/env python3
"""Mode-switch ENFORCEMENT (2026-10-30): execute the dispatch hook's
advisory seat plans as trap-restore windows.

The hook (mode_hook.py) is lookup + annotation ONLY: it resolves a
task_kind to a mode and annotates results. Enforcement goes one step
further for modes whose seat plan names a SERIAL seat: it parks the
residents, loads the serial seat, serves the generation through it,
stops it, and restores the residents -- with a window record attached
to the annotation. Everything else is served normally and annotated.

SCOPE BOUNDARY (binding): enforcement executes SEAT PLANS ONLY. It
does NOT build decider logic. Escalation rules (lean null/conf<0.8,
insurance, disagreement handling) stay ANNOTATIONS, exactly as the
hook reports them today. The decider build is a separate later item.

Semantics (implemented exactly):
  1. task_kind + hook ON + enforce ON -> resolve mode.
  2. Seat plan names an ALREADY-RESIDENT seat -> serve normally,
     annotate (no swap).
  3. Seat plan names a SERIAL seat with a registered, proven start
     command -> trap-restore window (baseline -> park -> load ->
     serve -> stop -> restore -> verify baseline). Annotate with
     the window record.
  4. Seat plan is UNBUILT or NONE -> annotate ONLY with an explicit
     "not enforceable" marker. NEVER fake execution, never
     substitute another seat silently.
  5. Unknown/missing/empty/wrong-case kind, hook OFF, or enforce OFF
     -> current behavior, zero change (plain fn(), no record).

Toggles (both stat'ed per call; no restart needed to flip either):
  annotation: ~/llm/voice-loop/multiplex/mode_hook.enabled
      disable: rm ~/llm/voice-loop/multiplex/mode_hook.enabled
  enforcement: ~/llm/voice-loop/multiplex/mode_enforce.enabled
      disable: rm ~/llm/voice-loop/multiplex/mode_enforce.enabled
  Enforcement defaults OFF: the enforce toggle file must not exist
  unless every test has passed. Removing the enforce toggle alone
  restores exactly the hook-only behavior (annotation, no windows).

Safety: any window failure (park/load/serve/restore/verify) restores
the baseline, annotates the failure, and never leaves residents
parked or a serial seat running. A corrupt/missing index means no
enforcement (current behavior). The window runs inside the router's
generation lock, so no other generation can interleave it; the cost
is that queued generations wait for the whole window (park + load +
serve + restore), bounded by downstream timeouts. Concurrent runs in
flight (runs_in_flight != 1) skip the window and serve normally --
parking under a queued generation would strand it.

Shepherd :8206 and verifier :8199 are PAUSED on perg's tap: the park
list contains ONLY the five proven resident units, and enforcement
never touches shepherd/verifier units, args, or state.
"""

import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENFORCE_TOGGLE = HERE / "mode_enforce.enabled"
WINDOW_MARKER = HERE / "mode_enforce.window"

try:
    import mode_hook
except Exception:  # pragma: no cover - hook absent
    mode_hook = None

try:
    import profiles
except Exception:  # pragma: no cover - profiles absent (harness)
    profiles = None

try:
    import multiplex
except Exception:  # pragma: no cover - multiplex absent (harness)
    multiplex = None


# ---------------------------------------------------------------- toggles
def enabled():
    """Enforce toggle: ON iff the toggle file exists. Never raises."""
    try:
        return ENFORCE_TOGGLE.exists()
    except Exception:
        return False


# ------------------------------------------------------- serial seat registry
# ONLY proven seats may be registered here: a mode naming a serial port
# with no registry entry is NOT ENFORCEABLE (never invent a start
# command). Values are the exact proven command from the 2026-10-30
# trial-2 window (dispatch-hook/trial2.sh): binary, model file, args,
# and the free-VRAM gate after parking.
def _home():
    return Path.home()


SERIAL_SEATS = {
    18010: {
        "name": "Qwen3-Coder-30B-A3B abliterated (Huihui)",
        "binary": "llm/llama.cpp-new/build/bin/llama-server",
        "model": (".cache/huggingface/hub/models--mradermacher--"
                  "Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-GGUF/"
                  "snapshots/d7da518983e85529d7e7de7c9ef3901bcf9d8f51/"
                  "Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated."
                  "Q4_K_M.gguf"),
        "args": ["-c", "8192", "-ngl", "99", "--log-disable"],
        "gate_free_mib": 21000,
        "healthy_timeout_s": 300,
        "port_down_timeout_s": 60,
    },
}

# Resident park list, in trial-2 order. Units/ports are read from
# profiles.BENCH at runtime; these NAMES are the allowlist. Shepherd
# and verifier are deliberately absent.
PARK_NAMES = ["agent", "code", "quicklook", "mathwhiz", "draft"]
RESIDENT_PORTS = {8201, 8202, 8203, 8207, 8211}
RESTORE_TIMEOUT_S = 120
RESIDUE_TOLERANCE_MIB = 150

_PORT_RE = re.compile(r":(\d{4,5})\b")


def _port_of(text):
    if not isinstance(text, str):
        return None
    m = _PORT_RE.search(text)
    return int(m.group(1)) if m else None


def _all_ports_of(text):
    """All 4-5 digit numbers in the text (the fleet-baseline string
    lists ports bare: '8201/8202/8203/8207/8211')."""
    if not isinstance(text, str):
        return []
    return [int(m) for m in re.findall(r"\b(\d{4,5})\b", text)]


# ------------------------------------------------------------ classification
def classify(mode):
    """Classify a resolved mode dict into an enforcement plan.

    Returns {"plan": ..., "port": int|None, "seat": str, "detail": str}.
    Plans: "serial" | "resident" | "rule-layer" | "not-enforceable".
    Fail-closed: an unrecognized seat-plan shape is not-enforceable,
    never guessed. Deterministic over the mode dict only.
    """
    seats = (mode or {}).get("seats") or {}
    if not isinstance(seats, dict):
        return {"plan": "not-enforceable", "port": None, "seat": "",
                "detail": "seat plan is not a mapping; refusing to guess"}
    # 1. A "primary" seat governs code modes.
    if "primary" in seats:
        text = seats.get("primary") or ""
        port = _port_of(text)
        if port in SERIAL_SEATS:
            return {"plan": "serial", "port": port, "seat": text,
                    "detail": "primary seat is a registered serial seat"}
        if port in RESIDENT_PORTS:
            return {"plan": "resident", "port": port, "seat": text,
                    "detail": "primary seat is already resident"}
        return {"plan": "not-enforceable", "port": port, "seat": text,
                "detail": "primary seat names a port with no registered "
                          "proven start command; refusing to invent one"}
    # 2. Blocking markers: unbuilt lane / rig-blocked live decoder.
    for key in ("status", "live_status"):
        val = seats.get(key)
        if isinstance(val, str):
            low = val.lower()
            if "unbuilt" in low or "rig-blocked" in low \
                    or "no live decoder" in low:
                return {"plan": "not-enforceable", "port": None,
                        "seat": seats.get("designated_lane", "")
                        or seats.get("interim_reader", ""),
                        "detail": f"{key}: {val}"}
    # 3. Rule layer: no model seat by design.
    position = seats.get("position")
    if isinstance(position, str) and "no model seat" in position.lower():
        return {"plan": "rule-layer", "port": None, "seat": "",
                "detail": "routing is a rule layer with no model seat"}
    # 4. Single designated/proposer/reader seat: resident port?
    for key in ("designated", "proposer", "interim_reader"):
        if key in seats:
            text = seats.get(key) or ""
            port = _port_of(text)
            if port in RESIDENT_PORTS:
                return {"plan": "resident", "port": port, "seat": text,
                        "detail": f"{key} seat is already resident"}
            return {"plan": "not-enforceable", "port": port, "seat": text,
                    "detail": f"{key} seat names a non-resident port "
                             "with no registered proven start command"}
    # 5. Fleet baseline: a "residents" list naming all five resident
    # ports (API :8198 and novelty :8212 are infrastructure, not
    # seats, and are allowed in the same string).
    if "residents" in seats:
        ports = _all_ports_of(seats.get("residents") or "")
        allowed = RESIDENT_PORTS | {8198, 8212}
        if ports and RESIDENT_PORTS <= set(ports) \
                and all(p in allowed for p in ports):
            return {"plan": "resident", "port": None,
                    "seat": seats.get("residents") or "",
                    "detail": "fleet-operation baseline is the resident "
                             "cast; serial programming seats are per-task, "
                             "not per-fleet-operation"}
    # 6. Fail closed.
    return {"plan": "not-enforceable", "port": None, "seat": "",
            "detail": "unrecognized seat-plan shape; refusing to guess"}


# --------------------------------------------------------------------- ops
# Small operations, module-level so the harness can replace them with
# fakes. Real implementations use profiles' proven primitives (the
# same systemctl --user / health-poll calls the bench controls use
# in-process) and the trial-2 serial-server command.

def ops_residents_healthy():
    """(all_healthy, detail): every park-list unit healthy (llama ok)."""
    if profiles is None:
        return False, "profiles unavailable"
    bad = []
    for name in PARK_NAMES:
        entry = profiles.BENCH[name]
        try:
            ok = profiles._healthy(entry)
        except Exception as e:  # noqa: BLE001
            ok = False
            bad.append(f"{name}: {type(e).__name__}")
        if not ok:
            bad.append(name)
    return (not bad), ("all healthy" if not bad else
                       "unhealthy: " + ", ".join(bad))


def ops_port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except Exception:
        return False


def ops_gpu_used_mib():
    if profiles is None:
        return None
    try:
        return profiles._gpu_used_mib("mrfloppy")
    except Exception:
        return None


def ops_gpu_free_mib():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def ops_park():
    """Stop each resident unit, wait for it to go down. Returns a list
    of {name, unit, port, stopped, seconds}."""
    out = []
    for name in PARK_NAMES:
        entry = profiles.BENCH[name]
        profiles._stop_one(name)
        waited = profiles._wait_down(name, 60)
        out.append({"name": name, "unit": entry["unit"],
                    "port": entry["port"],
                    "stopped": waited is not None,
                    "seconds": waited})
    return out


def ops_restore():
    """Start each resident unit, wait for health. Returns a list of
    {name, unit, port, healthy, seconds}."""
    out = []
    for name in PARK_NAMES:
        entry = profiles.BENCH[name]
        profiles._start_one(name)
        waited = profiles._wait_healthy(name, RESTORE_TIMEOUT_S)
        out.append({"name": name, "unit": entry["unit"],
                    "port": entry["port"],
                    "healthy": waited is not None,
                    "seconds": waited})
    return out


def ops_start_serial(spec, port):
    """Start the serial llama-server. Returns the Popen handle."""
    binary = _home() / spec["binary"]
    model = _home() / spec["model"]
    cmd = [str(binary), "-m", str(model),
           "--host", "127.0.0.1", "--port", str(port)] + list(spec["args"])
    return subprocess.Popen(cmd, start_new_session=True,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def ops_serial_healthy(port, timeout_s):
    """Poll :port/health until llama status ok. Returns seconds or None."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=3) as r:
                data = json.loads(r.read().decode() or "{}")
            if data.get("status") == "ok":
                return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(2)
    return None


def ops_stop_serial(proc, port):
    """Terminate the serial server; sweep leftovers; wait for the port
    to go down. Returns {"stopped": bool, "seconds": float|None}."""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
    except Exception:
        pass
    subprocess.run(["pkill", "-f", f"llama-server.*port {port}"],
                   capture_output=True, timeout=15)
    t0 = time.time()
    while time.time() - t0 < 60:
        if not ops_port_open(port):
            return {"stopped": True, "seconds": round(time.time() - t0, 1)}
        time.sleep(1)
    return {"stopped": False, "seconds": None}


def ops_runs_in_flight():
    if profiles is None:
        return -1
    try:
        return profiles.runs_in_flight()
    except Exception:
        return -1


def ops_is_staging():
    if profiles is None:
        return True
    try:
        return profiles.is_staging()
    except Exception:
        return True


# ------------------------------------------------------------- window core
class _WindowFailure(Exception):
    """A window phase failed before serving; caller falls back to a
    normal resident serve and annotates the failure."""


def _restore_all(record):
    """Best-effort full restore: stop any serial server, restart
    residents, verify. Never raises. Records outcome in `record`."""
    stopped = None
    proc = record.get("_proc")
    if proc is not None:
        try:
            stopped = ops_stop_serial(proc, record["port"])
        except Exception as e:  # noqa: BLE001
            stopped = {"stopped": False, "error": f"{type(e).__name__}: {e}"}
        record["window"]["serial_stop"] = stopped
        record["_proc"] = None
    try:
        restored = ops_restore()
    except Exception as e:  # noqa: BLE001
        restored = [{"error": f"{type(e).__name__}: {e}"}]
    record["window"]["restore"] = restored
    record["window"]["restore_ok"] = bool(
        restored and all(r.get("healthy") for r in restored))
    try:
        record["window"]["gpu_used_after_mib"] = ops_gpu_used_mib()
        before = record["window"].get("gpu_used_before_mib")
        after = record["window"]["gpu_used_after_mib"]
        if isinstance(before, int) and isinstance(after, int):
            record["window"]["residue_mib"] = after - before
    except Exception:
        pass
    try:
        WINDOW_MARKER.unlink(missing_ok=True)
    except Exception:
        pass
    return record["window"]["restore_ok"]


def _run_serial_window(fn, task_kind, plan):
    """Execute the trap-restore window. Returns (result, record).
    Raises _WindowFailure for pre-serve failures (caller falls back);
    re-raises serve exceptions after restoring (no double-serve);
    returns the genuine result for post-serve restore problems."""
    port = plan["port"]
    spec = SERIAL_SEATS[port]
    window = {"port": port, "seat": spec["name"],
              "phases": [], "restore_ok": None}
    record = {"enforced": False, "task_kind": task_kind,
              "plan": "serial", "seat": plan["seat"],
              "detail": plan["detail"], "window": window,
              "fallback": None, "_proc": None, "port": port}
    t_start = time.time()

    def phase(name, ok, extra=None):
        entry = {"phase": name, "ok": bool(ok),
                 "seconds": round(time.time() - t_start, 1)}
        if extra:
            entry.update(extra)
        window["phases"].append(entry)
        return ok

    # ---- pre-checks (no state changed yet) ----
    rif = ops_runs_in_flight()
    if rif != 1:
        phase("precheck-runs-in-flight", False, {"runs_in_flight": rif})
        record["fallback"] = {
            "reason": "concurrent runs in flight; refusing to park",
            "served_by": "resident-normal"}
        return fn(), record
    if ops_is_staging():
        phase("precheck-staging", False)
        record["fallback"] = {"reason": "profile staging in progress",
                              "served_by": "resident-normal"}
        return fn(), record
    healthy, detail = ops_residents_healthy()
    if not healthy:
        phase("precheck-residents-healthy", False, {"detail": detail})
        record["fallback"] = {
            "reason": "resident baseline unhealthy; refusing to park",
            "served_by": "resident-normal"}
        return fn(), record
    if ops_port_open(port):
        phase("precheck-serial-port-free", False, {"port": port})
        record["fallback"] = {"reason": "serial port already occupied",
                              "served_by": "resident-normal"}
        return fn(), record
    binary = _home() / spec["binary"]
    model = _home() / spec["model"]
    if not binary.exists() or not model.exists():
        phase("precheck-serial-files", False,
              {"binary_exists": binary.exists(),
               "model_exists": model.exists()})
        record["fallback"] = {"reason": "serial seat binary/model missing",
                              "served_by": "resident-normal"}
        return fn(), record
    window["gpu_used_before_mib"] = ops_gpu_used_mib()
    phase("precheck", True,
          {"gpu_used_before_mib": window["gpu_used_before_mib"]})

    # ---- window: from here on, restore is mandatory ----
    try:
        try:
            WINDOW_MARKER.write_text(json.dumps(
                {"pid": os.getpid(), "task_kind": task_kind,
                 "port": port, "ts": time.time()}))
        except Exception:
            pass
        parked = ops_park()
        window["park"] = parked
        if not all(p.get("stopped") for p in parked):
            phase("park", False)
            raise _WindowFailure("a resident unit did not stop")
        phase("park", True)
        free = ops_gpu_free_mib()
        window["gpu_free_after_park_mib"] = free
        if free is None or free < spec["gate_free_mib"]:
            phase("gate-free-vram", False,
                  {"free_mib": free, "gate_mib": spec["gate_free_mib"]})
            raise _WindowFailure("free-VRAM gate failed after parking")
        phase("gate-free-vram", True,
              {"free_mib": free, "gate_mib": spec["gate_free_mib"]})
        proc = ops_start_serial(spec, port)
        record["_proc"] = proc
        window["serial_pid"] = getattr(proc, "pid", None)
        healthy_s = ops_serial_healthy(port, spec["healthy_timeout_s"])
        window["serial_healthy_seconds"] = healthy_s
        if healthy_s is None:
            phase("load-serial", False)
            raise _WindowFailure("serial seat did not become healthy")
        phase("load-serial", True, {"seconds": healthy_s})
        # ---- serve through the serial seat ----
        if multiplex is None:
            raise _WindowFailure("multiplex override unavailable")
        multiplex.set_enforce_base_url(f"http://127.0.0.1:{port}")
        try:
            result = fn()
        finally:
            multiplex.clear_enforce_base_url()
        phase("serve", True)
        # ---- stop serial + restore ----
        stopped = ops_stop_serial(proc, port)
        record["_proc"] = None
        window["serial_stop"] = stopped
        if not stopped.get("stopped"):
            phase("stop-serial", False, stopped)
        else:
            phase("stop-serial", True, stopped)
        restored = ops_restore()
        window["restore"] = restored
        window["restore_ok"] = bool(
            restored and all(r.get("healthy") for r in restored))
        phase("restore", window["restore_ok"])
        window["gpu_used_after_mib"] = ops_gpu_used_mib()
        before = window.get("gpu_used_before_mib")
        after = window["gpu_used_after_mib"]
        if isinstance(before, int) and isinstance(after, int):
            window["residue_mib"] = after - before
            window["residue_within_tolerance"] = \
                abs(after - before) <= RESIDUE_TOLERANCE_MIB
        healthy2, detail2 = ops_residents_healthy()
        window["verify_residents_healthy"] = healthy2
        window["verify_detail"] = detail2
        phase("verify", healthy2, {"detail": detail2})
        record["enforced"] = True
        record["detail"] = ("serial window executed; result served by "
                            f"{spec['name']} :{port}")
        if not window["restore_ok"] or not healthy2:
            record["detail"] += (" -- WARNING: restore/verify imperfect; "
                                 "see window record")
        return result, record
    except _WindowFailure as e:
        # Pre-serve failure: restore, then fall back to a normal serve.
        _restore_all(record)
        record["fallback"] = {"reason": f"window failed: {e}",
                              "served_by": "resident-normal"}
        record["detail"] = f"window failed ({e}); fell back to resident"
        return fn(), record
    except Exception:
        # Serve (or unexpected) exception: restore, then propagate --
        # no double-serve, no silent substitution.
        _restore_all(record)
        raise
    finally:
        try:
            WINDOW_MARKER.unlink(missing_ok=True)
        except Exception:
            pass


# --------------------------------------------------------------- dispatch
def dispatch(fn, task_kind):
    """Enforced dispatch for one /run-family request.

    Returns (result, record|None). record None means: behave exactly
    as the hook-only router (annotation unchanged, no enforcement
    key). A record is returned only when a mode resolved AND enforce
    is ON; the caller attaches it under result["mode"]["enforcement"].
    Never raises for toggle/index problems; window failures degrade
    to a normal serve with a failure record (see _run_serial_window).
    """
    if mode_hook is None:
        return fn(), None
    if not isinstance(task_kind, str) or not task_kind:
        return fn(), None
    if not mode_hook.enabled():
        return fn(), None
    if not enabled():
        return fn(), None
    try:
        mode = mode_hook.resolve(task_kind)
    except Exception:
        return fn(), None
    if mode is None:
        return fn(), None
    try:
        plan = classify(mode)
    except Exception:
        return fn(), None
    if plan["plan"] == "serial":
        result, record = _run_serial_window(fn, task_kind, plan)
        record.pop("_proc", None)
        record.pop("port", None)
        return result, record
    # Resident / rule-layer / not-enforceable: serve normally, annotate.
    result = fn()
    if plan["plan"] == "resident":
        detail = ("resident seat; served normally, no swap: "
                  + plan["detail"])
    elif plan["plan"] == "rule-layer":
        detail = "rule layer; no model seat to execute: " + plan["detail"]
    else:
        detail = "NOT ENFORCEABLE: " + plan["detail"]
    record = {"enforced": False, "task_kind": task_kind,
              "plan": plan["plan"], "seat": plan["seat"],
              "detail": detail, "window": None, "fallback": None}
    return result, record


def status():
    """Enforcement status for GET /mode and GET /health. Never raises."""
    try:
        plans = {}
        if mode_hook is not None and mode_hook.enabled():
            for kind in ("code-patch-correct", "code-which-better",
                         "fleet-operation", "math-verification",
                         "research-synthesis", "routing-decision",
                         "shape-reconstruction", "signal-decoding"):
                try:
                    mode = mode_hook.resolve(kind)
                    plans[kind] = classify(mode)["plan"] \
                        if mode is not None else None
                except Exception:
                    plans[kind] = None
        return {"enabled": enabled(),
                "toggle": str(ENFORCE_TOGGLE),
                "disable_command": f"rm {ENFORCE_TOGGLE}",
                "serial_seats": sorted(SERIAL_SEATS),
                "plans": plans}
    except Exception as e:  # noqa: BLE001
        return {"enabled": False, "error": f"{type(e).__name__}: {e}"}
