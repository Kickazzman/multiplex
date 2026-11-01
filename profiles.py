#!/usr/bin/env python3
"""Profile manager for the GLaDOS specialist multiplexer.

Perg's design step (2026-10-20): the tiny specialists become
INTERCHANGEABLE per task. A *bench* registry describes every servable
model (where it lives, how it is started/stopped, what it costs in
VRAM - measured, not guessed). A *profile* is a named cast: the set of
bench models staged resident for a class of task, with a per-host VRAM
budget check. Staging stops what the cast does not need, starts what
it does, waits for health, verifies VRAM against nvidia-smi, and
records the active profile on the blackboard.

Perg's standing rules honoured here:
- Every service has an on/off switch WITH ITS COST VISIBLE:
  GET /bench and GET /profiles report VRAM per model/profile, and
  every bench model is individually start/stoppable.
- Staging never fights a running job: if a verified run is in flight
  (tracked by multiplex_api via run_begin/run_end), staging and bench
  stops are refused with 409-style errors.
- The hardening gates/defences in multiplex.py are untouched; this
  module only decides WHICH models are resident. Cast enforcement at
  the API layer turns a call to an unstaged specialist into a clean
  503 ("not staged in profile X") instead of a connection error.

Managers per bench entry:
- systemd   : systemctl --user unit (local, or over ssh on frank)
- docker    : the infinity container on frank (embed)
- process   : the Skywork verifier (nohup process, no unit exists)
- percall   : F5-TTS - a CLI subprocess per call, no resident process;
              its switch is an enable/disable flag in the state file.

State: .profile-state.json next to this file {active_profile,
disabled[], updated_ts}. Pin files (.vision-pinned / .trm-pinned)
tell the 180s reapers to leave a profile-pinned model alone.
"""
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / ".profile-state.json"
FRANK_HOST = os.environ.get("FRANK_HOST", "[REDACTED-IP]")
FRANK_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
             f"frank@{os.environ.get('FRANK_HOST', '[REDACTED-IP]')}"]

HOST_CAPACITY_MIB = {"mrfloppy": 24564, "frank": 8192}
# Reserved for driver/desktop/non-bench GPU users on each host.
HOST_MARGIN_MIB = {"mrfloppy": 1024, "frank": 512}

# ------------------------------------------------------------------ bench
# vram_mib: MEASURED per-process footprints (nvidia-smi, 2026-10-20).
# tts is per-call transient: measured peak 1324 MiB (2026-10-20,
# senses profile, where it has headroom). In the full balanced cast
# only ~1.2 GiB is free and F5 OOMs - a pre-existing headroom limit
# that profiles exist to solve. Never resident: counts 0 in budgets.
BENCH = {
    "agent": {
        "specialist": "agent", "model": "Hammer 2.1 3B (NousResearch)",
        "role": "agent / tool use", "host": "mrfloppy",
        "manager": "systemd", "unit": "multiplex-hammer.service",
        "port": 8201, "health": "http://127.0.0.1:8201/health",
        "health_kind": "llama", "vram_mib": 2664,
    },
    "code": {
        "specialist": "code", "model": "MagicoderS-DS-6.7B (UIUC)",
        "role": "code generation", "host": "mrfloppy",
        "manager": "systemd", "unit": "multiplex-magicoder.service",
        "port": 8202, "health": "http://127.0.0.1:8202/health",
        "health_kind": "llama", "vram_mib": 6410,
    },
    "quicklook": {
        "specialist": "quicklook", "model": "Qwen3-4B (16K ctx)",
        "role": "quicklook / second drafter", "host": "mrfloppy",
        "manager": "systemd", "unit": "multiplex-qwen.service",
        "port": 8203, "health": "http://127.0.0.1:8203/health",
        "health_kind": "llama", "vram_mib": 5222,
    },
    "vision": {
        "specialist": "vision", "model": "MiniCPM-V-4.5 (OpenBMB)",
        "role": "vision / OCR", "host": "mrfloppy",
        "manager": "systemd", "unit": "multiplex-minicpm.service",
        "port": 8204, "health": "http://127.0.0.1:8204/health",
        "health_kind": "llama", "vram_mib": 6680,
        "gate": "vision_gate.sh", "pin_file": ".vision-pinned",
    },
    "shepherd": {
        "specialist": "shepherd",
        "model": "Math-Shepherd PRM (Renmin Univ., 4-bit)",
        "role": "step-by-step verifier", "host": "mrfloppy",
        "manager": "systemd", "unit": "multiplex-shepherd.service",
        "port": 8206, "health": "http://127.0.0.1:8206/health",
        "health_kind": "ok", "vram_mib": 5074,
    },
    "verifier": {
        "specialist": None,
        "model": "Skywork-Reward-V2-Llama-3.2-1B",
        "role": "gate infrastructure: reward scoring (style only "
                "post-final-gate) + run_verified plumbing",
        "host": "mrfloppy", "manager": "process",
        "proc_match": "multiplex-env/bin/python verifier_server.py",
        "port": 8199, "health": "http://127.0.0.1:8199/health",
        "health_kind": "ok", "vram_mib": 3200,
    },
    "trm": {
        "specialist": "trm", "model": "TRM (Samsung SAIL, smoke ckpt)",
        "role": "ARC puzzle solver", "host": "mrfloppy",
        "manager": "systemd", "unit": "multiplex-trm.service",
        "port": 8210, "health": "http://127.0.0.1:8210/health",
        "health_kind": "ok", "vram_mib": 898,
        "gate": "trm_gate.sh", "pin_file": ".trm-pinned",
    },
    "ouro": {
        "specialist": "ouro", "model": "Ouro-2.6B-Thinking (4-bit)",
        "role": "looped reasoner", "host": "frank",
        "manager": "systemd", "unit": "multiplex-ouro.service",
        "port": 8208, "health": f"http://{FRANK_HOST}:8208/health",
        "health_kind": "ok", "vram_mib": 1998,
    },
    "stt": {
        "specialist": "stt", "model": "Kyutai stt-1b-en_fr",
        "role": "speech-to-text", "host": "frank",
        "manager": "systemd", "unit": "multiplex-kyutai.service",
        "port": 8209, "health": f"http://{FRANK_HOST}:8209/health",
        "health_kind": "ok", "vram_mib": 2128,
    },
    "embed": {
        "specialist": "embed", "model": "bge-m3 + bge-reranker (infinity)",
        "role": "embeddings / ranking", "host": "frank",
        "manager": "docker", "container": "infinity",
        "port": 7997, "health": f"http://{FRANK_HOST}:7997/health",
        "health_kind": "any200", "vram_mib": 3282,
    },
    "mathwhiz": {
        "specialist": "mathwhiz",
        "model": "MiniCPM5-2B (OpenBMB, Q4_K_M)",
        "role": "math seat (Phase-2 sweep math champion)",
        "host": "mrfloppy",
        "manager": "systemd", "unit": "multiplex-minicpm5.service",
        "port": 8207, "health": "http://127.0.0.1:8207/health",
        "health_kind": "llama", "vram_mib": 2002,  # measured 2026-10-22 @4096 ctx
    },
    "draft": {
        "specialist": "draft",
        "model": "Phi-4-mini-instruct 3.8B (Microsoft, Q4_K_M)",
        "role": "drafting / generalist seat (sweep's steadiest "
                "generalist)",
        "host": "mrfloppy",
        "manager": "systemd", "unit": "multiplex-phi4.service",
        "port": 8211, "health": "http://127.0.0.1:8211/health",
        "health_kind": "llama", "vram_mib": 3384,  # measured 2026-10-22 @4096 ctx
    },
    "tts": {
        "specialist": "tts", "model": "F5-TTS v1 Base (SJTU)",
        "role": "text-to-speech (per-call CLI, transient VRAM)",
        "host": "mrfloppy", "manager": "percall",
        "port": None, "health": None, "health_kind": None,
        "vram_mib": 1324, "transient": True,
    },
}

# --------------------------------------------------------------- profiles
# The verifier is listed in every cast on purpose: run_verified /
# run_re consult it, so it is costed infrastructure, not a freebie.
PROFILES = {
    "balanced": {
        "description": "Default cast - the full duplex as built "
                       "2026-10-20. Vision + TRM on demand via gates.",
        "resident": {
            "mrfloppy": ["verifier", "agent", "code", "quicklook",
                         "shepherd"],
            "frank": ["embed", "ouro", "stt"],
        },
        "on_demand": ["vision", "trm", "tts"],
        "pin": [],
    },
    "re": {
        "description": "Reverse-engineering cast: the final-gate "
                       "lineup (dual drafters + step checker). "
                       "Vision/TRM not in cast.",
        "resident": {
            "mrfloppy": ["verifier", "agent", "code", "quicklook",
                         "shepherd"],
            "frank": [],
        },
        "on_demand": ["tts"],
        "pin": [],
    },
    "browser": {
        "description": "Browser-driving cast: vision RESIDENT "
                       "(pinned, no swap delay), code swapped out.",
        "resident": {
            "mrfloppy": ["verifier", "agent", "quicklook", "vision"],
            "frank": [],
        },
        "on_demand": ["tts"],
        "pin": ["vision"],
    },
    "research": {
        "description": "Research cast: quicklook + agent drafting, "
                       "frank embeddings for retrieval.",
        "resident": {
            "mrfloppy": ["verifier", "agent", "quicklook"],
            "frank": ["embed"],
        },
        "on_demand": ["tts"],
        "pin": [],
    },
    "math": {
        "description": "Math cast: Ouro looped reasoner checked "
                       "step-by-step by Shepherd, quicklook assists.",
        "resident": {
            "mrfloppy": ["verifier", "shepherd", "quicklook"],
            "frank": ["ouro"],
        },
        "on_demand": ["tts"],
        "pin": [],
    },
    "unlock-research": {
        "description": "PS5 unlock research cast (2026-10-22): the "
                       "directed-shaping lineup - agent + code for RE "
                       "work, quicklook for evidence drafting, mathwhiz "
                       "for derivations/counts, draft for prose, the "
                       "Skywork gate resident; frank embed+ouro for "
                       "triage/reasoning; hal/strata is the deep seat "
                       "via the gateway (not resident, never staged).",
        "resident": {
            "mrfloppy": ["verifier", "agent", "code", "quicklook",
                         "mathwhiz", "draft"],
            "frank": ["embed", "ouro"],
        },
        "on_demand": ["vision", "tts"],
        "pin": [],
    },
    "triage": {
        "description": "Fuzz/log triage cast (2026-10-22): frank "
                       "embeddings cluster the corpus, code parses and "
                       "reduces, quicklook classifies, mathwhiz counts, "
                       "ouro reasons over families; verifier resident "
                       "for gates. Draft/shepherd stay on the shelf.",
        "resident": {
            "mrfloppy": ["verifier", "agent", "code", "quicklook",
                         "mathwhiz"],
            "frank": ["embed", "ouro"],
        },
        "on_demand": ["vision", "tts"],
        "pin": [],
    },
    "senses": {
        "description": "Voice/vision session cast: ears on frank, "
                       "eyes resident on mrfloppy, F5 voice per call.",
        "resident": {
            "mrfloppy": ["verifier", "vision"],
            "frank": ["stt"],
        },
        "on_demand": ["tts"],
        "pin": ["vision"],
    },
}

START_TIMEOUT_S = {"embed": 300, "ouro": 300, "stt": 300,
                   "shepherd": 240, "verifier": 240}
DEFAULT_START_TIMEOUT_S = 180
STOP_TIMEOUT_S = 90

# ------------------------------------------------------------ run tracking
_run_lock = threading.Lock()
_run_state = {"active": 0, "staging": False}
_stage_lock = threading.Lock()


class BusyError(Exception):
    """Raised when a run is attempted while staging is in progress."""


def run_begin():
    with _run_lock:
        if _run_state["staging"]:
            raise BusyError("profile switch in progress")
        _run_state["active"] += 1


def run_end():
    with _run_lock:
        _run_state["active"] = max(0, _run_state["active"] - 1)


def runs_in_flight():
    with _run_lock:
        return _run_state["active"]


def is_staging():
    with _run_lock:
        return _run_state["staging"]


# ------------------------------------------------------------------ state
def _load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"active_profile": None, "disabled": [], "updated_ts": 0}


def _save_state(st):
    st["updated_ts"] = time.time()
    STATE_FILE.write_text(json.dumps(st, indent=1))


def active_profile():
    return _load_state().get("active_profile")


def _disabled():
    return set(_load_state().get("disabled", []))


# --------------------------------------------------------------- primitives
def _run_cmd(args, timeout=120):
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout)


def _svc(host, action, unit):
    if host == "mrfloppy":
        return _run_cmd(["systemctl", "--user", action, unit],
                        timeout=180)
    return _run_cmd(FRANK_SSH + [f"systemctl --user {action} {unit}"],
                    timeout=180)


def _docker(action, container):
    return _run_cmd(FRANK_SSH + [f"docker {action} {container}"],
                    timeout=300)


def _healthy(entry, timeout=3):
    url = entry.get("health")
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if r.status != 200:
                return False
            if entry["health_kind"] == "any200":
                return True
            data = json.loads(r.read().decode() or "{}")
    except Exception:
        return False
    if entry["health_kind"] == "llama":
        return data.get("status") == "ok"
    if entry["health_kind"] == "ok":
        return bool(data.get("ok"))
    return False


def _is_resident(name):
    """Manager-level liveness (unit active / container running /
    process present). Health is a separate, stronger check."""
    e = BENCH[name]
    mgr = e["manager"]
    try:
        if mgr == "systemd":
            r = _svc(e["host"], "is-active", e["unit"])
            return r.stdout.strip() == "active"
        if mgr == "docker":
            r = _run_cmd(FRANK_SSH + [
                "docker inspect -f '{{.State.Running}}' "
                + e["container"]], timeout=30)
            return r.stdout.strip() == "true"
        if mgr == "process":
            r = _run_cmd(["pgrep", "-f", e["proc_match"]], timeout=15)
            return r.returncode == 0
    except Exception:
        return False
    return False  # percall: never resident


def _start_one(name):
    e = BENCH[name]
    mgr = e["manager"]
    if mgr == "systemd":
        _svc(e["host"], "start", e["unit"])
    elif mgr == "docker":
        _docker("start", e["container"])
    elif mgr == "process":
        cmd = (f"cd {HERE} && nohup /home/joshua/llm/multiplex-env/bin/"
               f"python verifier_server.py >> verifier.log 2>&1 &")
        subprocess.Popen(["bash", "-c", cmd], start_new_session=True,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    # percall: nothing to start


def _stop_one(name):
    e = BENCH[name]
    mgr = e["manager"]
    if mgr == "systemd":
        _svc(e["host"], "stop", e["unit"])
    elif mgr == "docker":
        _docker("stop", e["container"])
    elif mgr == "process":
        _run_cmd(["pkill", "-f", e["proc_match"]], timeout=30)


def _wait_healthy(name, timeout):
    e = BENCH[name]
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _healthy(e):
            return round(time.time() - t0, 1)
        time.sleep(2)
    return None


def _wait_down(name, timeout=STOP_TIMEOUT_S):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not _is_resident(name):
            return round(time.time() - t0, 1)
        time.sleep(1)
    return None


def _gpu_used_mib(host):
    try:
        if host == "mrfloppy":
            r = _run_cmd(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"], timeout=30)
        else:
            r = _run_cmd(FRANK_SSH + [
                "nvidia-smi --query-gpu=memory.used "
                "--format=csv,noheader,nounits"], timeout=30)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


# ------------------------------------------------------------------- bench
def bench_state(name):
    e = BENCH[name]
    disabled = name in _disabled()
    pinned = bool(e.get("pin_file") and (HERE / e["pin_file"]).exists())
    if e["manager"] == "percall":
        state = "disabled" if disabled else "on-demand"
        healthy = None
    elif _is_resident(name):
        state = "resident"
        healthy = _healthy(e)
    elif e.get("gate"):
        state = "on-demand"
        healthy = False
    else:
        state = "off"
        healthy = False
    if disabled and state == "resident":
        state = "resident (flagged disabled)"
    return {
        "name": name, "specialist": e["specialist"], "model": e["model"],
        "role": e["role"], "host": e["host"], "manager": e["manager"],
        "unit": e.get("unit") or e.get("container") or e.get("proc_match"),
        "port": e["port"], "vram_mib": e["vram_mib"],
        "transient": bool(e.get("transient")),
        "state": state, "healthy": healthy, "pinned": pinned,
        "disabled": disabled,
    }


def list_bench():
    return {"bench": [bench_state(n) for n in BENCH],
            "host_capacity_mib": HOST_CAPACITY_MIB,
            "host_margin_mib": HOST_MARGIN_MIB,
            "gpu_used_mib": {h: _gpu_used_mib(h) for h in HOST_CAPACITY_MIB}}


def bench_action(name, action):
    """Individual on/off switch for one bench model."""
    if name not in BENCH:
        return {"ok": False, "error": f"unknown bench model {name!r}",
                "kind": "unknown"}
    e = BENCH[name]
    if action == "stop" and runs_in_flight() > 0:
        return {"ok": False, "kind": "runs_in_flight",
                "error": f"{runs_in_flight()} run(s) in flight; "
                         "refusing to stop a model mid-run"}
    if is_staging():
        return {"ok": False, "kind": "busy",
                "error": "profile switch in progress"}
    st = _load_state()
    dis = set(st.get("disabled", []))
    t0 = time.time()
    if e["manager"] == "percall":
        if action == "stop":
            dis.add(name)
        else:
            dis.discard(name)
        st["disabled"] = sorted(dis)
        _save_state(st)
        return {"ok": True, "name": name, "action": action,
                "note": "per-call engine: toggled the enable flag "
                        "(no resident process)",
                "seconds": 0.0, "state": bench_state(name)}
    if action == "start":
        dis.discard(name)
        st["disabled"] = sorted(dis)
        _save_state(st)
        if _is_resident(name) and _healthy(e):
            return {"ok": True, "name": name, "action": "start",
                    "seconds": 0.0, "note": "already resident + healthy",
                    "state": bench_state(name)}
        _start_one(name)
        waited = _wait_healthy(name,
                                START_TIMEOUT_S.get(name,
                                                    DEFAULT_START_TIMEOUT_S))
        if waited is None:
            return {"ok": False, "name": name, "action": "start",
                    "error": f"started but not healthy within timeout",
                    "seconds": round(time.time() - t0, 1),
                    "state": bench_state(name)}
        return {"ok": True, "name": name, "action": "start",
                "seconds": waited, "state": bench_state(name)}
    # stop
    if e.get("pin_file"):
        (HERE / e["pin_file"]).unlink(missing_ok=True)
    if not _is_resident(name):
        return {"ok": True, "name": name, "action": "stop",
                "seconds": 0.0, "note": "already off",
                "state": bench_state(name)}
    _stop_one(name)
    waited = _wait_down(name)
    if waited is None:
        return {"ok": False, "name": name, "action": "stop",
                "error": "still resident after stop timeout",
                "seconds": round(time.time() - t0, 1),
                "state": bench_state(name)}
    return {"ok": True, "name": name, "action": "stop",
            "seconds": waited, "state": bench_state(name)}


# ---------------------------------------------------------------- profiles
def profile_math(cast_resident):
    """VRAM budget math for a resident cast, per host."""
    out = {}
    for host, cap in HOST_CAPACITY_MIB.items():
        names = cast_resident.get(host, [])
        items = {n: BENCH[n]["vram_mib"] for n in names
                 if not BENCH[n].get("transient")}
        total = sum(items.values())
        margin = HOST_MARGIN_MIB[host]
        out[host] = {"models": items, "total_mib": total,
                     "capacity_mib": cap, "margin_mib": margin,
                     "headroom_mib": cap - total,
                     "fits": total + margin <= cap}
    return out


def list_profiles():
    cur = active_profile()
    profs = {}
    for name, p in PROFILES.items():
        profs[name] = {
            "description": p["description"],
            "resident": p["resident"], "on_demand": p["on_demand"],
            "pin": p["pin"], "math": profile_math(p["resident"]),
            "active": name == cur,
        }
    return {"active_profile": cur, "profiles": profs}


def _resolve_cast(name, cast_override):
    if cast_override:
        resident = {"mrfloppy": list(cast_override.get("mrfloppy", [])),
                    "frank": list(cast_override.get("frank", []))}
        return {"description": "ad-hoc cast", "resident": resident,
                "on_demand": [], "pin": []}
    if name not in PROFILES:
        return None
    return PROFILES[name]


def stage_profile(name, cast_override=None):
    """Stage a profile: refuse if runs are in flight or the cast does
    not fit; otherwise stop/start to match, verify, record."""
    prof = _resolve_cast(name, cast_override)
    if prof is None:
        return {"ok": False, "kind": "unknown",
                "error": f"unknown profile {name!r} "
                         f"(have {sorted(PROFILES)})"}
    for host, names in prof["resident"].items():
        for n in names:
            if n not in BENCH:
                return {"ok": False, "kind": "unknown",
                        "error": f"cast names unknown bench model {n!r}"}
    math = profile_math(prof["resident"])
    for host, m in math.items():
        if not m["fits"]:
            return {"ok": False, "kind": "budget",
                    "error": f"profile {name!r} does not fit on {host}: "
                             f"needs {m['total_mib']} MiB + "
                             f"{m['margin_mib']} MiB margin > "
                             f"{m['capacity_mib']} MiB capacity",
                    "math": math}
    if runs_in_flight() > 0:
        return {"ok": False, "kind": "runs_in_flight",
                "error": f"{runs_in_flight()} run(s) in flight; "
                         "staging refused (retry when idle)"}
    if not _stage_lock.acquire(blocking=False):
        return {"ok": False, "kind": "busy",
                "error": "another profile switch is in progress"}
    with _run_lock:
        _run_state["staging"] = True
    t_start = time.time()
    actions = []
    try:
        target = {n for names in prof["resident"].values() for n in names}
        # pins for the incoming profile go on BEFORE any stop/start so
        # a reaper can never reap a model the new cast wants resident.
        for n in prof.get("pin", []):
            pin = BENCH[n].get("pin_file")
            if pin:
                (HERE / pin).touch()
        for n, e in BENCH.items():
            if e.get("pin_file") and n not in prof.get("pin", []):
                (HERE / e["pin_file"]).unlink(missing_ok=True)
        current = {n for n in BENCH
                   if BENCH[n]["manager"] != "percall"
                   and _is_resident(n)}
        # stops first: free VRAM before asking for more
        for n in sorted(current - target):
            t0 = time.time()
            _stop_one(n)
            waited = _wait_down(n)
            actions.append({"model": n, "action": "stop",
                            "seconds": waited if waited is not None
                            else round(time.time() - t0, 1),
                            "ok": waited is not None})
        for n in sorted(target - current):
            waited = None
            if not (_is_resident(n) and _healthy(BENCH[n])):
                _start_one(n)
                waited = _wait_healthy(
                    n, START_TIMEOUT_S.get(n, DEFAULT_START_TIMEOUT_S))
            else:
                waited = 0.0
            actions.append({"model": n, "action": "start",
                            "seconds": waited,
                            "ok": waited is not None})
        failed = [a for a in actions if not a["ok"]]
        gpu = {h: _gpu_used_mib(h) for h in HOST_CAPACITY_MIB}
        over = {h: gpu[h] for h in gpu
                if gpu[h] is not None and gpu[h] > HOST_CAPACITY_MIB[h]}
        # record the active profile (state file + blackboard)
        st = _load_state()
        st["active_profile"] = name
        st["disabled"] = [d for d in st.get("disabled", [])
                          if d not in target]
        _save_state(st)
        try:
            import blackboard
            blackboard.write("task_state", "active_profile", {
                "profile": name, "resident": prof["resident"],
                "on_demand": prof["on_demand"],
                "gpu_used_mib": gpu,
                "staged_ts": time.time()})
        except Exception:
            pass
        return {"ok": not failed and not over,
                "profile": name, "actions": actions,
                "math": math, "gpu_used_mib": gpu,
                "expected_resident_mib": {
                    h: math[h]["total_mib"] for h in math},
                "over_capacity": over,
                "failed_actions": failed,
                "total_seconds": round(time.time() - t_start, 1)}
    finally:
        with _run_lock:
            _run_state["staging"] = False
        _stage_lock.release()


# ------------------------------------------------------------ cast checking
def check_cast(bench_names):
    """Problems list for calling the given bench models right now.
    Gate-managed and per-call models are always callable (unless
    flagged disabled); service models must be resident."""
    problems = []
    dis = _disabled()
    for n in bench_names:
        e = BENCH[n]
        if n in dis:
            problems.append(f"{n} is toggled OFF")
        elif e["manager"] == "percall" or e.get("gate"):
            continue
        elif not _is_resident(n):
            problems.append(
                f"{n} is not staged in profile "
                f"{active_profile()!r} (see GET /bench)")
    return problems


def check_run(kind, specialist=None):
    """Cast requirements per API run kind. Returns a problems list."""
    if kind == "run":
        return check_cast([specialist]) if specialist in BENCH else []
    if kind == "verify":
        return check_cast(["verifier"])
    if kind == "run_verified":
        names = ["verifier", "quicklook"]
        if specialist in BENCH:
            names.append(specialist)
        return check_cast(sorted(set(names)))
    if kind == "run_re":
        return check_cast(["agent", "code", "quicklook", "verifier"])
    return []


def health_state():
    return {"active_profile": active_profile(),
            "runs_in_flight": runs_in_flight(),
            "staging": is_staging()}
