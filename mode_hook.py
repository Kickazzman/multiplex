#!/usr/bin/env python3
"""Mode-switch dispatch hook: task-kind tag -> use-map mode.

The use-map (use-map-full-content.md) is the single source of truth;
use-map-index.json is its derived, machine-readable index. This module
is the ONLY place the API consults the index. It never changes seat
state, never blocks a run, never reroutes: it resolves a mode and
annotates a dispatch result with the mode's seat plan, escalation
rule, tests, avoid list, and cost posture. Executing a seat plan
(parking/restoring residents, starting a serial seat) stays with the
batch runner's trap-restore windows, never with this hook.

Toggle (on/off switch, per perg's rule that every service has one):
    ON  iff the file mode_hook.enabled exists next to this module.
    OFF iff it is absent (the default state).
    Toggling needs NO restart: enabled() stats the file on every call.
    Disable in one command:
        rm /home/joshua/llm/voice-loop/multiplex/mode_hook.enabled
Cost/behavior when ON: one stat() per dispatch; responses for KNOWN
kinds gain a "mode" block (the full mode dict, ~5-8 KB JSON).
Unknown, missing, empty, wrong-case, or non-string kinds resolve to
NO mode and dispatch behavior is exactly current (no defaulting, no
inference of kind). When OFF: one stat() per dispatch, zero behavior
change. The index is loaded lazily and cached by mtime; a corrupt or
missing index degrades to no-mode, never an exception.

Tagging rule (from the index): the task kind is declared by the task
author in the request ("task_kind"); the dispatcher never infers it.
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOGGLE_FILE = HERE / "mode_hook.enabled"
INDEX_FILE = HERE / "mode_index.json"

_index_cache = {"mtime_ns": None, "data": None}


def enabled():
    """Hook on/off switch: presence of the toggle file. Never raises."""
    try:
        return TOGGLE_FILE.exists()
    except Exception:
        return False


def _load_index():
    """Load the deployed index copy, cached by mtime. None on any
    problem (missing/corrupt/wrong shape) — the hook degrades to
    no-mode rather than failing a dispatch."""
    try:
        st = INDEX_FILE.stat()
    except Exception:
        return None
    if (_index_cache["mtime_ns"] == st.st_mtime_ns
            and _index_cache["data"] is not None):
        return _index_cache["data"]
    try:
        data = json.loads(INDEX_FILE.read_text())
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(
            data.get("modes"), dict):
        return None
    _index_cache["mtime_ns"] = st.st_mtime_ns
    _index_cache["data"] = data
    return data


def resolve(task_kind):
    """Strict mode lookup. Exact string match against index mode keys
    only. Returns the mode dict, or None for: hook OFF, missing /
    empty / wrong-case / non-string kind, missing/corrupt index.
    Never raises, never infers, never defaults."""
    try:
        if not enabled():
            return None
        if not isinstance(task_kind, str) or not task_kind:
            return None
        data = _load_index()
        if data is None:
            return None
        mode = data["modes"].get(task_kind)
        return mode if isinstance(mode, dict) else None
    except Exception:
        return None


def annotate(result, task_kind):
    """Attach {"mode": {task_kind, ...mode}} to a dict result when a
    mode resolves; otherwise return the result unchanged. Never
    mutates the input, never raises, never wraps non-dict results."""
    try:
        mode = resolve(task_kind)
        if mode is None or not isinstance(result, dict):
            return result
        out = dict(result)
        block = copy.deepcopy(mode)
        block["task_kind"] = task_kind
        out["mode"] = block
        return out
    except Exception:
        return result


def status():
    """Introspection for GET /mode. Never raises."""
    try:
        on = enabled()
        data = _load_index() if on else None
        modes = sorted(data["modes"]) if data is not None else []
        return {
            "enabled": on,
            "toggle_file": str(TOGGLE_FILE),
            "index_file": str(INDEX_FILE),
            "index_loaded": data is not None,
            "modes": modes,
        }
    except Exception:
        return {"enabled": False, "toggle_file": str(TOGGLE_FILE),
                "index_file": str(INDEX_FILE), "index_loaded": False,
                "modes": []}
