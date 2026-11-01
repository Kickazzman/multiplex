"""Shared-context blackboard for the GLaDOS specialist multiplexer.

Perg's design: the giant context lives in RAM, shared by many small
specialist agents. This module is that shared context.

Storage: JSON section files under /dev/shm/glados_blackboard/ (tmpfs).
- Memory-speed, no disk I/O.
- Survives agent process restarts (the files persist while the machine
  is up; a reboot clears them, which is fine - it's working context,
  not long-term memory).
- Sections: task_state, findings, scratch, results, briefing.
- Reads are lock-free. Writes take an flock and merge, so concurrent
  specialists never clobber each other.

Usage:
    from blackboard import read, write, append, snapshot
    write("findings", "weather_columbus", {"temp_c": 9.8})
    data = read("findings")
"""
import json
import os
import time
import fcntl
import contextlib
from pathlib import Path

ROOT = Path(os.environ.get("GLADOS_BLACKBOARD_DIR", "/dev/shm/glados_blackboard"))
SECTIONS = ("task_state", "findings", "scratch", "results", "briefing")
_LOCKFILE = ROOT / ".lock"


def _ensure():
    ROOT.mkdir(parents=True, exist_ok=True)
    for s in SECTIONS:
        p = ROOT / f"{s}.json"
        if not p.exists():
            p.write_text("{}")


@contextlib.contextmanager
def _locked():
    _ensure()
    with open(_LOCKFILE, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _section_path(section: str) -> Path:
    if section not in SECTIONS:
        raise KeyError(f"unknown blackboard section: {section!r} (have {SECTIONS})")
    return ROOT / f"{section}.json"


def read(section: str) -> dict:
    """Return the whole section as a dict: {key: {value, ts, by}}."""
    _ensure()
    try:
        return json.loads(_section_path(section).read_text())
    except Exception:
        return {}


def get(section: str, key: str, default=None):
    ent = read(section).get(key)
    return default if ent is None else ent.get("value", default)


def write(section: str, key: str, value) -> None:
    """Set section[key] = value (stamped with time + writer)."""
    with _locked():
        data = read(section)
        data[key] = {
            "value": value,
            "ts": time.time(),
            "by": os.environ.get("AGENT_NAME", "unknown"),
        }
        _section_path(section).write_text(json.dumps(data, indent=1, default=str))


def append(section: str, key: str, value) -> None:
    """Append value to the list stored at section[key]."""
    with _locked():
        data = read(section)
        ent = data.get(key) or {"value": [], "ts": 0, "by": "system"}
        cur = ent.get("value")
        if not isinstance(cur, list):
            cur = [cur] if cur is not None else []
        cur.append(value)
        ent.update(value=cur, ts=time.time(),
                   by=os.environ.get("AGENT_NAME", "unknown"))
        data[key] = ent
        _section_path(section).write_text(json.dumps(data, indent=1, default=str))


def snapshot() -> dict:
    """All sections, for a merged view of shared context."""
    return {s: read(s) for s in SECTIONS}


def clear(section: str) -> None:
    with _locked():
        _section_path(section).write_text("{}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        print(json.dumps(snapshot(), indent=1, default=str)[:4000])
    else:
        write("scratch", "selftest", {"ok": True})
        print("selftest:", read("scratch").get("selftest"))
