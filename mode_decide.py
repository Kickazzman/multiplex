#!/usr/bin/env python3
"""JEV decider (2026-10-30): lean escalation + calculator resolution,
in the dispatch layer, for CODE kinds only.

Scope (binding): code-patch-correct + code-which-better ONLY -- the
kinds where JEV measured the rule. All other modes keep enforcement +
annotation; their decider paths are NOT built (no oracles exist for
them yet). This module never changes seat state, never blocks a
generation beyond the bounded calculator subprocess, never rewrites
a result: it reads the served fast result, applies the rule, and
returns a record the API attaches under result["mode"]["decider"].
The fast result text is ALWAYS served unchanged (fail-closed).

Rules (implemented exactly, from the JEV program):
  LEAN (default): escalate to the calculator when the fast seat
      returns null/empty (including any non-contract result) OR
      fast confidence < 0.8. Nothing else escalates.
  FULL-INSURANCE (separate toggle, default OFF): when ON, escalate
      ALSO on seat disagreement (fast decision vs a supplied second
      opinion). The insurance call stays perg's: the switch exists,
      default is lean, the switch is never removed.
  MERGE with top = "disagree": when fast and slow disagree and the
      calculator cannot resolve, the merge result is "disagree"
      (top), never max-by-confidence, never a silent pick.
  CALCULATOR = the test/oracle side: the applicable test suite run
      against the candidate (tests verdict, not model opinion).
      Failure floor P = 0.010, Wilson (0.0018, 0.0545) is a STATED
      uncertainty attached to calculator records, never a
      correction applied to any verdict.
  HAMMER CONFIDENCE NEVER ENTERS THE RULE in any form (raw,
      recalibrated, logprob-gated). Structural guarantee: decide()
      takes a slow DECISION string only -- there is no confidence
      parameter for any second opinion anywhere in this module. A
      second opinion's confidence, if a caller supplies one, is
      dropped before decide() is called and recorded as ignored.
  TERNARY verdicts: confirmed / undecided-void / refuted. A void
      (no signal) never collapses into refuted.
  FAIL-CLOSED: calculator unavailable, kind unknown, or any decider
      error -> serve the fast result with the mode annotation marked
      decider-skipped + reason. Never block a generation, never
      fabricate a verdict. Decider toggle OFF is NOT a failure: it
      means the module is absent -- no decider record at all, exactly
      the enforcement-only behavior (the enforcement precedent: toggle
      removed -> its key is absent, not stamped skipped).

Live shape of the rule:
  * The fast result is parsed for the JEV contract: a JSON object
    {decision, confidence} (bare or fenced). ANYTHING else -- prose,
    bare words, code blocks, empty -- is a NULL (contract failure),
    exactly as the harness treated bare-word/empty seats.
  * The calculator needs tests. They arrive in the request's
    optional "decider" payload: {"tests": <source>, "candidates":
    {...} | "candidate": <source>, "second_opinion": {"decision":
    ...}, "timeout_s": N}. Tests import the candidate as `impl`
    (the JEV convention). One candidate -> pass/fail verdict. Two
    candidates (keys "A"/"B") -> pick verdict; both-pass or
    both-fail is AMBIGUOUS, not a pick (the cf-b30 both-pass
    ambiguity is the calculator-floor failure mode). No tests
    supplied -> calculator unavailable -> skipped record.
  * The full-mode disagreement trigger needs a SUPPLIED second
    opinion decision (the caller consulted a second seat). NO live
    second-opinion generation is built here: no second seat is
    consulted automatically, because no live second-opinion seat
    meets the precondition (varying confidence + fast-wrong
    disagreements) and the live path carries no contract prompt to
    consult one with. When no second opinion is supplied, the full
    disagreement trigger is unevaluable and recorded as such; lean
    rules still apply. Building an automatic second-opinion call
    is a separate decision.
  * The calculator runs the tests in a subprocess (stdlib-only
    runner with a minimal pytest stub supporting approx/raises;
    temp dir; timeout; AST import allowlist). It never imports the
    API process and never touches seat state.

Toggles (both stat'ed per call; no restart needed to flip either):
    decider:      mode_decide.enabled      (master, default OFF)
    full:         mode_decide_full.enabled (insurance, default OFF)
    Disable decider: rm <multiplex>/mode_decide.enabled
    Disable full:    rm <multiplex>/mode_decide_full.enabled
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECIDE_TOGGLE = HERE / "mode_decide.enabled"
FULL_TOGGLE = HERE / "mode_decide_full.enabled"

try:
    import mode_hook
except Exception:  # pragma: no cover - hook absent (harness/replay)
    mode_hook = None

CODE_KINDS = ("code-patch-correct", "code-which-better")
CONFIDENCE_THRESHOLD = 0.8

# Calculator failure floor (item 1): STATED uncertainty only. Never
# applied as a correction to any verdict or error rate.
FLOOR = {"p": 0.010, "wilson_low": 0.0018, "wilson_high": 0.0545}

_DECISIONS = {"pass": "pass", "fail": "fail", "a": "A", "b": "B"}

# Imports the calculator runner permits in candidate/test sources.
# "impl" is the candidate module the tests import (JEV convention);
# "pytest" is the stubbed test framework (approx/raises only).
_ALLOWED_IMPORTS = {
    "pytest", "impl", "math", "re", "collections", "itertools",
    "functools", "statistics", "datetime", "decimal", "fractions",
    "string", "heapq", "bisect", "json", "random", "copy", "operator",
    "typing", "dataclasses", "enum", "unicodedata", "numbers", "abc",
    "contextlib",
}

_CALC_TIMEOUT_DEFAULT_S = 60
_CALC_TIMEOUT_MIN_S = 5
_CALC_TIMEOUT_MAX_S = 300
_MAX_SOURCE_BYTES = 200000

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------- toggles
def enabled():
    """Decider master toggle: ON iff the toggle file exists."""
    try:
        return DECIDE_TOGGLE.exists()
    except Exception:
        return False


def full_enabled():
    """Full-insurance toggle: ON iff the toggle file exists."""
    try:
        return FULL_TOGGLE.exists()
    except Exception:
        return False


# ------------------------------------------------------------- fast parsing
def _strip_think(text):
    return _THINK_RE.sub("", text)


def _parse_contract_json(text):
    """Parse one contract object from text. Returns the dict, or None
    when no JSON object parses. Tries the whole text first, then each
    fenced block in order."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            return obj
    for _info, body in _FENCE_RE.findall(text):
        try:
            obj = json.loads(body.strip())
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_fast(text):
    """Parse a fast result's text into a contract decision.

    Returns {"decision": str, "confidence": float} or None (null).
    Null covers: empty, prose, bare words, code, malformed JSON,
    missing/non-numeric/out-of-range confidence, unknown decision
    word. No confidence is ever inferred; no decision is ever
    guessed from prose.
    """
    if not isinstance(text, str):
        return None
    clean = _strip_think(text).strip()
    if not clean:
        return None
    obj = _parse_contract_json(clean)
    if obj is None:
        return None
    raw_dec = obj.get("decision")
    if not isinstance(raw_dec, str):
        return None
    decision = _DECISIONS.get(raw_dec.strip().lower())
    if decision is None:
        return None
    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return None
    conf = float(conf)
    if not (0.0 <= conf <= 1.0):
        return None
    return {"decision": decision, "confidence": conf}


def _normalize_decision(value):
    """Normalize a supplied second-opinion decision word. None when
    absent, non-string, or not a code-(a)/(b) decision word."""
    if not isinstance(value, str):
        return None
    return _DECISIONS.get(value.strip().lower())


def extract_candidate(text):
    """First fenced code block (python/py/bare fence) from text, or
    None. Used only when the request supplies no explicit candidate."""
    if not isinstance(text, str):
        return None
    for info, body in _FENCE_RE.findall(text):
        lang = info.strip().lower()
        if lang in ("", "python", "py", "python3"):
            body = body.strip("\n")
            if body.strip():
                return body
    return None


def _fast_text(result):
    """The served text of a run-family result dict. Specialists return
    {"text": ...}; run_verified-style results carry "response"."""
    if not isinstance(result, dict):
        return ""
    for key in ("text", "response"):
        val = result.get(key)
        if isinstance(val, str):
            return val
    msg = result.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    return ""


# ------------------------------------------------------------ pure decision
def decide(fast, slow_decision, calculator, full):
    """Pure JEV decider. No I/O, no seat calls, no confidence from
    any second opinion (there is no such parameter).

    fast: None (null) or {"decision": str, "confidence": float|None}.
    slow_decision: str ("pass"/"fail"/"A"/"B") or None.
    calculator: None (not run) or {"status": "ok"/"failed"/
        "ambiguous"/"unavailable", "verdict": str|None, ...}.
    full: bool, full-insurance mode (disagreement escalates).

    Returns a record dict. The caller (apply/replay) owns running
    the calculator; this function only resolves what it is given.
    """
    fast_null = fast is None
    fast_dec = None if fast_null else fast.get("decision")
    fast_conf = None if fast_null else fast.get("confidence")
    disagree = (not fast_null and slow_decision is not None
                and slow_decision != fast_dec)
    triggers = []
    if fast_null:
        triggers.append("null")
    if (not fast_null and fast_conf is not None
            and fast_conf < CONFIDENCE_THRESHOLD):
        triggers.append("confidence")
    if full and disagree:
        triggers.append("disagreement")
    escalated = bool(triggers)

    rec = {
        "fast_decision": fast_dec,
        "fast_confidence": fast_conf,
        "fast_null": fast_null,
        "slow_decision": slow_decision,
        "disagreement": disagree,
        "full_mode": bool(full),
        "escalation_triggers": triggers,
        "escalated": escalated,
        "calculator": None,
        "final_decision": None,
        "verdict": None,
        "merge": None,
        "basis": None,
    }
    if not escalated:
        # Lean, confident, unchallenged: the fast decision stands.
        # Recorded as confirmed on a fast-standalone basis -- no
        # independent check ran. The floor caveat travels with the
        # no-escalation path in the mode annotation, not here.
        rec.update({"final_decision": fast_dec, "verdict": "confirmed",
                    "merge": "fast", "basis": "fast-standalone",
                    "disagreement_recorded": disagree})
        return rec
    rec["disagreement_recorded"] = disagree
    if calculator is None:
        # Escalated but the calculator never ran: void, never refuted.
        rec.update({"calculator": "not-run",
                    "final_decision": fast_dec,
                    "verdict": "undecided-void", "merge": "none",
                    "basis": "escalated-no-calculator"})
        return rec
    status = calculator.get("status")
    verdict = calculator.get("verdict")
    rec["calculator"] = status
    if status == "ok" and verdict is not None:
        rec.update({"final_decision": verdict,
                    "merge": "calculator", "basis": "calculator"})
        if fast_null or verdict == fast_dec:
            rec["verdict"] = "confirmed"
        else:
            rec["verdict"] = "refuted"
        return rec
    # Calculator failed or ambiguous: it cannot resolve.
    if disagree:
        # Merge with top = "disagree". Contention is real and
        # unresolved: void, never a silent pick, never refuted.
        rec.update({"final_decision": "disagree",
                    "verdict": "undecided-void", "merge": "top-disagree",
                    "basis": "calculator-unresolved-disagreement"})
        return rec
    rec.update({"final_decision": fast_dec,
                "verdict": "undecided-void", "merge": "none",
                "basis": "calculator-unresolved-fast-stands"})
    return rec


# ------------------------------------------------------------- calculator
_RUNNER_SCRIPT = textwrap.dedent('''\
    import json, sys, types, traceback

    # Minimal pytest stub: approx + raises only. Anything else the
    # tests use (fixtures, parametrize, marks) fails loudly at
    # import/call time -- never silently.
    _pytest = types.ModuleType("pytest")

    class _RaisesContext:
        def __init__(self, exc):
            self._exc = exc
        def __enter__(self):
            return self
        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(
                    "DID NOT RAISE %s" % (self._exc,))
            return issubclass(et, self._exc)

    def _raises(exc, *a, **k):
        return _RaisesContext(exc)

    class _Approx:
        def __init__(self, expected, rel=1e-6, abs=1e-12):
            self.expected = expected
            self.rel = rel
            self.abs = abs
        def __eq__(self, other):
            tol = max(self.rel * abs(self.expected), self.abs)
            try:
                return abs(other - self.expected) <= tol
            except Exception:
                return False
        def __ne__(self, other):
            return not self.__eq__(other)
        def __repr__(self):
            return "approx(%r)" % (self.expected,)

    def _approx(expected, rel=1e-6, abs=1e-12):
        return _Approx(expected, rel=rel, abs=abs)

    _pytest.raises = _raises
    _pytest.approx = _approx
    sys.modules["pytest"] = _pytest

    def _main():
        try:
            import candidate_tests as _t
        except Exception:
            print(json.dumps({"status": "failed",
                              "detail": "test import error: " +
                              traceback.format_exc(limit=2)[-400:]}))
            return
        import impl as _impl  # noqa: F401  (candidate must import)
        names = [n for n in dir(_t) if n.startswith("test_")]
        if not names:
            print(json.dumps({"status": "ambiguous",
                              "detail": "no test_ functions collected"}))
            return
        failed = []
        for name in sorted(names):
            fn = getattr(_t, name)
            if not callable(fn):
                continue
            try:
                fn()
            except Exception as e:
                failed.append({"test": name,
                               "error": "%s: %s" % (type(e).__name__, e)})
        if failed:
            print(json.dumps({"status": "fail", "failed": failed[:5],
                              "n_failed": len(failed),
                              "n_total": len(names)}))
        else:
            print(json.dumps({"status": "pass", "n_total": len(names)}))

    _main()
    ''')


def _check_imports(source, label):
    """AST import allowlist. Returns an error string or None."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return "%s: syntax error: %s" % (label, e)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    return "%s: disallowed import %r" % (label, root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    return "%s: disallowed import %r" % (label, root)
    return None


def _run_one_candidate(candidate_src, tests_src, timeout_s):
    """Run the tests against one candidate. Returns (outcome, detail)
    with outcome in {"pass", "fail", "ambiguous", "failed"}."""
    err = _check_imports(candidate_src, "candidate")
    if err:
        return "failed", err[:300]
    err = _check_imports(tests_src, "tests")
    if err:
        return "failed", err[:300]
    try:
        with tempfile.TemporaryDirectory(prefix="jev-calc-") as td:
            Path(td, "impl.py").write_text(candidate_src)
            Path(td, "candidate_tests.py").write_text(tests_src)
            Path(td, "_calc_run.py").write_text(_RUNNER_SCRIPT)
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                   "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONNOUSERSITE": "1",
                   "HOME": td}
            try:
                proc = subprocess.run(
                    [sys.executable, "_calc_run.py"],
                    cwd=td, env=env, capture_output=True, text=True,
                    timeout=timeout_s)
            except subprocess.TimeoutExpired:
                return "failed", "timeout after %ss" % (timeout_s,)
            out = (proc.stdout or "").strip()
            if proc.returncode != 0:
                detail = (proc.stderr or out or "runner exit %d"
                          % proc.returncode)[-300:]
                return "failed", "runner error: %s" % detail
            try:
                data = json.loads(out.splitlines()[-1])
            except Exception:
                return "failed", "unparseable runner output: %s" % out[-200:]
            status = data.get("status")
            if status == "pass":
                return "pass", "%d tests passed" % data.get("n_total", 0)
            if status == "fail":
                return "fail", "%d/%d tests failed" % (
                    data.get("n_failed", 0), data.get("n_total", 0))
            if status == "ambiguous":
                return "ambiguous", data.get("detail", "")[:200]
            return "failed", data.get("detail", "unknown")[:300]
    except Exception as e:  # noqa: BLE001
        return "failed", "%s: %s" % (type(e).__name__, e)


def run_calculator(payload, fast_text):
    """Run the test/oracle calculator.

    payload: {"tests": str, "candidate": str | None,
              "candidates": {"A": str, "B": str} | None,
              "timeout_s": number | None}.
    fast_text: served fast text, used ONLY to extract a candidate
    when the payload names none.

    Returns {"status": "ok"/"failed"/"ambiguous"/"unavailable",
             "verdict": str|None, "detail": str, "runs": [...]}.
    "unavailable" means the calculator could not run at all (no
    tests, no candidate, oversize source). "failed" means it ran
    but errored (timeout, runner error, disallowed import).
    "ambiguous" means the tests ran but do not discriminate
    (both-pass / both-fail on two candidates, or zero tests).
    """
    if not isinstance(payload, dict):
        return {"status": "unavailable", "verdict": None,
                "detail": "no calculator payload", "runs": []}
    tests = payload.get("tests")
    if not isinstance(tests, str) or not tests.strip():
        return {"status": "unavailable", "verdict": None,
                "detail": "no tests supplied", "runs": []}
    if len(tests.encode("utf-8", "ignore")) > _MAX_SOURCE_BYTES:
        return {"status": "unavailable", "verdict": None,
                "detail": "tests source oversize", "runs": []}
    timeout = payload.get("timeout_s", _CALC_TIMEOUT_DEFAULT_S)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        timeout = _CALC_TIMEOUT_DEFAULT_S
    timeout = max(_CALC_TIMEOUT_MIN_S,
                  min(_CALC_TIMEOUT_MAX_S, float(timeout)))

    candidates = None
    if isinstance(payload.get("candidates"), dict):
        candidates = payload["candidates"]
    elif isinstance(payload.get("candidate"), str):
        candidates = {"candidate": payload["candidate"]}
    if candidates is None:
        extracted = extract_candidate(fast_text)
        if extracted is None:
            return {"status": "unavailable", "verdict": None,
                    "detail": "no candidate code found (none supplied, "
                             "none extractable from fast result)",
                    "runs": []}
        candidates = {"candidate": extracted}
    if not candidates or len(candidates) > 2:
        return {"status": "unavailable", "verdict": None,
                "detail": "candidates must name 1 or 2 implementations",
                "runs": []}
    for key, src in candidates.items():
        if not isinstance(src, str) or not src.strip():
            return {"status": "unavailable", "verdict": None,
                    "detail": "candidate %r is empty" % (key,), "runs": []}
        if len(src.encode("utf-8", "ignore")) > _MAX_SOURCE_BYTES:
            return {"status": "unavailable", "verdict": None,
                    "detail": "candidate %r oversize" % (key,), "runs": []}

    if len(candidates) == 1:
        key = next(iter(candidates))
        outcome, detail = _run_one_candidate(candidates[key], tests,
                                             timeout)
        runs = [{"candidate": key, "outcome": outcome,
                 "detail": detail[:200]}]
        if outcome == "pass":
            return {"status": "ok", "verdict": "pass", "detail": detail,
                    "runs": runs}
        if outcome == "fail":
            return {"status": "ok", "verdict": "fail", "detail": detail,
                    "runs": runs}
        if outcome == "ambiguous":
            return {"status": "ambiguous", "verdict": None,
                    "detail": detail, "runs": runs}
        return {"status": "failed", "verdict": None, "detail": detail,
                "runs": runs}

    # Two candidates: keys must be A/B (the which-better pick labels).
    keys = sorted(candidates)
    if keys != ["A", "B"]:
        return {"status": "unavailable", "verdict": None,
                "detail": "two-candidate payloads must use keys A and B",
                "runs": []}
    outcomes = {}
    runs = []
    for key in ("A", "B"):
        outcome, detail = _run_one_candidate(candidates[key], tests,
                                             timeout)
        outcomes[key] = outcome
        runs.append({"candidate": key, "outcome": outcome,
                     "detail": detail[:200]})
    if outcomes["A"] == "pass" and outcomes["B"] == "fail":
        return {"status": "ok", "verdict": "A",
                "detail": "A passes, B fails", "runs": runs}
    if outcomes["A"] == "fail" and outcomes["B"] == "pass":
        return {"status": "ok", "verdict": "B",
                "detail": "A fails, B passes", "runs": runs}
    if outcomes["A"] in ("pass", "fail") \
            and outcomes["B"] in ("pass", "fail"):
        return {"status": "ambiguous", "verdict": None,
                "detail": "both candidates %s; tests do not discriminate"
                         % outcomes["A"], "runs": runs}
    return {"status": "failed", "verdict": None,
            "detail": "calculator error on A/B: %s / %s"
                     % (outcomes["A"], outcomes["B"]), "runs": runs}


# ------------------------------------------------------------ orchestration
def _skipped(reason, task_kind, extra=None):
    rec = {"status": "skipped", "reason": reason,
           "task_kind": task_kind, "rule": None,
           "fast_decision": None, "fast_confidence": None,
           "fast_null": None, "escalated": False,
           "final_decision": None, "verdict": None}
    if extra:
        rec.update(extra)
    return rec


def _apply(result, task_kind, decider_input):
    """Decider orchestration. Never mutates `result`. Returns
    (result, record|None). record None = no decider involvement
    (not a code kind, hook off, or unresolvable kind)."""
    if not isinstance(result, dict):
        return result, None
    if task_kind not in CODE_KINDS:
        return result, None
    if mode_hook is None or not mode_hook.enabled():
        return result, None
    try:
        mode = mode_hook.resolve(task_kind)
    except Exception:
        return result, None
    if mode is None:
        return result, None
    if not enabled():
        # Toggle OFF = module absent: no record at all (rollback
        # criterion: enforcement-only behavior, no decider key).
        # Skipped records are for the ON-but-unable cases below.
        return result, None
    full = full_enabled()
    text = _fast_text(result)
    fast = parse_fast(text)

    # Second opinion: decision ONLY. Any confidence supplied with it
    # is dropped here, before decide(), and recorded as ignored --
    # hammer/second-opinion confidence never enters the rule.
    slow = None
    slow_conf_ignored = False
    if isinstance(decider_input, dict):
        so = decider_input.get("second_opinion")
        if isinstance(so, dict):
            slow = _normalize_decision(so.get("decision"))
            if "confidence" in so:
                slow_conf_ignored = True

    # Escalation check without the calculator (pure trigger logic).
    fast_null = fast is None
    fast_conf = None if fast_null else fast.get("confidence")
    disagree = (not fast_null and slow is not None
                and slow != fast["decision"])
    triggers = []
    if fast_null:
        triggers.append("null")
    if fast_conf is not None and fast_conf < CONFIDENCE_THRESHOLD:
        triggers.append("confidence")
    if full and disagree:
        triggers.append("disagreement")
    if not triggers:
        rec = decide(fast, slow, None, full)
        rec.update({"status": "decided", "task_kind": task_kind,
                    "rule": "full" if full else "lean",
                    "second_opinion_confidence_ignored":
                        slow_conf_ignored})
        return result, rec

    # Escalated: the calculator must run, or fail closed (skipped).
    calc = run_calculator(decider_input, text)
    if calc["status"] == "unavailable":
        return result, _skipped(
            "escalation triggered (%s) but calculator unavailable: %s"
            % (",".join(triggers), calc["detail"]),
            task_kind,
            {"fast_decision": None if fast_null else fast["decision"],
             "fast_confidence": fast_conf, "fast_null": fast_null,
             "escalation_triggers": triggers,
             "second_opinion_confidence_ignored": slow_conf_ignored})
    rec = decide(fast, slow, calc, full)
    rec.update({"status": "decided", "task_kind": task_kind,
                "rule": "full" if full else "lean",
                "calculator_detail": calc["detail"][:200],
                "calculator_runs": calc["runs"],
                "calculator_floor": dict(FLOOR),
                "second_opinion_confidence_ignored":
                    slow_conf_ignored})
    if full and slow is None:
        rec["full_disagreement_trigger"] = (
            "unevaluable: no second opinion supplied; lean rules applied")
    return result, rec


def apply(result, task_kind, decider_input=None):
    """Decider entry point for the API. Never raises, never mutates
    the result. Returns (result, record|None)."""
    try:
        return _apply(result, task_kind, decider_input)
    except Exception as e:  # noqa: BLE001 - last-resort fail-closed
        if isinstance(result, dict) and task_kind in CODE_KINDS:
            return result, _skipped(
                "decider error: %s: %s" % (type(e).__name__, e),
                task_kind)
        return result, None


def status():
    """Decider status for GET /mode and GET /health. Never raises."""
    try:
        return {
            "enabled": enabled(),
            "toggle": str(DECIDE_TOGGLE),
            "disable_command": "rm %s" % DECIDE_TOGGLE,
            "full_enabled": full_enabled(),
            "full_toggle": str(FULL_TOGGLE),
            "full_disable_command": "rm %s" % FULL_TOGGLE,
            "scope": list(CODE_KINDS),
            "rule": ("escalate on fast null OR confidence < 0.8; "
                     "full mode also escalates on disagreement"),
            "calculator_floor": dict(FLOOR),
        }
    except Exception as e:  # noqa: BLE001
        return {"enabled": False, "full_enabled": False,
                "error": "%s: %s" % (type(e).__name__, e)}
