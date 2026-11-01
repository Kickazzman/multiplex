#!/usr/bin/env python3
"""job_router — ONE routing surface for the expanded combined multiplex.

Perg's design (2026-10-22): a caller submits ONE task with a task_type;
the router casts it deterministically across ALL live compute — perg's
local seats via the multiplex API :8198, frank's lanes (reached through
the same API's specialists), and hal/strata as the deep seat via the
blackboard gateway (hal_dispatch's rule, unchanged: deep routes ONLY
when the gateway reports the lane live; lane off = QUEUED, never
silently rerouted to a weaker seat).

ROUTE-GATES (2026-10-22, perg's order: "Wire the gates into /route for
every task type"): NO task type reaches a naked seat anymore. After
the seat answers, a deterministic gate runs BEFORE the response
returns, and every response gains a "gate" object
{kind, verdict, detail}. Gate verdicts are appended to the router log
alongside the call record. A gate FAIL (or an UNGATED verdict where a
gate exists but could not certify) does NOT silently pass the seat
answer through: the answer is returned clearly marked
"gate_failed": true / "answer_status", with the gate detail attached —
the caller sees the failure; nothing is hidden. A type with NO honest
gate available returns a clean refusal in the cast-refusal shape
(error/kind/http_status 503) instead of a naked answer.

Gate table (kind — what it certifies / what it does not):

  math    ARITH-RECOMPUTE — if the prompt carries a pure arithmetic
          expression, it is re-evaluated in Python (AST-whitelisted
          safe eval) and the seat's final number must match. Certifies
          the arithmetic. Otherwise CROSS-CHECK via /run_verified.
  code    CODE-EXECUTION — the seat's code block is executed by the
          existing multiplex._execute_code_gate machinery (60s sandbox
          pattern, gate-checks dir). Exit != 0 = FAIL. Optional kw:
          test_code (appended before execution), expect_in_output
          (substring required in stdout). No extractable code block =
          UNGATED, never a fake pass. Certifies "it runs (and the
          stated expectation held)"; does not certify style or
          algorithmic optimality.
  lookup  CLAIM-CHECK — the question is independently re-run through
          /run_verified (quicklook + Layer-2 deterministic gates);
          the seat's load-bearing numbers and quoted strings must be
          supported by the verified path's answer, and the verified
          path must itself pass its gates. The Skywork score is
          reported in the detail but is NEVER decisive.
  draft   PROSE-GATE — deterministic prose checks (length floor,
          placeholder/lorem scan, task/source keyword overlap +
          source-word containment — the worker's Jaccard defense
          pattern) PLUS a CLAIM-CHECK of the draft's factual items
          via /run_verified. Certifies: substantive, on-task,
          non-placeholder prose whose checkable items agree with an
          independent gated path. Does NOT certify style, tone, or
          facts that carry no checkable item.
  reason  CLAIM-CHECK — same machinery as lookup.
  embed / DETERMINISM-GATE — the ranking is run twice; ranking order
  triage  must be identical, returned cosine scores must agree within
          epsilon (0.01), backend must be identical across runs, and
          the embedding dimension must be sane/consistent. (The seat
          API exposes ranked cosine scores, not raw vectors, so
          self-similarity is checked on those scores.) Certifies the
          retrieval path is deterministic and stable; does not
          certify that the ranking is *relevant*.
  vision  CLAIM-CHECK when the request supplies ground-truth context
          (kw ground_truth or context): the vision output's checkable
          items are cross-checked against a /run_verified answer
          derived from that context. WITHOUT context there is no
          honest gate: the route REFUSES (UNGATED-VISION, 503 shape)
          rather than returning a naked vision answer. MiniCPM-V
          stays on-demand behind its existing gate either way.
  verify/ VERIFIED-GATE — unchanged: /run_verified itself (Layer-2
  rederive deterministic gates + Skywork). The gate object summarizes
          its gates_passed / verified / score fields.
  deep    CLAIM-CHECK (cross-bench final gate) — after hal/strata
          answers via the gateway, the answer's extractable claims
          are claim-checked LOCALLY via /run_verified (federation
          rule: a final gate is a re-derivation on the other bench).
          Lane not live = QUEUED, unchanged, gate verdict QUEUED.

Every routed call appends one JSON line to the blackboard router log
(ts, task_type, target, status, elapsed, prompt sha256, gate kind +
verdict when a gate ran). Deep calls are additionally logged by
hal_dispatch in the trial log, as before.
"""
import argparse, ast, hashlib, json, os, re, sys, time
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LOCAL_API = "http://127.0.0.1:8198"
LOG_FILE = os.path.expanduser(
    "~/llm/blackboard/derived/router-log-2026-10-22.jsonl")

# task_type -> (kind, specialist) for :8198-served types.
_LOCAL = {
    "math":   ("run", "mathwhiz"),
    "code":   ("run", "code"),
    "lookup": ("run", "quicklook"),
    "draft":  ("run", "draft"),
    "reason": ("run", "ouro"),
    "embed":  ("run", "embed"),
    "triage": ("run", "embed"),
    "vision": ("run", "vision"),
}
_VERIFY = {"verify", "rederive"}
_KW_DEFAULTS = {"lookup": {"no_think": True}}

_GATE_KIND = {
    "math": "ARITH-RECOMPUTE/CROSS-CHECK",
    "code": "CODE-EXECUTION",
    "lookup": "CLAIM-CHECK",
    "draft": "PROSE-GATE",
    "reason": "CLAIM-CHECK",
    "embed": "DETERMINISM-GATE",
    "triage": "DETERMINISM-GATE",
    "vision": "CLAIM-CHECK/UNGATED-VISION",
    "deep": "CLAIM-CHECK",
}


def _log(task_type, target, status, elapsed, prompt, gate=None):
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "task_type": task_type, "target": target,
               "status": status, "elapsed_s": round(elapsed, 2),
               "prompt_sha256":
                   hashlib.sha256(prompt.encode()).hexdigest()[:16]}
        if gate:
            rec["gate_kind"] = gate.get("kind")
            rec["gate_verdict"] = gate.get("verdict")
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass  # logging must never break routing


def _post(path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(LOCAL_API + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"error": str(e)}


# ------------------------------------------------------------- numbers
def _numbers(text):
    out = []
    for tok in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text or ""):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def _final_number(text):
    nums = _numbers(text)
    return nums[-1] if nums else None


def _num_eq(a, b):
    if a is None or b is None:
        return False
    if a == b:
        return True
    if float(a).is_integer() and float(b).is_integer():
        # Integers must match EXACTLY. A relative tolerance here is a
        # hole: at 1e8 scale, 1e-6 relative = +/-100, and the gate
        # passed an off-by-one wrong answer in its first unit proof.
        return False
    # Non-integral values: tight tolerance for seat rounding (0.333
    # vs 1/3), far too tight to pass a genuinely different value.
    return abs(a - b) <= 1e-3 * max(1.0, abs(a), abs(b))


def _quoted(text):
    return [a or b for a, b in re.findall(
        r'"([^"\n]{3,})"|`([^`\n]{3,})`', text or "")]


# ------------------------------------------------- arithmetic safe eval
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                   ast.Mod, ast.Pow)


def _safe_eval_arith(expr):
    """Evaluate a pure arithmetic expression. AST whitelist: numeric
    constants, + - * / // % **, unary +/-, parentheses. Nothing else."""
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and \
                    not isinstance(node.value, bool):
                return node.value
            raise ValueError("non-numeric constant")
        if isinstance(node, ast.BinOp) and \
                isinstance(node.op, _ALLOWED_BINOPS):
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                return l / r
            if isinstance(node.op, ast.FloorDiv):
                return l // r
            if isinstance(node.op, ast.Mod):
                return l % r
            if isinstance(node.op, ast.Pow):
                if abs(r) > 12 or abs(l) > 10 ** 6:
                    raise ValueError("pow out of safe bounds")
                return l ** r
        if isinstance(node, ast.UnaryOp) and \
                isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        raise ValueError("disallowed node " + type(node).__name__)

    return ev(tree)


def _find_arith_expr(prompt):
    """Longest arithmetic-looking run in the prompt: >=2 numbers joined
    by operators. Returns a cleaned expression or None."""
    best = None
    pat = re.compile(
        r"\d[\d,]*(?:\.\d+)?"
        r"(?:\s*[+\-*/x×^%]\s*\(?\s*-?\d[\d,]*(?:\.\d+)?\s*\)?)+")
    for m in pat.finditer(prompt or ""):
        s = m.group(0)
        if len(re.findall(r"\d", s)) >= 2:
            if best is None or len(s) > len(best):
                best = s
    if not best:
        return None
    expr = best.replace(",", "").replace("x", "*").replace("×", "*")
    expr = expr.replace("^", "**")
    expr = re.sub(r"\s+", "", expr)
    while expr and expr[-1] not in "0123456789)":
        expr = expr[:-1]
    if expr.count("(") != expr.count(")"):
        return None
    return expr or None


# ------------------------------------------------------------ claim check
def _claim_check(prompt, seat_text, criteria=""):
    """Independent re-run of the question through /run_verified
    (quicklook + Layer-2 deterministic gates), then DETERMINISTIC
    agreement: the seat answer's load-bearing numbers and quoted
    strings must be supported by the verified path's answer, and the
    verified path must have passed its own gates. The Skywork score
    is carried in the detail for the record; it is never decisive."""
    status, out = _post("/run_verified", {
        "specialist": "quicklook", "prompt": prompt,
        "criteria": criteria, "kw": {"no_think": True}}, timeout=1500)
    if status != 200 or not isinstance(out, dict) or "text" not in out:
        return {"kind": "CLAIM-CHECK", "verdict": "UNGATED",
                "detail": "verified path unavailable (http %s): %s"
                          % (status, str(out)[:220])}
    vtext = out.get("text") or ""
    gates_passed = out.get("gates_passed")
    seat_nums = _numbers(seat_text)
    ver_nums = _numbers(vtext)
    unsupported = [n for n in seat_nums
                   if not any(_num_eq(n, m) for m in ver_nums)]
    vlow = vtext.lower()
    bad_q = [q for q in _quoted(seat_text) if q.lower() not in vlow]
    parts = ["verified-path gates_passed=%s" % gates_passed,
             "seat numbers checked=%d" % len(seat_nums),
             "verified score (non-decisive)=%s" % out.get("score")]
    if out.get("gate_failures"):
        parts.append("verified-path gate failures: %s"
                     % "; ".join(out["gate_failures"])[:300])
    if unsupported:
        parts.append("UNSUPPORTED seat numbers: %s"
                     % ", ".join("%g" % n for n in unsupported[:6]))
    if bad_q:
        parts.append("UNSUPPORTED seat quotes: %s"
                     % "; ".join(repr(q[:50]) for q in bad_q[:3]))
    if gates_passed and not unsupported and not bad_q:
        return {"kind": "CLAIM-CHECK", "verdict": "PASS",
                "detail": "; ".join(parts)}
    return {"kind": "CLAIM-CHECK", "verdict": "FAIL",
            "detail": "; ".join(parts)}


# ----------------------------------------------------------------- gates
def gate_math(prompt, text):
    expr = _find_arith_expr(prompt)
    if expr:
        try:
            expected = _safe_eval_arith(expr)
        except Exception:
            expected = None
        if expected is not None:
            got = _final_number(text)
            if got is None:
                return {"kind": "ARITH-RECOMPUTE", "verdict": "FAIL",
                        "detail": "prompt expression %r recomputes to "
                                  "%s; seat answer contains no number"
                                  % (expr, expected)}
            if _num_eq(got, expected):
                return {"kind": "ARITH-RECOMPUTE", "verdict": "PASS",
                        "detail": "recomputed %s = %s in Python; seat "
                                  "final number %s matches"
                                  % (expr, expected, got)}
            return {"kind": "ARITH-RECOMPUTE", "verdict": "FAIL",
                    "detail": "recomputed %s = %s in Python; seat "
                              "final number %s does NOT match"
                              % (expr, expected, got)}
    g = _claim_check(prompt, text,
                     criteria="Answer the arithmetic question exactly "
                              "and give the final number.")
    g["kind"] = "CROSS-CHECK (run_verified)"
    return g


def _extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "",
                        re.DOTALL)
    if blocks:
        return blocks[0]
    t = text or ""
    if re.match(r"\s*(#!/|import |from |def |class )", t):
        return t
    return None


def gate_code(prompt, text, kw):
    """Execute the seat's code via multiplex._execute_code_gate (the
    existing gate machinery: gate-checks dir, 60s timeout). Lazy
    import — inside the API server this is the same in-process module;
    the CLI stays light."""
    import multiplex
    run_text = text or ""
    if kw.get("test_code"):
        code = _extract_code(text)
        if code:
            run_text = "```python\n" + code.rstrip() + "\n" + \
                str(kw["test_code"]).rstrip() + "\n```"
    ex = multiplex._execute_code_gate({"text": run_text})
    if not ex.get("ran"):
        return {"kind": "CODE-EXECUTION", "verdict": "UNGATED",
                "detail": "no executable code block could be "
                          "extracted/executed: %s"
                          % ex.get("note", "unknown")}
    if ex.get("exit") != 0:
        tail = (ex.get("stderr_tail") or ex.get("note") or "")[-300:]
        return {"kind": "CODE-EXECUTION", "verdict": "FAIL",
                "detail": "deliverable exited %s: %s"
                          % (ex.get("exit"), tail)}
    expect = kw.get("expect_in_output")
    if expect and str(expect) not in (ex.get("output_tail") or ""):
        return {"kind": "CODE-EXECUTION", "verdict": "FAIL",
                "detail": "executed cleanly (exit 0) but expected "
                          "output %r not in stdout tail %r"
                          % (expect, (ex.get("output_tail") or "")[-200:])}
    return {"kind": "CODE-EXECUTION", "verdict": "PASS",
            "detail": "executed via _execute_code_gate, exit 0; "
                      "stdout tail: %r"
                      % (ex.get("output_tail") or "")[-200:]}


_PLACEHOLDER = re.compile(
    r"lorem ipsum|placeholder|\bTODO\b|\[insert|\bTBD\b|xxx{2,}",
    re.IGNORECASE)


def _words(s):
    return set(re.findall(r"[a-z0-9]{3,}", (s or "").lower()))


def gate_draft(prompt, text, kw):
    stripped = (text or "").strip()
    fails = []
    if len(stripped) < 200:
        fails.append("length floor: %d chars < 200" % len(stripped))
    if _PLACEHOLDER.search(stripped):
        fails.append("placeholder/lorem text present")
    src = _words(prompt) | _words(kw.get("source", ""))
    outw = _words(stripped)
    jac = (len(src & outw) / len(src | outw)) if src and outw else 0.0
    containment = (len(src & outw) / len(outw)) if outw else 0.0
    if jac < 0.02 and containment < 0.35:
        fails.append("keyword defenses: jaccard %.3f and source-word "
                     "containment %.3f — output appears unrelated to "
                     "the task/source" % (jac, containment))
    cc = _claim_check(prompt, stripped,
                      criteria="Check the draft's factual claims "
                               "against the source/task; invented "
                               "facts must not be supported.")
    det = ("chars=%d; jaccard=%.3f; containment=%.3f"
           % (len(stripped), jac, containment))
    if fails:
        verdict = "FAIL"
    elif cc["verdict"] == "FAIL":
        verdict = "FAIL"
    elif cc["verdict"] == "PASS":
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    return {"kind": "PROSE-GATE", "verdict": verdict,
            "detail": "certifies: substantive (>=200 chars), "
                      "non-placeholder, on-task prose whose checkable "
                      "items agree with an independent gated path. "
                      "Does NOT certify style/tone or facts with no "
                      "checkable item. [%s] claim-check: %s — %s"
                      % (det, cc["verdict"], cc["detail"])}


def gate_embed(prompt, kw, first_out):
    """Determinism gate: second identical run must reproduce the
    ranking order, the returned cosine scores within epsilon, and the
    backend; dimension must be sane and consistent."""
    status, out2 = _post("/run", {"specialist": "embed",
                                  "prompt": prompt, "kw": kw},
                         timeout=300)
    if status != 200 or not isinstance(out2, dict):
        return {"kind": "DETERMINISM-GATE", "verdict": "UNGATED",
                "detail": "second embed run failed (http %s): %s"
                          % (status, str(out2)[:200])}
    b1, b2 = first_out.get("backend"), out2.get("backend")
    if b1 != b2:
        return {"kind": "DETERMINISM-GATE", "verdict": "FAIL",
                "detail": "backend changed between runs: %s vs %s "
                          "(nondeterministic serving path)" % (b1, b2)}
    if "ranked" in first_out and "ranked" in out2:
        r1 = [(r["text"], r["score"]) for r in first_out["ranked"]]
        r2 = [(r["text"], r["score"]) for r in out2["ranked"]]
        order_same = [t for t, _ in r1] == [t for t, _ in r2]
        drift = None
        if len(r1) == len(r2) and r1:
            drift = max(abs(a - b) for (_, a), (_, b) in zip(r1, r2))
        ok = order_same and drift is not None and drift <= 0.01
        return {"kind": "DETERMINISM-GATE",
                "verdict": "PASS" if ok else "FAIL",
                "detail": "backend=%s; ranking order identical=%s; "
                          "max cosine-score drift=%s (epsilon 0.01); "
                          "scores are the seat's returned cosine "
                          "similarities (raw vectors are not exposed "
                          "by the seat API, so self-similarity is "
                          "checked on the scores)"
                          % (b1, order_same, drift)}
    d1, d2 = first_out.get("embedding_dim"), out2.get("embedding_dim")
    ok = d1 is not None and d1 == d2 and d1 > 0
    return {"kind": "DETERMINISM-GATE",
            "verdict": "PASS" if ok else "FAIL",
            "detail": "backend=%s; embedding_dim %s vs %s across two "
                      "runs (no corpus supplied, dimension-sanity "
                      "form of the gate)" % (b1, d1, d2)}


def _attach_gate(out, gate):
    out["gate"] = gate
    if gate.get("verdict") == "FAIL":
        out["gate_failed"] = True
        out["answer_status"] = "gate_failed"
    elif gate.get("verdict") in ("UNGATED", "PARTIAL"):
        out["gate_failed"] = True
        out["answer_status"] = "ungated"
    elif gate.get("verdict") == "PASS":
        out["answer_status"] = "gated"
    return out


# ------------------------------------------------------------------ route
def route(task_type, prompt, kw=None, criteria="", max_tokens=None):
    kw = dict(kw or {})
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    t0 = time.time()

    if task_type == "deep":
        import hal_dispatch
        out = hal_dispatch.dispatch_deep(
            prompt, max_tokens=max_tokens or 1024)
        out = dict(out)
        if out.get("routed") == "queued":
            gate = {"kind": "CLAIM-CHECK", "verdict": "QUEUED",
                    "detail": "deep lane not live (%s); nothing "
                              "dispatched, nothing to gate"
                              % out.get("reason", "")}
            out["gate"] = gate
            out["answer_status"] = "queued"
        else:
            gate = _claim_check(
                prompt, out.get("text", ""),
                criteria="Cross-check the deep answer's extractable "
                         "claims; unsupported claims fail.")
            gate["kind"] = "CLAIM-CHECK (cross-bench)"
            _attach_gate(out, gate)
        _log(task_type, out.get("routed", "hal/strata"),
             out.get("status", "queued"), time.time() - t0, prompt,
             gate=gate)
        out["task_type"] = task_type
        return out

    if task_type in _VERIFY:
        status, out = _post("/run_verified", {
            "specialist": "quicklook", "prompt": prompt,
            "criteria": criteria, "kw": kw}, timeout=1500)
        out = dict(out) if isinstance(out, dict) else {"result": out}
        out["routed"] = "local/quicklook+verified-gate"
        out["http_status"] = status
        if status == 200 and "text" in out:
            verdict = "PASS" if out.get("verified") else "FAIL"
            gate = {"kind": "VERIFIED-GATE", "verdict": verdict,
                    "detail": "run_verified unchanged: gates_passed=%s "
                              "verified=%s score=%s gate_failures=%s"
                              % (out.get("gates_passed"),
                                 out.get("verified"), out.get("score"),
                                 out.get("gate_failures"))}
            _attach_gate(out, gate)
        else:
            gate = {"kind": "VERIFIED-GATE", "verdict": "ERROR",
                    "detail": "verified path failed (http %s): %s"
                              % (status, str(out)[:220])}
            out["gate"] = gate
        _log(task_type, out["routed"], status, time.time() - t0,
             prompt, gate=gate)
        out["task_type"] = task_type
        return out

    if task_type == "vision":
        ctx = kw.get("ground_truth") or kw.get("context")
        img = kw.get("image_path") or kw.get("image_b64")
        if not img or not ctx:
            missing = "image (kw image_path/image_b64)" if not img \
                else "ground-truth context (kw ground_truth/context)"
            gate = {"kind": "UNGATED-VISION", "verdict": "REFUSED",
                    "detail": "no honest gate available without %s; "
                              "refusing to return a naked vision "
                              "answer" % missing}
            out = {"error": "vision route refused: missing %s — an "
                            "ungated vision answer is not served"
                            % missing,
                   "kind": "ungated", "gate": gate,
                   "http_status": 503,
                   "routed": "local-or-frank/vision"}
            _log(task_type, out["routed"], 503, time.time() - t0,
                 prompt, gate=gate)
            out["task_type"] = task_type
            return out
        status, out = _post("/run", {"specialist": "vision",
                                     "prompt": prompt, "kw": kw},
                            timeout=900)
        out = dict(out) if isinstance(out, dict) else {"result": out}
        out["routed"] = "local-or-frank/vision"
        out["http_status"] = status
        if status != 200 or "text" not in out:
            gate = {"kind": "CLAIM-CHECK", "verdict": "ERROR",
                    "detail": "vision seat call failed (http %s): %s"
                              % (status, str(out)[:220])}
            out["gate"] = gate
            _log(task_type, out["routed"], status, time.time() - t0,
                 prompt, gate=gate)
            out["task_type"] = task_type
            return out
        gate = _claim_check(
            "Ground-truth context:\n%s\n\nQuestion: %s" % (ctx, prompt),
            out.get("text", ""),
            criteria="Answer strictly from the ground-truth context.")
        gate["kind"] = "CLAIM-CHECK (vision, ground-truth)"
        _attach_gate(out, gate)
        _log(task_type, out["routed"], status, time.time() - t0,
             prompt, gate=gate)
        out["task_type"] = task_type
        return out

    if task_type in _LOCAL:
        kind, specialist = _LOCAL[task_type]
        for k, v in _KW_DEFAULTS.get(task_type, {}).items():
            kw.setdefault(k, v)
        status, out = _post("/run", {"specialist": specialist,
                                     "prompt": prompt, "kw": kw},
                            timeout=900)
        out = dict(out) if isinstance(out, dict) else {"result": out}
        out["routed"] = f"local-or-frank/{specialist}"
        out["http_status"] = status
        seat_ok = status == 200 and "error" not in out and (
            "text" in out or "ranked" in out or "embedding_dim" in out)
        if not seat_ok:
            gate = {"kind": _GATE_KIND[task_type], "verdict": "ERROR",
                    "detail": "seat call failed (http %s): %s"
                              % (status, str(out)[:220])}
            out["gate"] = gate
            _log(task_type, out["routed"], status, time.time() - t0,
                 prompt, gate=gate)
            out["task_type"] = task_type
            return out
        text = out.get("text", "")
        if task_type == "math":
            gate = gate_math(prompt, text)
        elif task_type == "code":
            gate = gate_code(prompt, text, kw)
        elif task_type == "lookup":
            gate = _claim_check(
                prompt, text,
                criteria=criteria or "Answer the question factually "
                                     "and concisely.")
        elif task_type == "reason":
            gate = _claim_check(
                prompt, text,
                criteria=criteria or "Reason carefully and give a "
                                     "definite final answer.")
        elif task_type == "draft":
            gate = gate_draft(prompt, text, kw)
        elif task_type in ("embed", "triage"):
            gate = gate_embed(prompt, kw, out)
        else:  # pragma: no cover — _LOCAL keys are all handled above
            gate = {"kind": _GATE_KIND[task_type], "verdict": "UNGATED",
                    "detail": "no gate implemented for this type"}
        _attach_gate(out, gate)
        _log(task_type, out["routed"], status, time.time() - t0,
             prompt, gate=gate)
        out["task_type"] = task_type
        return out

    return {"error": f"unknown task_type {task_type!r}",
            "known": sorted(_LOCAL) + sorted(_VERIFY) + ["deep"],
            "task_type": task_type}


ROUTES = {**{k: v[1] for k, v in _LOCAL.items()},
          "verify": "quicklook+verified-gate",
          "rederive": "quicklook+verified-gate",
          "deep": "hal/strata (gateway, live-flag gated)"}

GATES = dict(_GATE_KIND, verify="VERIFIED-GATE", rederive="VERIFIED-GATE")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-type", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--criteria", default="")
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--max-tokens", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(route(a.task_type, a.prompt, json.loads(a.kw),
                           a.criteria, a.max_tokens), indent=1)[:8000])
