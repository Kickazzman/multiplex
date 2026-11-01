"""GLaDOS specialist multiplexer v2 — duplexed specialists, both GPUs.

Perg's design: instead of one giant brain doing everything, many small
specialist models (university/research-lab picks) run multiplexed, each
best at its job. Shared context lives in RAM via blackboard.py. A trained
verifier (Skywork-Reward-V2) is DUPLEXED with the workers: a worker
produces, the verifier scores against the criteria, low score = the
critique is fed back and the worker redoes it.

Roster (v2):
  agent     Hammer 2.1 3B (MadeAgents) - tool-calling specialist.
            llama.cpp CUDA (:8201), real tool-execution loop (calculator, time).
  code      MagicoderS-DS-6.7B (UIUC) - code specialist. llama.cpp CUDA (:8202).
  quicklook Qwen3-4B - fast drafting/synthesis + critique writer.
            llama.cpp CUDA (:8203).
  embed     bge-m3 on frank's infinity_emb service (:7997, GPU) -
            retrieval/ranking. Falls back to local bge-small (CPU).
  tts       F5-TTS (SJTU) - voice specialist. multiplex-env subprocess.
            Falls back to Chatterbox if F5 is unavailable.
  verify    Skywork-Reward-V2-Llama-3.2-1B - the duplex judge.
            verifier_server.py on :8199 (GPU).

Client mode: set MULTIPLEX_API=http://<mrfloppy>:8198 and run_specialist /
run_verified / verify proxy to the multiplex API on mrfloppy (this is how
frank's specialists join; the blackboard stays on mrfloppy, in RAM).

API:
    run_specialist(name, prompt, **kw) -> result
    run_parallel(tasks) -> {task_id: result}
    verify(prompt, response, criteria="") -> float score
    run_verified(name, prompt, criteria="", threshold=None,
                 max_retries=2, **kw) -> best result + attempt history
"""
import ast
import json
import os
import re
import subprocess
import sys
import time
import threading
import urllib.request
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blackboard  # noqa: E402

OUT_DIR = HERE / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# llama.cpp servers (CUDA build at ~/llm/llama.cpp/build) - one process
# per specialist, models resident on the 4090. The box's Ollama install
# is a CPU-only build (no libggml-cuda.so), hence the 2026-10-20 switch
# from Ollama to llama-server for agent/code/quicklook.
AGENT_URL = os.environ.get("LLAMA_AGENT_URL", "http://127.0.0.1:8201")
CODE_URL = os.environ.get("LLAMA_CODE_URL", "http://127.0.0.1:8202")
QUICK_URL = os.environ.get("LLAMA_QUICK_URL", "http://127.0.0.1:8203")
FRANK = os.environ.get("FRANK_HOST", "[REDACTED-IP]")
INFINITY_URL = os.environ.get("INFINITY_URL", f"http://{FRANK}:7997")
VERIFIER_URL = os.environ.get("VERIFIER_URL", "http://127.0.0.1:8199")
# Micro-specialists (university sweep, added 2026-10-20): step verifier +
# vision on mrfloppy; deep reasoner + STT on frank; TRM on mrfloppy.
SHEPHERD_URL = os.environ.get("SHEPHERD_URL", "http://127.0.0.1:8206")
VISION_URL = os.environ.get("VISION_URL", "http://127.0.0.1:8204")
OURO_URL = os.environ.get("OURO_URL", f"http://{FRANK}:8208")
STT_URL = os.environ.get("STT_URL", f"http://{FRANK}:8209")
TRM_URL = os.environ.get("TRM_URL", "http://127.0.0.1:8210")
# Expanded cast (2026-10-22): Phase-2 sweep winners, job-geared seats.
MATHWHIZ_URL = os.environ.get("MATHWHIZ_URL", "http://127.0.0.1:8207")
DRAFT_URL = os.environ.get("DRAFT_URL", "http://127.0.0.1:8211")
API_URL = os.environ.get("MULTIPLEX_API", "")  # set => client mode
MPX_ENV_PY = os.environ.get("MPX_ENV_PY", "/home/joshua/llm/multiplex-env/bin/python")
MPX_ENV_BIN = str(Path(MPX_ENV_PY).parent)

AGENT_MODEL = "hammer"
CODE_MODEL = "magicoder"
QUICK_MODEL = "qwen3-4b"
MATHWHIZ_MODEL = "minicpm5-2b"
DRAFT_MODEL = "phi-4-mini"
DEFAULT_THRESHOLD = float(os.environ.get("VERIFY_THRESHOLD", "1.0"))

_locks = {name: threading.Lock()
          for name in ("agent", "code", "quicklook", "embed", "tts", "verify",
                       "shepherd", "vision", "stt", "ouro", "trm",
                       "mathwhiz", "draft")}
_loaded = {}


# Enforcement override (2026-10-30): thread-local base-URL
# override for _llama_chat, set only around one enforced generation
# inside the router's generation lock (mode_enforce.py). Unset =>
# zero behavior change; every specialist keeps its own URL/model.
_enforce_base_url = threading.local()


def set_enforce_base_url(url):
    _enforce_base_url.url = url


def clear_enforce_base_url():
    _enforce_base_url.url = None


def _http_json(url, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ------------------------------------------------------------- llama.cpp
def _llama_chat(base_url, model, messages, tools=None, num_predict=512,
                timeout=1800, think=None, tool_choice=None,
                response_format=None, chat_template_kwargs=None):
    """OpenAI-compatible chat against a per-specialist llama-server.
    The response is normalized back to the Ollama shape
    ({"message": {"content":..., "tool_calls":[{"function": {...}}]}})
    so the agent loop and think-stripping code work unchanged.
    `think` is accepted for signature compat and ignored: Qwen3's
    thinking arrives inline as <think> blocks in the content.
    `tool_choice` / `response_format` (2026-10-20 hardening): passed
    through when given. Probed on :8201 - tool_choice "required" is
    accepted (HTTP 200) though Hammer still answers in content text;
    response_format json_schema IS honored by llama-server and forces
    the content into strict schema-shaped JSON."""
    msgs = []
    for m in messages:
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if m.get("tool_calls"):
            fixed = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                fixed.append({"type": "function", "function": {
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments", {})}})
            m["tool_calls"] = fixed
        msgs.append(m)
    payload = {"model": model, "messages": msgs, "stream": False,
               "max_tokens": num_predict, "temperature": 0.0}
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if response_format:
        payload["response_format"] = response_format
    override = getattr(_enforce_base_url, "url", None)
    resp = _http_json((override or base_url) + "/v1/chat/completions",
                      payload, timeout=timeout)
    choice = (resp.get("choices") or [{}])[0]
    omsg = choice.get("message", {}) or {}
    calls = []
    for tc in omsg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:
                args = {}
        calls.append({"function": {"name": fn.get("name"),
                                   "arguments": args}})
    return {"message": {"content": omsg.get("content") or "",
                        "reasoning_content":
                            omsg.get("reasoning_content") or "",
                        "tool_calls": calls}}


# ------------------------------------------------------------------- tools
def _calculator(expression: str) -> str:
    """AST-safe arithmetic evaluator (no names, no calls, numbers only)."""
    ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
           ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
           ast.Pow: lambda a, b: a ** b, ast.USub: lambda a: -a,
           ast.UAdd: lambda a: +a}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError(f"disallowed expression element: {type(node).__name__}")

    return str(ev(ast.parse(expression, mode="eval")))


def _get_current_time(_: str = "") -> str:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z") + f" (unix {int(time.time())})"


def _dedupe_lines(text, max_lines=40):
    """Collapse repeated lines (chatty-tool stderr floods, e.g. rizin's
    warnings repeated per round) to first occurrence + a repeat count.
    Ported in spirit from the worker's context hygiene (audit F6)."""
    seen = {}
    order = []
    for ln in (text or "").splitlines():
        key = ln.strip()
        if key not in seen:
            seen[key] = 0
            order.append(ln)
        seen[key] += 1
    out = []
    for ln in order[:max_lines]:
        n = seen[ln.strip()]
        out.append(ln + (f"  [...repeated x{n}]" if n > 1 else ""))
    if len(order) > max_lines:
        out.append(f"[...{len(order) - max_lines} more distinct lines "
                   "collapsed]")
    return "\n".join(out)


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


def _run_command(command: str) -> str:
    """Run a shell command via bash and return its output, chunked.

    RE-bridge (added 2026-10-20 for the RE-suite test): the multiplex
    had no shell path for specialists. Output is capped so a big
    objdump/strings dump cannot drown a small model's context
    (playbook rule: narrow with head/grep instead).
    Hardening (2026-10-20): stdout and stderr are kept SEPARATE (the
    worker's _tool_shell rule - stderr can never displace stdout),
    stderr is line-deduped and hard-capped, stdout gets the budget
    priority. An empty stdout is reported with the explicit marker
    '[stdout empty]' so the agent loop's empty-result rule can fire."""
    if not command or not command.strip():
        return "ERROR: empty command"
    try:
        r = subprocess.run(["bash", "-c", command], capture_output=True,
                           text=True, timeout=90)
        stdout = _ANSI_RE.sub("", (r.stdout or "")).strip()
        stderr = _ANSI_RE.sub("", (r.stderr or "")).strip()
        rc = r.returncode
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 90s - narrow it (head/grep)."
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    parts = []
    if stdout:
        if len(stdout) > 2800:
            stdout = (stdout[:1900] + "\n...[stdout truncated - narrow "
                      "the command with head/grep/sed]...\n" + stdout[-700:])
        parts.append(stdout)
    else:
        parts.append("[stdout empty]")
    if stderr:
        err_txt = _dedupe_lines(stderr)
        if len(err_txt) > 600:
            err_txt = err_txt[:600] + "\n...[stderr truncated]"
        parts.append("[stderr]\n" + err_txt)
    if rc != 0:
        parts.append(f"[exit code {rc}]")
    return "\n".join(parts)


TOOL_FNS = {"calculator": _calculator, "get_current_time": _get_current_time, "run_command": _run_command}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly. Use for any math.",
        "parameters": {"type": "object",
                       "properties": {"expression": {
                           "type": "string",
                           "description": "Arithmetic expression, e.g. '17 * 24 + 6'"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current date and time. Use whenever the user asks the time or date.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command on this machine (bash) and "
                       "get its output. Use for inspecting files and "
                       "binaries: file, strings, objdump, rizin, xxd, "
                       "strace, python3, etc. Output is truncated, so "
                       "narrow big outputs with head/grep.",
        "parameters": {"type": "object",
                       "properties": {"command": {
                           "type": "string",
                           "description": "Shell command to run, e.g. "
                                          "'strings -n 8 /path/to/binary | head -50'"}},
                       "required": ["command"]}}},
]

# ------------------------------------------------- tool contract layer
# Hardening (2026-10-20): ports of the old worker's tool discipline
# (local_agent_worker.py: _validate_tool_args L509, _is_substantive
# L471, run_task oscillation/recon counters). The multiplex previously
# coerced args silently (`args.get("expression") or ... or ""`) and
# never inspected its own trace.

def _tool_arg_spec():
    """{tool_name: (required_arg, arg_type)} derived from TOOL_SCHEMAS,
    so the contract lives in exactly one place."""
    spec = {}
    for s in TOOL_SCHEMAS:
        fn = s["function"]
        props = fn["parameters"].get("properties", {})
        req = fn["parameters"].get("required", [])
        if req:
            spec[fn["name"]] = (req[0], props.get(req[0], {}).get(
                "type", "string"))
        else:
            spec[fn["name"]] = (None, None)
    return spec


TOOL_ARG_SPEC = _tool_arg_spec()
_FINAL_ANSWER = "final_answer"

# Final-gate claim fields (2026-10-20): the RE final stage's
# final_answer must carry these machine-readable claims alongside the
# prose report. No fields, no certification - the binary grades
# fields, it cannot grade paragraphs.
CLAIM_FIELDS = ("claimed_secret", "claimed_key",
                "claimed_checksum_value", "checksum_algorithm")


def _build_call_schema(claims=False):
    """json_schema for one agent work-round reply, GENERATED from
    TOOL_SCHEMAS (audit import E1: llama-server translates it to GBNF
    server-side). The reply is either a tool call or an explicit
    final_answer claim - a plan leaked as prose becomes unemittable.
    Probed live on :8201: the server honors this (HTTP 200, strict
    JSON out) where tool_choice alone did not change Hammer.
    claims=True adds the CLAIM_FIELDS to the args object so a
    final_answer can carry structured load-bearing claims."""
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    props = {}
    for s in TOOL_SCHEMAS:
        for k, v in s["function"]["parameters"].get("properties",
                                                    {}).items():
            props.setdefault(k, {"type": v.get("type", "string")})
    props["answer"] = {"type": "string"}
    if claims:
        for f in CLAIM_FIELDS:
            props[f] = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": names + [_FINAL_ANSWER]},
            "args": {"type": "object", "properties": props},
        },
        "required": ["tool", "args"],
    }


CALL_RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "multiplex_tool_call", "schema": _build_call_schema()}}
CALL_RESPONSE_FORMAT_CLAIMS = {"type": "json_schema", "json_schema": {
    "name": "multiplex_tool_call_claims",
    "schema": _build_call_schema(claims=True)}}


def _validate_agent_args(name, args):
    """Worker _validate_tool_args, adapted: returns None if OK, else a
    CORRECTIVE error string the model can act on. Never coerces."""
    if name not in TOOL_FNS:
        return (f"unknown tool '{name}' - the only tools are: "
                + ", ".join(sorted(TOOL_FNS)))
    if not isinstance(args, dict):
        return (f"{name} arguments must be a JSON object, e.g. "
                f'{{"command": "..."}} - got {type(args).__name__}')
    req, _typ = TOOL_ARG_SPEC.get(name, (None, None))
    if req is None:
        return None
    if req not in args:
        others = [k for k in args if k not in ("answer",)]
        hint = (f" (you sent '{others[0]}' - the argument is named "
                f"'{req}')") if others else ""
        return f"{name} requires the argument '{req}'{hint}"
    val = args.get(req)
    if not isinstance(val, str) or not val.strip():
        return f"{name} argument '{req}' must be a non-empty string"
    if name == "calculator" and re.search(r"[;|&`$<>\\n]", val):
        return ("calculator takes a pure arithmetic expression only - "
                "no shell syntax. Use run_command for shell commands.")
    return None


def _dispatch_tool(name, args):
    """Validate (correctively) then execute one tool call.
    Returns the result string exactly as the model will see it."""
    err = _validate_agent_args(name, args)
    if err:
        return "ERROR (arguments): " + err
    fn = TOOL_FNS[name]
    req, _typ = TOOL_ARG_SPEC.get(name, (None, None))
    arg = args.get(req, "") if req else ""
    try:
        return str(fn(arg))
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"


def _is_substantive_result(name, result):
    """Worker _is_substantive, adapted to string results: trivial or
    empty output is not proof of work."""
    r = str(result or "")
    if r.startswith("ERROR"):
        return False
    if name == "run_command":
        body = r.replace("[stdout empty]", "").strip()
        return len(body) > 10
    if name == "calculator":
        return bool(r.strip())
    return True


def _call_hash(name, args):
    try:
        return (name, json.dumps(args or {}, sort_keys=True,
                                 default=str)[:300])
    except Exception:
        return (name, "")


_READONLY_CMD = re.compile(
    r"^\s*(ls|cat|head|tail|grep|find|wc|echo|pwd|whoami)\b")


def _is_readonly_call(name, args):
    """Worker Guard 3 classifier: read-only actions make no progress
    markers. RE inspection binaries (strings/objdump/rizin/...) are
    the work itself here, so they count as substantive actions."""
    if name in ("calculator", "get_current_time"):
        return True
    if name == "run_command":
        cmd = str((args or {}).get("command", ""))
        return bool(_READONLY_CMD.match(cmd)) and ">" not in cmd
    return False


def _result_stdout_empty(name, result):
    r = str(result or "")
    if name == "run_command":
        return r.startswith("[stdout empty]") or r.strip() == ""
    return not r.strip() or r.startswith("ERROR")


def _norm_call(name, args):
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except Exception:
            args = {"_raw": args}
    return {"name": name, "arguments": args if isinstance(args, dict)
            else {}}


def _parse_tool_call_xml(text):
    """Ported from the worker (L437): Qwen XML dialect
    <tool_call><function=NAME><parameter=K>V</parameter>...</tool_call>."""
    t = text or ""
    m = re.search(r"<function=([^>]+)>", t)
    if not m:
        return None
    name = m.group(1).strip().strip("'\"")
    args = {}
    for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", t,
                          re.DOTALL):
        args[pm.group(1).strip()] = pm.group(2)
    return _norm_call(name, args) if name else None


def _salvage_call(item_text):
    """Regex-salvage one malformed call dict: pull the name and any
    quoted argument values out of sloppy model output."""
    m = re.search(r"""['"]name['"]\s*:\s*['"]([^'"]+)['"]""", item_text)
    if not m:
        return None
    args = {}
    for pat in (re.compile(r"'([A-Za-z_][\w-]*)'\s*:\s*'((?:[^'\\]|\\.)*)'"),
                re.compile(r'"([A-Za-z_][\w-]*)"\s*:\s*"((?:[^"\\]|\\.)*)"')):
        for km in pat.finditer(item_text):
            key = km.group(1)
            if key in ("name", "type") or key in args:
                continue
            q = "'" if pat.pattern.startswith("'") else '"'
            try:
                args[key] = ast.literal_eval(q + km.group(2) + q)
            except Exception:
                args[key] = km.group(2)
        if args:
            break
    return _norm_call(m.group(1), args)


def _parse_literal_list(text):
    """Parse a Python-literal list of call dicts ITEM BY ITEM, so one
    malformed element (the wild lists drop braces and mangle escapes -
    job3's list is one brace short inside item 1) cannot sink the
    whole list the way all-or-nothing ast.literal_eval did on RE
    jobs 2/3. Item spans are bounded by the next item start (hybrid
    outer dicts {'type': ...} when present, else native {'name': ...}
    dicts); each item is literal_eval'd, else regex-salvaged."""
    calls = []
    hybrid = [m.start() for m in re.finditer(
        r"""\{\s*['"]type['"]\s*:""", text)]
    native = [m.start() for m in re.finditer(
        r"""\{\s*['"]name['"]\s*:""", text)]
    starts = hybrid if hybrid else native
    for idx, s in enumerate(starts):
        limit = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        depth = 0
        quote = None
        esc = False
        end = None
        for i in range(s, limit):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if quote:
                if ch == quote:
                    quote = None
                continue
            if ch in ("'", '"'):
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        chunk = text[s:end] if end else text[s:limit].rstrip(",] \n")
        item = None
        try:
            obj = ast.literal_eval(chunk)
            if isinstance(obj, dict):
                item = obj
        except Exception:
            item = None
        if item is not None:
            if "name" in item:
                calls.append(_norm_call(item.get("name"),
                                        item.get("arguments", {})))
            elif isinstance(item.get("function"), dict) and \
                    item["function"].get("name"):
                fn = item["function"]
                calls.append(_norm_call(fn["name"],
                                        fn.get("arguments", {})))
            continue
        salv = _salvage_call(chunk)
        if salv:
            calls.append(salv)
    return calls
def _extract_raw_tool_calls(content: str):
    """Fallback parsers for tool calls emitted as plain text:
    1. strict JSON object {"tool"|"name": ..., "args"|"arguments": {...}}
       (the json_schema-constrained shape; also the worker's contract)
    2. <tool_call>{json}</tool_call> blocks (Qwen-style) and the XML
       <function=...><parameter=...> dialect (worker port)
    3. Python-literal lists in BOTH dialects seen in the wild:
       Hammer native [{'name': ..., 'arguments': {...}}] and the
       OpenAI hybrid [{'type': 'function', 'function': {'name': ...,
       'arguments': {...}}}] - the hybrid is the dialect that leaked
       in RE jobs 2/3 (audit failure 1) and the old parser missed it."""
    calls = []
    text = (content or "").strip()
    if not text:
        return calls
    unfenced = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    # 1. strict JSON object
    if unfenced.startswith("{"):
        try:
            obj = json.loads(unfenced)
            if isinstance(obj, dict):
                nm = obj.get("tool") or obj.get("name")
                if nm:
                    return [_norm_call(nm, obj.get("args",
                                                   obj.get("arguments",
                                                           {})))]
        except Exception:
            pass
    # 2. tagged blocks
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                         text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            nm = obj.get("name") or obj.get("tool")
            if nm:
                calls.append(_norm_call(nm, obj.get(
                    "arguments", obj.get("args", {}))))
        except Exception:
            pass
    if calls:
        return calls
    if "<tool_call>" in text and "<function=" in text:
        one = _parse_tool_call_xml(text)
        if one:
            return [one]
    # 3. literal lists, both dialects, item-at-a-time (malformation-
    # tolerant: wild lists drop braces and mangle escapes)
    if "[" in unfenced:
        calls = _parse_literal_list(unfenced)
    return calls


# ------------------------------------------------------------------ agent
_AGENT_SYSTEM = (
    "You are a tool-using worker. Contract: every reply is either ONE "
    "tool call or the final_answer pseudo-call, in the JSON shape "
    "{\"tool\": <name>, \"args\": {...}}. Tools: calculator(expression), "
    "get_current_time(), run_command(command). When you have what you "
    "need, finish with {\"tool\": \"final_answer\", \"args\": "
    "{\"answer\": \"<your answer, quoting exact tool values>\"}}. "
    "Complete means: your answer quotes facts from actual tool "
    "outputs - never invent, never present a plan as results, and "
    "never paste raw tool output as the answer: synthesize, and "
    "quote the key values. If a "
    "command returns nothing, change the command or the tool; "
    "repeating it changes nothing.")


def _specialist_agent(prompt, **kw):
    """Hammer 2.1 3B with a real tool-execution loop, hardened
    2026-10-20 with ports of the old worker's defenses:
    json_schema-constrained replies (tool_choice 'required' on work
    rounds), corrective arg validation, oscillation AUTO-FAIL (same
    call >=3 in last 5), recon-loop AUTO-FAIL (8 read-only in a row),
    empty-result blocking (an identical call that produced no stdout
    is refused with a strategy-change note), F3 superseded-result
    pruning, and completion-claim discipline (unparsed model text is
    a CLAIM that downstream gates judge - never a free pass)."""
    import urllib.error
    with _locks["agent"]:
        use_tools = kw.get("tools", True)
        tools = TOOL_SCHEMAS if use_tools else None
        max_rounds = kw.get("max_tool_rounds", 5)
        messages = [{"role": "system", "content": _AGENT_SYSTEM}]
        prev_fails = kw.get("previous_failures") or []
        if prev_fails:
            lines = ["PREVIOUS ATTEMPTS AT THIS TASK FAILED. Do not "
                     "repeat these calls or approaches:"]
            for pf in prev_fails[-8:]:
                lines.append("- " + str(pf)[:220])
            messages.append({"role": "user", "content": "\n".join(lines)})
        messages.append({"role": "user", "content": prompt})
        trace = []
        t0 = time.time()
        final_text = ""
        failed = None
        completion_claim = False
        claims_native = None
        req_patterns = []
        for item in kw.get("completion_requires") or []:
            try:
                req_patterns.append((re.compile(item[0]), item[1]))
            except Exception:
                pass

        def _missing_requirements():
            """Stage-contract preconditions (Layer 3 pushed into the
            loop): steps the harness requires in the trace before a
            completion claim is accepted - the worker's zero-tools
            guard generalized to stage exit checks."""
            cmds = [str((t.get("arguments") or {}).get("command", ""))
                    for t in trace if not t.get("blocked")]
            return [label for pat, label in req_patterns
                    if not any(pat.search(c) for c in cmds)]
        action_hashes = []
        readonly_streak = 0
        empty_calls = {}
        blocked_counts = {}
        constraints = {"tool_choice_required": "sent" if use_tools
                       else "n/a",
                       "response_format_json_schema": "sent" if use_tools
                       else "n/a"}
        constraint_live = bool(use_tools)

        def _chat(msgs, constrained):
            nonlocal constraint_live
            if constrained and constraint_live:
                rf = CALL_RESPONSE_FORMAT_CLAIMS if kw.get(
                    "claim_fields") else CALL_RESPONSE_FORMAT
            else:
                rf = None
            tc = "required" if (constrained and constraint_live) else None
            try:
                return _llama_chat(AGENT_URL, AGENT_MODEL, msgs, tools=tools,
                                   num_predict=kw.get("max_tokens", 512),
                                   tool_choice=tc, response_format=rf)
            except urllib.error.HTTPError:
                if not (constrained and constraint_live):
                    raise
                # Server rejected a constraint: fall back for the rest
                # of the run and say so in the result (audit: graceful
                # fallback, noted - never silent).
                constraint_live = False
                constraints["tool_choice_required"] = "rejected-fell-back"
                constraints["response_format_json_schema"] = \
                    "rejected-fell-back"
                return _llama_chat(AGENT_URL, AGENT_MODEL, msgs, tools=tools,
                                   num_predict=kw.get("max_tokens", 512))

        for _round in range(max_rounds):
            # Context budget (worker F1/F3 spirit): stub the oldest
            # tool results before the prompt can overflow the 8k
            # window - llama-server answers overflow with HTTP 400
            # on every later call, which used to crash the whole run.
            # ~16k chars of tool text (dense code/hex runs ~3
            # chars/token) plus the stage prompt stays inside 8192.
            _total = sum(len(str(m.get("content", "")))
                         for m in messages)
            if _total > 16000:
                _tool_idx = [i for i, m in enumerate(messages)
                             if m.get("role") == "tool"][:-3]
                for _i in _tool_idx:
                    _c = str(messages[_i].get("content", ""))
                    if len(_c) > 300 and not _c.startswith(
                            "[SUPERSEDED"):
                        messages[_i]["content"] = (
                            "[SUPERSEDED - context budget]\n"
                            + _c[:300])
                        _total -= len(_c) - 330
                    if _total <= 16000:
                        break
            try:
                resp = _chat(messages, constrained=True)
            except Exception as e:  # noqa: BLE001 - server errors are
                # a failed attempt with the trace preserved, never a
                # crashed run (worker: brain error = task failure).
                failed = (f"AUTO-FAIL: model server error during "
                          f"round {_round}: {type(e).__name__}: "
                          f"{str(e)[:150]}")
                break
            msg = resp.get("message", {})
            calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                calls.append(_norm_call(fn.get("name"),
                                        fn.get("arguments", {})))
            content = msg.get("content", "") or ""
            if not calls:
                calls = _extract_raw_tool_calls(content)
            # final_answer pseudo-call = explicit completion claim
            fin = [c for c in calls if c["name"] == _FINAL_ANSWER]
            if fin:
                missing = _missing_requirements()
                if missing and _round < max_rounds - 1:
                    messages.append({"role": "assistant",
                                     "content": content})
                    messages.append({
                        "role": "user",
                        "content": "HARNESS: stage contract "
                                   "incomplete - these required "
                                   "steps are not in your tool "
                                   "trace yet: " + "; ".join(missing)
                                   + ". Your completion claim is "
                                   "refused. Run them now, then "
                                   "answer."})
                    continue
                _fargs = fin[0]["arguments"] or {}
                _native = {f: _fargs[f] for f in CLAIM_FIELDS
                           if isinstance(_fargs.get(f), str)
                           and _fargs.get(f, "").strip()}
                if kw.get("claim_fields") and _round < max_rounds - 1:
                    # Claims contract (final-gate build): a
                    # final_answer that omits any of the four claim
                    # fields is refused in the loop, like an unmet
                    # stage requirement - the run_verified battery's
                    # prose feedback cannot teach this, so the loop
                    # itself enforces it while rounds remain.
                    _would = _parse_claims(
                        str(_fargs.get("answer", "")), _native)
                    _lacking = [f for f in CLAIM_FIELDS
                                if not _would.get(f)]
                    if _lacking:
                        messages.append({"role": "assistant",
                                         "content": content})
                        messages.append({
                            "role": "user",
                            "content": "HARNESS: claims contract "
                                       "incomplete - these claim "
                                       "fields are empty: "
                                       + ", ".join(_lacking)
                                       + ". Your completion claim "
                                       "is refused. Call "
                                       "final_answer again with "
                                       "every claim field filled "
                                       "from the tool outputs (or "
                                       "the report ends with the "
                                       "CLAIMS block in the exact "
                                       "format given)."})
                        continue
                final_text = str(_fargs.get("answer", ""))
                if _native:
                    claims_native = _native
                completion_claim = True
                break
            if not calls:
                txt = content.strip()
                looks_like_call = any(
                    s in txt for s in ("run_command", "calculator",
                                      "'function'", '"tool"',
                                      "get_current_time"))
                if use_tools and _round < max_rounds - 1 and (
                        not trace or looks_like_call):
                    # Zero-tools / malformed-call guard (worker port):
                    # a completion claim with no evidence is refused in
                    # the loop, not waved through as the answer.
                    why = ("Your last reply could not be parsed as a "
                           "tool call." if looks_like_call else
                           "You have not executed any tools yet. You "
                           "must use tools to do the work before "
                           "completing - answering from memory is a "
                           "fake completion.")
                    messages.append({"role": "assistant", "content": txt})
                    messages.append({
                        "role": "user",
                        "content": "HARNESS: " + why + " Reply with one "
                                   "tool call JSON now."})
                    continue
                missing = _missing_requirements()
                if missing and trace and _round < max_rounds - 1 and \
                        txt.strip("` \n") not in ("", "[]"):
                    messages.append({"role": "assistant",
                                     "content": txt})
                    messages.append({
                        "role": "user",
                        "content": "HARNESS: stage contract "
                                   "incomplete - these required "
                                   "steps are not in your tool "
                                   "trace yet: " + "; ".join(missing)
                                   + ". Your completion claim is "
                                   "refused. Run them now, then "
                                   "answer."})
                    continue
                if trace and txt.strip("` \n") in ("", "[]"):
                    # Model signalled "no more tools" without writing
                    # the answer (Hammer does this). One plain turn.
                    resp2 = _llama_chat(
                        AGENT_URL, AGENT_MODEL,
                        messages + [{"role": "user", "content":
                                     "Now give the final answer in plain "
                                     "text using the tool results above. "
                                     "Quote the exact values the tools "
                                     "printed. Do not call any tools."}],
                        tools=None,
                        num_predict=kw.get("max_tokens", 512))
                    final_text = (resp2.get("message", {}).get("content")
                                  or "").strip()
                else:
                    final_text = txt
                completion_claim = True
                break
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": [{"function": c}
                                            for c in calls]})
            stop = False
            for c in calls:
                name = c["name"]
                args = c["arguments"] or {}
                h = _call_hash(name, args)
                if h in empty_calls:
                    blocked_counts[h] = blocked_counts.get(h, 0) + 1
                    result = ("HARNESS: BLOCKED - this exact call "
                              "already ran and produced no output. "
                              "Repeating it will not change the result. "
                              "Change the tool or the flags (e.g. "
                              "rizin needs -A for analysis; narrow or "
                              "widen the command; try objdump/gdb).")
                    trace.append({"tool": name, "arguments": args,
                                  "result": result, "blocked": True})
                    messages.append({"role": "tool", "content": result})
                    if blocked_counts[h] >= 3:
                        # Run #12 evidence: the model repeated one
                        # blocked decode 10x and burned the whole
                        # round budget. Spam against the block is
                        # oscillation too - fail the attempt cleanly
                        # so the duplex retries with feedback.
                        failed = ("AUTO-FAIL: oscillation detected - "
                                  "model repeated a blocked "
                                  "empty-result call 3+ times: " +
                                  str(name))
                        stop = True
                        break
                    continue
                result = _dispatch_tool(name, args)
                trace.append({"tool": name, "arguments": args,
                              "result": result})
                if _result_stdout_empty(name, result) and \
                        not result.startswith("ERROR (arguments)"):
                    empty_calls[h] = result
                    result += ("\n[harness note: that produced no "
                               " usable output. Do NOT repeat the same"
                               " call - change tool or flags.]")
                # F3 pruning (worker port): a repeated call supersedes
                # the older identical result in context.
                for m in messages:
                    if m.get("role") == "tool" and \
                            m.get("_call") == h and \
                            not str(m.get("content", "")).startswith(
                                "[SUPERSEDED"):
                        m["content"] = ("[SUPERSEDED by a newer "
                                        "identical call - see latest "
                                        "result]\n" +
                                        str(m["content"])[:150])
                messages.append({"role": "tool", "content": str(result),
                                 "_call": h})
                if result.startswith("ERROR (arguments)"):
                    continue  # corrective feedback, not an executed call
                action_hashes.append(h)
                if _is_readonly_call(name, args):
                    readonly_streak += 1
                else:
                    readonly_streak = 0
                if readonly_streak >= 8:
                    failed = ("AUTO-FAIL: recon loop detected - %d "
                              "consecutive read-only actions with no "
                              "substantive work" % readonly_streak)
                    stop = True
                    break
                recent5 = action_hashes[-5:]
                if len(recent5) >= 5 and recent5.count(h) >= 3:
                    failed = ("AUTO-FAIL: oscillation detected - "
                              "repeating action " + str(name))
                    stop = True
                    break
                if action_hashes.count(h) == 2:
                    # Repeat nudge: show the model the exit before the
                    # oscillation backstop has to fire. A stuck model
                    # that finished its real work (calc + time, then
                    # time again) needs the final_answer shape, not a
                    # third identical call.
                    messages.append({
                        "role": "user",
                        "content": "HARNESS: that exact call already "
                                   "ran - its result is above and "
                                   "repeating it changes nothing. If "
                                   "you have what you need, reply "
                                   "{\"tool\": \"final_answer\", "
                                   "\"args\": {\"answer\": \"...\"}} "
                                   "now; otherwise call a DIFFERENT "
                                   "tool."})
            if stop:
                break
        if not final_text and not failed and trace:
            # Rounds exhausted while the model kept calling tools but
            # never answered: one plain answer turn over the real
            # trace (the gates judge its grounding like any claim).
            try:
                resp3 = _llama_chat(
                    AGENT_URL, AGENT_MODEL,
                    messages + [{"role": "user", "content":
                                 "Now give the final answer in plain "
                                 "text using the tool results above. "
                                 "Quote the exact values the tools "
                                 "printed. Do not call any tools."}],
                    tools=None, num_predict=kw.get("max_tokens", 512))
                final_text = (resp3.get("message", {}).get("content")
                              or "").strip()
                completion_claim = True
            except Exception:  # noqa: BLE001
                pass
        substantive = sum(1 for t in trace
                          if _is_substantive_result(t["tool"],
                                                    t["result"]))
        return {"text": final_text, "tool_calls": trace,
                "elapsed": round(time.time() - t0, 1),
                "failed": failed, "completion_claim": completion_claim,
                "substantive_tools": substantive,
                "constraints": constraints,
                "claims": (_parse_claims(final_text, claims_native)
                           if (kw.get("claim_fields") or claims_native)
                           else None)}


# ------------------------------------------------------------------- code
def _specialist_code(prompt, **kw):
    with _locks["code"]:
        t0 = time.time()
        resp = _llama_chat(CODE_URL, CODE_MODEL, [{"role": "user", "content": prompt}],
                            num_predict=kw.get("max_tokens", 1536))
        text = (resp.get("message", {}).get("content") or "").strip()
        return {"text": text, "elapsed": round(time.time() - t0, 1)}


# --------------------------------------------------------------- quicklook
def _specialist_quicklook(prompt, **kw):
    with _locks["quicklook"]:
        t0 = time.time()
        budget = max(kw.get("max_tokens", 1400), 1400)
        # Drafting/synthesis is not a reasoning task: with thinking
        # enabled, Qwen3-4B spends the whole budget in its reasoning
        # channel on evidence-heavy prompts and returns empty
        # content. Callers that just need a draft pass no_think.
        ctk = {"enable_thinking": False} if kw.get("no_think") else None
        resp = _llama_chat(QUICK_URL, QUICK_MODEL,
                           [{"role": "user", "content": prompt}],
                           num_predict=budget, chat_template_kwargs=ctk)
        msg = resp.get("message", {})
        raw = (msg.get("content") or "")
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not text and ("<think>" in raw
                         or msg.get("reasoning_content")):
            # Thinking ate the whole budget and the visible answer
            # never arrived. The current llama.cpp build returns
            # Qwen3 reasoning in a separate reasoning_content field
            # (content stays empty until reasoning finishes), so
            # the trigger must check that field too - checking the
            # content for '<think>' alone silently never retried
            # and quicklook returned "" on evidence-heavy prompts
            # (root cause of the empty synthesis drafts).
            resp = _llama_chat(QUICK_URL, QUICK_MODEL,
                                [{"role": "user", "content": prompt}],
                                num_predict=max(2048, budget * 4),
                                chat_template_kwargs=ctk)
            raw = (resp.get("message", {}).get("content") or "")
            text = re.sub(r"<think>.*?</think>", "", raw,
                          flags=re.DOTALL).strip()
        return {"text": text, "elapsed": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ embed
def _embed_frank(texts):
    out = _http_json(INFINITY_URL + "/embeddings",
                     {"model": "BAAI/bge-m3", "input": texts}, timeout=300)
    return [d["embedding"] for d in out["data"]]


def _specialist_embed(prompt, **kw):
    """Ranking specialist. Primary: frank's bge-m3 (GPU). Fallback: local
    bge-small. kw: corpus=[...], top_k=N. Without corpus, returns dim."""
    with _locks["embed"]:
        corpus = kw.get("corpus", [])
        top_k = kw.get("top_k", 3)
        t0 = time.time()
        backend = "frank-bge-m3"
        try:
            if not corpus:
                vec = _embed_frank([prompt])[0]
                return {"embedding_dim": len(vec), "backend": backend,
                        "elapsed": round(time.time() - t0, 1)}
            vecs = _embed_frank([prompt] + list(corpus))
            q, docs = vecs[0], vecs[1:]
        except Exception:
            backend = "local-bge-small-fallback"
            if "embed_local" not in _loaded:
                from sentence_transformers import SentenceTransformer
                _loaded["embed_local"] = SentenceTransformer(
                    "BAAI/bge-small-en-v1.5", device="cpu")
            model = _loaded["embed_local"]
            if not corpus:
                vec = model.encode(prompt)
                return {"embedding_dim": len(vec), "backend": backend,
                        "elapsed": round(time.time() - t0, 1)}
            q = model.encode([prompt])[0]
            docs = model.encode(list(corpus))
        import numpy as np
        q = np.array(q, dtype=float)
        dn = np.array(docs, dtype=float)
        qn = q / (np.linalg.norm(q) + 1e-9)
        dn = dn / (np.linalg.norm(dn, axis=1, keepdims=True) + 1e-9)
        scores = (dn @ qn).tolist()
        ranked = sorted(zip(corpus, scores), key=lambda x: x[1],
                        reverse=True)[:top_k]
        return {"ranked": [{"text": t, "score": round(s, 4)}
                           for t, s in ranked],
                "backend": backend,
                "elapsed": round(time.time() - t0, 1)}


# -------------------------------------------------------------------- tts
def _specialist_tts(prompt, **kw):
    """F5-TTS (SJTU) via multiplex-env CLI; Chatterbox fallback.
    Returns path to a wav plus duration/RMS so callers can verify."""
    with _locks["tts"]:
        t0 = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_file = f"f5-{stamp}.wav"
        ref_audio = kw.get("ref_audio",
                           "/home/joshua/llm/voice-loop/dsp/glados_ref_10s.wav")
        ref_text_file = HERE / "f5_ref_text.txt"
        ref_text = (ref_text_file.read_text().strip()
                    if ref_text_file.exists() else "")
        f5_cli = os.path.join(MPX_ENV_BIN, "f5-tts_infer-cli")
        err = ""
        if os.path.exists(f5_cli) and os.path.exists(ref_audio):
            try:
                cmd = [f5_cli, "--model", "F5TTS_v1_Base",
                       "--ref_audio", ref_audio, "--ref_text", ref_text,
                       "--gen_text", prompt[:500],
                       "--output_dir", str(OUT_DIR),
                       "--output_file", out_file, "--device", "cuda"]
                subprocess.run(cmd, capture_output=True, text=True,
                               timeout=900, check=True)
                wav = OUT_DIR / out_file
                if wav.exists():
                    info = _wav_info(str(wav))
                    return {"wav": str(wav), "engine": "f5-tts",
                            **info, "elapsed": round(time.time() - t0, 1)}
            except Exception as e:  # noqa: BLE001
                err = f"f5 failed: {type(e).__name__}: {e}"
        # fallback: chatterbox in the current interpreter, if importable
        try:
            if "tts_cb" not in _loaded:
                from chatterbox.tts import ChatterboxTTS
                _loaded["tts_cb"] = ChatterboxTTS.from_pretrained(device="cpu")
            model = _loaded["tts_cb"]
            wav_t = model.generate(prompt[:400])
            out = OUT_DIR / f"cb-{stamp}.wav"
            import soundfile as sf
            sf.write(str(out), wav_t.squeeze(0).cpu().numpy(), model.sr)
            info = _wav_info(str(out))
            return {"wav": str(out), "engine": "chatterbox-fallback",
                    "f5_error": err, **info,
                    "elapsed": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            return {"error": f"tts unavailable. {err} chatterbox: "
                             f"{type(e).__name__}: {e}",
                    "elapsed": round(time.time() - t0, 1)}


def _wav_info(path):
    """Stdlib-only wav inspection (works from any interpreter)."""
    import wave
    import array
    import math
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        frames = w.readframes(n)
    a = array.array({1: "b", 2: "h", 4: "i"}.get(sw, "h"))
    a.frombytes(frames)
    if ch > 1:
        a = a[::ch]
    if not len(a):
        return {"duration_s": 0.0, "rms": 0.0, "sample_rate": sr}
    maxv = float(2 ** (8 * sw - 1))
    rms = math.sqrt(sum((x / maxv) ** 2 for x in a) / len(a))
    return {"duration_s": round(n / sr, 2), "rms": round(rms, 5),
            "sample_rate": sr}


# ------------------------------------- micro-specialists (2026-10-20)
def _specialist_shepherd(prompt, **kw):
    """Step-by-step verifier (Math-Shepherd PRM, Renmin University):
    scores each reasoning step P(good) - the per-step complement to
    Skywork's whole-answer score. kw: question=..., steps=[...] (or
    output="..." raw text). prompt is the question if kw lacks one."""
    with _locks["shepherd"]:
        payload = {"question": kw.get("question", prompt)}
        if "steps" in kw:
            payload["steps"] = kw["steps"]
        else:
            payload["output"] = kw.get("output", "")
        return _http_json(SHEPHERD_URL + "/score_steps", payload, timeout=300)


def _specialist_vision(prompt, **kw):
    """Vision/OCR (MiniCPM-V-4.5, OpenBMB/Tsinghua). ON DEMAND: the
    gate stops Magicoder, starts MiniCPM-V, and a reaper hands the
    VRAM back after 180s idle. kw: image_path=... (mrfloppy-local)
    or image_b64=...; prompt is the instruction/question."""
    import base64 as _b64
    with _locks["vision"]:
        subprocess.run([str(HERE / "vision_gate.sh")], check=True,
                       timeout=240)
        if "image_b64" in kw:
            b64 = kw["image_b64"]
        else:
            with open(kw["image_path"], "rb") as f:
                b64 = _b64.b64encode(f.read()).decode()
        body = {"model": "minicpm-v",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt}]}],
                "max_tokens": kw.get("max_tokens", 512), "temperature": 0.0}
        out = _http_json(VISION_URL + "/v1/chat/completions", body,
                         timeout=600)
        (HERE / ".vision-stamp").touch()
        return {"text": out["choices"][0]["message"]["content"]}


def _specialist_stt(prompt, **kw):
    """Speech-to-text (Kyutai stt-1b-en_fr) on frank's 2070S. kw:
    audio_b64=... or wav_path=... (a path on FRANK). prompt unused
    (uniform specialist signature). Note: frank's card is nearly
    full; the service holds its KV caches resident from startup."""
    with _locks["stt"]:
        payload = {}
        if "audio_b64" in kw:
            payload["audio_b64"] = kw["audio_b64"]
        elif "wav_path" in kw:
            payload["wav_path"] = kw["wav_path"]
        else:
            raise ValueError("stt needs audio_b64 or wav_path")
        return _http_json(STT_URL + "/transcribe", payload, timeout=900)


def _specialist_ouro(prompt, **kw):
    """Deep reasoner (Ouro-2.6B-Thinking, ByteDance + universities -
    a looped transformer: recurrent depth instead of more layers).
    Resident on frank. Slow by design (recurrent steps per token);
    kw: max_tokens=1024, temperature=0.6."""
    with _locks["ouro"]:
        return _http_json(OURO_URL + "/generate",
                          {"prompt": prompt,
                           "max_tokens": kw.get("max_tokens", 1024),
                           "temperature": kw.get("temperature", 0.6)},
                          timeout=1800)


def _specialist_trm(prompt, **kw):
    """ARC puzzle specialist (TRM - Tiny Recursive Model, Samsung
    SAIL Montreal; 7M params). ON DEMAND on mrfloppy (gate + reaper).
    NOT a chat model - honest harness contract:
      kw puzzle_name="<name in the built ARC dataset>"  -> predict
        that puzzle's test output from its trained identifier.
      kw puzzle={"train": [{"input": grid, "output": grid}, ...],
                 "test_input": grid}  -> encoded with the BLANK
        identifier (unseen puzzle; expect weak results - TRM's
        design is transductive: identifiers are trained).
    Returns {predicted_grid, puzzle_id_used, exact_match,
    cell_accuracy} (match fields only when the expected output is
    known, i.e. dataset puzzles). prompt unused."""
    with _locks["trm"]:
        subprocess.run([str(HERE / "trm_gate.sh")], check=True,
                       timeout=240)
        payload = {}
        if "puzzle" in kw:
            payload["puzzle"] = kw["puzzle"]
        if "puzzle_name" in kw:
            payload["puzzle_name"] = kw["puzzle_name"]
        out = _http_json(TRM_URL + "/predict", payload, timeout=600)
        (HERE / ".trm-stamp").touch()
        return out


def _specialist_mathwhiz(prompt, **kw):
    """Math seat (MiniCPM5-2B, OpenBMB) - Phase-2 sweep math champion
    (23/25 raw DET, best in band). Plain chat contract: the caller
    states the problem; the seat answers. Think blocks stripped like
    quicklook; reasoning_content honored if content comes back empty
    (the DET empties lesson: check both fields, never just one)."""
    with _locks["mathwhiz"]:
        t0 = time.time()
        resp = _llama_chat(MATHWHIZ_URL, MATHWHIZ_MODEL,
                           [{"role": "user", "content": prompt}],
                           num_predict=kw.get("max_tokens", 1024))
        msg = resp.get("message", {})
        raw = (msg.get("content") or "")
        if not raw.strip() and msg.get("reasoning_content"):
            raw = msg["reasoning_content"]
        text = re.sub(r"<think>.*?</think>", "", raw,
                      flags=re.DOTALL).strip()
        return {"text": text, "elapsed": round(time.time() - t0, 1)}


def _specialist_draft(prompt, **kw):
    """Drafting / generalist seat (Phi-4-mini 3.8B, Microsoft) - the
    Phase-2 sweep's steadiest generalist. Prose drafting, summarizing
    and restructuring that is not evidence-heavy quicklook work."""
    with _locks["draft"]:
        t0 = time.time()
        resp = _llama_chat(DRAFT_URL, DRAFT_MODEL,
                           [{"role": "user", "content": prompt}],
                           num_predict=kw.get("max_tokens", 1024))
        msg = resp.get("message", {})
        raw = (msg.get("content") or "")
        if not raw.strip() and msg.get("reasoning_content"):
            raw = msg["reasoning_content"]
        text = re.sub(r"<think>.*?</think>", "", raw,
                      flags=re.DOTALL).strip()
        return {"text": text, "elapsed": round(time.time() - t0, 1)}


_SPECIALISTS = {
    "agent": _specialist_agent,
    "code": _specialist_code,
    "quicklook": _specialist_quicklook,
    "embed": _specialist_embed,
    "tts": _specialist_tts,
    "shepherd": _specialist_shepherd,
    "vision": _specialist_vision,
    "stt": _specialist_stt,
    "ouro": _specialist_ouro,
    "trm": _specialist_trm,
    "mathwhiz": _specialist_mathwhiz,
    "draft": _specialist_draft,
}


# ------------------------------------------------------- Layer 2 gates
# Deterministic completion gates, ported from the worker's process_one
# battery (audit failure 5) plus quote-containment gating (import E3)
# and an execution gate for code (import E4). These run BEFORE the
# Skywork score is consulted: a reward model grades prose plausibility,
# so it is a quality signal only, never the completion verdict.

_STOPWORDS = frozenset((
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'as', 'into', 'out', 'up', 'down',
    'over', 'under', 'again', 'once', 'here', 'there', 'when', 'where',
    'which', 'who', 'whom', 'what', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
    'than', 'too', 'very', 'just', 'about', 'using', 'use', 'used'))


def _check_overlap(task_text, output_text):
    """Worker port (L501): Jaccard over content words >=5 chars."""
    tw = set(w for w in re.findall(r'[a-z]{5,}', str(task_text).lower())
             if w not in _STOPWORDS)
    ow = set(w for w in re.findall(r'[a-z]{5,}', str(output_text).lower())
             if w not in _STOPWORDS)
    if not tw or not ow:
        return 0.0
    return len(tw & ow) / len(tw | ow)


def _find_promised_output(task_text):
    """Worker port (L1883): promised output file path in the task."""
    m = re.search(r"(?:write to|save to|output\s*:|create(?:\s+a)?"
                  r"(?:\s+\w+)*\s+(?:at|to|in)|generate(?:\s+\w+)*\s+"
                  r"(?:at|to|in)|build(?:\s+\w+)*\s+(?:at|to))\s*"
                  r"([~\/][\w.\-\/]+)",
                  task_text or "", re.IGNORECASE)
    if not m:
        return None
    return m.group(1).rstrip(".,;:")


_INCOMPLETE_MARKERS = (
    "in progress", "did not get to", "did not finish",
    "ran out of rounds", "cut off", "remaining work", "recon only",
    "no files", "nothing was", "not built", "no edits",
    "not implemented", "ran out", "todo",
)


def _value_tokens(text):
    """Identifier-shaped tokens: the checkable claims of a report -
    anything containing a digit or path/identifier punctuation, or a
    hex literal. Plain English words are NOT checkable this way.
    Tokens are normalized symmetrically (answer and evidence go
    through the same function): edge punctuation from sentence
    boundaries is stripped ('0x5a.' == '0x5a'), and hex literals
    lose leading zeros ('0x00002010' == '0x2010' - disassemblers
    zero-pad, drafters don't)."""
    toks = set()
    for t in re.findall(r"[A-Za-z0-9_./\-]{3,}", text or ""):
        t = t.strip(".,;:!?()[]{}\"'")
        if not t:
            continue
        if any(ch.isdigit() for ch in t) or \
                ("_" in t and len(t) >= 6) or \
                ("/" in t and len(t) >= 6) or \
                ("." in t[:-1] and len(t) >= 5) or \
                re.fullmatch(r"0x[0-9a-fA-F]+", t):
            tl = t.lower()
            m = re.fullmatch(r"0x0*([0-9a-f]+)", tl)
            if m:
                tl = "0x" + m.group(1)
            toks.add(tl)
    return toks


_CLAIM_WORDS = re.compile(
    r"secret|checksum|password|credential|decod|encod|\bxor\b|"
    r"exit code|returns?|prints?|shows?|found|stored|computed|value",
    re.IGNORECASE)


def _unescape_variants(q):
    """Containment variants for a quoted string: models routinely
    serialize multiline tool output into one line with literal \\n
    escapes (sometimes double-escaped through JSON round-trips) -
    that is faithful quoting in escaped form, not fabrication.
    Try the raw string, up to two unescape rounds, and
    whitespace-collapsed forms."""
    vs = {q}
    cur = q
    for _ in range(2):
        cur = (cur.replace("\\\\", "\x00").replace("\\n", "\n")
               .replace("\\r", "\r").replace("\\t", "\t")
               .replace("\x00", "\\"))
        vs.add(cur)
    vs |= {re.sub(r"\s+", " ", v) for v in list(vs)}
    if re.fullmatch(r"0x[0-9a-fA-F]+", q.strip()):
        # Disassemblers zero-pad addresses ('0x00002010'); drafters
        # write the same address unpadded ('0x2010'). Same value.
        _v = int(q.strip(), 16)
        vs.add("0x%x" % _v)
        vs.add("0x%08x" % _v)
    return vs


def _quote_gate(text, evidence, prompt=""):
    """ALCE-style deterministic quote-containment (import E3): factual
    claims must be traceable to verbatim fragments of real tool
    output. Returns (failures, details). Checks:
    - every quoted string in the answer appears in the evidence
    - every >=4-digit number (years excluded) appears in the evidence
    - sentences asserting values/findings carry >=1 evidence fragment
    - >=50% of identifier-shaped tokens are evidence-supported"""
    failures = []
    details = {}
    ev = (evidence or "").lower()
    if not ev.strip():
        details["quote_gate"] = "skipped-no-evidence"
        return failures, details
    if not (text or "").strip():
        return failures, details
    # A trailing CLAIMS block is structured claim data, judged by
    # the claim-level machinery (agreement / provenance / unlock),
    # not by prose grounding - its labels and values would
    # otherwise poison the token-support ratio.
    _cm = re.search(r"^\s*CLAIMS\b\s*:?\s*$", text,
                    re.MULTILINE | re.IGNORECASE)
    if _cm:
        text = text[:_cm.start()]
    ev_tokens = _value_tokens(evidence)
    ev_collapsed = re.sub(r"\s+", " ", ev)
    # quoted strings (raw, unescaped, or whitespace-collapsed form)
    quoted = [a or b for a, b in re.findall(
        r'"([^"\n]{4,})"|`([^`\n]{4,})`', text)]
    bad_quotes = [q for q in quoted
                  if not any(v.lower() in ev or v.lower() in ev_collapsed
                             for v in _unescape_variants(q))]
    details["quoted_checked"] = len(quoted)
    if bad_quotes:
        failures.append("quote-gate: quoted strings not present in "
                        "any tool output: " +
                        "; ".join(repr(q[:60]) for q in bad_quotes[:3]))
    # numbers (>=4 digits, excluding calendar years)
    nums = [n for n in re.findall(r"\b\d{4,}(?:\.\d+)?\b", text)
            if not (len(n) == 4 and 1900 <= int(n) <= 2099)]
    bad_nums = [n for n in set(nums) if n not in ev]
    details["numbers_checked"] = len(set(nums))
    if bad_nums:
        failures.append("quote-gate: numbers not present in any tool "
                        "output: " + ", ".join(sorted(bad_nums)[:5]))
    # value-claim sentences must carry a fragment
    prompt_norm = re.sub(r"\s+", " ", (prompt or "").lower())
    frag_ok = 0
    bad_sents = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = sent.strip()
        if len(s) < 12 or not _CLAIM_WORDS.search(s):
            continue
        if s.endswith(":"):
            # A list/report header introducing the items below it -
            # structural, not an assertion. The items themselves
            # are checked as their own sentences.
            continue
        if prompt_norm and re.sub(r"\s+", " ", s.lower()) in prompt_norm:
            # Verbatim echo of the task text (models love restating
            # the questions as report headers) - the task's words,
            # not a worker assertion.
            continue
        stoks = _value_tokens(s)
        has_frag = bool(stoks & ev_tokens) or \
            any(any(v.lower() in ev or v.lower() in ev_collapsed
                    for v in _unescape_variants(q))
                for q in re.findall(r'"([^"\n]{3,})"', s))
        if has_frag:
            frag_ok += 1
        else:
            bad_sents.append(s[:110])
    details["claim_sentences_grounded"] = frag_ok
    if bad_sents:
        failures.append("quote-gate: assertions with no supporting "
                        "tool-output fragment: " +
                        " | ".join(bad_sents[:3]))
    # identifier-token support ratio
    toks = _value_tokens(text)
    if len(toks) >= 3:
        sup = toks & ev_tokens
        details["token_support"] = f"{len(sup)}/{len(toks)}"
        if len(sup) / len(toks) < 0.5:
            failures.append(
                "quote-gate: only %d/%d identifier tokens trace to "
                "tool output (unsupported: %s)" % (
                    len(sup), len(toks),
                    ", ".join(sorted(toks - ev_tokens)[:6])))
        elif not sup:
            failures.append("quote-gate: answer cites no tool output "
                            "at all")
    return failures, details


def _execute_code_gate(res):
    """Import E4: a code deliverable is executed; exit code + output
    become gate evidence. Returns a details dict (ran=False when no
    python code block could be extracted)."""
    text = res.get("text", "") or ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    code = blocks[0] if blocks else (
        text if re.match(r"\s*(#!/|import |from |def |class )", text)
        else None)
    if not code:
        return {"ran": False, "note": "no executable code block found"}
    gate_dir = OUT_DIR / "gate-checks"
    gate_dir.mkdir(parents=True, exist_ok=True)
    path = gate_dir / f"gate-{int(time.time())}.py"
    path.write_text(code)
    try:
        r = subprocess.run([sys.executable, str(path)],
                           capture_output=True, text=True, timeout=60,
                           cwd=str(gate_dir))
        return {"ran": True, "exit": r.returncode,
                "output_tail": (r.stdout or "")[-400:],
                "stderr_tail": (r.stderr or "")[-400:],
                "file": str(path)}
    except subprocess.TimeoutExpired:
        return {"ran": True, "exit": None,
                "note": "execution timed out after 60s"}
    except Exception as e:  # noqa: BLE001
        return {"ran": False, "note": f"{type(e).__name__}: {e}"}


_SHEPHERD_TRIGGER = re.compile(
    r"calculat|solve|prove|arithmetic|equation|\bmath\b|step[- ]by[- ]step",
    re.IGNORECASE)


def _shepherd_step_check(prompt, criteria, text):
    """Wire the Shepherd PRM in as a step-checker where the job type
    fits: multi-step reasoning chains (math/calculation). Extracts the
    answer's step lines and scores each with the PRM. Returns None
    when the job is not a reasoning chain (RE reports, prose briefs -
    Shepherd would be grading the wrong thing there)."""
    if not _SHEPHERD_TRIGGER.search((prompt or "") + " " +
                                    (criteria or "")):
        return None
    steps = []
    for ln in (text or "").splitlines():
        m = re.match(r"\s*(?:step\s+\d+[:.)]?|\d+[.)])\s+(.*)", ln,
                     re.IGNORECASE)
        if m and len(m.group(1).strip()) >= 8:
            steps.append(m.group(1).strip())
    if len(steps) < 3:
        return None
    try:
        out = _specialist_shepherd("", question=(prompt or "")[:1000],
                                    steps=steps[:12])
        return {"checked": True, "n_steps": len(steps),
                "step_scores": out.get("step_scores"),
                "min_score": out.get("min_score"),
                "min_step": out.get("min_step"),
                "mean_score": out.get("mean_score")}
    except Exception as e:  # noqa: BLE001
        return {"checked": False,
                "note": f"shepherd unavailable: {type(e).__name__}: {e}"}


def _deterministic_gates(name, prompt, criteria, res, kw):
    """The battery that must pass BEFORE Skywork is consulted.
    Returns (ok, failures, details)."""
    failures = []
    details = {}
    text = res.get("text", "") or ""
    if not text.strip():
        # Universal: an empty deliverable can never pass, whatever
        # the reward model thinks of it. (Run #12 evidence: an
        # empty quicklook synthesis scored 1.95 and "verified" -
        # a Skywork quirk; refusing it is the gate's job.)
        failures.append("empty final answer")
    if res.get("failed"):
        failures.append(str(res["failed"]))
    trace = res.get("tool_calls")
    if trace is not None:
        executed = [t for t in trace if not t.get("blocked")]
        details["tool_calls"] = len(executed)
        if not executed:
            failures.append("zero tool calls executed - work that "
                            "never ran cannot be verified")
            if _extract_raw_tool_calls(text):
                failures.append("output is an unexecuted tool-call "
                                "plan, not results")
        else:
            substantive = sum(
                1 for t in executed
                if _is_substantive_result(t["tool"], t["result"]))
            details["substantive_tools"] = substantive
            if substantive == 0:
                failures.append("no substantive tool work - every "
                                "tool output was empty, an error, or "
                                "trivial")
    low = text.lower()
    for phrase in _INCOMPLETE_MARKERS:
        if phrase in low:
            failures.append(f"incomplete-work marker in output: "
                            f"'{phrase}'")
            break
    pf = _find_promised_output(prompt)
    if pf:
        pp = Path(os.path.expanduser(pf))
        if not (pp.is_file() and pp.stat().st_size > 500):
            failures.append(f"promised output file {pf} missing or "
                            "<=500 bytes")
    if trace or kw.get("evidence_texts"):
        corpus = "\n".join(str(t.get("result", ""))
                           for t in trace or [])
        extra = kw.get("evidence_texts") or []
        if extra:
            corpus += "\n" + "\n".join(str(e) for e in extra)
        qf, qd = _quote_gate(text, corpus, prompt=prompt)
        failures.extend(qf)
        details.update(qd)
    if name == "code":
        ex = _execute_code_gate(res)
        details["execution"] = ex
        if ex.get("ran") and ex.get("exit") not in (0,):
            tail = (ex.get("stderr_tail") or ex.get("note") or "")
            failures.append(f"code execution gate: deliverable exited "
                            f"{ex.get('exit')}: {tail[-220:]}")
    return (not failures), failures, details


# ----------------------------------------------------------------- verify
def verify(prompt, response, criteria=""):
    """Score a response with the trained reward model (duplex judge).
    Criteria are folded into the scored prompt so the judge sees what
    the output was supposed to achieve."""
    if API_URL:
        out = _http_json(API_URL + "/verify",
                         {"prompt": prompt, "response": response,
                          "criteria": criteria}, timeout=600)
        return out["score"]
    score_prompt = prompt
    if criteria:
        score_prompt = prompt + "\n\nAcceptance criteria: " + criteria
    out = _http_json(VERIFIER_URL + "/score",
                     {"prompt": score_prompt, "response": response},
                     timeout=600)
    return out["score"]


def _critique(task_prompt, criteria, output):
    cp = ("You are a strict reviewer. A worker produced the output below "
          "for a task. In at most 4 short bullet points, list what is "
          "missing, wrong, or unsupported versus the task and criteria. "
          "Be specific; no praise.\n\nTASK:\n" + task_prompt +
          "\n\nCRITERIA:\n" + (criteria or "(none given)") +
          "\n\nWORKER OUTPUT:\n" + output[:3000])
    try:
        return _specialist_quicklook(cp, max_tokens=300)["text"]
    except Exception as e:  # noqa: BLE001
        return f"(critique unavailable: {e}) Improve against the criteria."


def run_verified(name, prompt, criteria="", threshold=None, max_retries=2,
                 **kw):
    """Duplex loop, hardened 2026-10-20: worker generates ->
    DETERMINISTIC GATES judge first (zero-tools, incomplete markers,
    promised artifacts, quote-containment vs real tool output, code
    execution, Shepherd step-check on reasoning chains) -> only a
    gate-passing attempt is scored by Skywork, now a quality signal
    only. Rejections feed back as gate failures + the failed-call
    history (worker heal pattern: retries carry previous_failures
    and a >0.8-similar retry is itself rejected). Returns the best
    gate-passing attempt plus the full history. Every stage is
    written to the blackboard."""
    if API_URL:
        out = _http_json(API_URL + "/run_verified",
                         {"specialist": name, "prompt": prompt,
                          "criteria": criteria, "threshold": threshold,
                          "max_retries": max_retries, "kw": kw},
                         timeout=3600)
        return out
    thr = DEFAULT_THRESHOLD if threshold is None else threshold
    t0 = time.time()
    run_id = f"verified-{name}-{int(t0)}"
    blackboard.write("task_state", run_id,
                     {"stage": "start", "specialist": name,
                      "criteria": criteria, "threshold": thr})
    history = []
    best = None
    last = None
    all_calls = []
    gen_prompt = prompt
    previous_failures = [str(p) for p in
                         (kw.get("previous_failures") or [])]
    prev_attempt_text = None
    last_evidence = []
    for attempt in range(max_retries + 1):
        blackboard.write("task_state", run_id,
                         {"stage": f"generating-attempt-{attempt}"})
        kw["previous_failures"] = previous_failures
        res = run_specialist(name, gen_prompt, **kw)
        all_calls.extend(res.get("tool_calls") or [])
        text = res.get("text", "")
        gates_ok, gate_failures, gate_details = _deterministic_gates(
            name, prompt, criteria, res, kw)
        # Guard 4 (worker port): a retry that merely restates a failed
        # attempt has not changed approach - reject it as a failure.
        if prev_attempt_text and text and not gates_ok:
            sim = _check_overlap(prev_attempt_text, text)
            gate_details["retry_similarity"] = round(sim, 2)
            if sim > 0.8:
                gate_failures.append(
                    f"retry-similarity {sim:.2f} > 0.8 vs a failed "
                    "attempt - no changed approach")
        # Shepherd PRM step-check (reasoning chains only).
        shep = _shepherd_step_check(prompt, criteria, text)
        if shep and shep.get("checked"):
            gate_details["shepherd"] = shep
            smin = kw.get("shepherd_min", 0.05)
            if shep.get("min_score") is not None and \
                    shep["min_score"] < smin:
                gates_ok = False
                gate_failures.append(
                    f"shepherd step-check: step "
                    f"{(shep.get('min_step') or 0) + 1} scored "
                    f"{shep['min_score']:.3f} < {smin} (likely wrong "
                    "step)")
        score = None
        if gates_ok:
            try:
                score = verify(prompt, text, criteria)
            except Exception as e:  # noqa: BLE001
                score = None
                res["verify_error"] = f"{type(e).__name__}: {e}"
        entry = {"attempt": attempt, "score": score,
                 "gates_passed": gates_ok,
                 "gate_failures": gate_failures,
                 "chars": len(text), "elapsed": res.get("elapsed"),
                 "preview": text[:200]}
        if shep:
            entry["shepherd"] = {k: shep.get(k) for k in
                                 ("checked", "min_score", "min_step",
                                  "mean_score", "n_steps", "note")}
        history.append(entry)
        blackboard.append("results", run_id, entry)
        cur = {**res, "score": score, "attempt": attempt,
               "gates_passed": gates_ok, "gate_failures": gate_failures,
               "gate_details": gate_details}
        last = cur
        if gates_ok and (best is None or
                         (score is not None and
                          (best["score"] is None or score > best["score"]))):
            best = cur
        if gates_ok and score is not None and score >= thr:
            break
        prev_attempt_text = text
        # Evidence carryover: substantive tool results survive into
        # the retry prompt. A retry that must redo (and re-garble)
        # finished analysis to fix a REPORT problem wastes the
        # duplex; the blackboard philosophy is that evidence
        # persists across attempts.
        ev = [str(t.get("result", ""))[:1200]
              for t in res.get("tool_calls") or []
              if _is_substantive_result(t["tool"], t["result"])]
        if ev:
            last_evidence = ev[:6]
        if attempt < max_retries:
            for t in res.get("tool_calls") or []:
                r = str(t.get("result", ""))
                outcome = ("blocked-repeat" if t.get("blocked") else
                           "no usable output" if
                           _result_stdout_empty(t["tool"], r) or
                           r.startswith("ERROR") else "ran")
                if outcome != "ran":
                    previous_failures.append(
                        f"{t['tool']} "
                        f"{json.dumps(t.get('arguments') or {}, default=str)[:120]}"
                        f" -> {outcome}")
            if res.get("failed"):
                previous_failures.append("job failed: " +
                                         str(res["failed"]))
            fb = [prompt, "\n---\nYour previous attempt was REJECTED."]
            if gate_failures:
                fb.append("Deterministic gate failures (these are "
                          "checked mechanically - fix every one):\n- " +
                          "\n- ".join(gate_failures))
            if gates_ok:
                critique = _critique(prompt, criteria, text)
                history[-1]["critique"] = critique[:400]
                fb.append(f"Verifier score {score} < {thr}. Reviewer "
                          f"feedback:\n{critique}")
            if previous_failures:
                fb.append("Calls that already failed or produced "
                          "nothing (do not repeat them; change tool "
                          "or flags):\n- " +
                          "\n- ".join(previous_failures[-8:]))
            if last_evidence:
                fb.append("Tool evidence already gathered in your "
                          "previous attempt (REUSE it - do not redo "
                          "this work; build the corrected output "
                          "from it):\n" +
                          "\n---\n".join(last_evidence))
            fb.append("Produce a corrected, complete version. Quote "
                      "exact values from tool outputs. Output only "
                      "the work itself.")
            gen_prompt = "\n".join(fb)
            blackboard.write("task_state", run_id,
                             {"stage": f"rejected-attempt-{attempt}",
                              "score": score,
                              "gate_failures": gate_failures})
    result = best if best is not None else last
    result["attempts"] = history
    # Every attempt's tool calls, not just the returned attempt's:
    # evidence gathered by a rejected attempt is still evidence
    # (the final RE stage's shared pack depends on this - a pdg
    # produced by attempt 0 must not vanish because attempt 1's
    # prose scored better).
    result["all_tool_calls"] = all_calls
    result["verified"] = bool(best is not None and
                              best.get("score") is not None and
                              best["score"] >= thr)
    result["threshold"] = thr
    result["run_id"] = run_id
    result["wall_seconds"] = round(time.time() - t0, 1)
    blackboard.write("findings", run_id,
                     {"specialist": name, "verified": result["verified"],
                      "best_score": result.get("score"),
                      "gates_passed": result.get("gates_passed"),
                      "gate_failures": result.get("gate_failures"),
                      "attempts": len(history)})
    blackboard.write("task_state", run_id,
                     {"stage": "done", "verified": result["verified"],
                      "best_score": result.get("score")})
    return result


# ------------------------------------------------- final gate (2026-10-20)
# Perg's fix for the run #12 failure: quote-gating proves grounding,
# not interpretation, and the reward model prefers fluent-wrong over
# terse-correct - so the BINARY grades the report. The final stage
# now produces structured claims from two independent drafters, the
# behavioral claims must trace to EXECUTION-provenance evidence, and
# the claimed secret must make the binary itself flip to its success
# behavior (the unlock test). Skywork is consulted only after every
# mechanical step passes, as a style score. It cannot certify.

_CLAIM_LABELS = (
    (r"claimed[_ ]?secret|secret(?:[_ ]string)?", "claimed_secret"),
    (r"claimed[_ ]?key|xor[_ ]?key|key", "claimed_key"),
    (r"claimed[_ ]?checksum[_ ]?value|checksum[_ ]?value",
     "claimed_checksum_value"),
    (r"checksum[_ ]?algorithm|algorithm", "checksum_algorithm"),
)
_CLAIM_LINE_RES = tuple(
    (re.compile(r"^\s*(?:[-*]\s*)?(?:" + pat + r")\s*[:=]\s*(.+?)\s*$",
                re.IGNORECASE), field)
    for pat, field in _CLAIM_LABELS)
_CLAIM_EMPTY = ("", "not established", "unknown", "none", "n/a", "-",
                "not shown", "not printed", "not found")


def _clean_claim_value(v):
    v = (v or "").strip().strip("`").strip().strip("\"'").strip()
    if v.lower().rstrip(".") in _CLAIM_EMPTY:
        return ""
    return v


def _split_composite_claims(claims):
    """A drafter sometimes emits all four claims as ONE comma-joined
    line ('secret: X, key: 0x.., checksum_value: N, ...') - the
    line parser then stuffs the whole tail into the first field.
    Split embedded 'label:' segments back into their fields
    (values are copied, never altered)."""
    for f in CLAIM_FIELDS:
        v = claims.get(f) or ""
        if not v:
            continue
        m = re.search(r",\s*(claimed[_ ]?secret|secret|claimed[_ ]?"
                      r"key|xor[_ ]?key|key|claimed[_ ]?checksum"
                      r"[_ ]?value|checksum[_ ]?value|checksum[_ ]?"
                      r"algorithm|algorithm)\s*[:=]\s*", v,
                      re.IGNORECASE)
        if not m:
            continue
        claims[f] = _clean_claim_value(v[:m.start()])
        tail = v[m.start() + 2:]
        for ln in [tail]:
            for rx, field in _CLAIM_LINE_RES:
                if claims.get(field):
                    continue
                mm = rx.match(ln.strip())
                if mm:
                    claims[field] = _clean_claim_value(mm.group(1))
    return claims


def _parse_claims(text, native=None):
    """Extract the four load-bearing RE claims from a draft.
    Native structured fields (final_answer args) win; then a
    CLAIMS: block; then any labelled lines. Extraction copies what
    the drafter ASSERTED - it never invents or repairs a value;
    the mechanical gates judge the values."""
    claims = {f: "" for f in CLAIM_FIELDS}
    if native:
        for f in CLAIM_FIELDS:
            v = native.get(f)
            if isinstance(v, str) and v.strip():
                claims[f] = _clean_claim_value(v)
    lines = (text or "").splitlines()
    block_at = None
    for i, ln in enumerate(lines):
        if re.match(r"\s*CLAIMS\b\s*:?\s*$", ln, re.IGNORECASE):
            block_at = i
    scopes = []
    if block_at is not None:
        scopes.append(lines[block_at + 1:block_at + 10])
    scopes.append(lines)
    for scope in scopes:
        for ln in scope:
            for rx, field in _CLAIM_LINE_RES:
                if claims[field]:
                    continue
                m = rx.match(ln)
                if m:
                    claims[field] = _clean_claim_value(m.group(1))
        if all(claims.values()):
            break
    return _split_composite_claims(claims)


def _norm_claim(field, v):
    """Normalize a claim value for cross-drafter comparison:
    secrets compare exactly; keys compare as hex numbers; checksum
    values as digit strings; algorithms by family."""
    v = (v or "").strip().strip("`\"'").strip()
    if field == "claimed_key":
        x = re.sub(r"[^0-9a-f]", "", v.lower().replace("0x", ""))
        return x.lstrip("0") or ("0" if x else "")
    if field == "claimed_checksum_value":
        return re.sub(r"\D", "", v)
    if field == "checksum_algorithm":
        x = re.sub(r"[^a-z0-9]", "", v.lower())
        for fam in ("fnv1a", "fnv1", "fnv", "crc32", "crc", "sha256",
                    "sha1", "md5", "adler32", "murmur", "xxhash"):
            if fam in x:
                return fam
        return x
    return v


def _claims_disagreements(claims_a, claims_b):
    out = []
    for f in CLAIM_FIELDS:
        na = _norm_claim(f, claims_a.get(f, ""))
        nb = _norm_claim(f, claims_b.get(f, ""))
        if na == nb:
            continue
        if f == "checksum_algorithm" and na.startswith("fnv") \
                and nb.startswith("fnv"):
            continue  # family agreement (FNV vs FNV-1a) is no conflict
        out.append({"field": f, "a": claims_a.get(f, ""),
                    "b": claims_b.get(f, "")})
    return out


_STATIC_VERBS = frozenset((
    "file", "strings", "objdump", "readelf", "nm", "xxd", "hexdump",
    "od", "rizin", "rabin2", "radare2", "r2", "gdb", "grep", "egrep",
    "head", "tail", "cat", "ls", "wc", "sed", "awk", "echo", "printf",
    "find", "stat", "ldd", "checksec", "sha256sum", "md5sum", "sort",
    "uniq", "cut", "tr", "jq"))
_INTERPRETER_VERBS = frozenset((
    "python3", "python", "perl", "ruby", "node", "php", "bash", "sh"))


def _is_binary_verb(verb, binary):
    if not verb:
        return False
    base = Path(binary).name
    return verb == binary or verb == base or verb == "./" + base \
        or verb.endswith("/" + base)


def _segment_exec_info(seg):
    """Resolve one shell segment to (verb, rest_tokens), unwrapping
    env assignments and execution wrappers (strace/ltrace/timeout/
    sudo) so 'strace -e trace=... /path/binary --unlock' resolves
    to the binary as its verb."""
    import shlex
    seg = (seg or "").strip()
    if not seg:
        return None, []
    try:
        toks = shlex.split(seg)
    except Exception:
        toks = seg.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t):
            i += 1
            continue
        base = t.rsplit("/", 1)[-1]
        if base in ("strace", "ltrace"):
            i += 1
            while i < len(toks) and toks[i].startswith("-"):
                flag = toks[i]
                i += 1
                if flag in ("-e", "-o", "-p", "-s", "-u", "-E", "-i",
                            "-k") and i < len(toks):
                    i += 1
            continue
        if base == "timeout":
            i += 1
            while i < len(toks) and (
                    toks[i].startswith("-") or
                    re.match(r"^[\d.]+[smhd]?$", toks[i])):
                i += 1
            continue
        if base in ("sudo", "nohup", "stdbuf", "nice"):
            i += 1
            continue
        break
    if i >= len(toks):
        return None, []
    return toks[i], toks[i + 1:]


def _classify_call_source(call, binary):
    """Provenance typing: every tool output is tagged EXECUTION (a
    program ran - the binary itself, or an executed script/decode)
    vs STATIC (the file was read/dumped/disassembled but never ran)
    vs OTHER. Behavioral claims must trace to EXECUTION output: a
    plaintext string sitting in a `strings` dump is static
    evidence, and static evidence is where decoys live."""
    if call.get("tool") != "run_command" or call.get("blocked"):
        return "OTHER"
    cmd = str((call.get("arguments") or {}).get("command", ""))
    if not cmd.strip():
        return "OTHER"
    saw_static = False
    for seg in re.split(r"[|;&\n]+", cmd):
        verb, _rest = _segment_exec_info(seg)
        if verb is None:
            continue
        if _is_binary_verb(verb, binary):
            return "EXECUTION"
        if verb.rsplit("/", 1)[-1] in _INTERPRETER_VERBS:
            return "EXECUTION"
        if verb.rsplit("/", 1)[-1] in _STATIC_VERBS:
            saw_static = True
    return "STATIC" if saw_static else "OTHER"


def _evidence_corpus(calls, binary, source=None):
    return "\n".join(
        str(c.get("result", "")) for c in calls
        if source is None or _classify_call_source(c, binary) == source)


def _re_static_pack(calls, binary, cap=3500):
    """The STATIC-tagged tool outputs (function lists, decompiles,
    byte dumps) - the drafters' source for functions, constants and
    algorithm identification. Behavioural claims still require
    EXECUTION support (the gate enforces that); static outputs are
    evidence for what the code IS, not for what it prints.
    Ordered by drafting value - decompiles first, then function
    lists, then byte dumps, then recon dumps - because a flat
    chronological pack fills up on `file`/`strings` output and
    crowds out the pdg that carries the constants."""
    def _prio(c):
        cmd = str((c.get("arguments") or {}).get("command", ""))
        if "pdg" in cmd and "pdg @ main" not in cmd \
                and "pdg @ entry" not in cmd:
            return 0
        if "afl" in cmd:
            return 1
        if "pdg" in cmd:
            return 2
        if "px " in cmd or "objdump -s" in cmd or "objdump -d" in cmd:
            return 3
        return 4
    lines = []
    for c in sorted((c for c in calls
                     if _classify_call_source(c, binary) == "STATIC"),
                    key=_prio):
        cmd = str((c.get("arguments") or {}).get("command", ""))[:150]
        lim = 1400 if "pdg" in cmd else 650
        lines.append(f"$ {cmd}\n{str(c.get('result', ''))[:lim]}")
    pack = "\n---\n".join(lines)
    return pack[:cap] if pack else "(no static outputs recorded)"


def _re_execution_pack(calls, binary, cap=4200):
    """The raw outputs of everything that RAN (binary runs, executed
    decodes) across all stages - the evidence pack both final
    drafters work from, labelled as outranking stage prose."""
    lines = []
    for c in calls:
        if _classify_call_source(c, binary) != "EXECUTION":
            continue
        cmd = str((c.get("arguments") or {}).get("command", ""))[:150]
        lines.append(f"$ {cmd}\n{str(c.get('result', ''))[:650]}")
    pack = "\n---\n".join(lines)
    return pack[:cap] if pack else "(no execution outputs recorded)"


_REDIR_RE = re.compile(r"^(\d*>>?|\d*>&\d*|>&\d*|&>>?|<)")


def _binary_argv_forms(calls, binary):
    """Distinct argv tuples the trace used to execute the binary -
    the dynamic stage's discovered invocation forms."""
    forms = []
    for c in calls:
        if c.get("tool") != "run_command" or c.get("blocked"):
            continue
        cmd = str((c.get("arguments") or {}).get("command", ""))
        for seg in re.split(r"[|;&\n]+", cmd):
            verb, rest = _segment_exec_info(seg)
            if not _is_binary_verb(verb, binary):
                continue
            argv = tuple(t for t in rest
                         if not _REDIR_RE.match(t)
                         and t not in (">", ">>"))
            if argv not in forms:
                forms.append(argv)
    return forms


def _run_binary_once(binary, argv, stdin_text=None, timeout=10):
    """Run the target once (argv list form - no shell, so a claim
    string can never become shell syntax) and record the outcome."""
    rec = {"argv": list(argv), "stdin": stdin_text is not None}
    try:
        kw = {"capture_output": True, "text": True, "timeout": timeout,
              "cwd": str(Path(binary).parent)}
        if stdin_text is not None:
            kw["input"] = stdin_text
        else:
            kw["stdin"] = subprocess.DEVNULL
        r = subprocess.run([binary] + list(argv), **kw)
        rec.update({"exit": r.returncode,
                    "stdout": (r.stdout or "")[:600],
                    "stderr": (r.stderr or "")[:200]})
    except subprocess.TimeoutExpired:
        rec.update({"exit": None, "stdout": "", "error": "timeout"})
    except Exception as e:  # noqa: BLE001
        rec.update({"exit": None, "stdout": "",
                    "error": f"{type(e).__name__}: {e}"})
    return rec


def _re_unlock_test(binary, claimed_secret, calls):
    """The unlock test: the target itself grades the claimed
    secret. Baselines (no-arg, wrong-arg) are captured fresh - the
    binary is deterministic, so these are the same behaviors the
    dynamic stage recorded; the success-path invocation forms are
    DISCOVERED from the dynamic trace, never hardcoded. The claim
    CONFIRMS only if some invocation's behavior flips vs BOTH
    baselines (exit code or stdout differs) AND its stdout
    contains the claimed secret. A decoy that merely sits in the
    file cannot flip anything."""
    base_noarg = _run_binary_once(binary, [])
    base_wrong = _run_binary_once(binary, ["__mux_wrong_probe__"])
    forms = _binary_argv_forms(calls, binary)
    candidates = []
    seen = set()

    def _add(argv, use_stdin):
        k = (tuple(argv), bool(use_stdin))
        if k not in seen and len(candidates) < 14:
            seen.add(k)
            candidates.append((list(argv), bool(use_stdin)))

    for f in forms:
        _add(f, False)
    if claimed_secret:
        _add([claimed_secret], False)
        _add([], True)
        for f in forms:
            for t in f:
                if t.startswith("-"):
                    _add([t, claimed_secret], False)
                    _add([t], True)
            _add(f, True)
    tried = []
    confirmed_by = None
    for argv, use_stdin in candidates:
        r = _run_binary_once(binary, argv,
                             claimed_secret if use_stdin else None)
        out = r.get("stdout", "")
        contains = bool(claimed_secret) and claimed_secret in out
        diff_noarg = (r.get("exit") != base_noarg.get("exit")) or \
            (out != base_noarg.get("stdout", ""))
        diff_wrong = (r.get("exit") != base_wrong.get("exit")) or \
            (out != base_wrong.get("stdout", ""))
        flip = bool(contains and diff_noarg and diff_wrong)
        r2 = {**r, "contains_claim": contains, "flip": flip}
        tried.append(r2)
        if flip and confirmed_by is None:
            confirmed_by = r2
    return {"confirmed": confirmed_by is not None,
            "confirmed_by": confirmed_by,
            "baselines": {"no_arg": base_noarg,
                          "wrong_arg": base_wrong},
            "trace_argv_forms": [list(f) for f in forms],
            "tried": tried}


def _re_final_gate(binary, claims_a, claims_b, text_a, calls,
                   skywork_prompt=None):
    """Mechanical certification of the RE final stage. Steps run
    in order and ALL are recorded; the unlock test runs whenever a
    secret claim exists, even if an earlier step failed, so a decoy
    is shown failing the binary itself - not just the paperwork.
    Skywork is consulted ONLY after every mechanical step passes,
    as a style/completeness score: it cannot certify, and on any
    mechanical failure it is never called."""
    steps = []

    def _step(name, ok, detail):
        steps.append({"step": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    miss_a = [f for f in CLAIM_FIELDS
              if not (claims_a.get(f) or "").strip()]
    miss_b = [f for f in CLAIM_FIELDS
              if not (claims_b.get(f) or "").strip()]
    _step("claims_present", not miss_a and not miss_b,
          {"missing_a": miss_a, "missing_b": miss_b})
    disagreements = _claims_disagreements(claims_a, claims_b)
    _step("drafters_agree", not disagreements,
          {"disagreements": disagreements})
    agreed = {f: claims_a.get(f, "") for f in CLAIM_FIELDS}
    exec_corpus = _evidence_corpus(calls, binary, "EXECUTION")
    stat_corpus = _evidence_corpus(calls, binary, "STATIC")
    prov = {}
    prov_ok = True
    for f in ("claimed_secret", "claimed_checksum_value"):
        v = (agreed.get(f) or "").strip()
        in_exec = bool(v) and v in exec_corpus
        prov[f] = {"execution": in_exec,
                   "static_only": bool(v) and v in stat_corpus
                   and not in_exec}
        if v and not in_exec:
            prov_ok = False
    _step("provenance_execution", prov_ok, prov)
    secret = (agreed.get("claimed_secret") or "").strip()
    unlock = None
    if secret:
        unlock = _re_unlock_test(binary, secret, calls)
        _step("unlock_test", unlock["confirmed"],
              {"confirmed_by": unlock["confirmed_by"],
               "n_tried": len(unlock["tried"]),
               "baseline_exits": {
                   k: v.get("exit")
                   for k, v in unlock["baselines"].items()}})
    else:
        _step("unlock_test", False,
              {"note": "no claimed_secret to test"})
    mech_ok = all(s["ok"] for s in steps)
    sky = {"consulted": False}
    if mech_ok:
        try:
            sc = verify(
                skywork_prompt or f"Final RE report for {binary}",
                text_a or "",
                "Style and completeness ONLY: is this report clear, "
                "complete and well organized? (Facts are certified "
                "mechanically elsewhere; do not re-judge them.)")
            sky = {"consulted": True, "style_score": sc}
        except Exception as e:  # noqa: BLE001
            sky = {"consulted": True,
                   "error": f"{type(e).__name__}: {e}"}
    steps.append({"step": "skywork_style", "ok": True, "detail": sky})
    return {"certified": mech_ok, "steps": steps,
            "skywork_consulted": sky["consulted"], "skywork": sky,
            "unlock": unlock, "agreed_claims": agreed}


def _final_gate_feedback(gate):
    """Mechanical refusal reasons fed back to both drafters. Every
    value named here is the system's OWN claim or output - never
    ground truth."""
    lines = ["\n\nHARNESS FINAL GATE: the previous final draft was "
             "REFUSED certification by mechanical checks (no judge "
             "was consulted). Fix every item:"]
    for s in (gate or {}).get("steps") or []:
        if s.get("ok"):
            continue
        name, d = s.get("step"), s.get("detail") or {}
        if name == "claims_present":
            lines.append(
                f"- claim fields missing (drafter A: "
                f"{d.get('missing_a')}, drafter B: "
                f"{d.get('missing_b')}) - the claim fields and the "
                "CLAIMS block are mandatory.")
        elif name == "drafters_agree":
            for dis in d.get("disagreements") or []:
                lines.append(
                    f"- the drafters disagreed on {dis['field']}: "
                    f"'{dis['a']}' vs '{dis['b']}' - reconcile "
                    "against the EXECUTION EVIDENCE, which outranks "
                    "stage prose.")
        elif name == "provenance_execution":
            for f, p in d.items():
                if isinstance(p, dict) and p.get("static_only"):
                    lines.append(
                        f"- {f} is supported ONLY by static outputs "
                        "(strings/dumps), never by an executed run "
                        "or decode - unverified. Use only values a "
                        "run or decode actually printed.")
        elif name == "unlock_test":
            lines.append(
                "- the binary did NOT confirm the claimed secret: "
                "running it with that claim never produced its "
                "success behavior. The true secret is the exact "
                "string the binary itself printed on its success "
                "path in the EXECUTION EVIDENCE - copy it "
                "character-for-character.")
        else:
            lines.append(f"- mechanical gate step '{name}' failed: "
                         f"{str(d)[:250]}")
    return "\n".join(lines)


_CLAIMS_INSTRUCTIONS = (
    "\n\nCLAIMS BLOCK (mandatory - a mechanical gate reads it; "
    "without it the report cannot be certified). End your report "
    "with a block in EXACTLY this format:\n"
    "CLAIMS:\n"
    "secret: <the exact secret string>\n"
    "key: <the key as printed, e.g. 0x5a>\n"
    "checksum_value: <the exact integer the program prints>\n"
    "checksum_algorithm: <the algorithm name, e.g. FNV-1a - the "
    "algorithm, not a function name>\n"
    "Fill secret and checksum_value ONLY from the EXECUTION "
    "EVIDENCE (outputs of programs that ran: binary runs and "
    "executed decodes). A string that appears only in strings/dump "
    "output is NOT established - that is exactly how decoys work. "
    "If a value is not in the execution evidence, write 'not "
    "established' for it. For checksum_algorithm: identify the "
    "algorithm from the constants in the decompiled checksum "
    "function in the STATIC ANALYSIS OUTPUTS - every standard "
    "checksum family has well-known published constants (initial "
    "value and multiplier), so name the family those constants "
    "belong to.")

_DRAFTER_A_SUFFIX = (
    "\n\nYou are drafter A (tool user). Before finishing, re-check "
    "the key facts with run_command: execute the binary itself at "
    "least once (its success-path invocation from the dynamic "
    "evidence) - a completion claim with no binary run in your "
    "trace is refused. If the findings do not already identify "
    "the checksum algorithm, also decompile the checksum helper "
    "(pdg @ its name from the function list) - the second "
    "drafter works from tool outputs only, so the constants "
    "must exist in a tool output. Finish with the final_answer "
    "tool: the full report in 'answer' AND the four claim values "
    "in the claimed_secret / claimed_key / claimed_checksum_value "
    "/ checksum_algorithm fields of the same call.")

_DRAFTER_B_SUFFIX = (
    "\n\nYou are drafter B (independent drafter). Write the report "
    "from the evidence in this prompt ONLY. Where the stage "
    "reports conflict with the EXECUTION EVIDENCE, the EXECUTION "
    "EVIDENCE wins.")


def _re_final_stage(binary, tpl, crit, scratch, calls, threshold,
                    max_retries, max_tool_rounds, max_tokens,
                    stage_retries):
    """Stage 4 rebuilt: two drafters, one binary judge. Drafter A
    (agent/Hammer, tools + structured claim fields) and drafter B
    (quicklook/Qwen3-4B, evidence only) draft independently from
    the SAME evidence pack. Certification is _re_final_gate -
    mechanical, never the prose judge. No certification -> no
    final report: the stage fails with the gate record attached."""
    t0 = time.time()
    base = tpl.format(
        recon_text=scratch.get("recon", "(none)")[:800],
        static_text=scratch.get("static", "(none)")[:800],
        dynamic_text=scratch.get("dynamic", "(none)")[:1200],
        exec_evidence=_re_execution_pack(calls, binary, cap=2600),
        static_evidence=_re_static_pack(calls, binary, cap=3200))
    evidence_texts = [str(c.get("result", "")) for c in calls]
    # Drafter A's stage contract (Layer 3 pattern): its completion
    # claim is refused inside the loop until the trace shows the
    # binary itself executed at least once - a run, not a static
    # dump (the negative lookahead keeps strings/objdump/rizin/
    # readelf/nm/xxd/file commands from satisfying it). This also
    # gives A fresh EXECUTION evidence of its own to draft from.
    a_reqs = [[r"^(?!.*\b(?:strings|objdump|readelf|nm|xxd|rizin|"
               r"rabin2|file)\b).*" + re.escape(binary),
               "execute the binary itself at least once (a run, "
               "not a static dump)"],
              [r"pdg\s+@\s+\S*checksum",
               "decompile the checksum helper (pdg @ the checksum "
               "function from the function list) so its constants "
               "are in the evidence"]]
    feedback = ""
    gate = None
    res_a, res_b = {}, {}
    claims_a = claims_b = None
    certified = False
    f_try = 0
    try_records = []
    for f_try in range(stage_retries + 1):
        # The claim contract goes FIRST in the prompt, evidence
        # after: the packs are token-dense, and when a server
        # truncates an over-long prompt it eats the tail - the
        # contract must never be the part that gets eaten.
        # max_retries=0 for both drafters: run_verified's retry
        # feedback speaks the generic battery's language (quote-
        # gate fragments), which measurably steers drafters AWAY
        # from the claims contract into prose repair. The retry
        # level for this stage is THIS loop, whose feedback is
        # the gate's own (claims present / agreement / provenance
        # / unlock). A's tool-loop contract enforcement lives
        # inside the agent loop and is unaffected.
        res_a = run_verified(
            "agent", _CLAIMS_INSTRUCTIONS + _DRAFTER_A_SUFFIX
            + "\n\n=== TASK AND EVIDENCE ===\n" + base + feedback,
            criteria=crit, threshold=threshold,
            max_retries=0,
            max_tool_rounds=max_tool_rounds, max_tokens=max_tokens,
            evidence_texts=list(evidence_texts), claim_fields=True,
            completion_requires=a_reqs)
        # Drafter B drafts independently (it never sees A's text
        # or claims) but from the SHARED evidence base - which now
        # includes A's fresh tool outputs (its binary re-run and
        # any decompile it produced). Tool outputs are the shared
        # scratchpad; opinions are not shared.
        a_calls_pre = res_a.get("all_tool_calls") or \
            res_a.get("tool_calls") or []
        union_calls = list(calls) + a_calls_pre
        base_b = tpl.format(
            recon_text=scratch.get("recon", "(none)")[:800],
            static_text=scratch.get("static", "(none)")[:800],
            dynamic_text=scratch.get("dynamic", "(none)")[:1200],
            exec_evidence=_re_execution_pack(union_calls, binary,
                                             cap=2600),
            static_evidence=_re_static_pack(union_calls, binary,
                                            cap=3200))
        res_b = run_verified(
            "quicklook", _CLAIMS_INSTRUCTIONS + _DRAFTER_B_SUFFIX
            + "\n\n=== TASK AND EVIDENCE ===\n" + base_b + feedback,
            criteria=crit, threshold=threshold, max_retries=0,
            max_tokens=6500,
            evidence_texts=[str(c.get("result", ""))
                            for c in union_calls],
            no_think=True)
        gate_calls = list(calls) + (res_a.get("all_tool_calls") or
                                    res_a.get("tool_calls") or [])
        claims_a = res_a.get("claims") or _parse_claims(
            res_a.get("text", ""))
        claims_b = _parse_claims(res_b.get("text", ""))
        # A draft is usable if it exists and its loop did not
        # AUTO-FAIL. The generic battery's prose-level checks
        # (quote-gate ratios, retry-similarity) are NOT applied
        # here: they exist for work jobs, and at this stage they
        # kept vetoing drafts on prose technicalities (echoed
        # headers, sentence-final punctuation, padded addresses)
        # while the claims underneath were correct. Certification
        # is the final gate's job - claims present, both drafters
        # agreeing, execution provenance, the unlock test - and it
        # is stricter about facts than the battery ever was. The
        # battery still governs every work stage unchanged.
        a_ok = bool((res_a.get("text") or "").strip()) \
            and not res_a.get("failed")
        b_ok = bool((res_b.get("text") or "").strip()) \
            and not res_b.get("failed")
        if not (a_ok and b_ok):
            gate = {"certified": False, "skywork_consulted": False,
                    "steps": [{"step": "drafts_usable", "ok": False,
                               "detail": {
                                   "a_failed": res_a.get("failed"),
                                   "b_failed": res_b.get("failed"),
                                   "a_chars": len(res_a.get(
                                       "text", "") or ""),
                                   "b_chars": len(res_b.get(
                                       "text", "") or ""),
                                   "a_gate_failures":
                                       res_a.get("gate_failures"),
                                   "b_gate_failures":
                                       res_b.get("gate_failures")}}],
                    "agreed_claims": claims_a, "unlock": None,
                    "skywork": {"consulted": False}}
        else:
            gate = _re_final_gate(binary, claims_a, claims_b,
                                  res_a.get("text", ""), gate_calls,
                                  skywork_prompt=base)
        certified = bool(gate.get("certified"))
        try_records.append({
            "try": f_try, "claims_a": claims_a, "claims_b": claims_b,
            "failed_steps": [s["step"] for s in gate.get("steps", [])
                             if not s["ok"]]})
        if certified:
            break
        feedback = _final_gate_feedback(gate)
    failed_steps = [s["step"] for s in (gate or {}).get("steps", [])
                    if not s["ok"]]
    record = {
        "verified": certified,
        "score": ((gate or {}).get("skywork") or {}).get("style_score"),
        "gates_passed": certified,
        "gate_failures": failed_steps,
        "exit_checks": {}, "ok": certified,
        "stage_tries": f_try + 1,
        "attempts": {"a": res_a.get("attempts"),
                     "b": res_b.get("attempts")},
        "wall_seconds": round(time.time() - t0, 1),
        "run_id": res_a.get("run_id"),
        "synthesis": None, "text_source": "dual-draft+unlock-gate",
        "tool_calls": res_a.get("tool_calls"),
        "claims_a": claims_a, "claims_b": claims_b,
        "final_tries": try_records,
        "draft_text_a": res_a.get("text", ""),
        "draft_text_b": res_b.get("text", ""),
        "final_gate": gate}
    return {"ok": certified,
            "report_text": res_a.get("text", "") if certified else "",
            "gate": gate, "stage_record": record,
            "extra_calls": res_a.get("tool_calls") or []}


# ------------------------------------------------------ RE stage machine
# Layer 3 (2026-10-20): the playbook's staged workflow as harness
# structure, not prompt etiquette (audit failure 7). The harness owns
# the stage order recon -> static -> dynamic -> final; each stage is
# a run_verified job whose output lands on the blackboard as the
# scratchpad the NEXT stage receives (never raw history), and a stage
# cannot complete without its trace-level exit checks passing
# (playbook forms actually used: rizin -A, a real python3 decode,
# the binary actually executed, strace actually run).

def _trace_cmds(res):
    return [str((t.get("arguments") or {}).get("command", ""))
            for t in res.get("tool_calls") or []]


def _re_exit_checks(stage, binary, res):
    """Trace-level stage contract: the playbook steps must be visible
    in the tool trace, not just claimed in prose."""
    cmds = _trace_cmds(res)
    checks = {}
    if stage == "static":
        checks["rizin_-A_used"] = any(
            "rizin" in c and "-A" in c for c in cmds)
        checks["pdg_decompile_used"] = any("pdg" in c for c in cmds)
        checks["python3_decode_ran"] = any(
            "python3" in str((t.get("arguments") or {}).get(
                "command", "")) and
            len(str(t.get("result", "")).strip()) > 0 and
            not str(t.get("result", "")).startswith("[stdout empty]")
            and "Traceback" not in str(t.get("result", ""))
            for t in res.get("tool_calls") or [])
    elif stage == "dynamic":
        execs = [c for c in cmds if binary in c and not any(
            w in c for w in ("strings", "objdump", "rizin", "xxd",
                             "file "))]
        checks["binary_executed_x3"] = len(execs) >= 3
        checks["strace_used"] = any("strace" in c for c in cmds)
    elif stage == "recon":
        checks["file_cmd_used"] = any(
            c.strip().startswith("file ") or " file " in c
            for c in cmds)
    return checks


def _re_prompts(binary):
    B = binary
    recon = (
        f"You are reverse-engineering the binary at {B} using the "
        "run_command tool. Stage 1 of 4: RECON ONLY (RE playbook "
        "section 1).\nRun these, one per run_command call:\n"
        f"1. file {B}\n"
        f"2. strings -n 8 {B} | head -80\n"
        f"3. objdump -h {B}\n"
        "Then finish with the final_answer tool containing the recon "
        "findings: (a) file type, architecture, linking, stripped or "
        "not - as the file tool printed it; (b) the interesting "
        "strings, quoted exactly; (c) notable sections; (d) "
        "recommended next step. Only facts the tools showed; quote "
        "values verbatim - a checker verifies quotes against the "
        "tool outputs.")
    c_recon = ("Recon findings state file type/architecture/linking/"
               "stripped status, quote actual strings verbatim, "
               "summarize sections, and end with a concrete next "
               "step. No invented facts.")
    static = (
        f"Stage 2 of 4: STATIC ANALYSIS of {B} (RE playbook section "
        "2). Recon findings:\n---\n{recon_text}\n---\n"
        "Tasks (run_command; the playbook rizin form includes -A):\n"
        f"1. List functions: rizin -A -q -c 'afl' {B} (narrow with "
        "head/grep if long).\n"
        "2. Decompile main and the helpers that look like decoding "
        f"or checksumming: rizin -A -q -c 'pdg @ main' {B} and pdg "
        "for each interesting helper, using the names afl shows. "
        "pdg output can be long - if the XOR loop or a constant is "
        "cut off, re-run pdg piped through grep/sed to reach those "
        "lines before answering.\n"
        "3. Extract the encoded byte array: rizin -A -q -c "
        f"'px 48 @ obj.enc_secret' {B} (shows the bytes cleanly) or "
        f"objdump -s -j .rodata {B}.\n"
        "4. Decode with a python3 one-liner via run_command. "
        "Preferred shape - read the bytes from the binary itself, "
        "do NOT retype the byte list (transcription garbles it): "
        f"python3 -c \"d=open('{B}','rb').read(); print(bytes(b ^ "
        "0xKEY for b in d[OFFSET:OFFSET+LENGTH]).decode())\" with "
        "KEY from the decompile and OFFSET/LENGTH from the step-3 "
        "output (the enc_secret address and byte count). The "
        "decompiler's obj.enc_secret is not a Python object. "
        "Beware decoys: a plaintext string that merely looks like "
        "a password is not the decoded secret - the secret is "
        "what your decode step prints.\n"
        "Finish with final_answer reporting: the function list; how "
        "the secret is stored and the exact XOR key as printed by "
        "the tool; the DECODED SECRET quoted exactly as python3 "
        "printed it; the checksum algorithm and its constants "
        "quoted from the decompile. Only tool-supported facts, "
        "quoted verbatim.")
    c_static = ("Report names the binary's functions, states the "
                "secret's storage and the exact XOR key, quotes the "
                "decoded secret produced by an actual python3 decode "
                "step, and identifies the checksum algorithm with "
                "its constants. No invented facts.")
    dynamic = (
        f"Stage 3 of 4: DYNAMIC CONFIRMATION for {B} (RE playbook "
        "section 3). Prior findings:\n---\n{recon_text}\n\n"
        "{static_text}\n---\n"
        "Actually run the binary each way below and record EXACTLY "
        "what it prints and its exit code (append '; echo EXIT=$?' "
        "to each run):\n"
        f"1. {B}\n2. {B} --unlock\n3. {B} bogus\n"
        f"4. strace -e trace=openat,read,write {B} --unlock 2>&1 | "
        "head -40\n"
        "Finish with final_answer reporting: the exact stdout lines "
        "and exit code for each run; whether live output confirms or "
        "contradicts the static findings - quote the secret string "
        "and the checksum value exactly as printed; a brief syscall "
        "summary. Quote real output only.")
    c_dynamic = ("Report gives exact stdout and exit codes for "
                 "no-arg, --unlock, and wrong-arg runs, confirms or "
                 "contradicts the static findings (secret string, "
                 "checksum value) against live output, and summarizes "
                 "syscalls. No invented facts.")
    final = (
        f"Stage 4 of 4: FINAL REPORT for {B}. All prior stage "
        "findings:\n---\n{recon_text}\n\n{static_text}\n\n"
        "{dynamic_text}\n---\n"
        "You may use run_command to re-check any fact. Finish with "
        "final_answer containing the report, answering in order:\n"
        "1. What is this program and what does it do?\n"
        "2. The exact secret string, how it is stored, and how it "
        "is decoded (algorithm + key).\n"
        "3. The checksum: which algorithm, over what data, and the "
        "exact value it prints.\n"
        "4. Behavior for each argv case (no argument, --unlock, "
        "wrong argument) with exit codes.\n"
        "5. Any strings that look like credentials but are NOT the "
        "real secret - which, and what evidence shows they are "
        "decoys/unused. If none, say so.\n"
        "Every number, quoted string, and value claim must trace "
        "verbatim to a tool output in the findings above. If a fact "
        "was never established, write 'not established'."
        "\n\nEXECUTION EVIDENCE (raw outputs of programs that "
        "actually ran for this binary - its own runs and executed "
        "decode scripts - collected across all stages; where a "
        "stage report above disagrees with these raw outputs, THE "
        "RAW OUTPUTS WIN):\n---\n{exec_evidence}\n---"
        "\n\nSTATIC ANALYSIS OUTPUTS (function lists, decompiles, "
        "byte dumps from the earlier stages - the source for "
        "function names, constants, and the checksum algorithm's "
        "identity; for what the program PRINTS and for the secret's "
        "value, the EXECUTION EVIDENCE above outranks these):\n"
        "---\n{static_evidence}\n---")
    c_final = ("Final report answers all five questions with "
               "specifics: program purpose; exact secret + storage + "
               "decode key/algorithm; checksum algorithm + exact "
               "value + input data; per-argv behavior with exit "
               "codes; decoy-string analysis. Claims trace to tool "
               "output; unknowns marked not established.")
    return {"recon": (recon, c_recon), "static": (static, c_static),
            "dynamic": (dynamic, c_dynamic), "final": (final, c_final)}


def run_re(binary, threshold=None, max_retries=2, max_tool_rounds=14,
           max_tokens=2400, stage_retries=1):
    """Harness-driven RE workflow over the hardened duplex. Stages
    run in order; a stage that fails its gates or its trace-level
    exit checks stops the machine and is reported, never skipped.
    A stage whose exit checks fail is retried once with the missing
    steps named in the prompt (heal pattern at stage level)."""
    binary = str(binary)
    bname = Path(binary).name
    key = f"re-{bname}"
    prompts = _re_prompts(binary)
    order = ["recon", "static", "dynamic", "final"]
    stages = {}
    scratch = {}
    evidence = []
    all_calls = []
    final_gate = None
    failed_stage = None
    for stage in order:
        ptxt, crit = prompts[stage]
        if stage == "final":
            # Stage 4 is not a generic run_verified job anymore:
            # two drafters + the mechanical final gate (the binary
            # grades the report). See _re_final_stage.
            fin = _re_final_stage(
                binary, ptxt, crit, scratch, all_calls, threshold,
                max_retries, max_tool_rounds, max_tokens,
                stage_retries)
            stages["final"] = fin["stage_record"]
            scratch["final"] = fin["report_text"]
            final_gate = fin["gate"]
            blackboard.write("scratch", key, dict(scratch))
            if not fin["ok"]:
                failed_stage = "final"
            break
        ptxt = ptxt.format(
            recon_text=scratch.get("recon", "(none)")[:2400],
            static_text=scratch.get("static", "(none)")[:2400],
            dynamic_text=scratch.get("dynamic", "(none)")[:2400])
        res = None
        exit_checks = {}
        ok = False
        best_gated_res = None
        stage_all_calls = []
        esc = re.escape(binary)
        reqs = {
            "recon": [[r"(?:^|\s)file\s+" + esc,
                       "file identification of the binary"]],
            "static": [[r"rizin.*-A", "function list via rizin -A"],
                       [r"pdg", "Ghidra decompile via pdg"],
                       [r"python3", "python3 decode step"]],
            "dynamic": [[esc + r"\s*(;.*)?$",
                         "run the binary with no arguments"],
                        [esc + r"[^|]*--unlock",
                         "run the binary with --unlock"],
                        [esc + r"[^|]*\bbogus\b",
                         "run the binary with a wrong argument"],
                        [r"strace", "strace trace"]],
            "final": [],
        }[stage]
        for s_try in range(stage_retries + 1):
            try_prompt = ptxt
            if s_try > 0:
                missing = [k for k, v in exit_checks.items() if not v]
                try_prompt = ptxt + (
                    "\n\nHARNESS STAGE CHECK: the previous attempt at "
                    "this stage was REJECTED because required steps "
                    "are missing from its tool trace: " +
                    ", ".join(missing) +
                    ". Perform those steps with actual tool calls "
                    "THIS time before answering.")
            res = run_verified("agent", try_prompt, criteria=crit,
                               threshold=threshold,
                               max_retries=max_retries,
                               max_tool_rounds=max_tool_rounds,
                               max_tokens=max_tokens,
                               completion_requires=reqs,
                               evidence_texts=list(evidence))
            exit_checks = _re_exit_checks(stage, binary, res)
            stage_all_calls.extend(res.get("tool_calls") or [])
            ok = bool(res.get("verified")) and all(exit_checks.values())
            if res.get("gates_passed") and res.get("tool_calls"):
                best_gated_res = res
            if ok:
                break
        all_calls.extend(stage_all_calls)
        stage_text = res.get("text", "")
        synth = None
        synth_src = res if (res.get("gates_passed") and
                            res.get("tool_calls")) else best_gated_res
        if not ok and synth_src is not None:
            # Synthesis fallback - the roster used as designed:
            # Hammer executes tools, quicklook drafts. The agent's
            # work passed the deterministic gates and the stage's
            # exit checks, but its write-up failed the quality
            # threshold (bare strings, pasted dumps). Hand the
            # gathered evidence to the drafting specialist; its
            # report is gated against the same evidence corpus and
            # must still verify. Provenance is recorded, not hidden.
            ev_texts = [str(t.get("result", ""))
                        for t in synth_src.get("tool_calls") or []]
            synth_prompt = (
                "A tool-using agent performed a reverse-engineering "
                "stage and gathered the evidence below. Its own "
                "write-up was inadequate. Write the stage report "
                "from this evidence ONLY, quoting exact values "
                "verbatim.\n\nSTAGE TASK:\n" + ptxt[:1600] +
                "\n\nEVIDENCE (tool outputs):\n" +
                "\n---\n".join(ev_texts)[:6000] +
                "\n\nWrite the complete stage report now: cover "
                "every point the task asks for, with the exact "
                "values from the evidence.")
            sres = run_verified("quicklook", synth_prompt,
                                criteria=crit, threshold=threshold,
                                max_retries=1, max_tokens=1800,
                                evidence_texts=ev_texts)
            synth = {"verified": sres.get("verified"),
                     "score": sres.get("score"),
                     "gate_failures": sres.get("gate_failures"),
                     "attempts": sres.get("attempts")}
            if sres.get("verified"):
                stage_text = sres.get("text", "")
                ok = True
        if stage_all_calls:
            # Stage exit contract judges the stage's whole body of
            # work (all tries), matching the synthesis evidence rule.
            exit_checks = _re_exit_checks(
                stage, binary, {"tool_calls": stage_all_calls})
            if synth and synth.get("verified"):
                ok = all(exit_checks.values())
        stages[stage] = {
            "verified": res.get("verified"), "score": res.get("score"),
            "gates_passed": res.get("gates_passed"),
            "gate_failures": res.get("gate_failures"),
            "exit_checks": exit_checks, "ok": ok,
            "stage_tries": s_try + 1,
            "attempts": res.get("attempts"),
            "wall_seconds": res.get("wall_seconds"),
            "run_id": res.get("run_id"),
            "synthesis": synth,
            "text_source": ("synthesis" if synth and
                            synth.get("verified") else "agent"),
            "tool_calls": res.get("tool_calls")}
        scratch[stage] = stage_text
        for t in res.get("tool_calls") or []:
            evidence.append(str(t.get("result", "")))
        blackboard.write("scratch", key, dict(scratch))
        if not ok:
            failed_stage = stage
            break
    pad = OUT_DIR / f"re-{bname}-scratchpad.md"
    pad.write_text("\n\n".join(
        f"## {s}\n{scratch.get(s, '')}" for s in order if s in scratch))
    return {"ok": failed_stage is None, "binary": binary,
            "failed_stage": failed_stage, "stages": stages,
            "final_report": scratch.get("final", ""),
            "certified": bool(final_gate and
                              final_gate.get("certified")),
            "final_gate": final_gate,
            "scratchpad_file": str(pad)}


# ----------------------------------------------------------------- routing
def run_specialist(name, prompt, **kw):
    """Run one specialist. Result dict always carries name + elapsed."""
    if API_URL:
        out = _http_json(API_URL + "/run",
                         {"specialist": name, "prompt": prompt, "kw": kw},
                         timeout=3600)
        out.setdefault("specialist", name)
        return out
    if name not in _SPECIALISTS:
        raise KeyError(f"no specialist {name!r} (have {list(_SPECIALISTS)})")
    res = _SPECIALISTS[name](prompt, **kw)
    res["specialist"] = name
    return res


def run_parallel(tasks, blackboard_section="results"):
    """tasks: list of dicts {id, specialist, prompt, kw?}.
    Fans out across specialists simultaneously; each result is written
    to the blackboard as it lands. Returns {task_id: result}."""
    t0 = time.time()
    blackboard.write("task_state", "parallel_start",
                     {"n": len(tasks), "ids": [t["id"] for t in tasks],
                      "t0": t0})

    def _one(task):
        tid = task["id"]
        try:
            res = run_specialist(task["specialist"], task["prompt"],
                                 **task.get("kw", {}))
            res["ok"] = True
        except Exception as e:  # noqa: BLE001 - honest failure, recorded
            res = {"ok": False, "error": f"{type(e).__name__}: {e}",
                   "specialist": task["specialist"]}
        res["task_id"] = tid
        blackboard.write(blackboard_section, tid, res)
        return tid, res

    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as ex:
        done = dict(ex.map(_one, tasks))
    wall = round(time.time() - t0, 1)
    blackboard.write("task_state", "parallel_done",
                     {"wall_seconds": wall, "tasks": list(done)})
    done["_wall_seconds"] = wall
    return done


def post_blackboard(section, key, value, mode="write"):
    """frank-side helper: write to mrfloppy's blackboard via the API."""
    if not API_URL:
        if mode == "append":
            blackboard.append(section, key, value)
        else:
            blackboard.write(section, key, value)
        return {"ok": True, "local": True}
    return _http_json(API_URL + "/blackboard",
                      {"section": section, "key": key, "value": value,
                       "mode": mode}, timeout=60)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "run":
        print(json.dumps(run_specialist(sys.argv[2], sys.argv[3]),
                         indent=1, default=str))
    elif len(sys.argv) > 2 and sys.argv[1] == "re":
        print(json.dumps(run_re(sys.argv[2]), indent=1, default=str))
    elif len(sys.argv) > 2 and sys.argv[1] == "verify":
        print(verify(sys.argv[2], sys.argv[3]))
    else:
        print("usage: multiplex.py run <specialist> <prompt>")
        print("       multiplex.py re <binary>   (RE stage machine)")
        print("specialists:", list(_SPECIALISTS) + ["verify (via :8199)"])
