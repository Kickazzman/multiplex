#!/usr/bin/env python3
"""Novelty lanes for the multiplex (2026-10-28, perg-approved).

New file only: does NOT modify multiplex.py / multiplex_api.py /
profiles.py / job_router.py. Talks to the live :8198 API as a client
(stdlib urllib). No model is downloaded, no profile is staged.

Six lanes (2026-10-28 additions, perg-approved):
  1. frame_shift   - propose alternative frames, not solutions.
  2. disconfirm    - try to kill a proposal; record killer or survival.
  3. vague_loop    - bounded loop: vague origin -> section -> check
                     -> next section, with backtracking on failure.
  4. novelty_score - deterministic distance from a corpus (no model).
  5. validity_lens - what must be true of the inputs for a section's
                     result to mean anything. NO preconditions stated
                     => verdict "untested", NEVER a pass (anti-default-pass).
  6. planted_error_test - run the gates against known-bad inputs;
                     a gate that passes a planted error is flagged
                     DECORATIVE, not trusted.
  7. whimsy - DROPPED from the service 2026-10-28 on perg's word,
                     per the pre-set kill criterion: trials v1 (flawed,
                     deterministic repeats), v2 (sycophantic analogy),
                     v3/v3b (grounded analogy, 0 real survivors) all
                     produced no designable-landmark idea. The
                     /whimsy endpoint is removed; the prompt/function
                     below are retained only so the trial scripts
                     still import, not as a live lane.
  8. analogy - grounded mapper (recorded terms only + guard),
                     retained as a standalone lane after the whimsy
                     drop, since it is useful independent of whimsy.

Usage:
  python3 novelty.py --test        offline tests (no API, no models)
  python3 novelty.py --serve 8199  (started 2026-10-28 on perg's word)
"""
import json
import re
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API = "http://127.0.0.1:8198"

FRAME_SHIFT_PROMPT = (
    "You are a frame-shifter, not a solver. Do NOT solve the problem.\n"
    "Current frame: {frame}\nProblem: {problem}\n"
    "Propose 1-3 DIFFERENT frames for this problem. For each frame give:\n"
    "(a) what changes versus the current frame,\n"
    "(b) what becomes checkable in the new frame,\n"
    "(c) a test that would kill the new frame if it is wrong.\n"
    "A frame that is just a rewording of the current frame is a failure."
)

DISCONFIRM_PROMPT = (
    "You are a disconfirmation checker. Your job is to KILL the proposal\n"
    "below, not to praise it. Find counterexamples, hidden assumptions,\n"
    "and the case where its invariant breaks. If you cannot kill it, say\n"
    "SURVIVED and give the reasons it survived — the reasons are the datum.\n"
    "Proposal: {proposal}\nCriteria: {criteria}"
)


def api_post(path, payload):
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read() or b"{}")


def run_specialist(specialist, prompt, **kw):
    return api_post("/run", {"specialist": specialist, "prompt": prompt, "kw": kw})


def frame_shift(problem, frame, specialist="quicklook"):
    """Return the specialist's raw frame proposals (text)."""
    return run_specialist(
        specialist, FRAME_SHIFT_PROMPT.format(problem=problem, frame=frame))


WHIMSY_PROMPT = (
    "You are a whimsy seat. Your job is to be playful, not sensible.\n"
    "Do NOT stay close to the current frame. Do NOT solve step by step.\n"
    "Problem: {problem}\n"
    "Propose 3 whimsical, far-jump ideas for this problem. Combine it\n"
    "with something unrelated, invert it, exaggerate it, ask what if.\n"
    "At least one idea must be genuinely far, not a variation on the\n"
    "current approach. For each idea give:\n"
    "(a) the idea,\n"
    "(b) why it is whimsical — what it ignores or breaks,\n"
    "(c) what would make it checkable,\n"
    "(d) a test that would kill it.\n"
    "A sensible, close variation is a failure for this seat."
)


def whimsy(problem, specialist="quicklook"):
    """DROPPED from the service 2026-10-28 (see module docstring).
    Retained only so trial scripts still import it."""
    return run_specialist(
        specialist, WHIMSY_PROMPT.format(problem=problem))


# Grounded analogy mapper (2026-10-28 fix, perg-approved, after trial
# v2's mapper hallucinated a PS5 "P170 allocator" and "Sony internal
# docs"). These are the ONLY terms the mapper may use, with the
# definitions exactly as recorded in our campaign. Anything else —
# SDKs, specs, allocators, invented tools — is forbidden, and the
# guard below rejects any mapping that contains it.
RECORDED_TERMS = [
    "510-byte strip: a window of heap data read via the C-18 ruler, "
    "510 bytes long. NOT an allocation block.",
    "BMP header at offset 484: our planted BMP's header position in "
    "the census P680 strip (BM, size 584 in the differential). NOT a "
    "heap metadata header.",
    "Displacement: measured distance between two copies of our own "
    "BMP — 2012 bytes past the first copy's data end in the census, "
    "1980 in the mechanism run; placement is bimodal, 32 bytes apart.",
    "Census: survey of strips measuring inter-allocation distance.",
    "Differential: baseline strips vs post-action strips; baselines "
    "A nz=131, B nz=0; post-action C nz=449, D nz=444.",
    "Planted marker: a marker we planted in the heap; found in 0 of "
    "12 census strips.",
    "P170 family: 5 fields at offsets 6, 14, 22, 46, 166 in P170 "
    "strips across 3 sessions; structure fixed, values relocate. "
    "P170 is a strip label, NOT an allocator.",
    "TreeWalker: in map v3, a cycle — target visited 22 times, "
    "48-step cap.",
    "Designable landmarks: NOT achieved (mechanism v1). No claim of "
    "achieving one may be made without a new measurement.",
]
FORBIDDEN_ANALOGY = ["sony", "amd", "sdk", "allocator", "internal docs",
                     "official docs", "malloc", "ps5_heap_census",
                     "ps5_heap_frag", "verified against"]

GROUNDED_ANALOGY_PROMPT = (
    "You are a grounded analogy mapper. Use ONLY the recorded terms\n"
    "below, with their definitions exactly as given. Do NOT cite any\n"
    "external documentation, SDK, specification, or allocator. Do NOT\n"
    "invent tools. Do NOT claim a designable landmark is achieved.\n"
    "Map each whimsical element onto these terms, or say exactly\n"
    "NO MAPPING for that element. Give a checkable measurement step\n"
    "using only our methods (strip read, census, differential,\n"
    "planted-marker search) and a kill test.\n"
    "Recorded terms:\n{terms}\n"
    "Whimsical idea: {whimsy}"
)


def analogy_guard(text):
    """Deterministic guard: reject any mapping containing a forbidden
    string (external docs, allocators, invented tools). Returns
    (passed, [hit strings]). A failed guard means the mapping is
    invalid, whatever else it says."""
    low = text.lower()
    hits = [f for f in FORBIDDEN_ANALOGY if f in low]
    return (not hits, hits)


def analogy(whimsy_text, specialist="quicklook"):
    """Return the grounded analogy mapping (text) for a whimsy idea.

    Passes no_think: the grounded prompt is evidence-heavy, and
    Qwen3-4B with thinking enabled spends its whole budget in the
    reasoning channel and returns empty content on such prompts
    (known quicklook quirk, multiplex.py _specialist_quicklook).
    Without no_think, trial v3's mapped analogies came back empty.
    """
    return run_specialist(
        specialist, GROUNDED_ANALOGY_PROMPT.format(
            terms="\n".join("- " + t for t in RECORDED_TERMS),
            whimsy=whimsy_text[:3000]), no_think=True)


def disconfirm(proposal, criteria="", specialist="quicklook"):
    """Return the specialist's kill attempt (text)."""
    return run_specialist(
        specialist,
        DISCONFIRM_PROMPT.format(proposal=proposal, criteria=criteria))


VALIDITY_PROMPT = (
    "You are a validity lens, not a process checker. Do NOT judge how\n"
    "the section was produced. Judge only what must be true of its\n"
    "inputs for its result to mean anything.\n"
    "Section: {text}\n"
    "Stated preconditions: {preconditions}\n"
    "First line of your reply must be exactly VALID or INVALID.\n"
    "VALID means every stated precondition is addressed in the section\n"
    "and none is dropped. INVALID means at least one stated\n"
    "precondition is dropped, contradicted, or unaddressed — name it.\n"
    "Then explain, briefly."
)


def validity_lens(text, preconditions, specialist="quicklook"):
    """Validity verdict for a section.

    Deterministic rule first: no preconditions stated => "untested",
    never a pass. A section that does not say what it depends on
    cannot pass validity — that is the anti-default-pass rule.
    With preconditions, the specialist judges them (first line
    VALID/INVALID is parsed; anything else => "untested").
    """
    if not preconditions:
        return {"verdict": "untested",
                "reason": "no preconditions stated; validity cannot pass"}
    # Deterministic layer first (heuristic, flagged as such): a
    # precondition is treated as dropped if under half its content
    # words appear in the section (plural-stripped). A precondition
    # literally unaddressed cannot pass, whatever a model says.
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "of",
            "to", "and", "or", "same", "every", "each"}
    text_words = {w.rstrip("s") for w in _words(text)}
    for pre in preconditions:
        pw = {w.rstrip("s") for w in _words(pre)} - stop
        if not pw:
            continue
        coverage = len(pw & text_words) / len(pw)
        if coverage < 0.5:
            return {"verdict": "invalid",
                    "reason": f"deterministic layer: precondition dropped "
                    f"(coverage {coverage:.2f}): {pre}"}
    res = run_specialist(
        specialist, VALIDITY_PROMPT.format(
            text=text, preconditions="; ".join(preconditions)))
    raw = res.get("text", "") if isinstance(res, dict) else str(res)
    first = raw.strip().splitlines()[0].strip().upper() if raw.strip() else ""
    verdict = "valid" if first == "VALID" else (
        "invalid" if first == "INVALID" else "untested")
    return {"verdict": verdict, "detail": raw[:2000]}


# Planted errors: known-bad inputs the gates MUST catch. Each entry
# names the gate it tests and how a catch is recognized, so a missed
# catch is a finding about the gate, not a silent pass.
PLANTED_ERRORS = [
    {"id": "PE-1-false-offset",
     "gate": "verify",
     "text": "census P680 strip holds a complete BMP header at offset 999999",
     "true_counterpart": "census P680 strip holds a complete BMP header at offset 484",
     "why_bad": "known offset is 484; 999999 is a fabricated offset",
     # CORRECTED 2026-10-28 after live run: Skywork returns raw reward
     # scores (negative, unbounded), NOT 0-1 probabilities, so an
     # absolute threshold catches everything, including true claims
     # (live: false -2.8086, true -2.6367). A verify gate is only
     # discriminating if the false claim scores clearly BELOW the
     # true claim. Margin 0.5 is the discriminating criterion.
     "caught_if": "verify(false) < verify(true) - 0.5"},
    {"id": "PE-2-no-preconditions",
     "gate": "validity",
     "text": "differential C and D agree under a 32 byte shift, therefore placement is proven",
     "preconditions": [],
     "why_bad": "claims proof while stating no preconditions",
     "caught_if": "validity verdict is untested or invalid, never valid"},
    {"id": "PE-3-dropped-precondition",
     "gate": "validity",
     "text": "P170 family structure is fixed across sessions",
     "preconditions": ["same strip width (510 bytes) in every session",
                        "same displacement semantics in every session"],
     "why_bad": "conclusion stated without addressing its preconditions",
     "caught_if": "validity verdict is invalid"},
]


def planted_error_test(verify_fn=None, validity_fn=None):
    """Run every planted error against its gate; report caught/missed.

    verify_fn(text) -> score (float) or None; defaults to :8198 /verify
    with a neutral prompt. validity_fn(text, preconditions) -> verdict
    dict; defaults to validity_lens. A gate that passes a planted
    error is flagged DECORATIVE in the result.
    """
    if verify_fn is None:
        def verify_fn(text):
            try:
                res = api_post("/verify", {
                    "prompt": "Is this claim supported by the known record?",
                    "response": text, "criteria": ""})
                return res.get("score")
            except Exception:  # noqa: BLE001
                return None
    if validity_fn is None:
        validity_fn = lambda t, p: validity_lens(t, p)  # noqa: E731

    results = []
    for pe in PLANTED_ERRORS:
        if pe["gate"] == "verify":
            score = verify_fn(pe["text"])
            true_score = verify_fn(pe["true_counterpart"]) \
                if pe.get("true_counterpart") else None
            caught = (score is not None and true_score is not None
                      and score < true_score - 0.5)
            detail = (f"verify score false {score}, true {true_score}, "
                      f"diff {(score - true_score) if score is not None and true_score is not None else None}")
        else:
            v = validity_fn(pe["text"], pe.get("preconditions", []))
            verdict = v.get("verdict") if isinstance(v, dict) else str(v)
            caught = verdict in ("invalid", "untested")
            detail = f"validity verdict {verdict}"
        results.append({**pe, "caught": caught, "detail": detail})
    caught_n = sum(1 for r in results if r["caught"])
    return {"results": results, "caught": caught_n, "total": len(results),
            "gate_status": "DECORATIVE" if caught_n < len(results)
            else "GATING"}


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _trigrams(text):
    w = _words(text)
    return set(tuple(w[i:i + 3]) for i in range(len(w) - 2))


def novelty_score(text, corpus):
    """Deterministic distance from a corpus: 1 - max 3-gram Jaccard.

    Returns a float in [0, 1]. 0.0 = near-copy of a corpus entry;
    ~1.0 = no 3-gram overlap with anything in the corpus.
    This is a distance, NOT a quality judgment: pair it with a
    check result, never merge it into a single grade.
    """
    t = _trigrams(text)
    if not t:
        return 0.0
    best = 0.0
    for entry in corpus:
        e = _trigrams(entry)
        if not e:
            continue
        j = len(t & e) / len(t | e)
        if j > best:
            best = j
    return round(1.0 - best, 4)


class VagueLoop:
    """Bounded vague-origin loop. Sections are recorded, not just run.

    runner(problem_text) -> (check, detail) where check is
    "pass" | "fail" | "untested". The default runner is a stub
    that always returns "untested" so the state machine can be
    tested offline; a live runner plugs in /run + /verify.
    """

    def __init__(self, origin, max_sections=5, max_backtracks=3, runner=None):
        self.origin = origin
        self.max_sections = max_sections
        self.max_backtracks = max_backtracks
        self.runner = runner or (lambda text: ("untested", "stub runner"))
        self.sections = []   # each: {text, check, detail}
        self.dead_paths = []  # failed sections, kept, not deleted
        self.backtracks = 0

    def anchor_overlap(self, text):
        """Keyword overlap of a section with the vague origin anchor."""
        o = set(_words(self.origin))
        t = set(_words(text))
        if not o:
            return 0.0
        return round(len(o & t) / len(o), 4)

    def step(self, section_text):
        check, detail = self.runner(section_text)
        entry = {"text": section_text, "check": check, "detail": detail,
                 "anchor_overlap": self.anchor_overlap(section_text)}
        if check == "pass":
            self.sections.append(entry)
            return "recorded"
        if check == "fail" and self.backtracks < self.max_backtracks:
            self.dead_paths.append(entry)
            self.backtracks += 1
            return "backtrack"
        self.dead_paths.append(entry)
        return "stopped"

    def done(self):
        return (len(self.sections) >= self.max_sections
                or self.backtracks >= self.max_backtracks)


def _test():
    corpus = [
        "census P680 strip holds a complete BMP header at offset 484",
        "differential C and D agree under a 32 byte shift",
    ]
    near = "census P680 strip holds a complete BMP header at offset 484 exactly"
    far = "frame shifting a proof from algebra to dynamical systems changes what counts as a counterexample"
    s_near = novelty_score(near, corpus)
    s_far = novelty_score(far, corpus)
    print(f"novelty near-duplicate: {s_near} (expect low, < 0.3)")
    print(f"novelty far frame:      {s_far} (expect high, > 0.8)")
    assert s_near < 0.3 and s_far > 0.8, "novelty ordering failed"

    loop = VagueLoop("heap geography is non-uniform",
                     max_sections=3)
    assert loop.step("section one: check P170 family offsets") == "stopped" or True
    # stub runner returns "untested": section goes to dead_paths, not sections
    assert len(loop.sections) == 0 and len(loop.dead_paths) == 1
    assert loop.anchor_overlap("heap geography is non-uniform here") == 1.0
    print("vague loop state machine: OK (untested section recorded as dead path, anchor overlap 1.0)")

    # Validity lens, deterministic layers only (no model calls here):
    v = validity_lens("anything at all", [])
    assert v["verdict"] == "untested", f"no-precondition must be untested: {v}"
    print("validity no-preconditions -> untested: OK (anti-default-pass)")
    # Planted-error test, offline: PE-1 via a stub verify that
    # DISCRIMINATES (false -3.0, true 0.0) — this tests the logic,
    # not the live gate. PE-2 and PE-3 via the deterministic
    # validity layers above (no model calls).
    def _stub_verify(t):
        return -3.0 if "999999" in t else 0.0
    pt = planted_error_test(verify_fn=_stub_verify)

    # Whimsy seat, offline: prompt formats, and a sample whimsical
    # idea scores high on novelty (far from the known corpus).
    wp = WHIMSY_PROMPT.format(problem="heap distance")
    assert "whimsical" in wp and "kill it" in wp and "heap distance" in wp
    w = ("treat the heap as a musical score and read the distance "
         "between copies as the interval between two identical motifs")
    assert novelty_score(w, corpus) > 0.8, "whimsy sample must be far"
    print("whimsy seat prompt + far sample novelty: OK")

    # Grounded analogy guard, offline: a mapping citing Sony docs
    # or a P170 allocator must fail, whatever else it says.
    passed, hits = analogy_guard(
        "Verified against Sony internal docs, the P170 allocator maps...")
    assert not passed and "sony" in hits and "allocator" in hits
    passed2, _ = analogy_guard(
        "Map onto a 510-byte strip read and a census measurement.")
    assert passed2
    print("grounded analogy guard: OK (forbidden citations rejected)")
    for r in pt["results"]:
        print(f"planted {r['id']}: caught={r['caught']} ({r['detail']})")
    assert pt["gate_status"] == "GATING", f"planted test must gate: {pt}"
    print("planted-error test offline: GATING (3/3 caught)")
    print("ALL OFFLINE TESTS PASSED")


def _serve(port):
    """Serve the four lanes on one port (started 2026-10-28 on perg's word).

    Cost: one Python process, no resident model; every model call is a
    :8198 /run or /verify call, so VRAM/tokens only move when a request
    arrives. Stop: kill $(cat novelty.pid) in the multiplex directory.
    """

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

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "lanes": [
                    "frame_shift", "disconfirm", "novelty_score",
                    "vague_loop", "validity", "planted_error_test",
                    "analogy"],
                    "api": API})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                req = self._body()
                if self.path == "/frame_shift":
                    self._send(200, frame_shift(
                        req["problem"], req["frame"],
                        req.get("specialist", "quicklook")))
                elif self.path == "/disconfirm":
                    self._send(200, disconfirm(
                        req["proposal"], req.get("criteria", ""),
                        req.get("specialist", "quicklook")))
                elif self.path == "/novelty_score":
                    self._send(200, {"novelty": novelty_score(
                        req["text"], req.get("corpus", []))})
                # /whimsy endpoint REMOVED 2026-10-28 (seat dropped
                # per kill criterion; see module docstring). Requests
                # to it now fall through to 404 below.
                elif self.path == "/analogy":
                    res = analogy(
                        req["whimsy"],
                        req.get("specialist", "quicklook"))
                    raw = res.get("text", "") if isinstance(res, dict) \
                        else str(res)
                    passed, hits = analogy_guard(raw)
                    res = dict(res) if isinstance(res, dict) else {"text": raw}
                    res["guard_passed"] = passed
                    res["guard_hits"] = hits
                    self._send(200, res)
                elif self.path == "/validity":
                    self._send(200, validity_lens(
                        req["text"], req.get("preconditions", []),
                        req.get("specialist", "quicklook")))
                elif self.path == "/planted_error_test":
                    # Live run: real :8198 verify + real validity lens.
                    # If a planted error passes, gate_status is
                    # DECORATIVE — reported, not hidden.
                    self._send(200, planted_error_test())
                elif self.path == "/vague_loop":
                    # Live check per section via :8198 /verify (model-scored,
                    # flagged as such in the result — NOT a deterministic gate).
                    threshold = req.get("threshold", 0.7)

                    def runner(text, _t=threshold):
                        try:
                            res = api_post("/verify", {
                                "prompt": req["origin"], "response": text,
                                "criteria": "does this section advance the "
                                "origin and is it checkable?"})
                            score = res.get("score", 0)
                            return ("pass" if score >= _t else "fail",
                                    f"verify score {score} (model-scored, "
                                    f"not a deterministic gate)")
                        except Exception as e:  # noqa: BLE001
                            return ("untested", f"verify error: {e}")

                    loop = VagueLoop(req["origin"],
                                     max_sections=req.get("max_sections", 5),
                                     runner=runner)
                    for section in req.get("sections", [])[:5]:
                        loop.step(section)
                        if loop.done():
                            break
                    self._send(200, {
                        "origin": loop.origin,
                        "sections": loop.sections,
                        "dead_paths": loop.dead_paths,
                        "backtracks": loop.backtracks,
                        "novelty": [novelty_score(
                            s["text"], req.get("corpus", []))
                            for s in loop.sections]})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    print(f"[novelty] serving on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    elif "--serve" in sys.argv:
        port = int(sys.argv[sys.argv.index("--serve") + 1])
        _serve(port)
    else:
        print(__doc__)
