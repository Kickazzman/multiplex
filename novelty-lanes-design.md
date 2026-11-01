# Novelty lanes for the multiplex — design, 2026-10-28 (perg: "lets do so")

Source of the four changes: assistant's 2026-10-28 proposal, approved by perg
the same night. Rule obeyed throughout: NO whole-file rewrite of
multiplex.py (134 KB) or multiplex_api.py. The lanes live in ONE new
file, novelty.py, next to them, and talk to the live :8198 API as a
client. The live API service is NOT restarted, NO profile is staged,
and NO persistent new service is started in this pass — the lane
files are built and tested offline, and starting them is a separate
word from perg (his service-toggle rule: every service shows its
cost next to its toggle).

Live state verified 2026-10-28 ~00:45 EDT: :8198 /health ok,
active_profile "balanced", runs_in_flight 0, specialists:
agent, code, quicklook, embed, tts, shepherd, vision, stt, ouro,
trm, mathwhiz, draft.

## 1. Frame-shifter seat
Job: propose a DIFFERENT frame for a problem, not a solution in the
given frame. Input: problem + current frame. Output: 1-3 alternative
frames, each with (a) what changes, (b) what becomes checkable in
the new frame, (c) a test that would kill the new frame.
Implementation: a prompt template over an existing generalist
specialist (default: quicklook; agent as fallback), via POST /run.
The frame is returned as text; the multiplex does NOT adopt it
until a checker passes it. No new model is downloaded.

## 2. Disconfirmation duplex
Job: try to KILL a proposal, not verify it. Input: proposal +
criteria. Output: counterexamples, hidden assumptions, and the
case where the invariant breaks — or "survived" with the reasons
it survived (the why is the datum). Implementation: a prompt
template over the checker side (default: run_verified's verifier
path, with criteria rewritten as "find the case that breaks this"),
plus a deterministic record: killed / survived / untested, with
the killing case quoted verbatim. A survived proposal keeps its
grade; a killed one is recorded as dead with its killer, not
silently dropped.

## 3. Vague-origin loop lane
Job: run the loop from the 2026-10-28 discussion as a lane.
State machine, all steps recorded to the blackboard:
  origin (vague text, kept verbatim as the anchor)
  -> propose next section (specialist, frame-shifter allowed)
  -> generate the section's algorithm/check (deterministic where
     possible; model-proposed where not, flagged as such)
  -> run the check (gate: pass / fail / untested)
  -> on pass: record section, generate next section
  -> on fail: record dead path, backtrack one section, try an
     alternative frame (frame-shifter), max 3 backtracks
  -> stop: max_sections reached, or no live path, or origin
     anchor drift (new section no longer references the anchor
     — measured by keyword overlap with the origin, flagged)
Loop is bounded: max_sections (default 5), max_backtracks (3),
and every section is a blackboard entry, so the loop is
auditable after the fact. The loop does NOT run models in a
tight loop on its own — each section is one /run call, and the
lane pauses between sections (coarse-grained, per the API's
own rule).

## 4. Novelty score (deterministic, no model)
novelty_score(text, corpus) = 1 - max_jaccard(word 3-grams of
text vs each corpus entry), clipped to [0, 1].
- 0.0 = text is a near-copy of something already in the corpus.
- ~1.0 = no 3-gram overlap with anything in the corpus.
Corpus default: the multiplex blackboard "findings" section
plus a supplied list of known frames (registry lines, prior
proposals). The score is a distance, not a quality judgment:
a high score with a failed check is still a failure; a low
score with a passed check is still a pass. Reported as
(novelty, check) pairs, never as a single merged grade.
Tested offline in this pass (see novelty.py --test): a
near-duplicate of a known line scores low, an unrelated
frame scores high, and the ordering is stable.

## Files
- multiplex/novelty.py (new, stdlib only, :8198 client + scorer
  + loop state machine + --test offline mode)
- This design doc (workspace copy; mrfloppy copy alongside
  novelty.py)

## Explicitly NOT done in this pass
- No edit to multiplex.py, multiplex_api.py, profiles.py,
  or job_router.py.
- No restart of multiplex-api.service, no profile staging,
  no new model download, no VRAM change.
- No persistent novelty service started. Start command,
  when perg says so:
    cd ~/llm/voice-loop/multiplex && nohup python3 novelty.py --serve 8199 > novelty.log 2>&1 &
  Cost to show next to its toggle: one Python process, no
  resident model (it calls :8198 per section), negligible
  VRAM, tokens only when a section runs.
