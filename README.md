# multiplex

The GLaDOS specialist multiplexer: serial model multiplexing over local
seats on mrfloppy/frank, with profiles (named resident casts with VRAM
budget checks), mode hooks, enforcement windows, and a lean decider.
Profiles stage bench models resident per task class; the decider escalates
on a fast null or confidence below 0.8 (full-insurance toggle default OFF).

## Boundary note — read this first

Multiplex internals were previously held from the community (Oct 27
boundary; Oct 22 in-circle confidentiality rule for the multiplexer).
On 2026-11-01 perg explicitly chose PUBLIC posting for the multiplexer,
knowingly, after the private-vs-public choice was put to him. This repo
is that public posting, under the MIT license. It supersedes the prior
boundary for the multiplexer only.

Excluded: model weights, caches, virtualenvs, logs, `__pycache__`,
backups, the mirrored blackboard tree, run outputs, and any file with
secret/credential content. Network addresses in code are redacted
(`[REDACTED-IP]`); restore your own peer addresses before running.
Service state (which seats are live, which profile is active) is not
asserted here; see RECORD.md for file dates and `docs/INDEX.md` for
where the tuning rules live in code.

## Contents

- `multiplex.py`, `multiplex_api.py`, `profiles.py`, `mode_hook.py`,
  `mode_decide.py`, `mode_enforce.py`, `job_router.py`, `hal_dispatch.py`,
  `blackboard.py`, `novelty.py`, `shepherd_server.py`, `trm_server.py`,
  `verifier_server.py`, `multiplexctl.py`, `test_*.py`, `whimsy-trial*.py`
  — multiplexer source and tests as staged.
- `mode_index.json` — derived mode index (425 map entries; regenerated
  from the use-map; the markdown map is the source of truth).
- `.profile-state.json` — profile state as published.
- `novelty-lanes-design.md` — novelty lanes design.
- `docs/INDEX.md` — tuning docs index: how tuning works, the math to
  optimize seats/fleet, and what is missing (no invented math).
- `docs/ops/` — related operational scripts from `~/llm/voice-loop/bin/`
  (`deep-mode.sh`, `model-swap.sh`, `glados.sh`), copied read-only.

## Credit

Research by Joshua McClain (Kickazzman).
Research partner: cobrajet (GitHub: cobrajet302gt).
License: MIT — see LICENSE. Free use for anyone, with the copyright and
permission notice kept. Provenance: see RECORD.md and REDACTION-LOG.md.
