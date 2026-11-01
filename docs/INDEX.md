# Tuning docs index — how tuning works, and the math to optimize seats/fleet

No standalone tuning document was found in the places searched (see
"Places searched" below), and the word "tuning" does not appear in the
multiplexer sources. Nothing here is invented: the tuning rules live in
code and configs, cited below by file and symbol. This index only points
at them and quotes the module docstring that states the design.

## How tuning works

A profile is a named cast: the set of bench models staged resident for a
class of task, with a per-host VRAM budget check (`profiles.py` module
docstring, quoted verbatim):

> Perg's design step (2026-10-20): the tiny specialists become
> INTERCHANGEABLE per task. A *bench* registry describes every servable
> model (where it lives, how it is started/stopped, what it costs in
> VRAM - measured, not guessed). A *profile* is a named cast: the set of
> bench models staged resident for a class of task, with a per-host VRAM
> budget check. Staging stops what the cast does not need, starts what
> it does, waits for health, verifies VRAM against nvidia-smi, and
> records the active profile on the blackboard.

Profile state as published: active profile `balanced` (see `.profile-state.json`;
state changes over time — the file, not this line, is current).
`mode_index.json` is the derived mode index (425 map entries; regenerated
from the use-map markdown, which is the single source of truth).

## The math to optimize them (as recorded in code)

- VRAM budget math — `profiles.py`, `profile_math()`: per host,
  `total = sum(vram_mib)` over the resident cast (transient models count
  0), `headroom = capacity - total`, `fits = (total + margin <= capacity)`.
  Capacities and margins live in `HOST_CAPACITY_MIB` / `HOST_MARGIN_MIB`
  in the same file; per-model `vram_mib` values in `BENCH` are MEASURED
  per-process footprints (nvidia-smi, 2026-10-20, per the file's comments).
  `stage_profile()` refuses a cast that does not fit, and refuses staging
  while a verified run is in flight.
- Decider threshold — `mode_decide.py`: `CONFIDENCE_THRESHOLD = 0.8`.
  Escalate on a fast null or fast confidence below the threshold.
  Full-insurance mode is a separate toggle, default OFF.
- Enforcement gates — `mode_enforce.py`: serial seat plans run in
  trap-restored windows; a free-VRAM gate runs after parking, and the
  window fails if the gate fails.
- Shared decode budget — `multiplex_api.py` (comment): concurrent decodes
  share a fixed ~245 tok/s budget, so one decode at a time finishes every
  job sooner (mean latency note in the same comment).
- Cost visibility — `profiles.py` docstring: every service has an on/off
  switch with its cost visible; `GET /bench` and `GET /profiles` report
  VRAM per model/profile. `novelty.py` notes its lane cost as one Python
  process with no resident model.
- Single-GPU swap — `docs/ops/deep-mode.sh` and `docs/ops/model-swap.sh`:
  one model on the card at a time; every failure path restores Strata
  (the voice brain) before exiting non-zero. `docs/ops/glados.sh`
  composes presence on/off with deep mode.

## Missing (reported, not filled in)

- No standalone VRAM/cost/latency formula document exists in the places
  searched. Cost and latency appear only as comments, measured values,
  and the rules above — no separate cost model or latency model was found.
- No seat-sizing document separate from `profiles.py` (`BENCH` +
  `profile_math()` is the sizing record) and no mode-selection document
  separate from the profile/config code cited above.

## Places searched (2026-11-01)

- `/home/joshua/llm/voice-loop/multiplex/` — top-level `.py`/`.json`/`.md`
  (the published selection); subdirs `blackboard-mirror/`, `out/`, `logs/`,
  `models/` excluded from this repo and not used as tuning sources.
- `~/llm/voice-loop/` root — no `.md`/`.txt` docs at the root.
- `~/llm/voice-loop/bin/` — 3 scripts (`deep-mode.sh`, `model-swap.sh`,
  `glados.sh`); copies in `docs/ops/`.
- Excluded as non-docs: `fish-speech/docs/` (third-party TTS docs),
  `notes/`, `re-notes/`, `agent_tasks/`, `research-library/`
  (working notes / delivered copies, not tuning docs).
