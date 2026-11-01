#!/bin/bash
# glados.sh on|off|status — master switch for GLaDOS's interactive presence.
# off (project mode): voice loop, voice brain (Strata) and TTS stop; the 4090
#      goes to deep-research mode (Qwen3.8-27B resident). Workers keep running -
#      the deep worker IS the project runner. Everything serves the queue.
# on:  deep-research mode off, voice brain + loop + TTS restored.
# This composes with bin/deep-mode.sh; it does not replace it.
set -u
DEEP_MODE="$HOME/llm/voice-loop/bin/deep-mode.sh"
PENDING="$HOME/llm/voice-loop/agent_tasks/pending"

svc() { systemctl --user "$1" "$2" 2>/dev/null; }
is_active() { systemctl --user is-active "$1" 2>/dev/null; }

queue_depth() { ls "$PENDING" 2>/dev/null | wc -l; }

case "${1:-status}" in
  off)
    echo "GLaDOS going quiet - project mode."
    svc stop voice-loop.service
    svc stop xtts-voice.service
    "$DEEP_MODE" on
    echo "presence off. deep worker owns the 4090. $(queue_depth) job(s) queued."
    ;;
  on)
    "$DEEP_MODE" off
    svc start xtts-voice.service
    svc start voice-loop.service
    echo "GLaDOS back. voice brain + loop + TTS starting."
    ;;
  status)
    for s in voice-loop.service strata.service xtts-voice.service glados-local-agent.service glados-quick-agent.service glados-slow-agent.service; do
      printf '%-28s %s\n' "$s" "$(is_active $s)"
    done
    "$DEEP_MODE" status
    echo "queued jobs: $(queue_depth)"
    ;;
  *) echo "usage: glados.sh on|off|status"; exit 1 ;;
esac
