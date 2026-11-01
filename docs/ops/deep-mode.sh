#!/bin/bash
# deep-mode.sh on|off|status — sustained deep-research mode for the 4090.
# on:  voice brain (Strata) unloaded, Qwen3.8-27B resident; flag file tells the
#      deep worker to skip its end-of-job voice restore, so the card stays on
#      the 27B across jobs until you turn it off.
# off: flag cleared, Strata restored (voice brain back, ~25s).
# The model-swap.sh safety invariant is untouched: failures still restore Strata.
set -u
BIN="$HOME/llm/voice-loop/bin/model-swap.sh"
FLAG="$HOME/llm/voice-loop/run/deep-mode"

card_owner() {
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8082/v1/models 2>/dev/null)" = "200" ]; then
        echo "Qwen3.8-27B (:8082)"
    elif [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8081/v1/models 2>/dev/null)" = "200" ]; then
        echo "Strata voice brain (:8081)"
    else
        echo "none/unknown"
    fi
}

case "${1:-status}" in
  on)
    "$BIN" to-27b && touch "$FLAG" && echo "deep-research mode ON - 4090 on 27B, voice brain off until 'deep-mode.sh off'"
    ;;
  off)
    rm -f "$FLAG"
    "$BIN" to-strata && echo "deep-research mode OFF - voice brain restored"
    ;;
  status)
    echo "card: $(card_owner)"
    if [ -f "$FLAG" ]; then echo "deep-mode: ON (flag set)"; else echo "deep-mode: off"; fi
    ;;
  *) echo "usage: deep-mode.sh on|off|status"; exit 1 ;;
esac
