#!/bin/bash
# model-swap.sh — swap the 4090 between Strata (voice brain) and Qwen3.8-27B (deep worker).
# Usage: model-swap.sh to-27b | to-strata
# Single-GPU rule: one model on the card at a time. The voice brain must NEVER
# be left down: every failure path restores Strata before exiting non-zero.
set -u
HOME_DIR="/home/joshua"
LLAMA_SERVER="$HOME_DIR/llama.cpp-eval/build/bin/llama-server"
MODEL="$HOME_DIR/llm/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
PIDFILE="$HOME_DIR/llm/voice-loop/run/llama-27b.pid"
LOG="$HOME_DIR/llm/voice-loop/logs/llama-27b.log"

health() { # $1 = base url; true if /v1/models answers 200
    [ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$1/v1/models")" = "200" ]
}

wait_for() { # $1 = base url, $2 = timeout secs
    local t=0
    while [ "$t" -lt "$2" ]; do
        if health "$1"; then return 0; fi
        sleep 5; t=$((t + 5))
    done
    return 1
}

stop_27b() {
    if [ -f "$PIDFILE" ]; then
        pid="$(cat "$PIDFILE")"
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 12); do kill -0 "$pid" 2>/dev/null || break; sleep 5; done
        kill -9 "$pid" 2>/dev/null
        rm -f "$PIDFILE"
    fi
    # bracket trick: pattern must not match pkill's own cmdline
    pkill -f "llama-server.*--port 808[2]" 2>/dev/null
    sleep 2
}

restore_strata() {
    echo "restoring Strata (voice brain)..." >&2
    stop_27b
    systemctl --user start strata.service
    if wait_for http://127.0.0.1:8081 240; then
        echo "strata :8081 healthy" >&2
    else
        echo "CRITICAL: strata :8081 did NOT come up" >&2
        return 1
    fi
    if [ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://127.0.0.1:8091/)" = "200" ]; then
        echo "voice loop :8091 healthy" >&2
    else
        echo "WARNING: voice loop :8091 not responding 200" >&2
    fi
    return 0
}

cmd="${1:-}"
case "$cmd" in
to-27b)
    if health http://127.0.0.1:8082; then echo "27B already up on :8082"; exit 0; fi
    echo "stopping strata.service..."
    systemctl --user stop strata.service
    mkdir -p "$(dirname "$PIDFILE")" "$(dirname "$LOG")"
    # Validated flags (2026-10-03 eval + grammar test): q8 KV, MTP spec decode, thinking off.
    setsid nohup "$LLAMA_SERVER" \
        -m "$MODEL" --host 127.0.0.1 --port 8082 \
        -c 32768 -ngl 999 \
        --cache-type-k q8_0 --cache-type-v q8_0 \
        -fa on \
        --spec-type draft-mtp --spec-draft-n-max 2 \
        --chat-template-kwargs '{"enable_thinking": false}' \
        --temp 0.7 \
        > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "27B starting (pid $(cat "$PIDFILE")), waiting up to 120s..."
    if wait_for http://127.0.0.1:8082 120; then
        echo "27B healthy on :8082"
        exit 0
    fi
    echo "27B FAILED to become healthy in 120s" >&2
    restore_strata
    exit 1
    ;;
to-strata)
    restore_strata
    ;;
*)
    echo "usage: $0 to-27b | to-strata" >&2
    exit 2
    ;;
esac
