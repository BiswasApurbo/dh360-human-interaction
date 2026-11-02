#!/usr/bin/env bash
# ==============================================================================
# KBCON Humanoid Robot - Fullscreen Kiosk Launcher for Raspberry Pi 5
# ==============================================================================
# Ensures backend readiness, disables screen blanking / DPMS, and launches
# Chromium in permanent fullscreen kiosk mode with automatic crash recovery.
# ==============================================================================

set -u

KIOSK_URL="http://127.0.0.1:8000/kiosk"
HEALTH_URL="http://127.0.0.1:8000/health"
MAX_WAIT_SECONDS=60
POLL_INTERVAL=1

export DISPLAY="${DISPLAY:-:0}"
if [ -f "${HOME}/.Xauthority" ]; then
    export XAUTHORITY="${HOME}/.Xauthority"
fi

echo "[Kiosk] Waiting for DH360 backend readiness at ${HEALTH_URL}..."
elapsed=0
backend_ready=0

while [ $elapsed -lt $MAX_WAIT_SECONDS ]; do
    if curl -s -f -m 2 "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "[Kiosk] Backend is ready after ${elapsed}s."
        backend_ready=1
        break
    fi
    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [ $backend_ready -eq 0 ]; then
    echo "[Kiosk] WARNING: Backend health check timed out after ${MAX_WAIT_SECONDS}s. Launching browser anyway..."
fi

# Disable screen blanking, screensaver, and DPMS power saving on Xorg
if command -v xset > /dev/null 2>&1; then
    xset -dpms || true
    xset s off || true
    xset s noblank || true
fi

# Hide mouse cursor if unclutter is installed
if command -v unclutter > /dev/null 2>&1; then
    pkill unclutter || true
    unclutter -idle 0.5 -root &
fi

# Resolve Chromium binary
CHROMIUM_BIN="/usr/bin/chromium"
if [ ! -x "$CHROMIUM_BIN" ]; then
    CHROMIUM_BIN="$(command -v chromium || command -v chromium-browser || true)"
fi

if [ -z "$CHROMIUM_BIN" ]; then
    echo "[Kiosk] ERROR: Chromium binary not found!" >&2
    exit 1
fi

apply_portrait_orientation() {
    echo "[Kiosk] Enforcing portrait display rotation (left)..."
    if command -v xrandr > /dev/null 2>&1; then
        for out in $(xrandr 2>/dev/null | grep ' connected' | awk '{print $1}'); do
            xrandr --output "$out" --rotate left 2>/dev/null || true
        done
    fi
    if command -v xinput > /dev/null 2>&1; then
        xinput set-prop 'QDtech MPI7003' 'Coordinate Transformation Matrix' 0 -1 1 1 0 0 0 0 1 2>/dev/null || true
    fi
}

# Apply portrait orientation before launching Chromium
apply_portrait_orientation

echo "[Kiosk] Starting Chromium fullscreen display at ${KIOSK_URL}..."

# Clean up any stale Chromium singleton lock or crashed profile flags
rm -f "${HOME}/.config/chromium/SingletonLock" || true

# Continuous supervisor loop: automatically restart if browser is closed or crashes
while true; do
    # Re-verify portrait orientation prior to each browser invocation
    apply_portrait_orientation

    "$CHROMIUM_BIN" \
        --kiosk \
        --noerrdialogs \
        --disable-infobars \
        --no-first-run \
        --fast \
        --fast-start \
        --disable-pinch \
        --overscroll-history-navigation=0 \
        --disable-features=TranslateUI \
        --disk-cache-dir=/tmp/chromium-kiosk-cache \
        --autoplay-policy=no-user-gesture-required \
        --check-for-update-interval=31536000 \
        --simulate-outdated-no-au='Tue, 31 Dec 2099 23:59:59 GMT' \
        "$KIOSK_URL" || true

    echo "[Kiosk] Browser exited. Restarting in 2 seconds..."
    sleep 2
done
