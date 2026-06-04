#!/usr/bin/env bash
# launch_demo.sh — one-command NaVILA voice-demo launcher.
#
# Runs FROM THE LAPTOP. Orchestrates all three machines:
#   - laptop  (this box): voice relay + venv
#   - desktop (100.119.64.76, RTX 5090): NaVILA server  [ALWAYS restarted]
#   - robot   (192.168.10.102, Booster K1): camera + MJPEG bridge
#
# Modes
#   ./launch_demo.sh                                    # voice mode (default)
#   ./launch_demo.sh --text "walk to the volleyball"    # one-shot text mode
#   ./launch_demo.sh --text-mode                        # interactive text loop
#
# Other flags are passed straight through to navila_voice_relay.py.
#
# Ctrl-C  -> clean shutdown: zero robot velocity, print "Demo stopped."
# To kill everything across all machines, use ./stop_demo.sh.

set -u
# NOTE: no `set -e` here. We want explicit error handling and a clean
# trap-driven shutdown; aborting on the first failed ssh would skip the
# `actuator zero` step.

# ────────────────────────────────────────────────────────────────────────────
# Connection details (hardcoded — single-tenant demo box)
# ────────────────────────────────────────────────────────────────────────────
DESKTOP_HOST="janga@100.119.64.76"
DESKTOP_PASS="${DESKTOP_PASS:?set DESKTOP_PASS to the desktop password}"
ROBOT_HOST="booster@192.168.10.102"
ROBOT_PASS="${ROBOT_PASS:?set ROBOT_PASS to the robot password}"
ROBOT_IP="192.168.10.102"
DESKTOP_IP="100.119.64.76"

# Paths on the LAPTOP (this box). $HOME resolves to /home/<laptop-user>.
NAVILA_DIR="$HOME/Projects/k1_research/experiments/navila"
VENV_DIR="$HOME/k1_relay_env"

# Paths on the DESKTOP — hardcoded because the desktop user is fixed
# (janga). The previous bug was using $NAVILA_DIR in the remote ssh
# command, which expanded to the LAPTOP's home dir on the laptop side
# and then failed `cd` on the desktop with "No such file or directory".
DESKTOP_NAVILA_DIR="/home/janga/Projects/k1_research/experiments/navila"

SERVER_LOG="/tmp/navila_server.log"
BRIDGE_LOG="/tmp/bridge.log"
MJPEG_URL="http://${ROBOT_IP}:8080/frame.jpg"

# SSH options as an array so quoting is unambiguous.
#   -n   : don't read from stdin — critical because some of the remote
#          commands background a long-running process; if ssh kept stdin
#          attached, sshd would hold the channel open and ssh wouldn't
#          return (and eventually fail with a non-zero exit).
#   -o BatchMode=no : sshpass needs the password prompt path to work; we
#          set this explicitly to override any ~/.ssh/config default.
#   We deliberately omit `LogLevel=ERROR` here so real ssh errors are
#   visible. Re-add it later if it gets too noisy.
SSH_OPTS=(
    -n
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o BatchMode=no
    -o ConnectTimeout=10
    -o ServerAliveInterval=15
)

LOAD_TIMEOUT_S=120

# ────────────────────────────────────────────────────────────────────────────
# Mode parsing
# ────────────────────────────────────────────────────────────────────────────
MODE="voice"           # voice | text-oneshot | text-loop
TEXT_INSTR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --text)
            MODE="text-oneshot"
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --text requires an instruction." >&2; exit 2; }
            TEXT_INSTR="$1"
            shift
            ;;
        --text-mode)
            MODE="text-loop"
            shift
            ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ────────────────────────────────────────────────────────────────────────────
# Pre-flight: sshpass + venv + dir
# ────────────────────────────────────────────────────────────────────────────
command -v sshpass >/dev/null || {
    echo "ERROR: sshpass missing. Install with: sudo apt install sshpass" >&2
    exit 2
}

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR" >&2
    echo "       (this script is meant to run on the laptop.)" >&2
    exit 2
fi

if [[ ! -f "$NAVILA_DIR/navila_voice_relay.py" ]]; then
    echo "ERROR: $NAVILA_DIR/navila_voice_relay.py missing." >&2
    echo "       scp the laptop files over from the desktop first." >&2
    exit 2
fi

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
say() { echo -e "\033[1;36m[demo]\033[0m $*"; }
warn() { echo -e "\033[1;33m[demo]\033[0m $*" >&2; }
die() { echo -e "\033[1;31m[demo]\033[0m $*" >&2; exit 1; }

# Every ssh helper closes stdin EXPLICITLY (< /dev/null).
#
# We already pass `-n` in SSH_OPTS, which is supposed to redirect ssh's
# own stdin from /dev/null. In practice, sshpass + an interactive parent
# shell can still leak the terminal stdin to the ssh child, and any
# bytes the user typed (including ENTERs) get consumed by ssh / sshd
# before the script reaches the voice relay. Result: the relay sees
# EOF on stdin and exits with "stopping (Ctrl-C / EOF)". The redirect
# here is what actually fixes that — it guarantees no ssh invocation
# can touch the controlling terminal's input queue.
dssh() { sshpass -p "$DESKTOP_PASS" ssh "${SSH_OPTS[@]}" "$DESKTOP_HOST" "$@" < /dev/null; }
rssh() { sshpass -p "$ROBOT_PASS"   ssh "${SSH_OPTS[@]}" "$ROBOT_HOST"   "$@" < /dev/null; }

# Quiet variants used inside the polling loop where ssh is called
# repeatedly and verbose noise would clutter the user's terminal.
dssh_quiet() { sshpass -p "$DESKTOP_PASS" ssh "${SSH_OPTS[@]}" \
                -o LogLevel=ERROR "$DESKTOP_HOST" "$@" < /dev/null; }

# ────────────────────────────────────────────────────────────────────────────
# Clean shutdown trap — printed after the relay exits (clean or via Ctrl-C).
# ────────────────────────────────────────────────────────────────────────────
# IMPORTANT: the voice relay MUST run in the foreground (no `&`, no
# `wait`). Non-interactive bash automatically redirects backgrounded
# processes' stdin to /dev/null — which the relay sees as instant EOF on
# the push-to-talk readline. Running in the foreground lets the relay
# inherit the controlling terminal directly. SIGINT is delivered to the
# whole foreground process group, so Ctrl-C reaches the relay's own
# signal handler (which zeros the actuator).
cleanup() {
    local rc=$?
    echo
    echo "Demo stopped."
    exit $rc
}
trap cleanup INT TERM EXIT

# ────────────────────────────────────────────────────────────────────────────
# 1. ALWAYS restart NaVILA server on desktop, then wait for "listening"
# ────────────────────────────────────────────────────────────────────────────

# 1a. Connectivity probe. Surfaces ssh / sshpass / network / auth errors
#     with their actual stderr instead of the generic "ssh failed" message.
say "probing ssh to desktop ($DESKTOP_HOST)..."
probe_out=$(dssh 'echo connected; uname -n' 2>&1)
probe_rc=$?
if (( probe_rc != 0 )); then
    warn "ssh probe failed (rc=$probe_rc). raw output:"
    printf '%s\n' "$probe_out" >&2
    die "cannot reach desktop. Check: tailscale ping 100.119.64.76, password, sshpass."
fi
say "  desktop reachable: $(printf '%s' "$probe_out" | tr '\n' ' ')"

# 1b. Restart the server. CRITICAL: the remote command backgrounds a
#     long-running process, so we must redirect ALL three fds away from
#     the ssh channel AND `disown` the job AND `exit 0` explicitly.
#     Without this, ssh waits for the channel's stdout/stderr to close
#     (which never happens because the server inherits them) and
#     eventually returns non-zero — that was the original bug.
say "restarting NaVILA server on desktop..."
# IMPORTANT details about the remote command below:
#
# 1. pkill pattern is anchored on 'navila_server\.py' (NOT just
#    'navila_server'). The bash invoked by ssh has the literal
#    substring 'navila_server' in its own argv because that substring
#    appears in the command we are sending; pkill -f matches argv, so
#    a plain 'navila_server' pattern would kill our own shell and ssh
#    would exit with rc=255.
#
# 2. The server is launched via `setsid ... &`, which puts the child in
#    its own session/process group. Plain `nohup ... &` is NOT enough
#    here: sshd keeps the channel open as long as any descendant of
#    the login shell has the channel's stdout/stderr — `setsid` fully
#    detaches the new process so ssh returns immediately.
restart_cmd="pkill -f 'navila_server\.py' >/dev/null 2>&1; sleep 2; \
rm -f $SERVER_LOG; \
cd $DESKTOP_NAVILA_DIR || exit 9; \
setsid bash -c 'nohup bash /home/janga/start_navila_server.sh \
    > $SERVER_LOG 2>&1 < /dev/null &' < /dev/null > /dev/null 2>&1; \
sleep 1; \
echo started; \
exit 0"

restart_out=$(dssh "$restart_cmd" 2>&1)
restart_rc=$?
if (( restart_rc != 0 )); then
    warn "server restart command failed (rc=$restart_rc). output:"
    printf '%s\n' "$restart_out" >&2
    die "could not restart NaVILA server."
fi
say "  $restart_out"

say "waiting for NaVILA model to load (timeout ${LOAD_TIMEOUT_S}s)..."
t_start=$SECONDS
while true; do
    tail_line=$(dssh_quiet "tail -1 $SERVER_LOG 2>/dev/null || true")
    if [[ "$tail_line" == *"listening"* ]]; then
        say "NaVILA server ready: ${tail_line}"
        break
    fi
    elapsed=$((SECONDS - t_start))
    if (( elapsed >= LOAD_TIMEOUT_S )); then
        warn "tail of server log:"
        dssh_quiet "tail -20 $SERVER_LOG" >&2 || true
        die "NaVILA server did not reach 'listening' within ${LOAD_TIMEOUT_S}s."
    fi
    # Print a heartbeat every 10s so the user knows we're alive.
    if (( elapsed % 10 == 0 )) && (( elapsed > 0 )); then
        say "  ...still loading (${elapsed}s elapsed, last line: ${tail_line:-<empty>})"
    fi
    sleep 2
done

# ────────────────────────────────────────────────────────────────────────────
# 2. Robot camera bring-up — single atomic sequence, RIGHT ORDER.
# ────────────────────────────────────────────────────────────────────────────
# The order MATTERS:
#   (a) kill any stale robot_video_bridge so it can't be holding the old
#       low-rate subscription.
#   (b) call StartVisionService — this is what flips the camera from
#       its idle ~0.67 fps state into full ~10 fps capture.
#   (c) sleep 2s so the vision pipeline is actually publishing at full
#       rate before the bridge subscribes.
#   (d) start the bridge fresh; it picks up the high-rate stream.
#
# All four steps run in ONE rssh call so an ssh hiccup between (b) and
# (d) can't leave the robot in a half-started state. We do NOT use a
# "is it already running?" short-circuit — running launch_demo.sh is
# explicitly a request to restart this pipeline.
#
# Two non-obvious things still apply:
#
# 1. pkill pattern is anchored on `^python` (not just
#    'robot_video_bridge\.py'). The wrapper bash that ssh runs has the
#    ENTIRE remote command as one argv string — including the literal
#    substring `python3 $HOME/robot_video_bridge.py`. A plain
#    'robot_video_bridge\.py' pattern matches that and pkill kills its
#    own wrapper shell → ssh returns rc=255 with no useful error.
#    Anchoring on `^python` excludes the wrapper bash (cmdline starts
#    with "bash") and matches ONLY python processes.
#
# 2. The bridge is launched inside a `source <setup.bash> &&` shell so
#    `import rclpy` / message-type imports inside robot_video_bridge.py
#    can resolve. Without that, the bridge ImportErrors immediately
#    and nothing binds to port 8080 — which produced the
#    "Connection refused" we saw earlier.
ROS_SETUP="/opt/booster/BoosterRos2Interface/install/setup.bash"
say "robot camera bring-up: kill bridge → vision-service → bridge..."
robot_cmd="\
echo '[a] killing stale bridge...'; \
pkill -f '^python.*robot_video_bridge\.py' >/dev/null 2>&1; \
sleep 1; \
echo '[b] calling StartVisionService...'; \
source $ROS_SETUP >/dev/null 2>&1 && \
    python3 \$HOME/vision_service_call.py start \
        || { echo '  vision-service start exited non-zero'; }; \
echo '[c] waiting 2s for full-rate publish...'; \
sleep 2; \
echo '[d] starting bridge fresh...'; \
rm -f $BRIDGE_LOG; \
setsid bash -c 'source $ROS_SETUP && \
    nohup python3 \$HOME/robot_video_bridge.py \
    --topic /boostercamera/head/raw/rgb --port 8080 \
    > $BRIDGE_LOG 2>&1 < /dev/null &' < /dev/null > /dev/null 2>&1; \
sleep 1; \
echo '    bridge launched.'; \
exit 0"
robot_out=$(rssh "$robot_cmd" 2>&1)
robot_rc=$?
# Print whatever the robot reported (the [a]/[b]/[c]/[d] steps make it
# clear which substep failed if any).
printf '%s\n' "$robot_out" | sed 's/^/[robot] /'
if (( robot_rc != 0 )); then
    die "robot camera bring-up failed (rc=$robot_rc)"
fi

# Poll the camera stream — up to 15s for the bridge to start serving.
# Don't just continue on failure: NaVILA needs frames, and without
# them the voice relay's first instruction will die immediately.
say "waiting for camera stream at $MJPEG_URL ..."
camera_ok=0
for i in $(seq 1 15); do
    if curl -s --max-time 2 "$MJPEG_URL" -o /dev/null < /dev/null; then
        say "camera stream verified at $MJPEG_URL (took ${i}s)"
        camera_ok=1
        break
    fi
    sleep 1
done
if (( camera_ok == 0 )); then
    warn "camera stream at $MJPEG_URL never responded. Bridge log tail:"
    rssh "tail -20 $BRIDGE_LOG 2>/dev/null || true" >&2 || true
    warn "continuing anyway — voice relay will fail on first frame."
fi

# ────────────────────────────────────────────────────────────────────────────
# 3. Activate venv, ensure GROQ_API_KEY, launch the relay
# ────────────────────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate" || die "could not activate $VENV_DIR"
say "venv active: $(which python3)"

# GROQ_API_KEY resolution order:
#   1. existing env var (highest priority — easy to override per-run)
#   2. ~/.bashrc (non-interactive shells don't source it, so grep + eval)
#   3. hardcoded fallback (so the demo runs out of the box on any laptop
#      where the venv exists but ~/.bashrc isn't set up).
# The hardcoded key is the project's shared demo key — rotate via the
# Groq console if it leaks.
GROQ_API_KEY_FALLBACK=""

if [[ -z "${GROQ_API_KEY:-}" ]]; then
    if grep -qE '^[[:space:]]*export[[:space:]]+GROQ_API_KEY=' "$HOME/.bashrc" 2>/dev/null; then
        # shellcheck disable=SC1091
        eval "$(grep -E '^[[:space:]]*export[[:space:]]+GROQ_API_KEY=' "$HOME/.bashrc")"
        say "GROQ_API_KEY loaded from ~/.bashrc"
    fi
fi
if [[ -z "${GROQ_API_KEY:-}" ]]; then
    GROQ_API_KEY="$GROQ_API_KEY_FALLBACK"
    say "GROQ_API_KEY using hardcoded fallback (set env var or ~/.bashrc to override)"
fi
export GROQ_API_KEY

COMMON_FLAGS=(
    --server "$DESKTOP_IP"
    --mode live
    --net 192.168.10.100
    --image-source mjpeg
    --mjpeg-url "$MJPEG_URL"
)

cd "$NAVILA_DIR" || die "cannot cd to $NAVILA_DIR"

# NOTE: all three modes invoke python3 in the FOREGROUND. Do not
# background with `&` — non-interactive bash redirects backgrounded
# stdin to /dev/null, which the relay sees as instant EOF on its
# push-to-talk readline (this was the original "[voice] stopping
# (Ctrl-C / EOF)" bug).
case "$MODE" in
    voice)
        say "launching voice relay (Ctrl-C to stop)..."
        python3 navila_voice_relay.py \
            "${COMMON_FLAGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
    text-oneshot)
        say "launching one-shot text relay: ${TEXT_INSTR}"
        python3 navila_voice_relay.py \
            --input-mode text --instruction "$TEXT_INSTR" \
            "${COMMON_FLAGS[@]}" \
            "${EXTRA_ARGS[@]}"
        ;;
    text-loop)
        say "interactive text mode — enter blank line / Ctrl-C / Ctrl-D to exit."
        # Loop: read a line, execute it as a one-shot. Each iteration opens
        # a fresh connection — desktop server resets the frame buffer per
        # set_instruction, so no stale state between commands.
        while true; do
            printf '\n\033[1;32minstruction>\033[0m '
            if ! IFS= read -r line; then
                echo
                break
            fi
            [[ -z "${line// }" ]] && break
            python3 navila_voice_relay.py \
                --input-mode text --instruction "$line" \
                "${COMMON_FLAGS[@]}" \
                "${EXTRA_ARGS[@]}"
        done
        ;;
esac
