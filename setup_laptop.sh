#!/usr/bin/env bash
#
# setup_laptop.sh — bootstrap the lab laptop for the NaVILA relay.
#
# WHAT THIS INSTALLS
#   - Booster Robotics SDK (C++ libs + Python bindings) — for B1LocoClient.
#   - Minimal Python deps: numpy, pillow, opencv-python — for camera +
#     display.  socket / struct / json are stdlib, no install needed.
#
# WHAT THIS DOES *NOT* INSTALL
#   - torch / CUDA — stays on the desktop.
#   - llava / NaVILA model weights — stays on the desktop.
#   - mujoco — sim is desktop-only.
#   - Isaac Lab / Isaac Sim — never needed on the relay.
#
# Run it from the laptop, once::
#
#     cd ~/Projects/k1_research/experiments/navila
#     ./setup_laptop.sh
#
# Optionally point it at a remote desktop to pull the latest source::
#
#     ./setup_laptop.sh --rsync-from janga@desktop.tail-abc123.ts.net
#
# The script tries to be idempotent — re-running is fine.

set -euo pipefail

# ---------------------------------------------------------------- defaults

K1RES_DIR="${K1RES_DIR:-$HOME/Projects/k1_research}"
RELAY_DIR="$K1RES_DIR/experiments/navila"
SDK_DIR="$K1RES_DIR/booster/booster_robotics_sdk"

RSYNC_FROM=""
SKIP_SDK="${SKIP_SDK:-0}"
SKIP_PIP="${SKIP_PIP:-0}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-$K1RES_DIR/.venv-relay}"

usage() {
    sed -n 's/^# \{0,1\}//; 1,/^$/p' "$0" | head -40
    cat <<'EOF'

Flags:
  --rsync-from HOST       rsync the experiments/navila/ tree from a desktop
                          (HOST is "user@host" reachable over Tailscale).
  --skip-sdk              don't try to install the Booster SDK.
  --skip-pip              don't pip install python deps.
  --no-venv               install python deps system-wide / --user.
  --venv-dir PATH         use a specific venv path (default: $K1RES_DIR/.venv-relay).
  -h, --help              this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rsync-from) RSYNC_FROM="$2"; shift 2;;
        --skip-sdk)   SKIP_SDK=1; shift;;
        --skip-pip)   SKIP_PIP=1; shift;;
        --no-venv)    USE_VENV=0; shift;;
        --venv-dir)   VENV_DIR="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "unknown arg: $1"; usage; exit 2;;
    esac
done

log()  { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FAIL ]\033[0m %s\n' "$*"; exit 1; }

# ---------------------------------------------------------------- sanity

[[ "$(uname -s)" == "Linux" ]] || die "this script is Linux-only (got $(uname -s))."

ARCH="$(uname -m)"
log "arch=$ARCH  user=$USER  home=$HOME"

if [[ ! -d "$K1RES_DIR" ]]; then
    log "creating $K1RES_DIR"
    mkdir -p "$K1RES_DIR"
fi

mkdir -p "$RELAY_DIR"

# ---------------------------------------------------------------- rsync src

if [[ -n "$RSYNC_FROM" ]]; then
    log "rsyncing experiments/navila/ from $RSYNC_FROM ..."
    # Pull only the relay-relevant Python files + this script. We
    # intentionally skip checkpoints/, NaVILA/, out*/, results/.
    rsync -av --delete \
        --include='navila_protocol.py' \
        --include='navila_server.py' \
        --include='navila_laptop_relay.py' \
        --include='navila_k1_realrobot.py' \
        --include='navila_k1_core.py' \
        --include='navila_k1_bridge.py' \
        --include='setup_laptop.sh' \
        --include='REMOTE_README.md' \
        --include='README.md' \
        --include='*.py' \
        --exclude='__pycache__' \
        --exclude='out*' \
        --exclude='results' \
        --exclude='checkpoints' \
        --exclude='*.mp4' \
        --exclude='*.npz' \
        "${RSYNC_FROM}:${RELAY_DIR}/" "$RELAY_DIR/"

    log "rsyncing booster_robotics_sdk/ from $RSYNC_FROM ..."
    mkdir -p "$SDK_DIR"
    rsync -av \
        --exclude='build' \
        --exclude='__pycache__' \
        "${RSYNC_FROM}:${SDK_DIR}/" "$SDK_DIR/"
else
    log "no --rsync-from; assuming files are already in $RELAY_DIR"
fi

# Required relay files — bail early if anything's missing.
for f in navila_protocol.py navila_laptop_relay.py \
         navila_k1_realrobot.py navila_k1_core.py navila_k1_bridge.py; do
    if [[ ! -f "$RELAY_DIR/$f" ]]; then
        die "missing $RELAY_DIR/$f — copy it from the desktop first \
(or pass --rsync-from)."
    fi
done
log "relay python files present."

# ---------------------------------------------------------------- apt deps

if [[ $SKIP_SDK -eq 0 ]]; then
    log "installing apt build deps for the Booster SDK ..."
    APT_PKGS=(
        git build-essential cmake
        libssl-dev libasio-dev libtinyxml2-dev
        python3 python3-pip python3-venv
    )
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y "${APT_PKGS[@]}" || \
            warn "apt-get install failed; continuing — you may need to do it manually."
    else
        warn "sudo not found; skipping apt step. Install manually if missing: ${APT_PKGS[*]}"
    fi
fi

# ---------------------------------------------------------------- venv

PY="python3"
PIP_INSTALL=(pip install --upgrade)

if [[ $USE_VENV -eq 1 ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        log "creating venv at $VENV_DIR"
        $PY -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PY="$VENV_DIR/bin/python"
    log "venv active: $(which python)  ($(python --version 2>&1))"
else
    PIP_INSTALL+=(--user)
fi

# ---------------------------------------------------------------- python deps

if [[ $SKIP_PIP -eq 0 ]]; then
    log "installing minimal Python deps ..."
    # Pinned to widely-available majors. opencv-python is the headless-safe
    # build; we use cv2 only for imshow + putText so the full build is fine.
    "$PY" -m "${PIP_INSTALL[@]}" \
        numpy \
        pillow \
        "opencv-python>=4.6"
fi

# ---------------------------------------------------------------- booster SDK

if [[ $SKIP_SDK -eq 0 ]]; then
    log "installing booster_robotics_sdk_python ..."
    # Try the pip path first — it's the path of least resistance and is
    # the recommended one in the Booster README.
    if "$PY" -m pip install --upgrade booster_robotics_sdk_python; then
        log "booster_robotics_sdk_python installed via pip."
    else
        warn "pip install failed; falling back to source build ..."
        if [[ ! -d "$SDK_DIR" ]]; then
            die "$SDK_DIR not found; can't build SDK from source.  \
Either pass --rsync-from or clone the SDK there yourself."
        fi
        if command -v sudo >/dev/null 2>&1; then
            (cd "$SDK_DIR" && sudo ./install.sh)
        else
            (cd "$SDK_DIR" && ./install.sh) || \
                die "SDK install.sh failed and no sudo to elevate."
        fi
        "$PY" -m pip install --upgrade pybind11 pybind11-stubgen
        mkdir -p "$SDK_DIR/build"
        (
            cd "$SDK_DIR/build"
            cmake .. -DBUILD_PYTHON_BINDING=on
            make -j"$(nproc)"
            if command -v sudo >/dev/null 2>&1; then
                sudo make install
            else
                make install || die "make install failed (no sudo)."
            fi
        )
        log "SDK built from source."
    fi
fi

# ---------------------------------------------------------------- verify

log "verifying imports ..."
"$PY" - <<'PY'
import importlib
import sys

ok = True
for mod in ["numpy", "PIL", "cv2"]:
    try:
        importlib.import_module(mod)
        print(f"  [ok]  {mod}")
    except Exception as e:
        print(f"  [FAIL] {mod}: {e!r}")
        ok = False

# Booster SDK is the only one that may be missing on bring-up — soft-warn.
try:
    importlib.import_module("booster_robotics_sdk_python")
    print("  [ok]  booster_robotics_sdk_python")
except Exception as e:
    print(f"  [warn] booster_robotics_sdk_python not importable: {e!r}")
    print("         You can still run --mode print until you fix it.")

# Make sure the relay file at least parses.
import os
relay = os.path.expanduser("~/Projects/k1_research/experiments/navila/navila_laptop_relay.py")
import py_compile
try:
    py_compile.compile(relay, doraise=True)
    print(f"  [ok]  {relay} compiles")
except Exception as e:
    print(f"  [FAIL] {relay}: {e!r}")
    ok = False

sys.exit(0 if ok else 1)
PY

# ---------------------------------------------------------------- summary

log "all done."
cat <<EOF

==============================================================================
Next steps:

1. On the desktop, start the server:
     python ~/Projects/k1_research/experiments/navila/navila_server.py \\
         --bind 0.0.0.0 --port 5555

2. On this laptop, activate the venv (if used) and start the relay:
     ${USE_VENV:+source $VENV_DIR/bin/activate && }\\
     python ~/Projects/k1_research/experiments/navila/navila_laptop_relay.py \\
         --mode print \\
         --image-source static \\
         --server <DESKTOP_TAILSCALE_IP> \\
         --instruction "walk to the red box"

   When the print smoke test works, escalate to --mode dry, then live.
   See REMOTE_README.md for the full pre-flight pyramid.
==============================================================================
EOF
