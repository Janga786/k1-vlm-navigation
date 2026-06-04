#!/usr/bin/env bash
# stop_demo.sh — kill everything started by launch_demo.sh.
#
# Runs FROM THE LAPTOP. Kills:
#   - navila_voice_relay / navila_laptop_relay on this laptop
#   - navila_server on desktop (100.119.64.76)
#   - robot_video_bridge on robot (192.168.10.102)
#
# Vision service on the robot is left running — stopping it requires the
# Booster ROS stack and is not symmetric with how launch_demo started it.

set -u

DESKTOP_HOST="janga@100.119.64.76"
DESKTOP_PASS="${DESKTOP_PASS:?set DESKTOP_PASS to the desktop password}"
ROBOT_HOST="booster@192.168.10.102"
ROBOT_PASS="${ROBOT_PASS:?set ROBOT_PASS to the robot password}"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          -o LogLevel=ERROR -o ConnectTimeout=10"

say() { echo -e "\033[1;36m[stop]\033[0m $*"; }
warn() { echo -e "\033[1;33m[stop]\033[0m $*" >&2; }

command -v sshpass >/dev/null || {
    echo "ERROR: sshpass missing." >&2
    exit 2
}

# 1. Laptop relays.
say "killing voice/laptop relays locally..."
pkill -f navila_voice_relay  2>/dev/null && say "  killed navila_voice_relay"  || say "  no navila_voice_relay running"
pkill -f navila_laptop_relay 2>/dev/null && say "  killed navila_laptop_relay" || say "  no navila_laptop_relay running"

# 2. Desktop NaVILA server.
say "killing navila_server on desktop..."
sshpass -p "$DESKTOP_PASS" ssh $SSH_OPTS "$DESKTOP_HOST" \
    "pkill -f navila_server && echo killed || echo 'no navila_server running'" \
    || warn "could not reach desktop."

# 3. Robot bridge.
say "killing robot_video_bridge on robot..."
sshpass -p "$ROBOT_PASS" ssh $SSH_OPTS "$ROBOT_HOST" \
    "pkill -f robot_video_bridge && echo killed || echo 'no robot_video_bridge running'" \
    || warn "could not reach robot."

echo
echo "All processes stopped."
