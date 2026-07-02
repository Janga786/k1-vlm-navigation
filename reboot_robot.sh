#!/bin/bash
# Reboot the K1 and wait until it is back + services have started.
# NOTE: with the supported /booster_video_stream feed (Head_Camera.md) a reboot
# is NOT part of normal camera bringup — the old "camera decays after ~1h" claim
# only applied to the deprecated raw-MIPI/StartVisionService path. Reboot only
# if the booster-video-stream service itself is dead (publisher count 0).
# Run FROM the lab box:  bash reboot_robot.sh
set -u
ROBOT=${ROBOT:-192.168.10.102}
PW=${PW:?set PW to the robot password}
SSH="sshpass -p $PW ssh -o StrictHostKeyChecking=no -o ConnectTimeout=6 booster@$ROBOT"

if ! ping -c1 -W2 "$ROBOT" >/dev/null 2>&1; then
  echo "Robot $ROBOT not reachable. Power/connect it first."; exit 1
fi

echo "Rebooting robot $ROBOT ..."
$SSH "echo $PW | sudo -S reboot" 2>/dev/null || true

echo "Waiting for it to go down, then come back (~90s)..."
sleep 30
tries=0
until ping -c1 -W2 "$ROBOT" >/dev/null 2>&1; do
  printf '.'; sleep 5; tries=$((tries+1))
  [ "$tries" -gt 60 ] && { echo " timed out waiting for robot"; exit 2; }
done
echo " robot is back online."
echo "Giving Booster services ~45s to start..."
sleep 45
echo "Done. Next:  bash bringup_camera.sh"
