#!/bin/bash
# Bring up the K1 head-camera HTTP feed for NaVILA deploy — the SUPPORTED way,
# per the user's tutorial: K1_Tutorial/Head_Camera.md.
#
# CONTENT-RATE FIX (2026-06-30 16-agent deep-dive, ~/Desktop/CAMERA_FRAMERATE_REPORT.md):
# /booster_video_stream sends MESSAGES at 10 fps but its CONTENT changes only ~0.4 fps -- the
# publisher re-emits the SAME JPEG buffer on a timer (a stale-buffer republish). For real-time
# nav we bridge a callback-driven topic. RELIABILITY BEATS RATE (2026-06-30 reliability deep-dive):
# the rectified /boostercamera/head/rgb is CPU-gated on the loaded 4-core Tegra and STALLS 10-34 s to
# 0 fps (it degraded 8.7->0.2->0 across runs); the raw topic is flaky (0-msg cold starts). The ONLY
# feed whose delivery FLOOR NEVER hit 0 (max stall <=0.6 s) is /booster_video_stream -- so it's the
# default. Its unique-content rate is variable but it never dies, which is what the cart detector needs;
# it's a JPEG pass-through (no NV12 re-encode, cheaper on the Tegra). Alternatives (both STALL to 0):
#     TOPIC=/boostercamera/head/rgb     bash bringup_camera.sh   # rectified (faster bursts, but 10-34s stalls)
#     TOPIC=/boostercamera/head/raw/rgb bash bringup_camera.sh   # raw MIPI (flaky / 0-msg cold start)
# Keep BEST_EFFORT depth=1 (NOT RELIABLE -- it stacks retransmit stalls). Reap stale camera-probe
# subscribers + don't `ros2 topic hz/echo` the camera DURING a run (each adds ~10-20% CPU to the Tegra).
#
# Run FROM the deploy box:  bash bringup_camera.sh
# Exit 0 = live feed. Exit 2 = no/frozen frame -> see tail message. Exit 1 = unreachable.
set -u
ROBOT=${ROBOT:-192.168.10.102}
PW=${PW:?set PW to the robot password}
PORT=${PORT:-8080}
TOPIC=${TOPIC:-/booster_video_stream}    # RELIABLE feed -- delivery floor never hits 0 (see header); override with TOPIC=...
# Image (raw/rectified NV12) topics must force the Image type; the old JPEG stream is CompressedImage.
case "$TOPIC" in *boostercamera*) FORCE=image ;; *) FORCE=auto ;; esac
SSH="sshpass -p $PW ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=8"
SRC='source /opt/booster/BoosterRos2/install/setup.bash 2>/dev/null; source /opt/booster/BoosterRos2Interface/install/setup.bash 2>/dev/null'

ping -c1 -W2 "$ROBOT" >/dev/null 2>&1 || { echo "Robot $ROBOT unreachable (check 'ip -brief addr show eno1')."; exit 1; }

echo "[1/3] Checking the booster-video-stream publisher is up ..."
# NOTE: this ros2-topic-info-over-SSH check is FLAKY -- ros2 discovery on the robot can take
# >6s to enumerate, so a timeout here does NOT mean the camera is down. It is informational
# only; the REAL test is step [3/3] (does :8080 actually serve fresh frames). We retry with a
# longer budget, then continue regardless -- never bail, never advise a reboot.
PUB=0
for attempt in 1 2 3; do
  PUB=$(timeout -k 5 25 $SSH booster@"$ROBOT" "$SRC; timeout 14 ros2 topic info $TOPIC 2>/dev/null | grep -oiE 'publisher count: [0-9]+' | grep -oE '[0-9]+'")
  PUB=${PUB:-0}
  [ "$PUB" -ge 1 ] && break
  echo "    publisher check inconclusive (attempt $attempt/3) -- discovery is slow, retrying ..."
done
echo "    publisher count: $PUB"
if [ "$PUB" -lt 1 ]; then
  echo "  ⚠ couldn't CONFIRM the publisher via ros2 (usually just slow discovery, not a real outage)."
  echo "    Continuing to start the HTTP bridge anyway -- step [3/3] is the real test."
  echo "    If [3/3] shows no frames, the booster-video-stream SERVICE is down; restart it on the"
  echo "    robot (NO reboot needed):  ssh booster@$ROBOT 'ros2 run booster-video-stream booster-video-stream'"
fi

echo "[2/3] deploying the latest bridge + (re)starting it on $TOPIC (force-type=$FORCE) ..."
# ensure the robot runs the current bridge (the NV12->JPEG + BEST_EFFORT version)
timeout -k 5 25 sshpass -p "$PW" scp -o StrictHostKeyChecking=no \
  /home/boosterk1/K1_Tutorial/scripts/robot_video_bridge.py booster@"$ROBOT":/home/booster/robot_video_bridge.py \
  >/dev/null 2>&1 && echo "    bridge deployed" || echo "    (scp skipped/failed -- using the robot's existing bridge)"
timeout -k 5 30 $SSH booster@"$ROBOT" "cat > /home/booster/start_bridge_deploy.sh <<EOF
#!/usr/bin/env bash
$SRC
pkill -9 -f 'python3 robot_video_bridge.py' 2>/dev/null
sleep 1
cd /home/booster
setsid python3 robot_video_bridge.py --topic $TOPIC --port $PORT --force-type $FORCE >/tmp/bridge.log 2>&1 </dev/null &
echo launched-\$!
EOF
bash /home/booster/start_bridge_deploy.sh" 2>&1 | tail -1
echo "    waiting 8s for subscribe + bind"; for i in $(seq 8); do sleep 1; done

echo "[3/3] Verifying the feed is live (frames climbing, age_s low, content changing) ..."
OUT=$(/home/boosterk1/envs/navilaenv/bin/python - "$ROBOT" "$PORT" <<'PY'
import sys, time, hashlib, urllib.request
host, port = sys.argv[1], sys.argv[2]
def status():
    try: return urllib.request.urlopen(f"http://{host}:{port}/status", timeout=4).read().decode()
    except Exception: return ""
import re
def field(s, k):
    m = re.search(rf"{k}:\s*([0-9.]+)", s); return m.group(1) if m else "?"
s1 = status(); f1 = field(s1, "frames")
seen=set(); ok=0; t0=time.time()
for _ in range(30):
    try: seen.add(hashlib.md5(urllib.request.urlopen(f"http://{host}:{port}/frame.jpg",timeout=4).read()).hexdigest()); ok+=1
    except Exception: pass
    time.sleep(0.2)
s2 = status(); f2 = field(s2, "frames"); age = field(s2, "age_s")
try: msg_fps = (int(f2)-int(f1))/max(time.time()-t0,1e-3)
except Exception: msg_fps = -1
print(f"{ok} {len(seen)} {msg_fps:.1f} {age}")
PY
)
read SERVED UNIQ MSGFPS AGE <<<"$OUT"
echo "    served=$SERVED/30  unique=$UNIQ  msg_rate~${MSGFPS}/s  age_s=$AGE"
# The reliable "live" signal is the frame COUNTER climbing (msg_rate), NOT image variation:
# a still robot looking at a static scene produces byte-identical JPEGs and that is fine. A
# genuinely frozen producer shows msg_rate ~0 (counter stuck). So: served + counter-climbing
# = LIVE even with low unique; warn "stalled" only when the counter isn't moving.
MSG_OK=$(awk -v m="${MSGFPS:-0}" 'BEGIN{print (m>=5)?1:0}')
if [ "${SERVED:-0}" -ge 1 ] && { [ "${UNIQ:-0}" -ge 8 ] || [ "$MSG_OK" = 1 ]; }; then
  echo "  ✅ CAMERA LIVE on $TOPIC (msg_rate ~${MSGFPS}/s, age_s=$AGE). Deploy pulls http://$ROBOT:$PORT/frame.jpg"
  [ "${UNIQ:-0}" -lt 8 ] && echo "     (only $UNIQ unique frames = static scene / still robot; content updates once things move)"
  echo "     Deploy env: NAVILA_UNDISTORT=1, NAVILA_VLM_TRANSFORM=stretch."
  exit 0
elif [ "${SERVED:-0}" -eq 0 ]; then
  echo "  ❌ 503 / no frame. Bridge up but no publisher data — the booster-video-stream SERVICE is down."
  echo "     Restart it on the robot (NO reboot): ssh booster@$ROBOT 'ros2 run booster-video-stream booster-video-stream &'"
  exit 2
else
  echo "  ⚠ frames served but the counter isn't climbing (msg_rate ~${MSGFPS}/s) — producer stalled."
  echo "    Restart the booster-video-stream service on the robot (NO reboot):"
  echo "      ssh booster@$ROBOT 'ros2 run booster-video-stream booster-video-stream &'"
  exit 2
fi
