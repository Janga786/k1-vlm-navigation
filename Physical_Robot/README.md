# Robot-Side Scripts

> ✅ **CAMERA DOCTRINE — FINAL (2026-07-06, verified live at ~29.5 fps).**
> The X5 head camera **boots into a low-rate idle mode (~0.4 fps)** and must
> be unlocked with **`StartVisionService` (face detection OFF — `noface`)**
> after every reboot; then bridge the raw topic
> **`/boostercamera/head/raw/rgb`** (~28–30 fps, 100% unique). Both steps =
> `bash ~/k1_qr_nav_ws/camera_up.sh`. The earlier "avoid StartVisionService,
> use `/booster_video_stream`" guidance (2026-06-26 era) was a misdiagnosis:
> `/booster_video_stream` sits behind the broken StereoNet rectifier and
> republishes stale JPEGs, and the old "decay after ~1 h" came from the
> unlock's face-detection YOLO starving the Jetson (hence `noface`) plus
> battery brownout. Evidence: `~/k1_qr_nav_ws/CAMERA.md` +
> `CAMERA_STEREONET_ISSUE.md` + `/home/booster/K1_camera_fix_notes.md`.
> Never re-unlock an already-live feed (repeated unlocks degrade the X5).

These scripts live on the K1 robot at /home/booster/ and need to be SCP'd from there when the robot is next powered on:

- vision_service_call.py — ROS2 service caller for StartVisionService (**the per-boot unlock** — see banner; call with `start noface`)
- x5_camera_rpc.py — X5 camera RPC diagnostic client (⚠ `ChangeMode(3)` on the X5 RPC breaks the stream)
- walk_test.py — Basic SDK walk test

Also grab:
- K1_camera_fix_notes.md — Full camera recovery procedure

To copy from robot:
scp booster@192.168.10.102:~/vision_service_call.py Physical_Robot/
scp booster@192.168.10.102:~/x5_camera_rpc.py Physical_Robot/
scp booster@192.168.10.102:~/walk_test.py Physical_Robot/
scp booster@192.168.10.102:~/K1_camera_fix_notes.md .
