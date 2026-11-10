# Robot-Side Scripts

> **CAMERA DOCTRINE — FINAL (2026-07-06, verified live at ~29.5 fps).**
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

## Contents

- `robot_video_bridge.py`: on-robot ROS 2 → HTTP JPEG bridge (`/frame.jpg`, `/stream`) so a
  non-ROS host can pull head-camera frames.
- `Lab Cheat Sheet`: bring-up order and emergency-stop options for the server/relay setup in
  [`../REMOTE_README.md`](../REMOTE_README.md).

Other robot-side helpers (`vision_service_call.py`, `x5_camera_rpc.py`,
`walk_test.py`, `camera_up.sh`, and the camera notes files) live on the robot and are not
included in this repository.
