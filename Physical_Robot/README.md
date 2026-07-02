# Robot-Side Scripts

> ⚠️ **CAMERA PATH SUPERSEDED (2026-06-26).** The StartVisionService /
> `/boostercamera/head/raw/rgb` route below is a DEAD END — throttled at
> startup, decays after ~1 h, and leads to a reboot loop. The supported
> live feed is the **`/booster_video_stream`** topic (auto-starts on boot,
> ~10 fps, no unlock call): see `~/K1_Tutorial/Head_Camera.md` and use
> `bringup_camera.sh` in the repo root. `robot_video_bridge.py` is still
> the bridge — just point it at `/booster_video_stream` (its default).

These scripts live on the K1 robot at /home/booster/ and need to be SCP'd from there when the robot is next powered on:

- vision_service_call.py — ROS2 service caller for StartVisionService (**deprecated** — see banner above)
- x5_camera_rpc.py — X5 camera RPC diagnostic client (deprecated with the raw-MIPI path)
- walk_test.py — Basic SDK walk test

Also grab:
- K1_camera_fix_notes.md — Full camera recovery procedure

To copy from robot:
scp booster@192.168.10.102:~/vision_service_call.py Physical_Robot/
scp booster@192.168.10.102:~/x5_camera_rpc.py Physical_Robot/
scp booster@192.168.10.102:~/walk_test.py Physical_Robot/
scp booster@192.168.10.102:~/K1_camera_fix_notes.md .
