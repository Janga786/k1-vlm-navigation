# Robot-Side Scripts

These scripts live on the K1 robot at /home/booster/ and need to be SCP'd from there when the robot is next powered on:

- vision_service_call.py — ROS2 service caller for StartVisionService (THE camera fix)
- x5_camera_rpc.py — X5 camera RPC diagnostic client
- walk_test.py — Basic SDK walk test

Also grab:
- K1_camera_fix_notes.md — Full camera recovery procedure

To copy from robot:
scp booster@192.168.10.102:~/vision_service_call.py Physical_Robot/
scp booster@192.168.10.102:~/x5_camera_rpc.py Physical_Robot/
scp booster@192.168.10.102:~/walk_test.py Physical_Robot/
scp booster@192.168.10.102:~/K1_camera_fix_notes.md .
