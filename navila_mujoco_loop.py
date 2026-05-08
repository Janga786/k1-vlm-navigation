#!/usr/bin/env python3
"""Closed-loop NaVILA + MuJoCo simulation for the Booster K1.

Pipeline per step:
    head_cam render -> append to 8-frame buffer
    -> NaVILA inference -> regex parse -> (vx, vy, vyaw)
    -> kinematically slide the floating base for action_duration
    -> repeat

Two cv2 windows:
    "K1 head_cam (NaVILA input)"  — what the VLM sees, 384x384
    "K1 third person"             — fixed camera tracking the trunk

The robot does NOT walk — its standing pose is held statically and the
floating base is translated/yawed in body frame. Goal is to validate the
VLM's perception/grounding loop before the real K1 (where B1LocoClient.Move
drives the actual on-board walker).

The MUJOCO_GL=egl backend is set before importing mujoco — required because
torch's CUDA init conflicts with GLFW (mujoco's default windowed backend).
That's why we use cv2 windows instead of the passive viewer.

Usage:
    /home/janga/miniconda3/envs/navila/bin/python navila_mujoco_loop.py \\
      --instruction "navigate to the red box"
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco imports

import argparse
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from navila_k1_bridge import (  # noqa: E402
    NUM_FRAMES, build_prompt, load_navila, parse_action, ACTION_DURATION,
)
from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import (  # noqa: E402
    KeywordsStoppingCriteria, process_images, tokenizer_image_token,
)

K1_XML = Path.home() / "Projects/booster/booster_assets/robots/K1/K1_22dof.xml"
DEFAULT_CKPT = Path.home() / "Projects/booster/NaVILA/checkpoints/navila-llama3-8b-8f"

# Standing pose from booster_deploy K1_CFG.prepare_state.joint_pos.
STANDING_QPOS = np.array([
    0.0, 0.0,                          # head: yaw, pitch
    0.0, -1.3, 0.0, 0.0,               # left arm
    0.0,  1.3, 0.0, 0.0,               # right arm
    0.0, 0.0, 0.0, 0.105, -0.10, 0.0,  # left leg (slight knee bend)
    0.0, 0.0, 0.0, 0.105, -0.10, 0.0,  # right leg
], dtype=np.float64)


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


def build_scene_xml(robot_xml_path: Path,
                    target_pos: tuple[float, float, float]) -> str:
    """Add head_cam to Head_2, scene_cam to worldbody, red target box, and a
    couple of distractor objects. Rewrite meshdir to absolute path so the
    string can be loaded without a working-directory dependency.
    """
    tree = ET.parse(robot_xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((robot_xml_path.parent / "meshes").resolve()))

    # Make the offscreen framebuffer big enough for the largest camera.
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_el = visual.find("global")
    if global_el is None:
        global_el = ET.SubElement(visual, "global")
    global_el.set("offwidth", "1280")
    global_el.set("offheight", "960")

    head2 = next((b for b in root.iter("body") if b.get("name") == "Head_2"), None)
    if head2 is None:
        raise RuntimeError("Head_2 body not found")
    cam = ET.SubElement(head2, "camera")
    cam.set("name", "head_cam")
    cam.set("pos", "0.07 0 0.08")
    cam.set("xyaxes", "0 -1 0 0 0 1")  # face body +X, image up = body +Z
    cam.set("fovy", "70")

    worldbody = root.find("worldbody")

    # Third-person tracking camera (fixed pos, reorients to Trunk CoM).
    scene_cam = ET.SubElement(worldbody, "camera")
    scene_cam.set("name", "scene_cam")
    scene_cam.set("mode", "targetbodycom")
    scene_cam.set("target", "Trunk")
    scene_cam.set("pos", "1.5 4.0 2.0")
    scene_cam.set("fovy", "60")

    # Red target box.
    target = ET.SubElement(worldbody, "body")
    target.set("name", "navigation_target")
    target.set("pos", f"{target_pos[0]} {target_pos[1]} {target_pos[2]}")
    g = ET.SubElement(target, "geom")
    g.set("type", "box")
    g.set("size", "0.20 0.20 0.30")
    g.set("rgba", "0.92 0.10 0.10 1")
    g.set("contype", "0")
    g.set("conaffinity", "0")

    # A few distractors to give NaVILA something to disambiguate.
    for name, pos, rgba, sz in [
        ("distractor_blue",  (2.0, -1.5, 0.25), "0.10 0.30 0.85 1", "0.18 0.18 0.25"),
        ("distractor_green", (1.5,  1.8, 0.25), "0.10 0.70 0.20 1", "0.18 0.18 0.25"),
    ]:
        b = ET.SubElement(worldbody, "body")
        b.set("name", name)
        b.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        gg = ET.SubElement(b, "geom")
        gg.set("type", "box")
        gg.set("size", sz)
        gg.set("rgba", rgba)
        gg.set("contype", "0")
        gg.set("conaffinity", "0")

    return ET.tostring(root, encoding="unicode")


def init_robot_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, 1.0)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:7 + len(STANDING_QPOS)] = STANDING_QPOS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def slide_base(model: mujoco.MjModel, data: mujoco.MjData,
               vx: float, vy: float, vyaw: float, dt: float) -> None:
    yaw = yaw_from_quat(data.qpos[3:7])
    cy, sy = np.cos(yaw), np.sin(yaw)
    data.qpos[0] += (vx * cy - vy * sy) * dt
    data.qpos[1] += (vx * sy + vy * cy) * dt
    data.qpos[3:7] = quat_from_yaw(yaw + vyaw * dt)
    mujoco.mj_forward(model, data)


def render_cam(renderer: mujoco.Renderer, data: mujoco.MjData,
               cam_name: str) -> np.ndarray:
    renderer.update_scene(data, camera=cam_name)
    return renderer.render()


def _spawn_mpv(width: int, height: int, fps: float, title: str
               ) -> subprocess.Popen:
    """Spawn an mpv subprocess that displays raw RGB frames from stdin."""
    return subprocess.Popen(
        [
            "mpv",
            "--demuxer=rawvideo",
            f"--demuxer-rawvideo-w={width}",
            f"--demuxer-rawvideo-h={height}",
            f"--demuxer-rawvideo-fps={fps}",
            "--demuxer-rawvideo-mp-format=rgb24",
            "--no-cache", "--untimed", "--osc=no", "--keep-open=no",
            f"--title={title}",
            "-",
        ],
        stdin=subprocess.PIPE,
    )


def _write_mpv(proc: subprocess.Popen | None, rgb: np.ndarray) -> bool:
    if proc is None or proc.stdin is None or proc.poll() is not None:
        return False
    try:
        proc.stdin.write(rgb.tobytes())
        return True
    except (BrokenPipeError, ValueError):
        return False


def show_windows(scene_rgb: np.ndarray, head_rgb: np.ndarray,
                 hud_text: list[str]) -> bool:
    """Return False if the user pressed q or closed a window."""
    scene_bgr = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)
    head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
    y = 25
    for line in hud_text:
        cv2.putText(scene_bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22
    cv2.imshow("K1 third person", scene_bgr)
    cv2.imshow("K1 head_cam (NaVILA input)", head_bgr)
    return cv2.waitKey(1) & 0xFF != ord("q")


def run_inference(tokenizer, model, image_processor,
                  frames: list[Image.Image], instruction: str,
                  max_new_tokens: int = 64) -> str:
    images_tensor = process_images(frames, image_processor, model.config).to(
        model.device, dtype=torch.float16)
    qs = build_prompt(instruction, len(frames))
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids,
            images=images_tensor.half().cuda(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
            pad_token_id=tokenizer.eos_token_id,
        )
    out = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    if out.endswith(stop_str):
        out = out[: -len(stop_str)].strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--model-path", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--target", type=float, nargs=3, default=(3.0, 0.0, 0.30),
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--action-duration", type=float, default=ACTION_DURATION)
    ap.add_argument("--render-hz", type=float, default=30.0)
    ap.add_argument("--head-w", type=int, default=640)
    ap.add_argument("--head-h", type=int, default=480)
    ap.add_argument("--scene-w", type=int, default=960)
    ap.add_argument("--scene-h", type=int, default=720)
    ap.add_argument("--no-display", action="store_true",
                    help="don't open cv2 windows (headless run)")
    ap.add_argument("--save-video", type=Path, default=None,
                    help="write scene_view.mp4 + head_view.mp4 to this dir")
    ap.add_argument("--mpv", action="store_true",
                    help="live view via mpv subprocesses (avoids cv2+EGL+CUDA "
                         "segfault). Implies --no-display.")
    args = ap.parse_args()
    if args.mpv:
        args.no_display = True

    print(f"Building scene: K1 + red target at {tuple(args.target)} ...")
    scene_xml = build_scene_xml(K1_XML, tuple(args.target))
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)
    init_robot_state(model, data)

    # Single renderer; resize per-camera by passing height/width is not supported,
    # so create at the larger of the two and crop in cv2 if needed.
    fb_h = max(args.head_h, args.scene_h)
    fb_w = max(args.head_w, args.scene_w)
    head_renderer = mujoco.Renderer(model, height=args.head_h, width=args.head_w)
    scene_renderer = mujoco.Renderer(model, height=args.scene_h, width=args.scene_w)
    del fb_h, fb_w  # framebuffer size is set in XML; renderers just request smaller

    scene_writer = head_writer = None
    if args.save_video is not None:
        args.save_video.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        scene_writer = cv2.VideoWriter(
            str(args.save_video / "scene_view.mp4"), fourcc, args.render_hz,
            (args.scene_w, args.scene_h))
        head_writer = cv2.VideoWriter(
            str(args.save_video / "head_view.mp4"), fourcc, args.render_hz,
            (args.head_w, args.head_h))
        print(f"Writing videos to {args.save_video}/")

    # cv2.imshow + EGL + CUDA all in one process segfaults on this machine.
    # --mpv streams raw RGB frames to two mpv subprocesses, which run in
    # isolated process space and have no Qt/GLFW conflict.
    scene_mpv = head_mpv = None
    if args.mpv:
        if shutil.which("mpv") is None:
            sys.exit("--mpv requested but `mpv` is not on PATH. "
                     "`sudo snap install mpv` or `sudo apt install mpv`.")
        scene_mpv = _spawn_mpv(args.scene_w, args.scene_h, args.render_hz,
                                "K1 third person")
        head_mpv = _spawn_mpv(args.head_w, args.head_h, args.render_hz,
                               "K1 head_cam (NaVILA input)")

    tokenizer, vlm, image_processor = load_navila(args.model_path)

    frames: deque[Image.Image] = deque(maxlen=NUM_FRAMES)
    bootstrap_rgb = render_cam(head_renderer, data, "head_cam")
    bootstrap_img = Image.fromarray(bootstrap_rgb)
    for _ in range(NUM_FRAMES):
        frames.append(bootstrap_img)

    period = 1.0 / args.render_hz
    print(f"\nInstruction: {args.instruction!r}")
    print(f"Render: head={args.head_w}x{args.head_h}  "
          f"scene={args.scene_w}x{args.scene_h} @ {args.render_hz} Hz")
    print(f"action_duration={args.action_duration}s\n"
          "Press 'q' in either window to quit.\n")

    last_label = "(starting)"
    last_text = ""
    try:
        for step in range(args.steps):
            head_rgb = render_cam(head_renderer, data, "head_cam")
            frames.append(Image.fromarray(head_rgb))

            t0 = time.perf_counter()
            text = run_inference(
                tokenizer, vlm, image_processor, list(frames), args.instruction)
            inf_ms = (time.perf_counter() - t0) * 1000

            vx, vy, vyaw, label = parse_action(text, args.action_duration)
            last_label, last_text = label, text
            pos = data.qpos[0:3].copy()
            yaw_deg = float(np.degrees(yaw_from_quat(data.qpos[3:7])))
            print(f"step {step:3d}  pos=({pos[0]:+.2f},{pos[1]:+.2f}) "
                  f"yaw={yaw_deg:+.0f}°  inf={inf_ms:5.0f}ms  raw={text!r}")
            print(f"            -> {label}  vx={vx:+.3f} vy={vy:+.3f} vyaw={vyaw:+.3f}")

            if label == "stop":
                print("NaVILA emitted stop. Holding pose.")
                end = time.perf_counter() + 1.5
                while time.perf_counter() < end:
                    scene_rgb = render_cam(scene_renderer, data, "scene_cam")
                    if not args.no_display and not show_windows(
                        scene_rgb, head_rgb,
                        [f"step {step}: STOP", f"raw: {text[:60]}"]):
                        return
                    time.sleep(period)
                break

            t_end = time.perf_counter() + args.action_duration
            next_tick = time.perf_counter()
            while time.perf_counter() < t_end:
                slide_base(model, data, vx, vy, vyaw, period)
                scene_rgb = render_cam(scene_renderer, data, "scene_cam")
                head_rgb_live = render_cam(head_renderer, data, "head_cam")
                hud = [
                    f"step {step}  {last_label}",
                    f"vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}",
                    f"pos=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) "
                    f"yaw={float(np.degrees(yaw_from_quat(data.qpos[3:7]))):+.0f}°",
                ]
                if scene_writer is not None:
                    scene_writer.write(cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR))
                    head_writer.write(cv2.cvtColor(head_rgb_live, cv2.COLOR_RGB2BGR))
                _write_mpv(scene_mpv, scene_rgb)
                _write_mpv(head_mpv, head_rgb_live)
                if not args.no_display and not show_windows(scene_rgb, head_rgb_live, hud):
                    return
                next_tick += period
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
    finally:
        if scene_writer is not None:
            scene_writer.release()
            head_writer.release()
        for proc in (scene_mpv, head_mpv):
            if proc is not None and proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
        head_renderer.close()
        scene_renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
