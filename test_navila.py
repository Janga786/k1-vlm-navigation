#!/usr/bin/env python3
"""Smoke-test NaVILA-llama3-8b-8f on a single image (replicated x8 frames).

Runs in the `navila` conda env. Loads the model with attn_implementation="sdpa"
to avoid the missing flash-attn Blackwell kernels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria, get_model_name_from_path,
    process_images, tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

DEFAULT_CKPT = Path.home() / "Projects/booster/NaVILA/checkpoints/navila-llama3-8b-8f"
DEFAULT_IMAGE = Path.home() / "Projects/k1_research/experiments/vla/test_image.jpg"
NUM_FRAMES = 8


def build_prompt(instruction: str, num_frames: int) -> str:
    image_token = "<image>\n"
    history_tokens = image_token * (num_frames - 1)
    return (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f"of historical observations {history_tokens}, and current observation <image>\n. "
        f'Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, which could be turning left "
        f"or right by a specific degree, moving forward a certain distance, or stop if the task is completed."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    ap.add_argument("--instruction", default="Navigate to the headset on the desk")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    if not args.model_path.exists():
        raise SystemExit(f"checkpoint not found: {args.model_path}")
    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")

    disable_torch_init()
    print(f"Loading NaVILA from {args.model_path} ...")
    model_name = get_model_name_from_path(str(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.model_path), model_name, model_base=None,
        attn_implementation="sdpa",  # Blackwell: no flash-attn 2.5.8 kernels
    )
    print("Model ready.")

    img = Image.open(args.image).convert("RGB")
    images = [img] * NUM_FRAMES
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16)

    qs = build_prompt(args.instruction, NUM_FRAMES)
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor.half().cuda(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    out = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if out.endswith(stop_str):
        out = out[: -len(stop_str)].strip()
    print("\n=== NaVILA raw output ===")
    print(out)


if __name__ == "__main__":
    main()
