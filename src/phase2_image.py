"""Phase 2 — Image generation (Stable Diffusion) and composite construction."""

from __future__ import annotations

import os

from PIL import Image


def generate_sd_image(pipe, prompt: str, save_path: str) -> Image.Image:
    """Generate a single image with Stable Diffusion and save it."""
    image = pipe(prompt=prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
    image.save(save_path)
    return image
