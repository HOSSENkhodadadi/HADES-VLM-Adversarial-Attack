"""Phase 3 — White-box gradient attack and inference on the target VLM."""

from __future__ import annotations

import random
import time
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Internal: build VLM inputs
# ---------------------------------------------------------------------------

def _build_vlm_inputs(processor, image_path: str, prompt: str, target_suffix: str | None = None):
    """Build processor inputs for the VLM, optionally appending a target suffix."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt},
    ]}]
    text_with_gen = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(messages)
    except ImportError:
        image_inputs = [Image.open(image_path).convert("RGB")]

    full_text = text_with_gen + target_suffix if target_suffix else text_with_gen
    inputs = processor(
        text=[full_text], images=image_inputs, padding=True, return_tensors="pt",
    )
    return inputs, text_with_gen


# ---------------------------------------------------------------------------
# Gradient attack
# ---------------------------------------------------------------------------

def white_box_gradient_attack(
    model,
    processor,
    image_path: str,
    prompt: str,
    affirmative_responses: List[str],
    steps: int = 200,
    alpha: float = 1.0 / 255,
    epsilon: float = 8.0 / 255,
) -> torch.Tensor:
    """Run PGD-style gradient attack on pixel_values. Returns optimised delta."""
    target_response = random.choice(affirmative_responses)

    inputs_full, _ = _build_vlm_inputs(processor, image_path, prompt, target_suffix=target_response)

    target_token_len = len(processor.tokenizer.encode(target_response, add_special_tokens=False))
    prompt_len = inputs_full["input_ids"].shape[1] - target_token_len

    labels = inputs_full["input_ids"].clone()
    labels[:, :prompt_len] = -100

    device = next(model.parameters()).device
    inputs_full = {k: v.to(device) for k, v in inputs_full.items()}
    labels = labels.to(device)

    pv_clean = inputs_full["pixel_values"].detach().clone().to(model.dtype)
    delta = torch.zeros_like(pv_clean, requires_grad=True)

    print(f'    target: "{target_response[:60]}..."')
    print(f"    pixel_values shape: {pv_clean.shape}")
    print(f"    running {steps} gradient steps ...")

    for p in model.parameters():
        p.requires_grad_(False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps):
        t_step = time.perf_counter()
        pv_adv = pv_clean + delta

        fwd = {k: v for k, v in inputs_full.items()}
        fwd["pixel_values"] = pv_adv
        fwd["labels"] = labels

        outputs = model(**fwd)
        loss = outputs.loss
        loss_val = loss.item()
        loss.backward()

        with torch.no_grad():
            delta.data = (delta.data - alpha * delta.grad.sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
        del outputs, loss, pv_adv

        if step % 10 == 0 or step == steps - 1:
            elapsed = time.perf_counter() - t0
            step_time = time.perf_counter() - t_step
            eta = step_time * (steps - step - 1)
            mem_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            mem_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            print(
                f"      step {step:>4d}/{steps}  loss={loss_val:.4f}  "
                f"step_time={step_time:.2f}s  elapsed={elapsed:.1f}s  ETA={eta:.1f}s  "
                f"GPU_alloc={mem_alloc:.2f}GB  GPU_reserved={mem_reserved:.2f}GB"
            )

    for p in model.parameters():
        p.requires_grad_(True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    print(f"    gradient attack done in {total:.1f}s ({total / steps:.2f}s/step)")
    return delta.detach()


# ---------------------------------------------------------------------------
# Adversarial image saving
# ---------------------------------------------------------------------------

def save_adv_image(processor, original_image_path: str, delta: torch.Tensor, image_grid_thw, save_path: str):
    """Reconstruct and save the adversarial image from the delta tensor."""
    import torchvision.transforms.functional as TF

    img_proc = processor.image_processor
    mean = torch.tensor(img_proc.image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(img_proc.image_std, dtype=torch.float32).view(3, 1, 1)
    patch_size = getattr(img_proc, "patch_size", 14)
    temporal_patch_size = getattr(img_proc, "temporal_patch_size", 2)

    grid_t = int(image_grid_thw[0, 0].item())
    grid_h = int(image_grid_thw[0, 1].item())
    grid_w = int(image_grid_thw[0, 2].item())
    H, W = grid_h * patch_size, grid_w * patch_size

    d = delta.detach().float().cpu()
    d = d.reshape(grid_t, grid_h, grid_w, temporal_patch_size, 3, patch_size, patch_size)
    d = d[0, :, :, 0, :, :, :]
    d = d.permute(2, 0, 3, 1, 4).contiguous()
    d = d.reshape(3, H, W)

    orig = Image.open(original_image_path).convert("RGB").resize((W, H), Image.BICUBIC)
    orig_tensor = TF.to_tensor(orig)
    orig_norm = (orig_tensor - mean) / std
    adv_norm = orig_norm + d
    adv = (adv_norm * std + mean).clamp(0, 1)

    img_arr = (adv.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img_arr).save(save_path)
    print(f"    Saved adversarial image: {save_path}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def generate_response_adv(
    model, processor, image_path: str, prompt: str,
    delta: torch.Tensor, max_new_tokens: int = 512, save_path: str | None = None,
) -> str:
    """Generate a response using adversarial pixel_values."""
    inputs, _ = _build_vlm_inputs(processor, image_path, prompt)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = (inputs["pixel_values"].float() + delta.to(device)).to(model.dtype)

    if save_path:
        try:
            save_adv_image(processor, image_path, delta, inputs["image_grid_thw"], save_path)
        except Exception as e:
            print(f"    WARNING: could not save adversarial image: {e}")

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0]


def generate_response_clean(
    model, processor, image_path: str, prompt: str, max_new_tokens: int = 512,
) -> str:
    """Generate a response without perturbation (baseline)."""
    inputs, _ = _build_vlm_inputs(processor, image_path, prompt)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0]
