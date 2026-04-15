#!/usr/bin/env python3
"""
HADES Attack — main entry point.

Runs the three-phase pipeline:
  Phase 1  Abliterated 9B  → keyword extraction & SD prompt generation
  Phase 2  SD 3.5 Medium   → harmful image generation & composite construction
  Phase 3  Target 27B VLM  → white-box gradient attack & inference
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from config import load_config
from src.data import load_behaviors
from src.models import load_abliterated_model, load_sd_model, load_target_model
from src.phase1_text import extract_keyword, generate_sd_prompt
from src.phase2_image import generate_sd_image
from src.phase3_attack import (
    generate_response_adv,
    generate_response_clean,
    white_box_gradient_attack,
)
from src.utils import unload
from src.utils.image import build_composite, modify_prompt
from evaluate import save_results, compute_summary


def run_phase1(cfg, behaviors, output_dir):
    """Phase 1: extract keywords and generate SD prompts."""
    print("=" * 60)
    print("PHASE 1 — keyword extraction & SD prompt generation")
    print("=" * 60)

    tokenizer, abl_model = load_abliterated_model(cfg.models.abliterated)

    prep = []
    for idx, item in enumerate(behaviors):
        instr = item["original_prompt"]
        kw = extract_keyword(tokenizer, abl_model, instr)
        sd_p = generate_sd_prompt(tokenizer, abl_model, instr) if cfg.pipeline.use_sd_images else ""
        mod_p = modify_prompt(instr, kw)
        print(f'  [{idx}] kw="{kw}"  sd_prompt="{sd_p[:70]}..."')
        prep.append({
            "id": item.get("id", idx),
            "original_prompt": instr,
            "keyword": kw,
            "sd_prompt": sd_p,
            "modified_prompt": mod_p,
            "image_path": item["image_path"][0],
            "category": item.get("main_category", ""),
            "subcategory": item.get("subcategory", ""),
        })

    prep_path = os.path.join(output_dir, "phase1_prep.json")
    with open(prep_path, "w") as f:
        json.dump(prep, f, indent=2, ensure_ascii=False)

    unload(abl_model, tokenizer)
    return prep


def run_phase2(cfg, prep, output_dir):
    """Phase 2: generate SD images and build composites."""
    print("\n" + "=" * 60)
    print("PHASE 2 — image generation & composite construction")
    print("=" * 60)

    sd_dir = os.path.join(output_dir, "sd_images")
    comp_dir = os.path.join(output_dir, "composite_images")
    os.makedirs(sd_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    if cfg.pipeline.use_sd_images:
        pipe = load_sd_model(cfg.models.stable_diffusion)
        for idx, pd in enumerate(prep):
            path = os.path.join(sd_dir, f"{pd['id']}.png")
            if not os.path.exists(path):
                print(f"  [{idx}] generating SD image ...")
                generate_sd_image(pipe, pd["sd_prompt"], path)
            else:
                print(f"  [{idx}] SD image exists, skip")
            pd["sd_image_path"] = path
        unload(pipe)
    else:
        print("  SD generation disabled; using dataset images.")

    for idx, pd in enumerate(prep):
        if cfg.pipeline.use_sd_images:
            content_path = os.path.join(sd_dir, f"{pd['id']}.png")
        else:
            content_path = pd["image_path"]

        comp_path = os.path.join(comp_dir, f"{pd['id']}.png")
        build_composite(content_path, pd["keyword"], comp_path, image_base_dir=cfg.data.image_base_dir)
        pd["composite_image_path"] = comp_path
        print(f"  [{idx}] composite saved: {comp_path}")

    images_path = os.path.join(output_dir, "phase2_images.json")
    with open(images_path, "w") as f:
        json.dump(prep, f, indent=2, ensure_ascii=False)

    return prep


def run_phase3(cfg, prep, output_dir):
    """Phase 3: white-box gradient attack + inference."""
    print("\n" + "=" * 60)
    print("PHASE 3 — white-box gradient attack & inference")
    print("=" * 60)

    adv_img_dir = os.path.join(output_dir, "adv_images")
    os.makedirs(adv_img_dir, exist_ok=True)

    model, processor = load_target_model(cfg.models.target)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    print("model loaded, starting attacks ...")

    results = []
    for idx, pd in enumerate(prep):
        cid = pd["id"]
        comp_path = pd.get("composite_image_path", "")
        if not comp_path or not os.path.exists(comp_path):
            print(f"  [{idx}] SKIP — no composite image")
            continue

        mod_prompt = pd["modified_prompt"]
        print(f"\n{'─' * 60}")
        print(f"[{idx + 1}/{len(prep)}] id={cid}")
        print(f"  original : {pd['original_prompt'][:80]}...")
        print(f"  modified : {mod_prompt[:80]}...")
        print(f"  keyword  : {pd['keyword']}")

        # --- gradient attack ---
        print(f"  >> Starting gradient attack ({cfg.attack.gradient_steps} steps) ...")
        delta = None
        try:
            delta = white_box_gradient_attack(
                model, processor, comp_path, mod_prompt,
                affirmative_responses=cfg.affirmative_responses,
                steps=cfg.attack.gradient_steps,
                alpha=cfg.attack.alpha,
                epsilon=cfg.attack.epsilon,
            )
            print("  >> Generating adversarial response ...")
            adv_save_path = os.path.join(adv_img_dir, f"{cid}.png")
            adv_resp = generate_response_adv(
                model, processor, comp_path, mod_prompt, delta,
                max_new_tokens=cfg.attack.max_new_tokens, save_path=adv_save_path,
            )
        except Exception as e:
            print(f"  ERROR in gradient attack: {e}")
            adv_resp = f"[ERROR] {e}"
        finally:
            if delta is not None:
                del delta
            torch.cuda.empty_cache()

        # --- clean baseline ---
        print("  >> Generating clean baseline response ...")
        try:
            clean_resp = generate_response_clean(
                model, processor, comp_path, mod_prompt,
                max_new_tokens=cfg.attack.max_new_tokens,
            )
        except Exception as e:
            clean_resp = f"[ERROR] {e}"
        finally:
            torch.cuda.empty_cache()

        print(f"  clean : {clean_resp[:120]}")
        print(f"  adv   : {adv_resp[:120]}")

        results.append({
            "id": cid,
            "original_prompt": pd["original_prompt"],
            "modified_prompt": mod_prompt,
            "keyword": pd["keyword"],
            "sd_prompt": pd.get("sd_prompt", ""),
            "composite_image_path": comp_path,
            "clean_response": clean_resp,
            "adversarial_response": adv_resp,
            "category": pd.get("category", ""),
            "subcategory": pd.get("subcategory", ""),
            "gradient_steps": cfg.attack.gradient_steps,
        })

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    avg_time = total_time / len(results) if results else 0

    print(f"\n{'=' * 60}")
    print(f"Done — {len(results)}/{len(prep)} processed")
    print(f"Total attack time: {total_time:.1f}s")
    print(f"Average delay per attack: {avg_time:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="HADES Attack Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--phases", type=int, nargs="+", default=None, help="Phases to run (1 2 3)")
    parser.add_argument("--num-behaviors", type=int, default=None, help="Limit number of behaviors")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.phases:
        cfg.pipeline.phases = args.phases
    if args.num_behaviors is not None:
        cfg.pipeline.num_behaviors = args.num_behaviors

    output_dir = cfg.data.output_dir
    os.makedirs(output_dir, exist_ok=True)

    behaviors = load_behaviors(cfg.data.behaviors, limit=cfg.pipeline.num_behaviors)
    print(f"Loaded {len(behaviors)} behaviors\n")

    prep = None

    # Phase 1
    if 1 in cfg.pipeline.phases:
        prep = run_phase1(cfg, behaviors, output_dir)

    # Phase 2
    if 2 in cfg.pipeline.phases:
        if prep is None:
            with open(os.path.join(output_dir, "phase1_prep.json"), "r") as f:
                prep = json.load(f)
        prep = run_phase2(cfg, prep, output_dir)

    # Phase 3
    if 3 in cfg.pipeline.phases:
        if prep is None:
            with open(os.path.join(output_dir, "phase2_images.json"), "r") as f:
                prep = json.load(f)
        results = run_phase3(cfg, prep, output_dir)

        results_path = os.path.join(output_dir, "results.json")
        save_results(results, results_path)
        summary = compute_summary(results)
        print(f"Summary: {summary}")

    print("HADES evaluation completed.")


if __name__ == "__main__":
    main()
