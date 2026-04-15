"""Model loaders for all three pipeline phases."""

from __future__ import annotations

import torch


def load_abliterated_model(model_path: str):
    """Load the abliterated text-only model for Phase 1 (keyword + SD prompt)."""
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    print(f"[Phase 1] Loading abliterated model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch = (cfg.architectures or [None])[0]
    ModelCls = getattr(transformers, arch, None) if arch else None
    ModelCls = ModelCls or transformers.AutoModelForCausalLM
    print(f"  Model class: {ModelCls.__name__}")
    model = ModelCls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def load_sd_model(model_path: str):
    """Load Stable Diffusion 3.5 Medium for Phase 2 (image generation)."""
    from diffusers import StableDiffusion3Pipeline

    print(f"[Phase 2] Loading SD model from {model_path} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def load_target_model(model_path: str):
    """Load Qwen3.5-27B VLM as the white-box target for Phase 3."""
    import transformers
    from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM

    print(f"[Target] Loading processor <- {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left", use_fast=True,
    )

    cfg_obj = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch_name = cfg_obj.architectures[0] if cfg_obj.architectures else None
    ModelCls = (
        getattr(transformers, arch_name, None) if arch_name else None
    ) or AutoModelForCausalLM
    print(f"[Target] Model class: {ModelCls.__name__}", flush=True)

    print(f"[Target] Loading model (bfloat16) <- {model_path}  device_map=auto", flush=True)
    model = ModelCls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    model.gradient_checkpointing_enable({"use_reentrant": False})
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i}: {mem:.1f}GB allocated", flush=True)
    print(f"[Target] Model loaded (bfloat16, gradient_checkpointing=ON)", flush=True)
    return model, processor
