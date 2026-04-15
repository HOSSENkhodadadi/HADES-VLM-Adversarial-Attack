"""Configuration loader — reads config/config.yaml and exposes a flat namespace."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class ModelPaths:
    target: str = ""
    abliterated: str = ""
    stable_diffusion: str = ""


@dataclass
class DataPaths:
    behaviors: str = ""
    image_base_dir: str = ""
    output_dir: str = "output"


@dataclass
class AttackParams:
    gradient_steps: int = 200
    alpha: float = 1.0 / 255
    epsilon: float = 8.0 / 255
    max_new_tokens: int = 512


@dataclass
class PipelineSettings:
    use_sd_images: bool = True
    num_behaviors: Optional[int] = None
    phases: List[int] = field(default_factory=lambda: [1, 2, 3])


@dataclass
class Config:
    models: ModelPaths = field(default_factory=ModelPaths)
    data: DataPaths = field(default_factory=DataPaths)
    attack: AttackParams = field(default_factory=AttackParams)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    affirmative_responses: List[str] = field(default_factory=list)


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file and return a Config dataclass."""
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        models=ModelPaths(**raw.get("models", {})),
        data=DataPaths(**raw.get("data", {})),
        attack=AttackParams(**raw.get("attack", {})),
        pipeline=PipelineSettings(**raw.get("pipeline", {})),
        affirmative_responses=raw.get("affirmative_responses", []),
    )

    # Ensure CUDA allocator is configured early
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    return cfg
