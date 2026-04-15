"""GPU / memory utilities."""

from __future__ import annotations

import gc

import torch


def unload(*objects) -> None:
    """Delete objects and aggressively free GPU memory."""
    for o in objects:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
