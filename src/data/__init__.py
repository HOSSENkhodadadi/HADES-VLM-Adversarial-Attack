"""Data loading helpers — behaviours dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_behaviors(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load the behaviours JSON file, optionally truncating to *limit* items."""
    with open(path, "r") as f:
        behaviors = json.load(f)
    if limit is not None:
        behaviors = behaviors[:limit]
    return behaviors
