"""Evaluation helpers — result aggregation and (placeholder) HarmBench judge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def save_results(results: List[Dict[str, Any]], path: str) -> None:
    """Persist results list to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {path}")


def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return basic aggregate statistics from a results list."""
    total = len(results)
    errors = sum(1 for r in results if r.get("adversarial_response", "").startswith("[ERROR]"))
    return {
        "total_samples": total,
        "errors": errors,
        "successful_attacks": total - errors,
    }


# ------------------------------------------------------------------
# Placeholder for HarmBench judge integration (to be added later)
# ------------------------------------------------------------------

def harmbench_judge(results: List[Dict[str, Any]], classifier_path: str | None = None) -> List[Dict[str, Any]]:
    """Score adversarial responses with the HarmBench classifier.

    This is a placeholder. Implement when the HarmBench-Llama-2-13b-cls
    classifier is integrated.
    """
    raise NotImplementedError(
        "HarmBench judge not yet integrated. "
        "Provide the classifier path and implement scoring logic."
    )
