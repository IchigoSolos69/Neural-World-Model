"""Shared position units and in-memory compatibility for historical checkpoints."""

from __future__ import annotations

import math
from pathlib import Path

import torch


POSITION_RMSE_DEFINITION = "euclidean_position_rmse_px_v1"
LEGACY_POSITION_RMSE_DEFINITION = "coordinate_position_rmse_px_v0"


def position_squared_error(prediction: torch.Tensor, target: torch.Tensor,
                           image_size: int = 64) -> torch.Tensor:
    """Return squared radial error in pixels, retaining every axis except xy/state."""
    if prediction.shape != target.shape or prediction.ndim < 1 or prediction.shape[-1] < 2:
        raise ValueError("Positions require matching shapes with at least two final coordinates")
    return ((prediction[..., :2] - target[..., :2]) * image_size).square().sum(-1)


def normalize_checkpoint_metrics(checkpoint: dict, tag: str | Path | None = None) -> dict:
    """Copy metric metadata into radial units, without modifying weights or the source.

    Historical dynamics checkpoints averaged over xy before taking a square root.
    Scaling every validation value by sqrt(2) preserves their selected epoch.
    """
    definition = checkpoint.get("position_rmse_definition")
    if definition not in (None, LEGACY_POSITION_RMSE_DEFINITION, POSITION_RMSE_DEFINITION):
        raise ValueError(f"Unknown position RMSE definition: {definition}")
    stage = checkpoint.get("stage") or (Path(tag).stem.split("_")[0] if tag else None)
    if stage not in ("learned", "hybrid") or definition == POSITION_RMSE_DEFINITION:
        return checkpoint
    result = dict(checkpoint)
    factor = math.sqrt(2)
    if "best_metric" in checkpoint:
        result["best_metric"] = checkpoint["best_metric"] * factor
    if "history" in checkpoint:
        result["history"] = [dict(row, validation=row["validation"] * factor)
                             if "validation" in row else dict(row)
                             for row in checkpoint["history"]]
    result["position_rmse_definition"] = POSITION_RMSE_DEFINITION
    result["position_rmse_compatibility"] = {
        "source_definition": LEGACY_POSITION_RMSE_DEFINITION,
        "target_definition": POSITION_RMSE_DEFINITION,
        "multiplier": factor,
        "converted_fields": ["best_metric", "history[].validation"],
    }
    return result
