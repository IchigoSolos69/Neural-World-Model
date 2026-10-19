"""State, contact, and marginal ensemble-uncertainty metrics."""

from __future__ import annotations

import math

import numpy as np
import torch

from .metric_definitions import position_squared_error


def _segment_distance(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    delta = end - start
    fraction = (-(start * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (start + fraction[..., None] * delta).norm(dim=-1)


def infer_contacts(initial: torch.Tensor, states: torch.Tensor, radius: float, dt: float,
                   actions: torch.Tensor | None = None) -> torch.Tensor:
    """Infer the same 15 contact channels for every predictor, without true events.

    Swept position segments detect crossings. A pre-step velocity projection plus
    velocity reversal also detects bounces that fall between rendered frames.
    This is a discretized trajectory metric, not access to a solver's contacts.
    """
    previous = torch.cat((initial[:, None], states[:, :-1]), dim=1)
    position, next_position = previous[..., :2], states[..., :2]
    velocity, next_velocity = previous[..., 2:].clone(), states[..., 2:]
    if actions is not None:
        velocity[:, :, 0] += actions
    projected = position + velocity * dt
    tolerance = 1e-5
    events = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        before = position[:, :, first] - position[:, :, second]
        after = next_position[:, :, first] - next_position[:, :, second]
        forecast = projected[:, :, first] - projected[:, :, second]
        before_speed = ((velocity[:, :, first] - velocity[:, :, second]) * before).sum(-1)
        after_speed = ((next_velocity[:, :, first] - next_velocity[:, :, second]) * after).sum(-1)
        approaching = (before_speed < 0) | (((after - before) * before).sum(-1) < 0)
        swept = (_segment_distance(before, after) <= 2 * radius + tolerance) & approaching
        bounce = (_segment_distance(before, forecast) <= 2 * radius + tolerance) & (before_speed < 0) & (after_speed >= 0)
        events.append(swept | bounce)
    for ball in range(3):
        for axis in range(2):
            before, after = position[:, :, ball, axis], next_position[:, :, ball, axis]
            forecast = projected[:, :, ball, axis]
            old_speed, new_speed = velocity[:, :, ball, axis], next_velocity[:, :, ball, axis]
            events.append(((torch.minimum(before, after) <= radius + tolerance) &
                           ((before > radius + tolerance) | (old_speed < 0))) |
                          ((forecast <= radius + tolerance) & (old_speed < 0) & (new_speed >= 0)))
            events.append(((torch.maximum(before, after) >= 1 - radius - tolerance) &
                           ((before < 1 - radius - tolerance) | (old_speed > 0))) |
                          ((forecast >= 1 - radius - tolerance) & (old_speed > 0) & (new_speed <= 0)))
    return torch.stack(events, dim=-1)


def collision_counts(predicted: torch.Tensor, target: torch.Tensor, tolerance: int = 1) -> tuple[int, int, int]:
    """Return TP/FP/FN after collapsing contact runs and matching each event once."""
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("Contact arrays must have matching [batch, time, channel] shapes")
    if tolerance < 0:
        raise ValueError("Contact tolerance must be nonnegative")
    arrays = [value.detach().bool().cpu().numpy() for value in (predicted, target)]
    onsets = [value & ~np.concatenate((np.zeros_like(value[:, :1]), value[:, :-1]), axis=1) for value in arrays]
    true_positive = false_positive = false_negative = 0
    for batch in range(predicted.shape[0]):
        for channel in range(predicted.shape[2]):
            pred = np.flatnonzero(onsets[0][batch, :, channel])
            truth = np.flatnonzero(onsets[1][batch, :, channel])
            p = t = 0
            while p < len(pred) and t < len(truth):
                if pred[p] < truth[t] - tolerance:
                    false_positive += 1
                    p += 1
                elif truth[t] < pred[p] - tolerance:
                    false_negative += 1
                    t += 1
                else:
                    true_positive += 1
                    p += 1
                    t += 1
            false_positive += len(pred) - p
            false_negative += len(truth) - t
    return true_positive, false_positive, false_negative


def conformal_multiplier(errors: torch.Tensor, spread: torch.Tensor, coverage: float = 0.9,
                         floor: float = 0.5) -> dict:
    """Calibrate pointwise radial intervals using the held-out split only."""
    if not 0 < coverage < 1 or floor <= 0:
        raise ValueError("Coverage must be in (0, 1), and the spread floor must be positive")
    if errors.shape != spread.shape or errors.numel() == 0:
        raise ValueError("Calibration errors and spread need matching nonempty shapes")
    scores = (errors / spread.clamp_min(floor)).detach().flatten().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("Calibration scores must be finite")
    requested_rank = math.ceil((scores.numel() + 1) * coverage)
    rank = min(scores.numel(), requested_rank)
    return {"multiplier": float(scores.sort().values[rank - 1]), "sample_count": scores.numel(),
            "coverage_target": coverage, "spread_floor_px": floor,
            "finite_sample_rank_supported": requested_rank <= scores.numel(),
            "maximum_supported_coverage": scores.numel() / (scores.numel() + 1)}


def uncertainty_metrics(errors: torch.Tensor, spread: torch.Tensor, calibration: dict) -> dict:
    if "objects" in calibration:
        per_object = [uncertainty_metrics(errors[..., index], spread[..., index], parameters)
                      for index, parameters in enumerate(calibration["objects"])]
        return {key: sum(item[key] for item in per_object) / len(per_object)
                for key in ("coverage", "mean_radius_px", "mean_diameter_px")} | {"per_object": per_object}
    radius = spread.clamp_min(calibration["spread_floor_px"]) * calibration["multiplier"]
    return {"coverage": float((errors <= radius).float().mean()),
            "mean_radius_px": float(radius.mean()), "mean_diameter_px": float((2 * radius).mean())}


class RolloutMetrics:
    """Accumulate batch statistics without retaining videos or rollout graphs."""

    def __init__(self, horizons: list[int], radius: float, dt: float, image_size: int = 64):
        self.horizons, self.radius, self.dt, self.image_size = horizons, radius, dt, image_size
        self.sums: dict[str, torch.Tensor] = {}
        self.count = 0
        self.contacts = {horizon: np.zeros(3, dtype=np.int64) for horizon in horizons}

    def update(self, states: torch.Tensor, target: torch.Tensor, initial: torch.Tensor,
               contacts: torch.Tensor, frames: torch.Tensor | None = None,
               target_frames: torch.Tensor | None = None, actions: torch.Tensor | None = None,
               pixel_mse: torch.Tensor | None = None) -> None:
        if states.shape != target.shape or states.ndim != 4 or states.shape[-2:] != (3, 4):
            raise ValueError("Rollouts must have matching [batch, time, 3, 4] shapes")
        if not torch.isfinite(states).all():
            raise ValueError("Prediction contains nonfinite states")
        position_sq = position_squared_error(states, target, image_size=1)
        velocity_sq = (states[..., 2:] - target[..., 2:]).square().sum(-1)
        outside = ((states[..., :2] < self.radius - 1e-5) |
                   (states[..., :2] > 1 - self.radius + 1e-5)).any(-1)
        overlaps = torch.stack([(states[:, :, i, :2] - states[:, :, j, :2]).norm(dim=-1) < 2 * self.radius - 1e-4
                                for i, j in ((0, 1), (0, 2), (1, 2))], dim=-1)
        values = {"position_sq": position_sq.mean(-1), "position_distance": position_sq.sqrt().mean(-1),
                  "velocity_sq": velocity_sq.mean(-1), "boundary": outside.float().mean(-1),
                  "overlap": overlaps.float().mean(-1)}
        if frames is not None and target_frames is not None:
            values["pixel_mse"] = (frames - target_frames).square().flatten(2).mean(-1)
        elif pixel_mse is not None:
            values["pixel_mse"] = pixel_mse
        for key, value in values.items():
            total = value.detach().double().sum(0).cpu()
            self.sums[key] = self.sums.get(key, torch.zeros_like(total)) + total
        inferred = infer_contacts(initial, states, self.radius, self.dt, actions)
        for horizon in self.horizons:
            self.contacts[horizon] += collision_counts(inferred[:, :horizon], contacts[:, :horizon])
        self.count += states.shape[0]

    def compute(self) -> dict:
        if not self.count:
            raise ValueError("No rollouts were evaluated")
        means = {key: value / self.count for key, value in self.sums.items()}
        result = {"episodes": self.count, "position_ade_curve_px": (means["position_distance"] * self.image_size).tolist(),
                  "horizons": {}}
        for horizon in self.horizons:
            tp, fp, fn = (int(value) for value in self.contacts[horizon])
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            row = {"position_rmse_px": float(means["position_sq"][:horizon].mean().sqrt() * self.image_size),
                   "position_ade_px": float(means["position_distance"][:horizon].mean() * self.image_size),
                   "position_fde_px": float(means["position_distance"][horizon - 1] * self.image_size),
                   "velocity_rmse": float(means["velocity_sq"][:horizon].mean().sqrt()),
                   "collision_precision": precision, "collision_recall": recall,
                   "collision_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                   "collision_tp": tp, "collision_fp": fp, "collision_fn": fn,
                   "boundary_violation_rate": float(means["boundary"][:horizon].mean()),
                   "overlap_rate": float(means["overlap"][:horizon].mean())}
            if "pixel_mse" in means:
                row["pixel_mse"] = float(means["pixel_mse"][:horizon].mean())
            result["horizons"][str(horizon)] = row
        return result
