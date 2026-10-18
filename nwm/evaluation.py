"""Fair autoregressive comparisons and separately held-out calibration."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import EpisodeDataset, OOD_CONDITIONS, validate_dataset
from .metrics import (RolloutMetrics, collision_counts, conformal_multiplier,
                      infer_contacts, uncertainty_metrics)
from .physics import PhysicsConfig
from .predict import constant_velocity_rollout, physics_rollout


def rollout_predictions(artifacts: dict, initial: torch.Tensor, actions: torch.Tensor,
                        cfg: PhysicsConfig, baselines: bool = True) -> list[dict]:
    """Use identical initial states/actions, never future states or contacts."""
    predictions = []
    if baselines:
        for name, function, parameters in (
            ("constant_velocity", constant_velocity_rollout, cfg),
            ("imperfect_physics", physics_rollout, replace(cfg, drag=0.0, restitution=1.0)),
            ("fitted_physics", physics_rollout, artifacts["fitted_physics"]),
        ):
            predictions.append({"model": name, "seed": None, "states": function(initial, actions, parameters)})
    hybrid_members = []
    for kind in ("learned", "hybrid"):
        for seed, model in artifacts[kind]:
            states = model(initial, actions)
            predictions.append({"model": kind, "seed": seed, "states": states})
            if kind == "hybrid":
                hybrid_members.append(states)
    members = torch.stack(hybrid_members)
    mean = members.mean(0)
    spread = (members[..., :2] - mean[None, ..., :2]).square().sum(-1).mean(0).sqrt()
    predictions.append({"model": "hybrid_ensemble", "seed": None, "states": mean, "spread": spread})
    return predictions


def decoded_pixel_mse(decoder, states: torch.Tensor, target_frames: torch.Tensor,
                      chunk_size: int = 64) -> torch.Tensor:
    """Keep decoder activations bounded even for 50-frame evaluation batches."""
    flat = states.reshape(-1, 3, 4)
    target = target_frames.reshape(-1, 3, 64, 64)
    errors = [(decoder(flat[start:start + chunk_size]) - target[start:start + chunk_size])
              .square().flatten(1).mean(-1)
              for start in range(0, len(flat), chunk_size)]
    return torch.cat(errors).reshape(states.shape[:2])


def _loader(config: dict, split: str) -> DataLoader:
    dataset = EpisodeDataset(config["data"]["root"], split, context=4,
                             horizon=max(config["evaluation"]["horizons"]),
                             windows_per_episode=1, render_frames=True)
    maximum = config["evaluation"].get("max_episodes")
    if maximum is not None and split != "calibration":
        dataset = Subset(dataset, range(min(int(maximum), len(dataset))))
    return DataLoader(dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False,
                      num_workers=0)


def _on_device(batch: dict, device: torch.device) -> dict:
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


def _uncertainty_samples(prediction: dict, target: torch.Tensor, horizons: list[int]) -> dict:
    errors = (prediction["states"][..., :2] - target[..., :2]).norm(dim=-1) * 64
    return {str(horizon): (errors[:, horizon - 1].detach().cpu(),
                           prediction["spread"][:, horizon - 1].detach().cpu() * 64)
            for horizon in horizons}


@torch.inference_mode()
def calibrate(config: dict, artifacts: dict) -> dict:
    """Only the calibration episodes determine the horizon-specific radii."""
    horizons = config["evaluation"]["horizons"]
    gathered = {str(horizon): ([], []) for horizon in horizons}
    for raw_batch in _loader(config, "calibration"):
        batch = _on_device(raw_batch, artifacts["device"])
        initial = artifacts["estimator"](batch["context_frames"], batch["context_actions"])
        members = torch.stack([model(initial, batch["future_actions"]) for _, model in artifacts["hybrid"]])
        mean = members.mean(0)
        spread = (members[..., :2] - mean[None, ..., :2]).square().sum(-1).mean(0).sqrt()
        for horizon, values in _uncertainty_samples({"states": mean, "spread": spread}, batch["target_states"], horizons).items():
            for destination, value in zip(gathered[horizon], values):
                destination.append(value)
    calibration = {}
    for horizon, (error_batches, spread_batches) in gathered.items():
        errors, spread = torch.cat(error_batches), torch.cat(spread_batches)
        calibration[horizon] = {"objects": [conformal_multiplier(errors[:, index], spread[:, index])
                                             for index in range(3)],
                                "object_order": ["red", "green", "blue"]}
    return {"split": "calibration", "initialization": "estimated", "horizons": calibration,
            "interpretation": "Separately calibrated marginal per-object/per-horizon radial intervals across "
                              "independent episodes, conditional on shared perception. ID split-conformal coverage "
                              "requires exchangeable episodes and sufficient calibration sample count, flagged per object. "
                              "No joint-object, joint-trajectory, or OOD coverage guarantee."}


def _seed_summary(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        if record["seed"] is not None:
            key = (record["split"], record["initialization"], record["actions"], record["model"])
            groups.setdefault(key, []).append(record)
    summary = []
    for (split, initialization, actions, model), members in groups.items():
        horizons = {}
        for horizon in members[0]["horizons"]:
            horizons[horizon] = {
                metric: {"mean": float(np.mean([item["horizons"][horizon][metric] for item in members])),
                         "std": float(np.std([item["horizons"][horizon][metric] for item in members]))}
                for metric in members[0]["horizons"][horizon]
            }
        summary.append({"split": split, "initialization": initialization, "actions": actions, "model": model,
                        "seeds": [item["seed"] for item in members], "horizons": horizons})
    return summary


def _plot(records: list[dict], destination: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(destination.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, initialization in zip(axes, ("estimated", "true")):
        groups = {}
        for row in records:
            if row["split"] == "test" and row["initialization"] == initialization and row["actions"] == "recorded":
                groups.setdefault(row["model"], []).append(row["position_ade_curve_px"])
        for model, curves in groups.items():
            values = np.asarray(curves)
            time = np.arange(1, values.shape[1] + 1)
            mean, std = values.mean(0), values.std(0)
            axis.plot(time, mean, label=model.replace("_", " "))
            if len(values) > 1:
                axis.fill_between(time, mean - std, mean + std, alpha=0.12)
        axis.set(title=f"{initialization.capitalize()} initial state", xlabel="Autoregressive future frame")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean position error (pixels)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


@torch.inference_mode()
def evaluate(config: dict, device: str = "auto") -> Path:
    from .training import load_artifacts

    validate_dataset(config)
    artifacts = load_artifacts(config, device)
    cfg = PhysicsConfig(**config["physics"])
    horizons = config["evaluation"]["horizons"]
    output = Path(config["outputs"]["root"]) / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    calibration = calibrate(config, artifacts)
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    records, uncertainty_rows, decoder_rows, contact_reference_rows = [], [], [], []
    for split in ("test", *OOD_CONDITIONS):
        accumulators = {}
        uncertainty = {str(horizon): ([], []) for horizon in horizons}
        reference_contacts = {horizon: np.zeros(3, dtype=np.int64) for horizon in horizons}
        decoder_error, decoder_count = None, 0
        for raw_batch in _loader(config, split):
            batch = _on_device(raw_batch, artifacts["device"])
            estimated = artifacts["estimator"](batch["context_frames"], batch["context_actions"])
            target = batch["target_states"]
            reference = infer_contacts(batch["initial_state"], target, cfg.radius, cfg.dt, batch["future_actions"])
            for horizon in horizons:
                reference_contacts[horizon] += collision_counts(reference[:, :horizon], batch["contacts"][:, :horizon])
            decoder_batch = decoded_pixel_mse(artifacts["decoder"], target, batch["target_frames"]).sum(0).cpu()
            decoder_error = decoder_batch if decoder_error is None else decoder_error + decoder_batch
            decoder_count += target.shape[0]
            for initialization, initial in (("estimated", estimated), ("true", batch["initial_state"])):
                for action_mode in ("recorded", "removed"):
                    # The ablation removes future interventions only; context remains observed.
                    actions = batch["future_actions"] if action_mode == "recorded" else torch.zeros_like(batch["future_actions"])
                    for prediction in rollout_predictions(artifacts, initial, actions, cfg, baselines=action_mode == "recorded"):
                        key = (initialization, action_mode, prediction["model"], prediction["seed"])
                        if key not in accumulators:
                            accumulators[key] = RolloutMetrics(horizons, cfg.radius, cfg.dt)
                        pixel_error = decoded_pixel_mse(artifacts["decoder"], prediction["states"], batch["target_frames"])
                        accumulators[key].update(prediction["states"], target, initial, batch["contacts"],
                                                 actions=actions, pixel_mse=pixel_error)
                        if initialization == "estimated" and action_mode == "recorded" and "spread" in prediction:
                            for horizon, values in _uncertainty_samples(prediction, target, horizons).items():
                                for destination, value in zip(uncertainty[horizon], values):
                                    destination.append(value)
        for (initialization, action_mode, model, seed), accumulator in accumulators.items():
            records.append({"split": split, "initialization": initialization, "actions": action_mode,
                            "model": model, "seed": seed, **accumulator.compute()})
        decoder_rows.append({"split": split, "pixel_mse_by_horizon": {
            str(horizon): float((decoder_error / decoder_count)[:horizon].mean()) for horizon in horizons}})
        reference_metrics = {}
        for horizon, counts in reference_contacts.items():
            tp, fp, fn = (int(value) for value in counts)
            reference_metrics[str(horizon)] = {
                "collision_tp": tp, "collision_fp": fp, "collision_fn": fn,
                "collision_precision": tp / (tp + fp) if tp + fp else 0.0,
                "collision_recall": tp / (tp + fn) if tp + fn else 0.0,
                "collision_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
        contact_reference_rows.append({"split": split, "horizons": reference_metrics})
        uncertainty_rows.append({"split": split, "initialization": "estimated", "model": "hybrid_ensemble", "horizons": {
            horizon: uncertainty_metrics(torch.cat(errors), torch.cat(spread), calibration["horizons"][horizon])
            for horizon, (errors, spread) in uncertainty.items()}})
        print(f"Evaluated {split}: {decoder_count} episodes", flush=True)
    report = {"format_version": 1, "records": records, "seed_summary": _seed_summary(records),
              "uncertainty": uncertainty_rows, "true_state_decoder_error": decoder_rows,
              "true_trajectory_contact_reference": contact_reference_rows,
              "calibration": calibration, "config": config,
              "definitions": {"position_rmse": "Root mean squared Euclidean distance, pooled over objects and prefix frames, in pixels",
                              "position_fde": "Mean Euclidean distance at the named horizon, in pixels",
                              "velocity_rmse": "Root mean squared Euclidean velocity distance, in world units/second",
                              "pixel_mse": "Common learned decoder output versus true raster frames, RGB scaled to [0,1]",
                              "collision": "Universally inferred swept trajectory contacts; runs deduplicated; one-to-one matching within +/-1 frame",
                              "boundary_violation_rate": "Fraction of object-frames with any coordinate outside legal center bounds",
                              "overlap_rate": "Fraction of pair-frames with separation below diameter minus 1e-4",
                              "actions_removed": "Only future actions zeroed; targets and past actions unchanged"}}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    flattened = [{**{key: row[key] for key in ("split", "initialization", "actions", "model", "seed", "episodes")},
                  "horizon": int(horizon), **metrics}
                 for row in records for horizon, metrics in row["horizons"].items()]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    _plot(records, output / "error_vs_horizon.png")
    return output
