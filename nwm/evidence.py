"""Frozen experiment provenance and compact episode-level research measurements."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import experiment_signature
from .metric_definitions import POSITION_RMSE_DEFINITION, position_squared_error
from .metrics import collision_counts, infer_contacts
from .provenance import capture_provenance, verify_provenance


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_snapshot(config):
    data = Path(config["data"]["root"])
    run = Path(config["outputs"]["root"])
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    checkpoints = sorted({*run.glob("checkpoints/*_best.pt"), *run.glob("checkpoints/*_latest.pt")})
    return {"experiment_signature": experiment_signature(config),
            "dataset_fingerprint": metadata["fingerprint"],
            "dataset_files": {path.name: sha256_file(path) for path in [data / "metadata.json", *sorted(data.glob("*.npz"))]},
            "checkpoints": {path.name: sha256_file(path) for path in checkpoints},
            "fitted_physics_sha256": sha256_file(run / "fitted_physics.json")}


def source_snapshot(config):
    return {**_input_snapshot(config), "execution_provenance": capture_provenance()}


def verify_source_snapshot(config, expected):
    actual = _input_snapshot(config)
    if any(actual[key] != value for key, value in expected.items() if key in actual):
        raise ValueError("Experiment inputs changed during evidence evaluation; preserve this incomplete output and use a fresh directory")
    verify_provenance(expected.get("execution_provenance"))


def model_hashes(artifacts):
    from .training import _model_hash
    models = {"estimator": artifacts["estimator"], "decoder": artifacts["decoder"]}
    models.update({f"{kind}_{seed}": model for kind in ("learned", "hybrid") for seed, model in artifacts[kind]})
    return {name: _model_hash(model) for name, model in models.items()}


def checkpoint_history(config):
    from .training import _load_checkpoint
    histories = {}
    for path in sorted(Path(config["outputs"]["root"]).glob("checkpoints/*_latest.pt")):
        value = _load_checkpoint(path, config)
        histories[path.stem.removesuffix("_latest")] = {
            key: value[key] for key in ("stage", "epoch", "best_epoch", "best_metric", "selection_metric",
                                        "position_rmse_definition", "position_rmse_compatibility", "training_provenance", "history") if key in value}
    return histories


def episode_measurements(prediction, target, initial, contacts, actions, pixel_mse, batch,
                         estimated_initial, cfg, horizons, calibration):
    states = prediction["states"]
    position_sq = position_squared_error(states, target)
    velocity_sq = (states[..., 2:] - target[..., 2:]).square().sum(-1)
    outside = ((states[..., :2] < cfg.radius - 1e-5) | (states[..., :2] > 1 - cfg.radius + 1e-5)).any(-1)
    overlap = torch.stack([(states[:, :, i, :2] - states[:, :, j, :2]).norm(dim=-1) < 2 * cfg.radius - 1e-4
                           for i, j in ((0, 1), (0, 2), (1, 2))], -1)
    inferred = infer_contacts(initial, states, cfg.radius, cfg.dt, actions)
    counts = np.asarray([[collision_counts(inferred[i:i + 1, :h], contacts[i:i + 1, :h])
                          for h in horizons] for i in range(len(states))], dtype=np.int32)
    ball_contact = contacts[..., :3].any(dim=(1, 2))
    wall_contact = contacts[..., 3:].any(dim=(1, 2))
    values = {"episode_seed": batch["episode_seed"],
              "position_sq_px": position_sq.mean(-1),
              "position_distance_px": position_sq.sqrt().mean(-1),
              "velocity_sq": velocity_sq.mean(-1), "pixel_mse": pixel_mse,
              "boundary_rate": outside.float().mean(-1), "overlap_rate": overlap.float().mean(-1),
              "contact_group": torch.where(ball_contact, 2, wall_contact.to(torch.uint8)).to(torch.uint8),
              "impulse_present": batch["future_actions"].ne(0).any(dim=(1, 2)),
              "context_position_error_px": position_squared_error(estimated_initial, batch["initial_state"]).mean(-1).sqrt()}
    if "spread" in prediction:
        errors, radii = [], []
        for horizon in horizons:
            parameters = calibration["horizons"][str(horizon)]["objects"]
            floor = states.new_tensor([value["spread_floor_px"] for value in parameters])
            multiplier = states.new_tensor([value["multiplier"] for value in parameters])
            errors.append(position_sq[:, horizon - 1].sqrt())
            radii.append((prediction["spread"][:, horizon - 1] * 64).clamp_min(floor) * multiplier)
        values.update(uncertainty_errors_px=torch.stack(errors, 1), uncertainty_radii_px=torch.stack(radii, 1))
    arrays = {name: value.detach().cpu().numpy() for name, value in values.items()}
    arrays["contact_counts"] = counts
    return arrays


class EvidenceWriter:
    """Keep one condition's compact arrays in memory, then compress each group."""

    def __init__(self, output, config, provenance, calibration):
        self.output = Path(output)
        self.config, self.provenance, self.calibration = config, provenance, calibration
        self.groups, self.gallery, self.chunks = [], [], {}
        self.candidates = []
        (self.output / "episodes").mkdir()
        (self.output / "gallery").mkdir()

    def add(self, split, initialization, action_mode, prediction, target, initial, contacts,
            actions, pixel_mse, batch, estimated_initial, cfg):
        key = (split, initialization, action_mode, prediction["model"], prediction["seed"])
        values = episode_measurements(prediction, target, initial, contacts, actions, pixel_mse,
                                      batch, estimated_initial, cfg, self.config["evaluation"]["horizons"], self.calibration)
        if initialization != "estimated" or action_mode != "recorded":
            values.pop("uncertainty_errors_px", None)
            values.pop("uncertainty_radii_px", None)
        self.chunks.setdefault(key, []).append(values)

    def add_gallery_batch(self, batch, predictions):
        learned = self.config["training"]["seeds"][0]
        chosen = [("constant_velocity", None), ("imperfect_physics", None), ("fitted_physics", None),
                  ("learned", learned), ("hybrid_ensemble", None)]
        keyed = {(row["model"], row["seed"]): row["states"].detach().cpu().numpy() for row in predictions}
        self.candidates.append({"episode_seed": batch["episode_seed"].detach().cpu().numpy(),
                                "context_states": batch["context_states"].detach().cpu().numpy(),
                                "target_states": batch["target_states"].detach().cpu().numpy(),
                                **{f"prediction_{i}": keyed[key] for i, key in enumerate(chosen)}})

    def finish_split(self, split):
        combined = {}
        for key, chunks in self.chunks.items():
            arrays = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
            group_id = f"g{len(self.groups):03d}"
            relative = f"episodes/{group_id}.npz"
            np.savez_compressed(self.output / relative, **arrays)
            condition, initialization, action_mode, model, seed = key
            self.groups.append({"key": group_id, "split": condition, "initialization": initialization,
                                "actions": action_mode, "model": model, "seed": seed,
                                "episodes": len(arrays["episode_seed"]), "file": relative})
            combined[key] = arrays
        rank_arrays = combined[(split, "estimated", "recorded", "hybrid_ensemble", None)]
        candidates = {name: np.concatenate([chunk[name] for chunk in self.candidates]) for name in self.candidates[0]}
        if not np.array_equal(rank_arrays["episode_seed"], candidates["episode_seed"]):
            raise ValueError("Gallery episodes do not align with their measurements")
        errors = np.sqrt(rank_arrays["position_sq_px"].mean(1, dtype=np.float64))
        order = np.lexsort((rank_arrays["episode_seed"], errors))
        # Nearest-rank percentiles, with episode ID breaking ties, are declared before inspecting results.
        for label, quantile in (("median", .5), ("p90", .9), ("worst", 1.0)):
            rank = max(0, math.ceil(quantile * len(order)) - 1)
            index = int(order[rank])
            relative = f"gallery/{split}_{label}"
            state_path = self.output / f"{relative}.npz"
            np.savez_compressed(state_path, **{name: value[index] for name, value in candidates.items()})
            video, note = render_gallery(state_path, self.config["physics"]["radius"],
                                         self.config["training"]["seeds"][0])
            self.gallery.append({"split": split, "selection": label, "episode_seed": int(candidates["episode_seed"][index]),
                                 "position_rmse_px": float(errors[index]), "rank": rank + 1, "population": len(order),
                                 "path": video.relative_to(self.output).as_posix(), "state_archive": f"{relative}.npz",
                                 "note": note})
        self.chunks.clear()
        self.candidates.clear()

    def complete(self):
        files = {path.relative_to(self.output).as_posix(): sha256_file(path)
                 for path in sorted(self.output.rglob("*")) if path.is_file()
                 and path.suffix in (".npz", ".mp4", ".gif", ".json") and path.name != "evidence.json"}
        evidence = {"format_version": 1, "complete": True, "metric_definition": POSITION_RMSE_DEFINITION,
                    "horizons": self.config["evaluation"]["horizons"], "config": self.config,
                    "provenance": self.provenance, "training_history": checkpoint_history(self.config),
                    "groups": self.groups, "gallery": self.gallery, "files": files,
                    "gallery_selection": "Estimated initialization, recorded actions, hybrid ensemble RMSE pooled over all 50 future frames; nearest-rank percentiles; ties by episode seed. Learned video panel uses the first configured seed, never test selection.",
                    "strata_definition": "Ground-truth contacts over the full 50-frame rollout: any ball contact takes precedence over wall-only, otherwise free flight; impulse flag uses supplied future actions."}
        temporary = self.output / "evidence.partial.json"
        temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.output / "evidence.json")
        return evidence


def render_gallery(state_path, radius, learned_seed):
    from .demo import _panels, _save_video
    from .physics import render_states
    with np.load(state_path, allow_pickle=False) as archive:
        context = torch.from_numpy(archive["context_states"])
        trajectories = [archive["target_states"], *(archive[f"prediction_{i}"] for i in range(5))]
    videos = [render_states(torch.cat((context, torch.from_numpy(states))), radius=radius) for states in trajectories]
    return _save_video(_panels(videos, ["Ground truth", "Constant velocity", "Imperfect physics", "Fitted physics",
                                      f"Learned seed {learned_seed}", "Hybrid ensemble"]), Path(state_path).with_suffix(".mp4"))
