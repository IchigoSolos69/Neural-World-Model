"""Publish selected, dataset-free inference weights as a verified bundle."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .config import experiment_signature
from .metric_definitions import POSITION_RMSE_DEFINITION, normalize_checkpoint_metrics, position_squared_error
from .models import HybridDynamics, LearnedDynamics, StateDecoder, StateEstimator
from .physics import PhysicsConfig
from .predict import EnsemblePredictor, WorldModelPredictor, constant_velocity_rollout, physics_rollout
from .training import _load_checkpoint, _model_hash, load_artifacts, resolve_device
from .provenance import capture_provenance, verify_provenance


def _checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verified_files(root, declared, expected=None):
    if not isinstance(declared, dict) or not declared or (expected is not None and set(declared) != expected):
        raise ValueError("Research evidence contains incomplete file hashes")
    root = Path(root).resolve(strict=True)
    for name, checksum in declared.items():
        relative = Path(name)
        if (not name or relative.is_absolute() or relative.drive or ".." in relative.parts
                or not isinstance(checksum, str) or len(checksum) != 64
                or any(char not in "0123456789abcdef" for char in checksum)):
            raise ValueError("Research evidence contains an invalid file path or SHA256")
        path = root / relative
        if (not path.resolve().is_relative_to(root) or any(parent.is_symlink() for parent in (path, *path.parents))
                or not path.is_file() or _checksum(path) != checksum):
            raise ValueError(f"Research evidence file hash mismatch: {name}")


def _research_evidence(config, evaluation):
    directory = Path(evaluation).resolve(strict=True)
    path = directory / "evidence.json"
    if path.is_symlink():
        raise ValueError("Research evidence cannot be a symbolic link")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("complete") is not True or evidence.get("metric_definition") != POSITION_RMSE_DEFINITION:
        raise ValueError("Research evidence must be complete and use the standard position RMSE")
    provenance = evidence.get("provenance", {})
    verify_provenance(provenance.get("execution_provenance"))
    if provenance.get("experiment_signature") != experiment_signature(config):
        raise ValueError("Research evidence belongs to another experiment")
    files = evidence.get("files", {})
    if not {"metrics.json", "calibration.json"}.issubset(files):
        raise ValueError("Research evidence must bind metrics and calibration files")
    _verified_files(directory, files)
    source = Path(config["outputs"]["root"])
    checkpoint_root = source / "checkpoints"
    expected_checkpoints = {item.name for pattern in ("*_best.pt", "*_latest.pt")
                            for item in checkpoint_root.glob(pattern)}
    required_best = {"perception_best.pt", "decoder_best.pt"} | {
        f"{kind}_{seed}_best.pt" for kind in ("learned", "hybrid") for seed in config["training"]["seeds"]}
    if not required_best.issubset(expected_checkpoints):
        raise ValueError("Research evidence source is missing selected checkpoints")
    _verified_files(checkpoint_root, provenance.get("checkpoints"), expected_checkpoints)
    data_root = Path(config["data"]["root"])
    expected_data = {"metadata.json"} | {item.name for item in data_root.glob("*.npz")}
    if not {"train.npz", "validation.npz", "calibration.npz", "test.npz"}.issubset(expected_data):
        raise ValueError("Research evidence requires all source dataset splits")
    _verified_files(data_root, provenance.get("dataset_files"), expected_data)
    fingerprint = json.loads((data_root / "metadata.json").read_text(encoding="utf-8")).get("fingerprint")
    if not fingerprint or provenance.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Research evidence dataset fingerprint mismatch")
    if provenance.get("fitted_physics_sha256") != _checksum(source / "fitted_physics.json"):
        raise ValueError("Research evidence fitted physics hash mismatch")
    binding = {"evidence_sha256": _checksum(path), "metrics_sha256": files["metrics.json"],
               "metric_definition": POSITION_RMSE_DEFINITION}
    if "primary_comparison" in evidence:
        binding["primary_comparison"] = evidence["primary_comparison"]
    return evidence, binding


def _calibration(value):
    if value.get("split") != "calibration" or value.get("initialization") != "estimated":
        raise ValueError("Serving calibration must use the held-out calibration split and estimated states")
    if set(value.get("horizons", {})) != {"10", "20", "50"}:
        raise ValueError("Serving calibration requires horizons 10, 20 and 50")
    for entry in value["horizons"].values():
        if entry.get("object_order") != ["red", "green", "blue"] or len(entry.get("objects", [])) != 3:
            raise ValueError("Calibration must contain the three fixed object identities")
        for parameters in entry["objects"]:
            for key in ("multiplier", "spread_floor_px", "coverage_target"):
                if not isinstance(parameters.get(key), (float, int)) or not math.isfinite(parameters[key]):
                    raise ValueError("Calibration parameters must be finite numbers")
            if (parameters["multiplier"] < 0 or parameters["spread_floor_px"] <= 0
                    or parameters["coverage_target"] != 0.9 or parameters.get("sample_count", 0) < 1):
                raise ValueError("Invalid 90% uncertainty calibration")
    return value


def calibrated_radii(position_spread, calibration):
    """Return per-object radii in pixels at calibrated horizons, without interpolation."""
    _calibration(calibration)
    spread = torch.as_tensor(position_spread)
    if spread.ndim != 3 or spread.shape[-1] != 3 or not torch.isfinite(spread).all() or (spread < 0).any():
        raise ValueError("position_spread must be finite nonnegative [batch, horizon, 3] world units")
    result = {}
    for horizon, entry in calibration["horizons"].items():
        if int(horizon) <= spread.shape[1]:
            floor = spread.new_tensor([p["spread_floor_px"] for p in entry["objects"]])
            multiplier = spread.new_tensor([p["multiplier"] for p in entry["objects"]])
            result[horizon] = (spread[:, int(horizon) - 1] * 64).clamp_min(floor) * multiplier
    return result


@torch.inference_mode()
def _select_default(config, artifacts, checkpoints):
    """Compare every predictor on validation when source episodes are available."""
    if not (Path(config["data"]["root"]) / "validation.npz").exists():
        candidates = {name: normalize_checkpoint_metrics(ckpt, name)["best_metric"]
                      for name, ckpt in checkpoints.items()}
        return min(candidates, key=lambda name: (candidates[name], name)), {
            "split": "validation", "metric": "position_rmse_px", "horizon": 20,
            "position_rmse_definition": POSITION_RMSE_DEFINITION,
            "candidates": candidates, "candidate_scope": "trained_single_seed_models",
            "note": "Source validation episodes unavailable; selection uses saved checkpoint validation metrics."}
    from torch.utils.data import DataLoader
    from .data import EpisodeDataset
    from .evaluation import rollout_predictions

    dataset = EpisodeDataset(config["data"]["root"], "validation", horizon=20, render_frames=True)
    loader = DataLoader(dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False, num_workers=0)
    totals, count = {}, 0
    for batch in loader:
        initial = artifacts["estimator"](batch["context_frames"], batch["context_actions"])
        for prediction in rollout_predictions(artifacts, initial, batch["future_actions"], PhysicsConfig(**config["physics"])):
            name = prediction["model"] if prediction["seed"] is None else f"{prediction['model']}_{prediction['seed']}"
            error = position_squared_error(prediction["states"], batch["target_states"])
            totals[name] = totals.get(name, 0.0) + float(error.sum())
        count += batch["target_states"].shape[0] * 20 * 3
    candidates = {name: math.sqrt(total / count) for name, total in totals.items()}
    if not candidates or any(not math.isfinite(metric) for metric in candidates.values()):
        raise ValueError("Validation selection did not produce finite model errors")
    return min(candidates, key=lambda name: (candidates[name], name)), {
        "split": "validation", "initialization": "estimated", "metric": "position_rmse_px",
        "position_rmse_definition": POSITION_RMSE_DEFINITION,
        "horizon": 20, "episodes": len(dataset), "candidates": candidates, "candidate_scope": "all_predictors"}


def export_serving(config, output, evaluation=None):
    """Atomically publish an immutable directory; never replace an existing bundle."""
    requested = Path(output)
    if ".." in requested.parts or requested.name in ("", ".", ".."):
        raise ValueError("Bundle output must be an explicit directory without parent traversal")
    destination = requested.absolute()
    if any(parent.is_symlink() for parent in (destination, *destination.parents)):
        raise ValueError("Bundle output cannot traverse symbolic links")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = destination.with_name(f".{destination.name}.export.lock")
    try:
        descriptor = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError("Another export is publishing this bundle") from error
    os.close(descriptor)
    temporary = None
    try:
        if destination.exists():
            raise FileExistsError(f"Bundle already exists: {destination}")
        export_provenance = capture_provenance()
        seeds = config["training"]["seeds"]
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("Serving requires three independently seeded dynamics members")
        evidence, binding = _research_evidence(config, evaluation) if evaluation is not None else (None, None)
        artifacts = load_artifacts(config, "cpu")
        root = Path(config["outputs"]["root"])
        calibration_path = Path(evaluation) / "calibration.json" if evidence else root / "evaluation/calibration.json"
        calibration = _calibration(json.loads(calibration_path.read_text(encoding="utf-8")))
        calibration_verified = evidence is not None or (Path(config["data"]["root"]) / "calibration.npz").is_file()
        if calibration_verified and evidence is None:
            from .evaluation import calibrate
            calibration = _calibration(calibrate(config, artifacts))
        checkpoints = {}
        for kind in ("learned", "hybrid"):
            for seed in seeds:
                name = f"{kind}_{seed}"
                checkpoint = _load_checkpoint(root / "checkpoints" / f"{name}_best.pt", config)
                if checkpoint.get("selection_metric") != "position_rmse_px":
                    raise ValueError(f"{name} has no comparable validation selection metric")
                metric = checkpoint.get("best_metric")
                if not isinstance(metric, (int, float)) or not math.isfinite(metric):
                    raise ValueError(f"{name} has an invalid validation metric")
                checkpoints[name] = checkpoint
        selected, selection = _select_default(config, artifacts, checkpoints)
        portable = deepcopy(config)
        portable.pop("config_path", None)
        portable.pop("outputs", None)
        portable["data"].pop("root", None)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        models = {"estimator": artifacts["estimator"], "decoder": artifacts["decoder"]}
        models.update({f"{kind}_{seed}": model for kind in ("learned", "hybrid")
                       for seed, model in artifacts[kind]})
        model_hashes = {name: _model_hash(model) for name, model in models.items()}
        if evidence is not None and evidence["provenance"].get("model_hashes") != model_hashes:
            raise ValueError("Research evidence model fingerprints differ from the selected checkpoints")
        for name, model in models.items():
            torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, temporary / f"{name}.pt")
        (temporary / "calibration.json").write_text(json.dumps(calibration, indent=2, allow_nan=False), encoding="utf-8")
        (temporary / "fitted_physics.json").write_text(json.dumps(asdict(artifacts["fitted_physics"]), indent=2), encoding="utf-8")
        full_sizes = {"train": 8000, "validation": 500, "calibration": 500, "test": 1000, "ood_per_condition": 500}
        full = all(config["data"].get(k, 0) >= v for k, v in full_sizes.items())
        training_complete = True
        for tag in ("perception", "decoder", *(f"{kind}_{seed}" for kind in ("learned", "hybrid") for seed in seeds)):
            latest = root / "checkpoints" / f"{tag}_latest.pt"
            if not latest.is_file():
                training_complete = False
                break
            checkpoint = _load_checkpoint(latest, config)
            stage = tag.split("_")[0]
            budget = config["training"][f"{stage}_epochs" if stage in ("perception", "decoder") else "dynamics_epochs"]
            if not checkpoint.get("stopped_early") and checkpoint.get("epoch", -1) + 1 < budget:
                training_complete = False
                break
        manifest = {"format_version": 1, "architecture": "nwm-fixed-three-discs-v1",
                    "export_provenance": export_provenance,
                    "evaluation_provenance": deepcopy(evidence["provenance"].get("execution_provenance")) if evidence else None,
                    "position_rmse_definition": POSITION_RMSE_DEFINITION,
                    "experiment_signature": experiment_signature(config), "config": portable,
                    "experiment_size": "full" if full and training_complete else "full_partial" if full else "smoke",
                    "training_complete": training_complete, "default_model": selected,
                    "selection": selection,
                    "calibration_binding": {"verified": calibration_verified,
                                            "source": "verified_research_evaluation" if evidence else "recomputed_on_calibration_split" if calibration_verified else "source_artifact_unverified",
                                            "model_hashes": {name: _model_hash(model) for name, model in models.items()
                                                             if name == "estimator" or name.startswith("hybrid_")}},
                    "model_hashes": model_hashes,
                    "files": {path.name: _checksum(path) for path in temporary.iterdir()}}
        if evidence is not None:
            manifest["research_evidence"] = binding
            if _research_evidence(config, evaluation)[1] != binding:
                raise ValueError("Research evidence changed while publishing the bundle")
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        verify_provenance(export_provenance)
        os.rename(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        reservation.unlink(missing_ok=True)


def load_serving(path, device="auto"):
    """Verify every file before constructing any model; no dataset access is required."""
    root = Path(path).resolve(strict=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Serving files must not be symbolic links")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("architecture") != "nwm-fixed-three-discs-v1":
        raise ValueError("Unsupported serving bundle compatibility version")
    if manifest.get("position_rmse_definition") not in (None, POSITION_RMSE_DEFINITION):
        raise ValueError("Unknown serving position RMSE definition")
    config = manifest["config"]
    if manifest.get("experiment_signature") != experiment_signature(config):
        raise ValueError("Serving experiment signature mismatch")
    seeds = config["training"]["seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Serving requires three distinct seeds")
    model_names = {"estimator", "decoder", *(f"{kind}_{seed}" for kind in ("learned", "hybrid") for seed in seeds)}
    expected = {f"{name}.pt" for name in model_names} | {"calibration.json", "fitted_physics.json"}
    if set(manifest.get("files", {})) != expected or set(manifest.get("model_hashes", {})) != model_names:
        raise ValueError("Serving manifest contains missing or unexpected file paths")
    if {item.name for item in root.iterdir()} != expected | {"manifest.json"}:
        raise ValueError("Serving directory contains missing or unexpected files")
    for name, checksum in manifest["files"].items():
        item = root / name
        if item.is_symlink() or not item.is_file() or _checksum(item) != checksum:
            raise ValueError(f"Serving checksum mismatch: {name}")
    calibration = _calibration(json.loads((root / "calibration.json").read_text(encoding="utf-8")))
    physics = PhysicsConfig(**config["physics"])
    fitted = PhysicsConfig(**json.loads((root / "fitted_physics.json").read_text(encoding="utf-8")))
    device = resolve_device(device)
    result = {"device": device, "config": config, "metadata": manifest,
              "calibration": calibration, "fitted_physics": fitted, "learned": [], "hybrid": []}
    constructors = {"estimator": StateEstimator, "decoder": StateDecoder}
    constructors.update({f"learned_{seed}": LearnedDynamics for seed in seeds})
    constructors.update({f"hybrid_{seed}": partial(HybridDynamics, physics) for seed in seeds})
    for name, constructor in constructors.items():
        model = constructor()
        model.load_state_dict(torch.load(root / f"{name}.pt", map_location="cpu", weights_only=True))
        if _model_hash(model) != manifest["model_hashes"][name]:
            raise ValueError(f"Serving model fingerprint mismatch: {name}")
        model = model.to(device).eval().requires_grad_(False)
        if name in ("estimator", "decoder"):
            result[name] = model
        else:
            kind, seed = name.rsplit("_", 1)
            result[kind].append((int(seed), model))
    dynamics = {"constant_velocity": partial(constant_velocity_rollout, cfg=physics),
                "imperfect_physics": partial(physics_rollout, cfg=replace(physics, drag=0.0, restitution=1.0)),
                "fitted_physics": partial(physics_rollout, cfg=fitted)}
    dynamics.update({f"{kind}_{seed}": model for kind in ("learned", "hybrid") for seed, model in result[kind]})
    result["predictors"] = {name: WorldModelPredictor(result["estimator"], result["decoder"], model, device)
                            for name, model in dynamics.items()}
    result["predictors"]["hybrid_ensemble"] = EnsemblePredictor(result["estimator"], result["decoder"],
                                                             [model for _, model in result["hybrid"]], device)
    if manifest.get("default_model") not in result["predictors"]:
        raise ValueError("Serving default model is unavailable")
    result["predictor"] = result["predictors"][manifest["default_model"]]
    return result
