"""Staged training, portable epoch-boundary resumption, and artifact loading."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import hashlib
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import experiment_signature
from .data import EpisodeDataset, validate_dataset
from .models import HybridDynamics, LearnedDynamics, StateDecoder, StateEstimator
from .physics import PhysicsConfig, render_states


def resolve_device(name="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install a compatible CUDA PyTorch wheel or use --device cpu")
    return device


def device_check(device="auto"):
    """Exercise actual CUDA kernels/backprop; availability alone is insufficient."""
    device = resolve_device(device)
    x = torch.randn(8, 8, device=device, requires_grad=True)
    x.square().mean().backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("Device forward/backward produced invalid gradients")
    result = {"device": str(device), "python": platform.python_version(), "torch": str(torch.__version__)}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                      capability=list(torch.cuda.get_device_capability(device)),
                      memory_gib=round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2))
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    # A checkpoint can move between Windows CUDA, Colab CUDA, and CPU.
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.pt")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path, config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("signature") != experiment_signature(config):
        raise ValueError(f"Checkpoint {path} belongs to a different experiment; choose another outputs.root")
    return checkpoint


def _model_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def normalized_state_loss(prediction, target):
    scale = prediction.new_tensor([0.25, 0.25, 0.5, 0.5])
    return ((prediction - target) / scale).square().mean()


def reconstruction_loss(prediction, target):
    # Discs occupy little of the frame; foreground weighting discourages a blank decoder.
    weights = 1 + 10 * (target.amax(dim=-3, keepdim=True) > 0.1)
    return (weights * (prediction - target).abs()).mean()


class _EstimatedWindows(Dataset):
    def __init__(self, source, estimates):
        self.source, self.estimates = source, estimates

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        item = self.source[index]
        return {"estimate": self.estimates[index], "future_actions": item["future_actions"],
                "target_states": item["target_states"]}


def _loader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=shuffle,
                      num_workers=config["training"].get("num_workers", 0))


def _estimate_windows(source, estimator, config, device):
    values = []
    estimator.eval()
    with torch.no_grad():
        for batch in _loader(source, config):
            frames = render_states(batch["context_states"].to(device), radius=source.radius)
            values.append(estimator(frames, batch["context_actions"].to(device)).cpu())
    return _EstimatedWindows(source, torch.cat(values))


def curriculum_horizon(epoch, epochs):
    return (1, 10, 20)[min(2, epoch * 3 // epochs)]


def _batch_loss(model, batch, stage, device, radius, horizon, training, residual_weight):
    if stage == "perception":
        frames = render_states(batch["context_states"].to(device), radius=radius)
        prediction = model(frames, batch["context_actions"].to(device))
        target = batch["initial_state"].to(device)
        loss = normalized_state_loss(prediction, target)
    elif stage == "decoder":
        candidates = torch.cat((batch["context_states"], batch["target_states"]), dim=1).to(device)
        indices = (torch.randint(candidates.shape[1], (len(candidates),), device=device) if training else
                   torch.full((len(candidates),), 3, device=device, dtype=torch.long))
        target_state = candidates[torch.arange(len(candidates), device=device), indices]
        target = render_states(target_state, radius=radius)
        prediction = model(target_state)
        loss = reconstruction_loss(prediction, target)
        return loss, (prediction - target).square().mean()
    else:
        prediction = model(batch["estimate"].to(device), batch["future_actions"][:, :horizon].to(device))
        target = batch["target_states"][:, :horizon].to(device)
        loss = normalized_state_loss(prediction, target)
        if stage == "hybrid" and training:
            loss = loss + residual_weight * model.last_residual.square().mean()
    position_mse = ((prediction[..., :2] - target[..., :2]) * 64).square().mean()
    return loss, position_mse


def _fit_stage(config, model, stage, seed, train_set, validation_set, device, resume, perception_hash=None):
    settings = config["training"]
    epochs = settings["dynamics_epochs" if stage in ("learned", "hybrid") else f"{stage}_epochs"]
    tag = f"{stage}_{seed}" if stage in ("learned", "hybrid") else stage
    directory = Path(config["outputs"]["root"]) / "checkpoints"
    latest_path, best_path = directory / f"{tag}_latest.pt", directory / f"{tag}_best.pt"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    start, best, stale, history, best_state, best_epoch = 0, math.inf, 0, [], None, -1
    if latest_path.exists():
        if not resume:
            raise FileExistsError(f"{latest_path} exists; pass --resume or use another outputs.root")
        previous = _load_checkpoint(latest_path, config)
        if previous.get("perception_hash") != perception_hash:
            raise ValueError(f"{tag} used a different perception checkpoint; choose a new outputs.root")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        start, best, stale = previous["epoch"] + 1, previous["best_metric"], previous["stale"]
        history = previous["history"]
        best_state, best_epoch = previous["best_model"], previous["best_epoch"]
        # latest embeds the selected weights so an interruption between the two
        # atomic writes cannot lose the best model or require replaying an epoch.
        _atomic_save({**previous, "model": best_state, "epoch": best_epoch}, best_path)
        _restore_rng(previous["rng"])
        if previous.get("stopped_early"):
            start = epochs
    train_loader, val_loader = _loader(train_set, config, True), _loader(validation_set, config)
    for epoch in range(start, epochs):
        began = time.monotonic()
        horizon = curriculum_horizon(epoch, epochs) if stage in ("learned", "hybrid") else 20
        if stage in ("learned", "hybrid") and epoch > 0 and horizon != curriculum_horizon(epoch - 1, epochs):
            stale = 0
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(model, batch, stage, device, config["physics"]["radius"], horizon,
                                  True, settings["residual_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss in {tag}, epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            size = len(batch["future_actions"])
            total += loss.item() * size
            count += size
        model.eval()
        validation, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                _, metric = _batch_loss(model, batch, stage, device, config["physics"]["radius"],
                                        20, False, settings["residual_weight"])
                size = len(batch["future_actions"])
                validation += metric.item() * size
                val_count += size
        value = validation / val_count
        value = math.sqrt(value) if stage != "decoder" else value
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite validation metric in {tag}")
        improved = value < best
        best, stale = (value, 0) if improved else (best, stale + 1)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if improved:
            best_state, best_epoch = state, epoch
        row = {"epoch": epoch + 1, "horizon": horizon, "loss": total / count,
               "validation": value, "seconds": round(time.monotonic() - began, 3)}
        history.append(row)
        print(f"{tag} epoch {epoch + 1}/{epochs} horizon={horizon}: loss={row['loss']:.5f}, val={value:.5f}", flush=True)
        # Give every dynamics model time in all three curriculum stages.
        stopped = stale >= settings["patience"] and (stage not in ("learned", "hybrid") or horizon == 20)
        checkpoint = {"format": 1, "signature": experiment_signature(config), "config": config,
                      "stage": stage, "seed": seed, "epoch": epoch, "best_metric": best,
                      "stale": stale, "stopped_early": stopped, "history": history,
                      "model": state, "best_model": best_state, "best_epoch": best_epoch,
                      "perception_hash": perception_hash,
                      "optimizer": optimizer.state_dict(), "rng": _rng_state()}
        _atomic_save(checkpoint, latest_path)
        if improved:
            _atomic_save(checkpoint, best_path)
        if stopped:
            break
    if not best_path.exists():
        raise RuntimeError(f"No valid checkpoint produced for {tag}")
    model.load_state_dict(_load_checkpoint(best_path, config)["model"])
    model.eval().requires_grad_(False)
    return model


def fit_physics(config):
    """Fit global drag on contact-free steps, restitution on single-wall impacts.

    Labels come exclusively from the training split. Scalar drag commutes with
    an isolated wall reflection, allowing restitution to be estimated exactly.
    """
    cfg = PhysicsConfig(**config["physics"])
    with np.load(Path(config["data"]["root"]) / "train.npz", allow_pickle=False) as archive:
        states, actions, contacts = archive["states"], archive["actions"], archive["contacts"]
    before = states[:, :-1, :, 2:].copy()
    before[:, :, 0] += actions
    after = states[:, 1:, :, 2:]
    free = ~contacts.any(axis=-1)
    denominator = np.square(before[free].astype(np.float64)).sum()
    if denominator <= 0:
        raise ValueError("Cannot fit drag without moving, contact-free training states")
    decay = np.clip((before[free].astype(np.float64) * after[free]).sum() / denominator, 1e-8, 1)
    drag = float(-np.log(decay) / cfg.dt)
    restitution_samples = []
    isolated = contacts.sum(axis=-1) == 1
    for obj in range(3):
        for wall in range(4):
            axis = 0 if wall < 2 else 1
            valid = isolated & contacts[:, :, 3 + obj * 4 + wall] & (np.abs(before[:, :, obj, axis]) > 1e-5)
            restitution_samples.extend((-after[:, :, obj, axis][valid] /
                                        (before[:, :, obj, axis][valid] * decay)).tolist())
    if not restitution_samples:
        raise ValueError("Cannot fit restitution: training split has no isolated wall impacts")
    fitted = replace(cfg, drag=drag, restitution=float(np.clip(np.median(restitution_samples), 0, 1)))
    path = Path(config["outputs"]["root"]) / "fitted_physics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"signature": experiment_signature(config), "physics": asdict(fitted),
                                "training_wall_samples": len(restitution_samples)}, indent=2), encoding="utf-8")
    return fitted


def train(config, device="auto", stage="all", resume=False, seeds=None):
    validate_dataset(config)
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    device = resolve_device(device)
    output = Path(config["outputs"]["root"])
    output.mkdir(parents=True, exist_ok=True)
    environment = device_check(device)
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps(environment), flush=True)
    root = config["data"]["root"]
    if not (Path(root) / "train.npz").exists():
        raise FileNotFoundError("Dataset missing; run python -m nwm generate first")
    train_set = EpisodeDataset(root, "train", horizon=20,
                               windows_per_episode=config["training"]["windows_per_episode"], render_frames=False)
    validation = EpisodeDataset(root, "validation", horizon=20, render_frames=False)
    seed_everything(config["training"]["seeds"][0])
    estimator = StateEstimator()
    if stage in ("all", "perception"):
        estimator = _fit_stage(config, estimator, "perception", 0, train_set, validation, device, resume)
    if stage in ("all", "decoder"):
        seed_everything(config["training"]["seeds"][0])
        _fit_stage(config, StateDecoder(), "decoder", 0, train_set, validation, device, resume)
    if stage in ("all", "learned", "hybrid"):
        if stage != "all":
            estimator.load_state_dict(_load_checkpoint(output / "checkpoints/perception_best.pt", config)["model"])
            estimator.to(device).eval().requires_grad_(False)
        estimated_train = _estimate_windows(train_set, estimator, config, device)
        estimated_validation = _estimate_windows(validation, estimator, config, device)
        selected_seeds = config["training"]["seeds"] if seeds is None else seeds
        if any(seed not in config["training"]["seeds"] for seed in selected_seeds):
            raise ValueError("Requested seed is not listed in training.seeds")
        for dynamics_stage in ("learned", "hybrid") if stage == "all" else (stage,):
            for seed in selected_seeds:
                seed_everything(seed)
                model = LearnedDynamics() if dynamics_stage == "learned" else HybridDynamics(PhysicsConfig(**config["physics"]))
                _fit_stage(config, model, dynamics_stage, seed, estimated_train, estimated_validation,
                           device, resume, perception_hash=_model_hash(estimator))
        fit_physics(config)
    return output


def load_artifacts(config, device="auto"):
    validate_dataset(config)
    device = resolve_device(device)
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    root = Path(config["outputs"]["root"])
    result = {"device": device}
    for name, constructor in (("estimator", StateEstimator), ("decoder", StateDecoder)):
        tag = "perception" if name == "estimator" else name
        model = constructor()
        path = root / "checkpoints" / f"{tag}_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; finish the training stages first")
        model.load_state_dict(_load_checkpoint(path, config)["model"])
        result[name] = model.to(device).eval().requires_grad_(False)
    for name, constructor in (("learned", LearnedDynamics), ("hybrid", HybridDynamics)):
        result[name] = []
        for seed in config["training"]["seeds"]:
            model = constructor() if name == "learned" else constructor(PhysicsConfig(**config["physics"]))
            checkpoint = _load_checkpoint(root / "checkpoints" / f"{name}_{seed}_best.pt", config)
            if checkpoint.get("perception_hash") != _model_hash(result["estimator"]):
                raise ValueError(f"{name}_{seed} was trained with a different perception checkpoint")
            model.load_state_dict(checkpoint["model"])
            result[name].append((seed, model.to(device).eval().requires_grad_(False)))
    fitted = json.loads((root / "fitted_physics.json").read_text(encoding="utf-8"))
    if fitted["signature"] != experiment_signature(config):
        raise ValueError("Fitted physics belongs to another experiment")
    result["fitted_physics"] = PhysicsConfig(**fitted["physics"])
    return result
