"""Small JSON configurations shared by the CLI and notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    root = path.parent.parent
    for section in ("data", "outputs"):
        value = Path(config[section]["root"])
        config[section]["root"] = str(value if value.is_absolute() else root / value)
    config["config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    data, train, physics = config["data"], config["training"], config["physics"]
    if data["frames"] < 54:
        raise ValueError("Episodes need at least 54 frames: four context plus 50 future frames")
    for split in ("train", "validation", "calibration", "test", "ood_per_condition"):
        if not isinstance(data[split], int) or data[split] < 1:
            raise ValueError(f"data.{split} must be a positive integer")
    seeds = train["seeds"]
    if not seeds or any(not isinstance(s, int) for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds must contain distinct integers")
    if train["batch_size"] < 1 or train["learning_rate"] <= 0:
        raise ValueError("Training batch size and learning rate must be positive")
    for key in ("perception_epochs", "decoder_epochs", "dynamics_epochs", "windows_per_episode"):
        if train[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if config["evaluation"]["horizons"] != [10, 20, 50]:
        raise ValueError("This experiment reports horizons [10, 20, 50]")
    if not 0 < physics["radius"] < 0.2 or not 0 <= physics["restitution"] <= 1:
        raise ValueError("Invalid radius or restitution")
    if physics["drag"] < 0 or physics["dt"] <= 0 or physics["substeps"] < 1:
        raise ValueError("Invalid physics integration parameters")


def experiment_signature(config: dict) -> str:
    """Exclude portable paths and extendable epoch budgets from resume identity."""
    training = {k: v for k, v in config["training"].items() if not k.endswith("epochs")}
    data = {k: v for k, v in config["data"].items() if k != "root"}
    identity = {"format": 1, "data": data, "physics": config["physics"], "training": training}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

