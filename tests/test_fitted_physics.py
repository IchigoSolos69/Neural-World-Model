import json
from pathlib import Path

import numpy as np
import pytest

from nwm.config import load_config
from nwm.data import generate_episodes
from nwm.physics import PhysicsConfig
from nwm.training import fit_physics


@pytest.fixture(params=[(0.05, 0.6), (0.4, 0.9)])
def training_trajectories(request, tmp_path):
    drag, restitution = request.param
    config = load_config("configs/smoke.json")
    config["data"]["root"] = str(tmp_path / "data")
    config["outputs"]["root"] = str(tmp_path / "output")
    root = Path(config["data"]["root"])
    root.mkdir()
    episodes = generate_episodes(np.arange(64), 64, PhysicsConfig(drag=drag, restitution=restitution))
    np.savez_compressed(root / "train.npz", **episodes)
    return config, episodes, drag, restitution


def test_fitted_physics_recovers_coefficients_with_impulses_and_wall_impacts(training_trajectories):
    config, episodes, drag, restitution = training_trajectories
    contacts = episodes["contacts"]
    isolated = contacts.sum(axis=-1) == 1
    wall_samples = (isolated[..., None] & contacts[..., 3:]).sum(axis=(0, 1))
    assert (wall_samples > 0).all()  # Exercise every object's four wall-contact channels.
    assert ((episodes["actions"] != 0).any(axis=-1) & ~contacts.any(axis=-1)).any()
    fitted = fit_physics(config)
    assert fitted.drag == pytest.approx(drag, abs=2e-5)
    assert fitted.restitution == pytest.approx(restitution, abs=2e-5)
    artifact = json.loads((Path(config["outputs"]["root"]) / "fitted_physics.json").read_text())
    assert artifact["training_wall_samples"] == int(wall_samples.sum())
    assert artifact["physics"]["drag"] == fitted.drag
    assert artifact["physics"]["restitution"] == fitted.restitution


def test_fitted_physics_ignores_held_out_trajectories(training_trajectories):
    config, episodes, _, _ = training_trajectories
    root = Path(config["data"]["root"])
    held_out = ("validation", "calibration", "test", "ood_drag_high")
    for split in held_out:
        np.savez_compressed(root / f"{split}.npz", **episodes)
    before = fit_physics(config)
    different_physics = generate_episodes(np.arange(100, 116), 64, PhysicsConfig(drag=1.2, restitution=0.3))
    for split in held_out:
        np.savez_compressed(root / f"{split}.npz", **different_physics)
    assert fit_physics(config) == before
