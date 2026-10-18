import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from nwm.config import experiment_signature, load_config
from nwm.models import HybridDynamics, LearnedDynamics, StateDecoder, StateEstimator
from nwm.physics import PhysicsConfig, render_states
from nwm.predict import EnsemblePredictor
from nwm.training import _model_hash, load_artifacts


@pytest.mark.parametrize("location", ["custom.json", "configs/custom.json", "nested/experiment.json"])
def test_moved_configuration_keeps_project_relative_paths(tmp_path, monkeypatch, location):
    source = Path(__file__).resolve().parents[1]
    config_path = tmp_path / location
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text((source / "configs/smoke.json").read_text())
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    assert Path(config["data"]["root"]) == source / "data/smoke"
    assert Path(config["outputs"]["root"]) == source / "runs/smoke"


def test_configuration_preserves_explicit_absolute_paths(tmp_path):
    source = Path(__file__).resolve().parents[1]
    config = json.loads((source / "configs/smoke.json").read_text())
    for section in ("data", "outputs"):
        config[section]["root"] = str(tmp_path / section)
    path = tmp_path / "absolute.json"
    path.write_text(json.dumps(config))
    loaded = load_config(path)
    for section in ("data", "outputs"):
        assert loaded[section]["root"] == config[section]["root"]


@pytest.fixture
def checkpoint_only_experiment(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    config["data"]["root"] = str(tmp_path / "missing-dataset")
    config["outputs"]["root"] = str(tmp_path / "artifacts")
    config["training"]["seeds"] = [0]
    root = Path(config["outputs"]["root"])
    (root / "checkpoints").mkdir(parents=True)
    estimator = StateEstimator()
    fingerprint = _model_hash(estimator)
    for name, model in (("perception", estimator), ("decoder", StateDecoder()),
                        ("learned_0", LearnedDynamics()), ("hybrid_0", HybridDynamics())):
        torch.save({"signature": experiment_signature(config), "model": model.state_dict(),
                    "perception_hash": fingerprint}, root / "checkpoints" / f"{name}_best.pt")
    (root / "fitted_physics.json").write_text(json.dumps({
        "signature": experiment_signature(config), "physics": asdict(PhysicsConfig())}))
    return config


def test_inference_loads_legacy_weights_without_dataset(checkpoint_only_experiment):
    config = checkpoint_only_experiment
    assert not Path(config["data"]["root"]).exists()
    artifacts = load_artifacts(config, "cpu")
    predictor = EnsemblePredictor(artifacts["estimator"], artifacts["decoder"],
                                 [model for _, model in artifacts["hybrid"]])
    state = torch.tensor([[0.25, 0.25, 0.1, 0.0], [0.75, 0.25, 0.0, 0.1], [0.5, 0.75, -0.1, 0.0]])
    frames = render_states(state.repeat(1, 4, 1, 1))
    prediction = predictor.predict(frames, torch.zeros(1, 3, 2), torch.zeros(1, 10, 2))
    assert prediction["states"].shape == (1, 10, 3, 4)
    assert torch.isfinite(prediction["frames"]).all()


@pytest.mark.parametrize("corrupt", ["checkpoint", "fitted_physics"])
def test_checkpoint_only_loading_retains_signature_checks(checkpoint_only_experiment, corrupt):
    config = checkpoint_only_experiment
    root = Path(config["outputs"]["root"])
    if corrupt == "checkpoint":
        path = root / "checkpoints/perception_best.pt"
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["signature"] = "different-experiment"
        torch.save(checkpoint, path)
    else:
        path = root / "fitted_physics.json"
        fitted = json.loads(path.read_text())
        fitted["signature"] = "different-experiment"
        path.write_text(json.dumps(fitted))
    with pytest.raises(ValueError, match="different experiment|another experiment"):
        load_artifacts(config, "cpu")
