from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nwm.config import experiment_signature, load_config
from nwm.models import HybridDynamics, LearnedDynamics, StateDecoder, StateEstimator
from nwm.physics import PhysicsConfig
from nwm.serving import calibrated_radii, export_serving, load_serving
from nwm.training import _model_hash


@pytest.fixture(scope="module")
def source_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("serving-source")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    config["data"]["root"] = str(root / "missing-data")
    config["outputs"]["root"] = str(root)
    (root / "checkpoints").mkdir()
    estimator = StateEstimator()
    models = {"perception": estimator, "decoder": StateDecoder()}
    for kind, constructor in (("learned", LearnedDynamics), ("hybrid", HybridDynamics)):
        models.update({f"{kind}_{seed}": constructor() for seed in config["training"]["seeds"]})
    for index, (name, model) in enumerate(models.items()):
        torch.save({"signature": experiment_signature(config), "model": model.state_dict(),
                    "perception_hash": _model_hash(estimator), "optimizer": {"never_export": True},
                    "selection_metric": "position_rmse_px", "best_metric": 100.0 - index},
                   root / "checkpoints" / f"{name}_best.pt")
    (root / "fitted_physics.json").write_text(json.dumps({
        "signature": experiment_signature(config), "physics": asdict(PhysicsConfig())}))
    (root / "evaluation").mkdir()
    calibration = {"split": "calibration", "initialization": "estimated", "horizons": {
        str(horizon): {"object_order": ["red", "green", "blue"], "objects": [
            {"multiplier": 2.0 + index, "spread_floor_px": 0.5, "coverage_target": 0.9, "sample_count": 12}
            for index in range(3)]} for horizon in (10, 20, 50)}}
    (root / "evaluation/calibration.json").write_text(json.dumps(calibration))
    return config


def test_bundle_roundtrip_needs_no_datasets_and_contains_selected_weights_only(source_experiment, tmp_path):
    path = export_serving(source_experiment, tmp_path / "bundle")
    loaded = load_serving(path, "cpu")
    assert loaded["metadata"]["experiment_size"] == "smoke"
    assert loaded["metadata"]["selection"]["candidate_scope"] == "trained_single_seed_models"
    assert loaded["metadata"]["default_model"] == "hybrid_2"
    assert "root" not in loaded["config"]["data"] and "outputs" not in loaded["config"]
    assert len(loaded["predictors"]) == 10
    for file in path.glob("*.pt"):
        weights = torch.load(file, weights_only=True)
        assert weights and all(isinstance(value, torch.Tensor) for value in weights.values())
    inputs = (torch.zeros(1, 4, 3, 64, 64), torch.zeros(1, 3, 2), torch.zeros(1, 10, 2))
    prediction = loaded["predictors"]["hybrid_ensemble"].predict(*inputs)
    assert prediction["states"].shape == (1, 10, 3, 4)
    assert prediction["frames"].shape == (1, 10, 3, 64, 64)
    assert torch.isfinite(prediction["frames"]).all()
    radii = calibrated_radii(prediction["position_spread"], loaded["calibration"])
    assert set(radii) == {"10"} and radii["10"].shape == (1, 3)
    floor = calibrated_radii(torch.zeros(2, 50, 3), loaded["calibration"])
    torch.testing.assert_close(floor["50"], torch.tensor([[1.0, 1.5, 2.0]]).expand(2, -1))


def test_selection_compares_baselines_and_ensemble_on_validation_only(source_experiment, tmp_path):
    from nwm.data import generate_episodes
    config = deepcopy(source_experiment)
    config["data"]["root"] = str(tmp_path / "data")
    data = Path(config["data"]["root"])
    data.mkdir()
    episodes = generate_episodes(np.arange(2), 64, PhysicsConfig())
    np.savez_compressed(data / "validation.npz", **episodes)
    path = export_serving(config, tmp_path / "bundle")
    manifest = json.loads((path / "manifest.json").read_text())
    selection = manifest["selection"]
    assert selection["split"] == "validation" and selection["episodes"] == 2
    assert selection["candidate_scope"] == "all_predictors"
    assert set(selection["candidates"]) == {
        "constant_velocity", "imperfect_physics", "fitted_physics", "hybrid_ensemble",
        "learned_0", "learned_1", "learned_2", "hybrid_0", "hybrid_1", "hybrid_2"}
    assert manifest["default_model"] == min(selection["candidates"], key=selection["candidates"].get)
    load_serving(path, "cpu")


@pytest.mark.parametrize("tamper", ["weights", "path", "version", "model_hash", "extra"])
def test_bundle_rejects_corruption_and_path_escape(source_experiment, tmp_path, tamper):
    path = export_serving(source_experiment, tmp_path / "bundle")
    manifest = json.loads((path / "manifest.json").read_text())
    if tamper == "weights":
        with (path / "estimator.pt").open("ab") as stream:
            stream.write(b"tampered")
    elif tamper == "path":
        manifest["files"]["../escape.pt"] = manifest["files"].pop("estimator.pt")
    elif tamper == "version":
        manifest["format_version"] = 2
    elif tamper == "model_hash":
        manifest["model_hashes"]["estimator"] = "wrong"
    else:
        (path / "extra.pt").write_bytes(b"not allowed")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum|unexpected|compatibility|fingerprint"):
        load_serving(path, "cpu")


def test_export_refuses_overwrite_and_parent_traversal(source_experiment, tmp_path):
    path = export_serving(source_experiment, tmp_path / "bundle")
    original = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        export_serving(source_experiment, path)
    assert (path / "manifest.json").read_bytes() == original
    with pytest.raises(ValueError, match="traversal"):
        export_serving(source_experiment, tmp_path / "nested/../escape")
    assert not list(tmp_path.glob(".*.export.lock"))


def test_calibration_rejects_negative_spread(source_experiment):
    calibration = json.loads((Path(source_experiment["outputs"]["root"]) / "evaluation/calibration.json").read_text())
    with pytest.raises(ValueError, match="nonnegative"):
        calibrated_radii(-torch.ones(1, 50, 3), calibration)
