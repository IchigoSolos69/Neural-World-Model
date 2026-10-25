from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from nwm.config import experiment_signature, load_config
from nwm.metric_definitions import POSITION_RMSE_DEFINITION
from nwm.models import HybridDynamics, LearnedDynamics, StateDecoder, StateEstimator
from nwm.physics import PhysicsConfig
from nwm.serving import _checksum, calibrated_radii, export_serving, load_serving
from nwm.training import _model_hash, load_artifacts


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
    assert loaded["metadata"]["selection"]["position_rmse_definition"] == POSITION_RMSE_DEFINITION
    assert loaded["metadata"]["selection"]["candidates"]["hybrid_2"] == pytest.approx(93 * 2 ** 0.5)
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
    assert selection["position_rmse_definition"] == POSITION_RMSE_DEFINITION
    assert set(selection["candidates"]) == {
        "constant_velocity", "imperfect_physics", "fitted_physics", "hybrid_ensemble",
        "learned_0", "learned_1", "learned_2", "hybrid_0", "hybrid_1", "hybrid_2"}
    assert manifest["default_model"] == min(selection["candidates"], key=lambda name: (selection["candidates"][name], name))
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


@pytest.fixture
def bound_evidence(source_experiment, tmp_path):
    from nwm.data import generate_episodes
    config = deepcopy(source_experiment)
    source = tmp_path / "source"
    shutil.copytree(config["outputs"]["root"], source)
    config["outputs"]["root"] = str(source)
    data = tmp_path / "data"
    data.mkdir()
    config["data"]["root"] = str(data)
    episodes = generate_episodes(np.arange(2), 64, PhysicsConfig())
    for split in ("train", "validation", "calibration", "test"):
        np.savez_compressed(data / f"{split}.npz", **episodes)
    (data / "metadata.json").write_text(json.dumps({"fingerprint": "test-data-fingerprint"}))
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    shutil.copy(source / "evaluation/calibration.json", evaluation / "calibration.json")
    (evaluation / "metrics.json").write_text(json.dumps({"metric_definition": POSITION_RMSE_DEFINITION}))
    artifacts = load_artifacts(config, "cpu")
    models = {name: artifacts[name] for name in ("estimator", "decoder")}
    models.update({f"{kind}_{seed}": model for kind in ("learned", "hybrid") for seed, model in artifacts[kind]})
    evidence = {"complete": True, "metric_definition": POSITION_RMSE_DEFINITION,
                "primary_comparison": {"difference_px": 1.0},
                "provenance": {"experiment_signature": experiment_signature(config),
                               "checkpoints": {path.name: _checksum(path) for path in (source / "checkpoints").glob("*.pt")},
                               "model_hashes": {name: _model_hash(model) for name, model in models.items()},
                               "dataset_fingerprint": "test-data-fingerprint",
                               "dataset_files": {path.name: _checksum(path) for path in data.iterdir()},
                               "fitted_physics_sha256": _checksum(source / "fitted_physics.json")},
                "files": {path.name: _checksum(path) for path in evaluation.iterdir()}}
    (evaluation / "evidence.json").write_text(json.dumps(evidence))
    return config, evaluation, evidence


def test_research_bound_bundle_reproduces_evaluated_weights(bound_evidence, tmp_path):
    config, evaluation, evidence = bound_evidence
    checkpoints = list((Path(config["outputs"]["root"]) / "checkpoints").glob("*.pt"))
    before = {path.name: _checksum(path) for path in checkpoints}
    bundle = export_serving(config, tmp_path / "published", evaluation=evaluation)
    loaded = load_serving(bundle, "cpu")
    metadata = loaded["metadata"]
    assert metadata["research_evidence"]["evidence_sha256"] == _checksum(evaluation / "evidence.json")
    assert metadata["research_evidence"]["metrics_sha256"] == evidence["files"]["metrics.json"]
    assert metadata["research_evidence"]["primary_comparison"] == evidence["primary_comparison"]
    assert metadata["calibration_binding"]["verified"]
    assert metadata["calibration_binding"]["source"] == "verified_research_evaluation"
    assert metadata["model_hashes"] == evidence["provenance"]["model_hashes"]
    assert metadata["export_provenance"]["environment"]["dependencies"]
    assert metadata["evaluation_provenance"] is None  # Never infer a legacy evaluation's source.
    assert before == {path.name: _checksum(path) for path in checkpoints}
    artifacts = load_artifacts(config, "cpu")
    for kind in ("learned", "hybrid"):
        for (_, published), (_, source) in zip(loaded[kind], artifacts[kind]):
            actions = torch.zeros(1, 10, 2)
            initial = torch.tensor([[[0.2, 0.2, 0.1, 0], [0.5, 0.5, 0, 0], [0.8, 0.8, 0, 0]]])
            torch.testing.assert_close(published(initial, actions), source(initial, actions), rtol=0, atol=0)


def test_bundle_retains_evaluation_provenance_separately_from_export(bound_evidence, tmp_path):
    from nwm.provenance import capture_provenance
    config, evaluation, evidence = bound_evidence
    recorded = capture_provenance()
    recorded["environment"]["python"] = "historical-python"
    recorded["environment"]["dependencies"]["torch"] = "historical-gpu-wheel"
    evidence["provenance"]["execution_provenance"] = recorded
    (evaluation / "evidence.json").write_text(json.dumps(evidence))
    bundle = export_serving(config, tmp_path / "published", evaluation=evaluation)
    metadata = load_serving(bundle, "cpu")["metadata"]
    assert metadata["evaluation_provenance"] == recorded
    assert metadata["export_provenance"]["environment"]["python"] != "historical-python"


def test_bundle_refuses_changed_evaluation_source(bound_evidence, tmp_path):
    from nwm.provenance import capture_provenance
    config, evaluation, evidence = bound_evidence
    recorded = capture_provenance()
    recorded["source_sha256"] = "0" * 64
    evidence["provenance"]["execution_provenance"] = recorded
    (evaluation / "evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="source or evaluation protocol changed"):
        export_serving(config, tmp_path / "published", evaluation=evaluation)
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("tamper", ["incomplete", "definition", "model", "checkpoint", "dataset", "metric_file", "path"])
def test_research_binding_rejects_changed_or_unbound_evidence(bound_evidence, tmp_path, tamper):
    config, evaluation, evidence = bound_evidence
    if tamper == "incomplete":
        evidence["complete"] = False
    elif tamper == "definition":
        evidence["metric_definition"] = "unknown"
    elif tamper == "model":
        evidence["provenance"]["model_hashes"]["estimator"] = "wrong"
    elif tamper == "checkpoint":
        checkpoint = Path(config["outputs"]["root"]) / "checkpoints/learned_0_best.pt"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    elif tamper == "dataset":
        (Path(config["data"]["root"]) / "test.npz").write_bytes(b"changed")
    elif tamper == "metric_file":
        (evaluation / "metrics.json").write_text("{}")
    else:
        evidence["files"]["../outside"] = "0" * 64
    (evaluation / "evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="evidence"):
        export_serving(config, tmp_path / "published", evaluation=evaluation)
    assert not (tmp_path / "published").exists()
    assert not list(tmp_path.glob(".*.export.lock"))
