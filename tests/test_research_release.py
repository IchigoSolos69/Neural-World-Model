from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from nwm.config import load_config
from nwm.data import OOD_CONDITIONS, generate_episodes
from nwm.experiment import ResourceBusyError
from nwm.metric_definitions import POSITION_RMSE_DEFINITION
from nwm.physics import PhysicsConfig
import nwm.research_release as release
from nwm.serving import export_serving
from test_serving import source_experiment


@pytest.fixture
def release_stages(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    config["data"]["root"] = str(tmp_path / "data")
    config["outputs"]["root"] = str(tmp_path / "run")
    config.pop("config_path")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    data, run = Path(config["data"]["root"]), Path(config["outputs"]["root"])
    data.mkdir()
    (run / "checkpoints").mkdir(parents=True)
    for split in ("train", "validation", "calibration", "test", *OOD_CONDITIONS):
        (data / f"{split}.npz").write_bytes(f"compact {split}".encode())
    (data / "metadata.json").write_text('{"fingerprint":"fixed-dataset"}')
    (run / "fitted_physics.json").write_text('{"drag":0.2,"restitution":0.9}')
    tags = ["perception", "decoder", *(f"{kind}_{seed}" for kind in ("learned", "hybrid")
                                        for seed in config["training"]["seeds"])]
    for tag in tags:
        for suffix in ("best", "latest"):
            (run / "checkpoints" / f"{tag}_{suffix}.pt").write_bytes(f"{tag}_{suffix}".encode())
    output, bundle = tmp_path / "evaluation", tmp_path / "bundle"
    calls = {"evaluate": 0, "report": 0, "export": 0, "parity": 0}
    model_hashes = {"estimator": "frozen-model"}

    def evaluate(config, device, *, output):
        calls["evaluate"] += 1
        output.mkdir()
        (output / "metrics.json").write_text('{"records":[]}')
        (output / "calibration.json").write_text('{"split":"calibration"}')
        evidence = {"format_version": 1, "complete": True, "metric_definition": POSITION_RMSE_DEFINITION,
                    "provenance": {
                        "checkpoints": {p.name: release._sha256(p) for p in (run / "checkpoints").glob("*_best.pt")},
                        "dataset_files": {p.name: release._sha256(p) for p in data.iterdir()},
                        "fitted_physics_sha256": release._sha256(run / "fitted_physics.json"),
                        "model_hashes": model_hashes},
                    "files": {name: release._sha256(output / name) for name in ("metrics.json", "calibration.json")}}
        (output / "evidence.json").write_text(json.dumps(evidence))
        return output

    def report(output):
        calls["report"] += 1
        (output / "report.html").write_text("Research report")
        return output / "report.html"

    def export(config, bundle, *, evaluation):
        calls["export"] += 1
        bundle.mkdir()
        (bundle / "manifest.json").write_text(json.dumps({
            "training_complete": True,
            "research_evidence": {"evidence_sha256": release._sha256(evaluation / "evidence.json"),
                                  "metrics_sha256": release._sha256(evaluation / "metrics.json")},
            "model_hashes": model_hashes}))
        return bundle

    def parity(config, bundle):
        calls["parity"] += 1
        return {"predictors": {"hybrid_ensemble": {"states_max_absolute_error": 0.0}}}

    report_module = ModuleType("nwm.research_report")
    report_module.build_report = report
    monkeypatch.setitem(sys.modules, "nwm.research_report", report_module)
    monkeypatch.setattr(release, "experiment_status", lambda path: {"state": "complete"})
    monkeypatch.setattr(release, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(release.evaluation, "evaluate", evaluate)
    monkeypatch.setattr(release.serving, "export_serving", export)
    monkeypatch.setattr(release.serving, "load_serving", lambda path, device: {
        "metadata": json.loads((path / "manifest.json").read_text())})
    monkeypatch.setattr(release, "verify_serving_parity", parity)
    return config_path, output, bundle, calls


@pytest.mark.parametrize("state", ["running", "complete"])
def test_release_leaves_live_experiment_undisturbed(release_stages, monkeypatch, state):
    config, output, bundle, calls = release_stages
    monkeypatch.setattr(release, "experiment_status", lambda path: {"state": state, "pid": 123})
    monkeypatch.setattr(release, "_pid_alive", lambda pid: bool(pid))
    with pytest.raises(ResourceBusyError, match="undisturbed"):
        release.release_research(config, output, bundle, "cpu")
    assert not output.exists() and not bundle.exists()
    assert not any(calls.values())


@pytest.mark.parametrize("state", ["failed", "interrupted", "not_started", "unknown"])
def test_release_never_restarts_a_failed_experiment(release_stages, monkeypatch, state):
    config, output, bundle, calls = release_stages
    monkeypatch.setattr(release, "experiment_status", lambda path: {"state": state})
    with pytest.raises(RuntimeError, match="never starts or restarts training"):
        release.release_research(config, output, bundle, "cpu", wait=True)
    assert not any(calls.values())
    status = json.loads((output.with_name(".evaluation.research-release") / "release_status.json").read_text())
    assert status["state"] == "failed"


def test_waits_until_complete_parent_has_exited_then_evaluates_once(release_stages, monkeypatch):
    config, output, bundle, calls = release_stages
    observed = iter([{"state": "running", "pid": 123}, {"state": "complete", "pid": 123}, {"state": "complete"}])
    monkeypatch.setattr(release, "experiment_status", lambda path: next(observed))
    monkeypatch.setattr(release, "_pid_alive", lambda pid: bool(pid))
    sleeps = []
    monkeypatch.setattr(release.time, "sleep", sleeps.append)
    assert release.release_research(config, output, bundle, "cpu", wait=True) == output
    assert sleeps == [30, 30]
    assert calls == {"evaluate": 1, "report": 1, "export": 1, "parity": 1}


def test_completed_release_can_repeat_without_inference_or_overwrite(release_stages):
    config, output, bundle, calls = release_stages
    release.release_research(config, output, bundle, "cpu")
    state_root = output.with_name(".evaluation.research-release")
    frozen = (state_root / "freeze.json").read_bytes()
    manifest = (bundle / "manifest.json").read_bytes()
    release.release_research(config, output, bundle, "cpu")
    assert calls == {"evaluate": 1, "report": 2, "export": 1, "parity": 2}
    assert (state_root / "freeze.json").read_bytes() == frozen
    assert (bundle / "manifest.json").read_bytes() == manifest
    status = json.loads((state_root / "release_status.json").read_text())
    assert status["state"] == "complete" and status["artifacts"]["report.html"]


def test_completed_evaluation_survives_publish_failure_without_duplicate_inference(release_stages, monkeypatch):
    config, output, bundle, calls = release_stages
    original = release.serving.export_serving

    def fail(*args, **kwargs):
        raise RuntimeError("publication interrupted")

    monkeypatch.setattr(release.serving, "export_serving", fail)
    with pytest.raises(RuntimeError, match="publication interrupted"):
        release.release_research(config, output, bundle, "cpu")
    monkeypatch.setattr(release.serving, "export_serving", original)
    release.release_research(config, output, bundle, "cpu")
    assert calls == {"evaluate": 1, "report": 2, "export": 1, "parity": 1}


def test_failed_parity_never_exposes_bundle_at_public_destination(release_stages, monkeypatch):
    config, output, bundle, calls = release_stages
    def fail(*args):
        raise ValueError("prediction parity failed")
    monkeypatch.setattr(release, "verify_serving_parity", fail)
    with pytest.raises(ValueError, match="parity failed"):
        release.release_research(config, output, bundle, "cpu")
    assert not bundle.exists()
    assert bundle.with_name(".bundle.research-staging").exists()
    assert calls["evaluate"] == 1


def test_incomplete_evaluation_is_preserved_without_rerunning(release_stages):
    config, output, bundle, calls = release_stages
    output.mkdir()
    original = b"preserve interrupted measurements"
    (output / "partial.npz").write_bytes(original)
    with pytest.raises(ValueError, match="incomplete"):
        release.release_research(config, output, bundle, "cpu")
    assert (output / "partial.npz").read_bytes() == original
    assert not any(calls.values())


def test_input_mutation_during_evaluation_prevents_publication(release_stages, monkeypatch):
    config, output, bundle, calls = release_stages
    evaluate = release.evaluation.evaluate

    def mutate(*args, **kwargs):
        evaluate(*args, **kwargs)
        source = Path(load_config(config)["outputs"]["root"]) / "checkpoints/hybrid_0_best.pt"
        source.write_bytes(b"changed after evaluation")

    monkeypatch.setattr(release.evaluation, "evaluate", mutate)
    with pytest.raises(ValueError, match="Frozen research inputs changed"):
        release.release_research(config, output, bundle, "cpu")
    assert calls["evaluate"] == 1 and calls["export"] == 0 and not bundle.exists()


@pytest.mark.parametrize("tamper", ["input", "artifact", "binding"])
def test_resume_refuses_mixed_or_tampered_release(release_stages, tamper):
    config, output, bundle, calls = release_stages
    release.release_research(config, output, bundle, "cpu")
    if tamper == "input":
        path = Path(load_config(config)["data"]["root"]) / "test.npz"
        path.write_bytes(b"different test episodes")
    elif tamper == "artifact":
        (output / "metrics.json").write_text("{}")
    else:
        (bundle / "manifest.json").write_text('{"research_evidence":{},"model_hashes":{}}')
    with pytest.raises(ValueError, match="Frozen|checksum|not bound"):
        release.release_research(config, output, bundle, "cpu")
    assert calls["evaluate"] == 1


def test_parity_checks_all_dataset_free_predictors_on_real_context(source_experiment, tmp_path):
    config = deepcopy(source_experiment)
    data = tmp_path / "validation-data"
    data.mkdir()
    config["data"]["root"] = str(data)
    episodes = generate_episodes(np.arange(1), 64, PhysicsConfig())
    np.savez_compressed(data / "validation.npz", **episodes)
    bundle = export_serving(config, tmp_path / "bundle")
    result = release.verify_serving_parity(config, bundle)
    assert result["split"] == "validation" and result["horizon"] == 50
    assert set(result["predictors"]) == {"constant_velocity", "imperfect_physics", "fitted_physics",
                                        "hybrid_ensemble", *(f"{kind}_{seed}" for kind in ("learned", "hybrid")
                                                              for seed in config["training"]["seeds"])}
    assert "position_spread_max_absolute_error" in result["predictors"]["hybrid_ensemble"]
    assert all(value == 0 for row in result["predictors"].values() for value in row.values())
