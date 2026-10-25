from datetime import datetime, timedelta, timezone
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from nwm.__main__ import main
import nwm.diagnostics as diagnostics
from nwm.config import experiment_signature, load_config
from nwm.models import LearnedDynamics, HybridDynamics, StateDecoder, StateEstimator
from nwm.physics import PhysicsConfig


@pytest.fixture
def project(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/smoke.json").read_text())
    config["data"]["root"] = str(tmp_path / "data")
    config["outputs"]["root"] = str(tmp_path / "run")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    app_path = tmp_path / "app.json"
    app_path.write_text(json.dumps({"root": str(tmp_path / "missing-drive/storage"),
                                    "bundles": [{"id": "smoke", "path": str(tmp_path / "bundle")}]}))
    monkeypatch.setattr(diagnostics, "_cuda", lambda: {"available": False, "kernel_execution": "not_checked"})
    monkeypatch.setattr(diagnostics, "_pid_alive", lambda pid: False)
    return config_path, app_path, tmp_path


def _state(root, value, release=False):
    path = root / ("run/.evaluation.research-release/release_status.json" if release else "run/process.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_fresh_checkout_and_missing_drive_are_warnings_without_any_mutation(project, monkeypatch):
    config, app, root = project
    before = _files(root)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Created directory")))
    monkeypatch.setattr("nwm.experiment._file_lock", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Acquired lock")))
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Started process")))
    report = diagnostics.diagnose(config, app)
    assert report["status"] == "warning" and report["exit_code"] == 0
    assert report["experiment"]["observed_state"] == "not_started"
    assert not report["storage"]["root_exists"]
    assert report["cuda"]["kernel_execution"] == "not_checked"
    assert all(not item["verified"] for item in report["checkpoints"])
    assert _files(root) == before


def test_dead_training_and_waiting_release_report_interruption_without_rewriting(project):
    config, app, root = project
    _state(root, {"state": "running", "pid": 123, "stage": "train"})
    _state(root, {"state": "waiting", "pid": 456}, release=True)
    log = root / "run/training.log"
    log.write_text("learned_0 epoch 2/3 horizon=10: loss=0.1\n")
    before = _files(root)
    report = diagnostics.diagnose(config, app)
    assert report["experiment"]["recorded_state"] == "running"
    assert report["experiment"]["observed_state"] == "interrupted"
    assert report["releases"][0]["observed_state"] == "interrupted"
    assert report["progress"]["epoch"] == 2 and "log-derived" in report["progress"]["source"]
    assert any("run-experiment" in command and "--resume" in command for command in report["next_commands"])
    assert _files(root) == before


@pytest.mark.parametrize("heartbeat", ["old", "invalid", "missing", "future", "naive"])
def test_old_or_bad_heartbeat_never_declares_a_live_process_dead(project, monkeypatch, heartbeat):
    config, app, root = project
    timestamps = {"old": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                  "invalid": "not-a-timestamp", "missing": None,
                  "future": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                  "naive": "2026-01-01T12:00:00"}
    _state(root, {"state": "running", "pid": 123, "heartbeat_at": timestamps[heartbeat]})
    monkeypatch.setattr(diagnostics, "_pid_alive", lambda pid: pid == 123)
    report = diagnostics.diagnose(config, app)
    assert report["experiment"]["observed_state"] == "running"
    assert report["experiment"]["pid_alive"]
    assert not any("--resume" in command for command in report["next_commands"])


def test_live_orphan_child_blocks_resumption_suggestion(project, monkeypatch):
    config, app, root = project
    _state(root, {"state": "running", "pid": 123, "child_pid": 456})
    monkeypatch.setattr(diagnostics, "_pid_alive", lambda pid: pid == 456)
    report = diagnostics.diagnose(config, app)
    assert report["experiment"]["observed_state"] == "interrupted"
    assert report["experiment"]["child_alive"]
    assert not any("--resume" in command for command in report["next_commands"])
    assert any("child is still alive" in item["message"] for item in report["checks"])


def test_malformed_status_and_config_are_errors_but_missing_stages_are_not(project):
    config, app, root = project
    status = _state(root, {})
    status.write_text("{bad-json")
    report = diagnostics.diagnose(config, app)
    assert report["status"] == "error" and report["exit_code"] == 1
    config.write_text('{"data":{}}')
    assert diagnostics.diagnose(config, app)["exit_code"] == 1


def test_deep_checkpoint_validation_is_cpu_only_and_quick_mode_never_loads_weights(project, monkeypatch):
    config_path, app, root = project
    config = load_config(config_path)
    checkpoint = root / "run/checkpoints/learned_0_latest.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"signature": experiment_signature(config), "epoch": 1, "best_epoch": 0,
                "model": LearnedDynamics().state_dict()}, checkpoint)
    original_load = torch.load
    loads = []
    def load(*args, **kwargs):
        loads.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    diagnostics.diagnose(config_path, app)
    assert loads == []
    before = _files(root)
    report = diagnostics.diagnose(config_path, app, verify_artifacts=True)
    assert report["exit_code"] == 0
    item = next(item for item in report["checkpoints"] if item["present"])
    assert item["verified"] and item["epoch"] == 1
    assert all(value["map_location"] == "cpu" and value["weights_only"] for value in loads)
    assert _files(root) == before
    checkpoint.write_bytes(b"not a checkpoint")
    assert diagnostics.diagnose(config_path, app, verify_artifacts=True)["exit_code"] == 1


def test_deep_validation_rejects_checkpoint_incompatible_signature_and_shape(project):
    config_path, app, root = project
    checkpoint = root / "run/checkpoints/learned_0_best.pt"
    checkpoint.parent.mkdir(parents=True)
    for value in ({"signature": "wrong", "model": {}},
                  {"signature": experiment_signature(load_config(config_path)), "model": {"wrong": torch.ones(1)}}):
        torch.save(value, checkpoint)
        report = diagnostics.diagnose(config_path, app, verify_artifacts=True)
        assert any(row["name"] == "checkpoint:learned_0_best.pt" and row["status"] == "error" for row in report["checks"])


def test_corrupt_dataset_metadata_and_bundle_manifest_are_errors(project):
    config, app, root = project
    (root / "data").mkdir()
    (root / "data/metadata.json").write_text('{"fingerprint":"wrong"}')
    (root / "bundle").mkdir()
    (root / "bundle/manifest.json").write_text("{}")
    before = _files(root)
    report = diagnostics.diagnose(config, app, verify_artifacts=True)
    assert report["exit_code"] == 1
    assert any(row["name"] == "dataset" and row["status"] == "error" for row in report["checks"])
    assert any(row["name"] == "bundle:smoke" and row["status"] == "error" for row in report["checks"])
    assert _files(root) == before


def test_deep_bundle_and_evidence_validators_are_used_without_inference(project, monkeypatch):
    config, app, root = project
    (root / "bundle").mkdir()
    (root / "bundle/manifest.json").write_text('{"format_version":1,"files":{"weights.pt":"fake-hash"}}')
    (root / "run/evaluation").mkdir(parents=True)
    (root / "run/evaluation/evidence.json").write_text('{"complete":true}')
    calls = []
    monkeypatch.setattr("nwm.serving.load_serving", lambda path, device: calls.append(("bundle", device)))
    monkeypatch.setattr("nwm.research_report._read_evidence", lambda path: calls.append(("evidence", str(path))))
    before = _files(root)
    report = diagnostics.diagnose(config, app, verify_artifacts=True)
    assert report["exit_code"] == 0
    assert ("bundle", "cpu") in calls
    assert any(call[0] == "evidence" for call in calls)
    assert _files(root) == before


def test_storage_reserve_violation_is_error(project, monkeypatch):
    config, app, root = project
    settings = json.loads(app.read_text())
    managed = Path(settings["root"])
    managed.mkdir(parents=True)
    class Usage:
        free = 1
    monkeypatch.setattr(diagnostics.shutil, "disk_usage", lambda path: Usage())
    report = diagnostics.diagnose(config, app)
    assert report["exit_code"] == 1
    assert report["storage"]["disk_free_bytes"] == 1


def test_cli_json_human_and_argument_exit_codes(project, capsys):
    config, app, _ = project
    for arguments in (["--json"], []):
        with pytest.raises(SystemExit) as exited:
            main(["doctor", "--config", str(config), "--app-config", str(app), *arguments])
        assert exited.value.code == 0
        output = capsys.readouterr().out
        if arguments:
            assert json.loads(output)["format_version"] == 1
        else:
            assert "Suggested commands (not executed" in output
    config.write_text("invalid")
    with pytest.raises(SystemExit) as exited:
        main(["doctor", "--config", str(config), "--app-config", str(app), "--json"])
    assert exited.value.code == 1
    with pytest.raises(SystemExit) as exited:
        main(["doctor", "--unknown"])
    assert exited.value.code == 2


def test_cuda_availability_does_not_execute_kernels(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: (_ for _ in ()).throw(AssertionError("Initialized GPU")))
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Executed kernels")))
    assert diagnostics._cuda()["kernel_execution"] == "not_checked"


def test_complete_compatible_source_and_deep_snapshot_without_inference(project, monkeypatch):
    from nwm.data import generate_dataset
    from nwm.training import _model_hash
    config_path, app, root = project
    config = load_config(config_path)
    generate_dataset(config)
    checkpoints = root / "run/checkpoints"
    checkpoints.mkdir(parents=True)
    estimator = StateEstimator()
    dependency = _model_hash(estimator)
    models = {"perception": estimator, "decoder": StateDecoder()}
    models.update({f"{kind}_{seed}": constructor() for kind, constructor in
                   (("learned", LearnedDynamics), ("hybrid", HybridDynamics)) for seed in (0, 1, 2)})
    for tag, model in models.items():
        stage = tag.split("_")[0]
        for suffix in ("best", "latest"):
            torch.save({"signature": experiment_signature(config), "model": model.state_dict(),
                        "stage": stage, "epoch": 1, "best_epoch": 0, "best_metric": 1.0,
                        "perception_hash": dependency}, checkpoints / f"{tag}_{suffix}.pt")
    fitted = root / "run/fitted_physics.json"
    fitted.write_text(json.dumps({"signature": experiment_signature(config), "physics": asdict(PhysicsConfig())}))
    for constructor in (StateEstimator, StateDecoder, LearnedDynamics, HybridDynamics):
        monkeypatch.setattr(constructor, "forward", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Neural inference")))
    before = _files(root)
    report = diagnostics.diagnose(config_path, app, verify_artifacts=True)
    assert report["exit_code"] == 0
    assert len(report["checkpoints"]) == 16
    assert all(item["verified"] for item in report["checkpoints"])
    assert any(item["name"] == "perception_binding" and item["status"] == "ok" for item in report["checks"])
    assert _files(root) == before
    changed = checkpoints / "hybrid_1_latest.pt"
    checkpoint = torch.load(changed, map_location="cpu", weights_only=True)
    checkpoint["perception_hash"] = "different-estimator"
    torch.save(checkpoint, changed)
    report = diagnostics.diagnose(config_path, app, verify_artifacts=True)
    assert report["exit_code"] == 1
    assert any(item["name"] == "checkpoint:hybrid_1_latest.pt" and "different perception" in item["message"] for item in report["checks"])
    fitted.write_text('{"signature":"wrong","physics":{}}')
    report = diagnostics.diagnose(config_path, app, verify_artifacts=True)
    assert any(item["name"] == "fitted_physics" and item["status"] == "error" for item in report["checks"])


def test_missing_selected_estimator_does_not_mislabel_dependency_as_verified(project):
    config, app, root = project
    checkpoint = root / "run/checkpoints/learned_0_best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"signature": experiment_signature(load_config(config)), "model": LearnedDynamics().state_dict()}, checkpoint)
    report = diagnostics.diagnose(config, app, verify_artifacts=True)
    assert report["exit_code"] == 0
    assert any(item["name"] == "perception_binding" and item["status"] == "warning" for item in report["checks"])
    assert any(item["name"] == "fitted_physics" and item["status"] == "warning" for item in report["checks"])


def test_deep_verifies_recorded_external_release_output_without_current_source_checks(project, monkeypatch):
    config, app, root = project
    external = root / "external-evaluation"
    external.mkdir()
    (external / "evidence.json").write_text('{"complete":true,"provenance":{"execution_provenance":{"old":"source"}}}')
    _state(root, {"state": "complete", "output": str(external)}, release=True)
    calls = []
    monkeypatch.setattr("nwm.research_report._read_evidence", lambda path: calls.append(path))
    before = _files(root)
    report = diagnostics.diagnose(config, app, verify_artifacts=True)
    assert report["exit_code"] == 0
    assert calls == [external]
    assert _files(root) == before


@pytest.mark.parametrize("output", [["bad-path"], "relative-path", "\x00invalid"])
def test_malformed_release_output_is_reported_without_crashing(project, output):
    config, app, root = project
    _state(root, {"state": "waiting", "output": output}, release=True)
    report = diagnostics.diagnose(config, app, verify_artifacts=True)
    assert report["exit_code"] == 1
    assert any(item["name"] == "release_evidence_path" and item["status"] == "error" for item in report["checks"])
