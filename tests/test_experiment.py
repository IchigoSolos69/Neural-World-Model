import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import nwm.experiment as experiment
from nwm.experiment import ResourceBusyError, experiment_status, gpu_lock, run_experiment


@pytest.fixture
def experiment_config(tmp_path):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/smoke.json").read_text())
    config["outputs"]["root"] = str(tmp_path / "run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, Path(config["outputs"]["root"])


def test_gpu_lock_blocks_other_process_and_releases_after_owner_exit(tmp_path):
    path = tmp_path / "gpu.lock"
    script = ("import sys; from nwm.experiment import gpu_lock, ResourceBusyError; "
              "\ntry:\n with gpu_lock(sys.argv[1]): pass\nexcept ResourceBusyError: sys.exit(7)")
    with gpu_lock(path):
        result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, timeout=15)
        assert result.returncode == 7, result.stderr.decode()
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode()


def test_status_recognizes_old_interruption_without_rewriting_file(experiment_config, monkeypatch):
    config, root = experiment_config
    root.mkdir()
    path = root / "process.json"
    path.write_text(json.dumps({"state": "running", "pid": 123, "stage": "train"}))
    before = path.read_bytes()
    monkeypatch.setattr(experiment, "_pid_alive", lambda pid: False)
    assert experiment_status(config)["state"] == "interrupted"
    assert path.read_bytes() == before


def test_supervisor_resumes_three_stages_appends_logs_and_updates_heartbeat(experiment_config, monkeypatch):
    config, root = experiment_config
    root.mkdir()
    (root / "training.log").write_text("preserved earlier training\n")
    calls, statuses = [], []
    original_write = experiment._write_status

    def write(path, value):
        statuses.append(dict(value))
        original_write(path, value)

    class Child:
        pid = 9876

        def __init__(self, args, **kwargs):
            self.waits = 0
            calls.append((args, kwargs))
            kwargs["stdout"].write(f"{args[4]} output\n".encode())

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            return 0

    monkeypatch.setattr(experiment.subprocess, "Popen", Child)
    monkeypatch.setattr(experiment, "_write_status", write)
    assert run_experiment(config, "cpu", resume=True) == root
    assert [args[4] for args, _ in calls] == ["train", "evaluate", "demo"]
    assert "--resume" in calls[0][0] and all("--resume" not in args for args, _ in calls[1:])
    assert all(kwargs["env"]["NWM_EXPERIMENT_SUPERVISOR_PID"] == str(os.getpid()) for _, kwargs in calls)
    assert (root / "training.log").read_text().startswith("preserved earlier training\n")
    assert any(status["child_pid"] == 9876 for status in statuses)
    final = experiment_status(config)
    assert final["state"] == "complete" and final["child_pid"] is None
    assert final["heartbeat_at"] and final["updated_at"]


def test_supervisor_records_failure_and_does_not_run_later_stages(experiment_config, monkeypatch):
    config, root = experiment_config
    calls = []

    class Child:
        pid = 9876

        def __init__(self, args, **kwargs):
            calls.append(args)

        def wait(self, timeout=None):
            return 3

        def poll(self):
            return 3

    monkeypatch.setattr(experiment.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="train exited with code 3"):
        run_experiment(config, "cpu")
    assert len(calls) == 1
    assert experiment_status(config)["state"] == "failed"


def test_supervisor_refuses_live_previous_process(experiment_config, monkeypatch):
    config, root = experiment_config
    root.mkdir()
    path = root / "process.json"
    path.write_text(json.dumps({"pid": os.getpid(), "state": "running"}))
    with pytest.raises(ResourceBusyError, match="still alive"):
        run_experiment(config, "cpu")
    assert json.loads(path.read_text())["state"] == "running"


def test_pid_check_detects_current_process():
    assert experiment._pid_alive(os.getpid())
    assert not experiment._pid_alive(None)
