"""Crash-released resource locks and resumable experiment supervision."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import load_config
from .files import replace_file


class ResourceBusyError(RuntimeError):
    """A different process owns the requested experiment or GPU."""


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _file_lock(path, resource):
    """Keep an OS lock open: process death releases ownership automatically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    stream = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    try:
        if path.stat().st_size == 0:
            stream.write(b" ")
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as error:
            raise ResourceBusyError(f"{resource} is busy in another process; retry after it finishes") from error
        owner = {"pid": os.getpid(), "resource": resource, "acquired_at": _now()}
        stream.seek(1)
        stream.write(json.dumps(owner).encode("utf-8"))
        stream.truncate()
        yield owner
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def gpu_lock(lock_path=None):
    """The app worker and training supervisor share this one local GPU lease."""
    path = Path(lock_path) if lock_path else Path(__file__).resolve().parents[1] / "runs/gpu.lock"
    return _file_lock(path, "GPU")


def _pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) does not provide a safe Windows liveness check.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def experiment_status(config_path):
    """Return observed state, overriding stale 'running' when the supervisor died."""
    config = load_config(config_path)
    path = Path(config["outputs"]["root"]) / "process.json"
    if not path.exists():
        return {"state": "not_started"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        return {"state": "unknown", "error": f"Cannot read experiment status: {error}"}
    if value.get("state") == "running":
        alive = _pid_alive(value.get("pid"))
        child_alive = _pid_alive(value.get("child_pid"))
        if not alive:
            value.update(state="interrupted", reason="Supervisor process exited", child_alive=child_alive)
        elif value.get("heartbeat_at"):
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(value["heartbeat_at"])).total_seconds()
                value["heartbeat_stale"] = age > 30
            except ValueError:
                value["heartbeat_stale"] = True
    return value


def _write_status(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.partial.json")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)


def run_experiment(config_path, device="cuda", resume=True):
    """Run train/evaluate/demo sequentially, with persistent logs and heartbeat."""
    from .training import resolve_device

    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    resolved_device = resolve_device(device)
    root = Path(config["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    previous = experiment_status(config_path)
    if previous.get("state") == "running" or previous.get("child_alive"):
        raise ResourceBusyError("An existing experiment process is still alive; do not start a duplicate")
    from contextlib import nullcontext
    with _file_lock(root / "experiment.lock", "Experiment"), (
        gpu_lock() if resolved_device.type == "cuda" else nullcontext()
    ):
        status_path = root / "process.json"
        status = {"pid": os.getpid(), "child_pid": None, "started_at": _now(), "heartbeat_at": _now(),
                  "state": "running", "stage": "starting", "resume": bool(resume),
                  "device": str(resolved_device), "previous_state": previous.get("state")}
        child = None
        _write_status(status_path, status)
        try:
            with (root / "training.log").open("ab", buffering=0) as stdout, (
                root / "training-errors.log"
            ).open("ab", buffering=0) as stderr:
                stdout.write(f"\nExperiment started {status['started_at']}, resume={resume}\n".encode())
                for stage in ("train", "evaluate", "demo"):
                    args = [sys.executable, "-u", "-m", "nwm", stage, "--config", str(config_path),
                            "--device", str(resolved_device)]
                    if stage == "train" and resume:
                        args.append("--resume")
                    environment = os.environ.copy()
                    environment["NWM_EXPERIMENT_SUPERVISOR_PID"] = str(os.getpid())
                    child = subprocess.Popen(args, cwd=Path(__file__).resolve().parents[1],
                                             stdout=stdout, stderr=stderr, env=environment)
                    status.update(stage=stage, child_pid=child.pid, heartbeat_at=_now())
                    _write_status(status_path, status)
                    while True:
                        try:
                            returncode = child.wait(timeout=2)
                            break
                        except subprocess.TimeoutExpired:
                            status["heartbeat_at"] = _now()
                            _write_status(status_path, status)
                    status["child_pid"] = None
                    if returncode:
                        raise RuntimeError(f"{stage} exited with code {returncode}; inspect training-errors.log")
                    child = None
            status.update(state="complete", stage="complete")
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            status.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                          error=str(error), child_pid=None)
            raise
        finally:
            status.update(updated_at=_now(), heartbeat_at=_now())
            _write_status(status_path, status)
    return root
