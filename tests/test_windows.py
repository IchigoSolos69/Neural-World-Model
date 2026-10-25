"""Native process/file behavior and the Windows launcher, without research jobs."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import venv

import pytest

from nwm.experiment import _pid_alive, gpu_lock
from nwm.files import replace_file


@contextmanager
def _child(tmp_path, body):
    ready = tmp_path / "ready.json"
    script = ("import json, os, pathlib, sys, time\n"
              "ready = pathlib.Path(sys.argv[1])\n" + body)
    child = subprocess.Popen([sys.executable, "-c", script, str(ready)],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    actual_pid = None
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if ready.exists():
                try:
                    actual_pid = json.loads(ready.read_text())["pid"]
                    break
                except (ValueError, OSError):
                    pass
            if child.poll() is not None:
                raise AssertionError(child.communicate(timeout=2)[1].decode())
            time.sleep(.02)
        assert actual_pid is not None, "Test child did not signal readiness"
        yield child, actual_pid
    finally:
        if actual_pid is not None and _pid_alive(actual_pid):
            os.kill(actual_pid, signal.SIGTERM)
        if child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        child.communicate(timeout=10)


def test_child_liveness_before_and_after_exit(tmp_path):
    with _child(tmp_path, "ready.write_text(json.dumps({'pid': os.getpid()}))\ntime.sleep(60)\n") as (child, pid):
        assert _pid_alive(pid)
        os.kill(pid, signal.SIGTERM)
        child.wait(timeout=10)
        assert not _pid_alive(pid)


def test_abrupt_owner_death_releases_resource_lock(tmp_path):
    lock_path = tmp_path / "resource.lock"
    body = ("from nwm.experiment import gpu_lock\n"
            f"with gpu_lock({str(lock_path)!r}):\n"
            "    ready.write_text(json.dumps({'pid': os.getpid()}))\n"
            "    time.sleep(60)\n")
    with _child(tmp_path, body) as (child, pid):
        os.kill(pid, signal.SIGTERM)
        child.wait(timeout=10)
        with gpu_lock(lock_path):
            pass


@pytest.mark.skipif(os.name != "nt", reason="Windows delete-sharing semantics")
def test_atomic_replace_preserves_destination_during_native_sharing_lock(tmp_path):
    import ctypes
    from ctypes import wintypes

    source, destination = tmp_path / "new.bin", tmp_path / "existing.bin"
    source.write_bytes(b"new checkpoint")
    destination.write_bytes(b"original checkpoint")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(destination), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    try:
        with pytest.raises(PermissionError):
            replace_file(source, destination)
        assert destination.read_bytes() == b"original checkpoint"
        assert source.read_bytes() == b"new checkpoint"
    finally:
        assert kernel.CloseHandle(handle)
    replace_file(source, destination)
    assert destination.read_bytes() == b"new checkpoint"
    assert not source.exists()


@pytest.fixture
def launcher_project(tmp_path):
    if os.name != "nt":
        pytest.skip("Windows batch launcher")
    project = tmp_path / "project with spaces"
    project.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "run.bat", project / "run.bat")
    venv.EnvBuilder(with_pip=False).create(project / ".venv")
    package = project / "nwm"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(
        "import json, os, pathlib, sys\n"
        "with pathlib.Path(os.environ['NWM_TEST_CALLS']).open('a') as stream:\n"
        "    stream.write(json.dumps({'args':sys.argv[1:], 'cwd':os.getcwd(), 'prefix':sys.prefix}) + '\\n')\n"
        "command = sys.argv[1]\n"
        "if command == os.environ['NWM_TEST_FAIL']:\n"
        "    sys.exit(19)\n"
        "if command != 'generate':\n"
        "    raise AssertionError('Stub must stop before training, browser or Explorer execution')\n"
    )
    return project


def _run_launcher(project, outside, arguments, fail="generate"):
    calls = project / "calls.jsonl"
    environment = os.environ.copy()
    environment.update(NWM_TEST_CALLS=str(calls), NWM_TEST_FAIL=fail)
    # CALL makes the quoted batch path work even when the checkout contains spaces.
    command = 'call "' + str(project / "run.bat") + '" ' + arguments
    # A sequence would apply C-runtime backslash quoting to CMD's embedded quotes.
    command_line = f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /c {command}'
    result = subprocess.run(command_line,
                            cwd=outside, env=environment, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=30)
    recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return result, recorded


@pytest.mark.parametrize("argument,expected", [("--help", 0), ("not-an-experiment", 1)])
def test_real_batch_help_and_invalid_mode_do_not_run_python(launcher_project, tmp_path, argument, expected):
    result, calls = _run_launcher(launcher_project, tmp_path, argument)
    assert result.returncode == expected, result.stdout + result.stderr
    assert not calls
    assert "Usage:" in result.stdout if expected == 0 else "Unknown experiment" in result.stdout


@pytest.mark.parametrize("mode", ["smoke", "full"])
@pytest.mark.parametrize("fail,expected_commands", [("generate", ["generate"]),
                                                     ("run-experiment", ["generate", "run-experiment"])])
def test_batch_uses_local_venv_and_stops_before_results_launch(launcher_project, tmp_path, mode, fail, expected_commands):
    result, calls = _run_launcher(launcher_project, tmp_path, mode, fail)
    assert result.returncode == 1, result.stdout + result.stderr
    assert [call["args"][0] for call in calls] == expected_commands
    for call in calls:
        assert Path(call["cwd"]).resolve() == launcher_project.resolve()
        assert Path(call["prefix"]).resolve() == (launcher_project / ".venv").resolve()
        expected = [call["args"][0], "--config", f"configs\\{mode}.json"]
        if call["args"][0] == "run-experiment":
            expected += ["--device", "auto", "--resume"]
        assert call["args"] == expected


def test_batch_app_forwards_arguments_without_opening_browser(launcher_project, tmp_path):
    result, calls = _run_launcher(launcher_project, tmp_path, "app", fail="serve")
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(calls) == 1
    assert calls[0]["args"] == ["serve", "--config", "configs\\app.json", "--open-browser"]
    assert Path(calls[0]["cwd"]).resolve() == launcher_project.resolve()
