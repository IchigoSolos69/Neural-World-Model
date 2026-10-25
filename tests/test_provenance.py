import builtins
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from nwm.provenance import capture_provenance, source_fingerprint, verify_provenance


@pytest.fixture
def source_tree(tmp_path):
    (tmp_path / "nwm").mkdir()
    (tmp_path / "nwm/second.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "nwm/first.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="fixture"\n', encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/cpu.txt").write_text("torch==2.7.0\n", encoding="utf-8")
    return tmp_path


def test_fingerprint_includes_untracked_sources_but_excludes_generated_and_docs(source_tree):
    original = source_fingerprint(source_tree)
    assert original == source_fingerprint(source_tree)
    assert list(original["source_files"]) == sorted(original["source_files"])
    for name in ("README.md", "runs/checkpoint.pt", "data/train.npz", "frontend/dist/index.js",
                 "nwm/__pycache__/cached.py"):
        path = source_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored\n", encoding="utf-8")
    assert source_fingerprint(source_tree) == original
    (source_tree / "nwm/untracked.py").write_text("ADDED = True\n", encoding="utf-8")
    changed = source_fingerprint(source_tree)
    assert changed["source_sha256"] != original["source_sha256"]
    assert "nwm/untracked.py" in changed["source_files"]
    (source_tree / "nwm/first.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert source_fingerprint(source_tree)["source_sha256"] != changed["source_sha256"]


def test_capture_works_without_git_or_runtime_imports(source_tree, monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git unavailable")
    monkeypatch.setattr("nwm.provenance.subprocess.run", missing_git)
    original_import = builtins.__import__
    def without_runtimes(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "numpy"}, f"Unexpected runtime import: {name}"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_runtimes)
    captured = capture_provenance(source_tree)
    assert captured["git"] == {"available": False, "revision": None, "dirty": None}
    assert captured["format_version"] == 1
    assert captured["environment"]["python"]
    assert isinstance(captured["environment"]["dependencies"], dict)
    json.dumps(captured, allow_nan=False)
    verify_provenance(captured, source_tree)


def test_verification_ignores_environment_and_git_but_rejects_source_and_protocol_changes(source_tree):
    captured = capture_provenance(source_tree)
    portable = deepcopy(captured)
    portable["environment"] = {"python": "another runtime"}
    portable["git"] = {"available": True, "revision": "another commit", "dirty": True}
    verify_provenance(portable, source_tree)
    changed = deepcopy(captured)
    changed["evaluation_protocol_version"] = "another-protocol"
    with pytest.raises(ValueError, match="protocol changed"):
        verify_provenance(changed, source_tree)
    (source_tree / "requirements/cpu.txt").write_text("torch==2.8.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source or evaluation protocol changed"):
        verify_provenance(captured, source_tree)
    verify_provenance(None, source_tree)
    with pytest.raises(ValueError, match="unsupported format"):
        verify_provenance({}, source_tree)


@pytest.mark.skipif(shutil.which("git") is None, reason="Git not installed")
def test_git_revision_and_dirty_state_are_read_only(source_tree):
    def git(*args):
        result = subprocess.run(["git", "-c", f"safe.directory={source_tree}", "-C", str(source_tree),
                                 "-c", "user.name=Provenance Test", "-c", "user.email=tests@example.invalid",
                                 *args], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()
    git("init")
    git("add", ".")
    git("commit", "-m", "Fixture source")
    expected_revision = git("rev-parse", "HEAD")
    clean = capture_provenance(source_tree)
    assert clean["git"] == {"available": True, "revision": expected_revision, "dirty": False}
    (source_tree / "nwm/new.py").write_text("NEW = 1\n", encoding="utf-8")
    dirty = capture_provenance(source_tree)
    assert dirty["git"]["dirty"] is True
    assert dirty["source_sha256"] != clean["source_sha256"]
    before = git("status", "--porcelain")
    capture_provenance(source_tree)
    assert git("status", "--porcelain") == before


def test_missing_sources_are_not_a_valid_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="missing package sources"):
        source_fingerprint(tmp_path)
