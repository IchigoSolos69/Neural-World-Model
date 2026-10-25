"""Read-only execution provenance; historical origins are never inferred."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess


EVALUATION_PROTOCOL_VERSION = "nwm-evidence-v1"


def source_fingerprint(root=None):
    """Hash execution sources, including untracked files, without requiring Git."""
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    paths = [*root.glob("nwm/**/*.py"), root / "pyproject.toml"]
    requirements = root / "requirements"
    if requirements.is_dir():
        paths.extend(path for path in requirements.rglob("*") if path.is_file())
    files = {}
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Cannot fingerprint a symbolic or external source: {path}")
        if "__pycache__" in path.parts:
            continue
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not any(name.startswith("nwm/") for name in files):
        raise ValueError(f"Cannot fingerprint missing package sources: {root}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"source_sha256": hashlib.sha256(encoded).hexdigest(), "source_files": files,
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION}


def _git_state(root):
    unavailable = {"available": False, "revision": None, "dirty": None}
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*arguments):
        return subprocess.run(["git", "-c", f"safe.directory={root}", "-C", str(root), *arguments],
                              capture_output=True, text=True, timeout=5, env=environment)

    try:
        top = git("rev-parse", "--show-toplevel")
        # An extracted source directory inside another checkout is not that repo.
        if top.returncode or Path(top.stdout.strip()).resolve() != root:
            return unavailable
        revision = git("rev-parse", "--verify", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.TimeoutExpired):
        return unavailable
    return {"available": True, "revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "dirty": bool(status.stdout) if status.returncode == 0 else None}


def capture_provenance(root=None):
    """Capture source identity and installed versions without importing runtimes."""
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    dependencies = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            dependencies[name.lower().replace("_", "-")] = str(distribution.version)
    return {"format_version": 1, **source_fingerprint(root), "git": _git_state(root),
            "environment": {"python": platform.python_version(),
                            "implementation": platform.python_implementation(),
                            "platform": platform.platform(),
                            "dependencies": dict(sorted(dependencies.items()))}}


def verify_provenance(expected, root=None):
    """Verify recorded source/protocol only; legacy missing provenance is unknown."""
    if expected is None:
        return
    if not isinstance(expected, dict) or expected.get("format_version") != 1:
        raise ValueError("Execution provenance has an unsupported format")
    current = source_fingerprint(root)
    if (expected.get("source_sha256") != current["source_sha256"]
            or expected.get("source_files") != current["source_files"]
            or expected.get("evaluation_protocol_version") != current["evaluation_protocol_version"]):
        raise ValueError("Execution source or evaluation protocol changed; preserve this release and use a fresh evaluation")
