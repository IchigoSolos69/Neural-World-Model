"""Read-only diagnostics; no repair, inference, resource leases, or CUDA kernels."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess

from .config import load_config
from .experiment import _pid_alive


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _process(path, now, stale_after):
    if not path.is_file():
        return {"path": str(path), "recorded_state": None, "observed_state": "not_started",
                "pid_alive": False, "child_alive": False, "heartbeat": "missing", "heartbeat_age_seconds": None}
    value = _json(path)
    pid_alive = _pid_alive(value.get("pid"))
    child_alive = _pid_alive(value.get("child_pid"))
    recorded = value.get("state", "unknown")
    active = recorded in {"running", "waiting", "starting", "evaluating", "reporting", "publishing"}
    observed = "interrupted" if active and not pid_alive else recorded
    heartbeat, age = "missing", None
    if value.get("heartbeat_at"):
        try:
            timestamp = datetime.fromisoformat(value["heartbeat_at"])
            if timestamp.tzinfo is None:
                raise ValueError("Heartbeat needs a timezone")
            age = (now - timestamp).total_seconds()
            if age < 0:
                raise ValueError("Heartbeat is in the future")
            heartbeat = "stale" if age > stale_after else "fresh"
        except (ValueError, TypeError):
            heartbeat, age = "invalid", None
    return {"path": str(path), "recorded_state": recorded, "observed_state": observed,
            "pid": value.get("pid"), "child_pid": value.get("child_pid"),
            "pid_alive": pid_alive, "child_alive": child_alive, "stage": value.get("stage"),
            "heartbeat": heartbeat, "heartbeat_age_seconds": age,
            "output": value.get("output"), "bundle": value.get("bundle")}


def _last_progress(path):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        match = re.search(r"^(\w+) epoch (\d+)/(\d+) horizon=(\d+):", line)
        if match:
            return {"source": "training.log; log-derived, not checkpoint-verified", "stage": match[1],
                    "epoch": int(match[2]), "epoch_budget": int(match[3]),
                    "horizon": int(match[4]), "line": line}
    return None


def _cuda():
    import torch
    return {"torch": str(torch.__version__), "runtime": torch.version.cuda,
            "available": bool(torch.cuda.is_available()), "kernel_execution": "not_checked"}


def _checkpoint(path, config):
    import torch
    from .models import StateEstimator, StateDecoder, LearnedDynamics, HybridDynamics
    from .physics import PhysicsConfig
    from .training import _load_checkpoint, _model_hash
    checkpoint = _load_checkpoint(path, config)
    tag = path.stem.rsplit("_", 1)[0]
    stage = tag.split("_")[0]
    constructors = {"perception": StateEstimator, "decoder": StateDecoder,
                    "learned": LearnedDynamics,
                    "hybrid": lambda: HybridDynamics(PhysicsConfig(**config["physics"]))}
    with torch.random.fork_rng(devices=[]):
        model = constructors[stage]()
    model.load_state_dict(checkpoint["model"])
    if not all(torch.isfinite(tensor).all().item() for tensor in model.state_dict().values()):
        raise ValueError("Nonfinite model weights")
    result = {"epoch": checkpoint.get("epoch"), "best_epoch": checkpoint.get("best_epoch"),
              "stopped_early": checkpoint.get("stopped_early", False), "verified": True}
    if stage == "perception":
        result["model_hash"] = _model_hash(model)
    elif stage in {"learned", "hybrid"}:
        result["perception_hash"] = checkpoint.get("perception_hash")
    return result


def diagnose(config_path="configs/full.json", app_config_path="configs/app.json", verify_artifacts=False):
    """Return versioned observations without creating or rewriting project files."""
    now = datetime.now(timezone.utc)
    report = {"format_version": 1, "checked_at": now.isoformat(),
              "verification": "artifacts" if verify_artifacts else "quick",
              "checks": [], "next_commands": [], "releases": [], "checkpoints": [], "bundles": []}

    def check(name, status, message):
        report["checks"].append({"name": name, "status": status, "message": message})

    def process_check(name, path, threshold):
        try:
            observed = _process(path, now, threshold)
            state = observed["observed_state"]
            warning = (state in {"not_started", "interrupted", "failed", "unknown"}
                       or observed["heartbeat"] in {"invalid", "stale", "missing"}
                       or (state == "complete" and (observed["pid_alive"] or observed["child_alive"])))
            message = f"recorded={observed['recorded_state']}, observed={state}; heartbeat={observed['heartbeat']}"
            if observed["child_alive"] and not observed["pid_alive"]:
                warning = True
                message += "; child is still alive: do not start another trainer"
            check(name, "warning" if warning else "ok", message)
            return observed
        except (OSError, ValueError, TypeError) as error:
            check(name, "error", str(error))
            return {"path": str(path), "observed_state": "unknown", "pid_alive": False, "child_alive": False}

    try:
        config = load_config(config_path)
        report["config_path"] = config["config_path"]
        check("experiment_config", "ok", config["config_path"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        config = None
        check("experiment_config", "error", str(error))

    if config is not None:
        run = Path(config["outputs"]["root"])
        report["experiment"] = process_check("experiment", run / "process.json", 30)
        report["progress"] = None
        try:
            report["progress"] = _last_progress(run / "training.log")
        except OSError as error:
            check("training_log", "error", str(error))
        tags = ["perception", "decoder", *(f"{kind}_{seed}" for kind in ("learned", "hybrid")
                                             for seed in config["training"]["seeds"])]
        for tag in tags:
            for suffix in ("best", "latest"):
                path = run / "checkpoints" / f"{tag}_{suffix}.pt"
                item = {"path": str(path), "present": path.is_file(), "verified": False}
                if item["present"] and verify_artifacts:
                    try:
                        item.update(_checkpoint(path, config))
                    except Exception as error:
                        check(f"checkpoint:{path.name}", "error", str(error))
                report["checkpoints"].append(item)
        missing = [Path(item["path"]).name for item in report["checkpoints"] if not item["present"]]
        check("checkpoint_presence", "warning" if missing else "ok",
              "Missing stages/checkpoints: " + ", ".join(missing) if missing else "All configured checkpoints are present")
        if verify_artifacts:
            selected = next((item for item in report["checkpoints"]
                             if Path(item["path"]).name == "perception_best.pt" and item["verified"]), None)
            dynamics = [item for item in report["checkpoints"] if item["verified"]
                        and Path(item["path"]).name.startswith(("learned_", "hybrid_"))]
            if dynamics and selected is None:
                check("perception_binding", "warning", "Selected perception is unavailable; dynamics dependency cannot be verified")
            elif dynamics:
                for item in dynamics:
                    if item.get("perception_hash") != selected["model_hash"]:
                        item["verified"] = False
                        check(f"checkpoint:{Path(item['path']).name}", "error", "Dynamics used a different perception checkpoint")
                if all(item["verified"] for item in dynamics):
                    check("perception_binding", "ok", "All present dynamics checkpoints match the selected visual estimator")
        for path in sorted(run.glob(".*.research-release/release_status.json")):
            report["releases"].append(process_check(f"release:{path.parent.name}", path, 90))
        if not report["releases"]:
            check("releases", "warning", "No research release status has been recorded")
        data = Path(config["data"]["root"])
        if verify_artifacts and (data / "metadata.json").is_file():
            try:
                from .data import validate_dataset
                validate_dataset(config)
                check("dataset", "ok", "Dataset compatibility and episode archives verified")
            except FileNotFoundError as error:
                check("dataset", "warning", f"Incomplete generation: {error}")
            except Exception as error:
                check("dataset", "error", str(error))
        elif (data / "metadata.json").is_file():
            check("dataset", "ok", "Metadata present; archive integrity not checked in quick mode")
        else:
            check("dataset", "error" if any(data.glob("*.npz")) else "warning", "Dataset metadata is missing")
        if verify_artifacts:
            fitted = run / "fitted_physics.json"
            if fitted.is_file():
                try:
                    from .config import experiment_signature
                    from .physics import PhysicsConfig
                    value = _json(fitted)
                    if value.get("signature") != experiment_signature(config):
                        raise ValueError("Fitted physics belongs to a different experiment")
                    PhysicsConfig(**value["physics"])
                    check("fitted_physics", "ok", "Fitted coefficients and signature verified")
                except Exception as error:
                    check("fitted_physics", "error", str(error))
            else:
                check("fitted_physics", "warning", "Training has not produced fitted coefficients")
            evidence_paths = set(run.glob("*/evidence.json"))
            for release in report["releases"]:
                output = release.get("output")
                if output is None:
                    continue
                try:
                    if not isinstance(output, str) or not output or not Path(output).is_absolute():
                        raise ValueError("Recorded release output must be an absolute directory path")
                    directory = Path(output)
                    path = directory / "evidence.json"
                    if directory.exists() and not directory.is_dir():
                        raise ValueError("Recorded release output is not a directory")
                    if path.is_file():
                        evidence_paths.add(path)
                    else:
                        check(f"evidence:{directory.name}", "warning", "Recorded release output has no complete evidence manifest")
                except (OSError, ValueError, TypeError) as error:
                    check("release_evidence_path", "error", str(error))
            for path in sorted(evidence_paths):
                try:
                    value = _json(path)
                    if value.get("complete") is not True:
                        check(f"evidence:{path.parent.name}", "warning", "Evaluation is incomplete")
                    else:
                        from .research_report import _read_evidence
                        _read_evidence(path.parent)
                        check(f"evidence:{path.parent.name}", "ok", "Saved evidence hashes and episode alignment verified")
                except Exception as error:
                    check(f"evidence:{path.parent.name}", "error", str(error))
        command = subprocess.list2cmdline(["python", "-m", "nwm", "experiment-status", "--config", config["config_path"]])
        report["next_commands"].append(command)
        process = report["experiment"]
        if (process["observed_state"] in {"interrupted", "failed", "not_started"}
                and not process["pid_alive"] and not process["child_alive"]):
            if not (data / "metadata.json").is_file():
                report["next_commands"].append(subprocess.list2cmdline(["python", "-m", "nwm", "generate", "--config", config["config_path"]]))
            report["next_commands"].append(subprocess.list2cmdline(["python", "-m", "nwm", "run-experiment", "--config", config["config_path"], "--device", "auto", "--resume"]))

    try:
        from .app_storage import app_config
        app = app_config(app_config_path)
        check("app_config", "ok", str(Path(app_config_path).resolve()))
        root = Path(app["root"])
        storage = {"root": str(root), "root_exists": root.is_dir(),
                   "quota_bytes": int(app.get("limit_gib", 27) * 1024**3),
                   "reserve_bytes": int(app.get("reserve_gib", 8) * 1024**3)}
        report["storage"] = storage
        if root.is_dir():
            storage["managed_used_bytes"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            storage["disk_free_bytes"] = shutil.disk_usage(root).free
            within_quota = storage["managed_used_bytes"] <= storage["quota_bytes"]
            reserve_ok = storage["disk_free_bytes"] >= storage["reserve_bytes"]
            check("storage", "ok" if within_quota and reserve_ok else "error",
                  f"managed={storage['managed_used_bytes']} / {storage['quota_bytes']} bytes; free={storage['disk_free_bytes']}, reserve={storage['reserve_bytes']} bytes")
        else:
            check("storage", "warning", "Managed storage root/drive does not exist; no directories were created")
        for bundle in app.get("bundles", []):
            path = Path(bundle["path"])
            item = {"id": bundle["id"], "path": str(path), "present": (path / "manifest.json").is_file(), "verified": False}
            report["bundles"].append(item)
            if not item["present"]:
                check(f"bundle:{bundle['id']}", "warning", "Optional serving bundle is not published")
                continue
            try:
                manifest = _json(path / "manifest.json")
                if manifest.get("format_version") != 1 or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
                    raise ValueError("Unsupported or incomplete serving manifest")
                if verify_artifacts:
                    from .serving import load_serving
                    load_serving(path, "cpu")
                    item["verified"] = True
                check(f"bundle:{bundle['id']}", "ok", "Bundle verified on CPU" if verify_artifacts else "Manifest present; weights not checked in quick mode")
            except Exception as error:
                check(f"bundle:{bundle['id']}", "error", str(error))
    except FileNotFoundError as error:
        check("app_config", "warning", str(error))
    except (OSError, ValueError, KeyError, TypeError) as error:
        check("app_config", "error", str(error))

    try:
        report["cuda"] = _cuda()
        check("cuda", "ok" if report["cuda"]["available"] else "warning",
              "Availability only; forward/backward kernels were not executed")
    except Exception as error:
        check("cuda", "warning", f"CUDA availability could not be inspected: {error}")
    report["next_commands"].append("python -m nwm check-device --device cuda")
    report["status"] = ("error" if any(item["status"] == "error" for item in report["checks"]) else
                        "warning" if any(item["status"] == "warning" for item in report["checks"]) else "ok")
    report["exit_code"] = int(report["status"] == "error")
    return report


def format_report(report):
    lines = [f"Neural World Model doctor: {report['status']} ({report['verification']})"]
    lines.extend(f"[{item['status'].upper()}] {item['name']}: {item['message']}" for item in report["checks"])
    if report.get("progress"):
        lines.append("Log-derived progress (not checkpoint-verified): " + report["progress"]["line"])
    verified = [item for item in report["checkpoints"] if item.get("verified")]
    lines.extend(f"Checkpoint {Path(item['path']).name}: epoch={item['epoch']}, best_epoch={item['best_epoch']} (zero-based)" for item in verified)
    lines.append("Suggested commands (not executed; run kernel checks only when the GPU is free):")
    lines.extend("  " + command for command in report["next_commands"])
    return "\n".join(lines)
