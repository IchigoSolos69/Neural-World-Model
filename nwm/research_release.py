"""One finite, restart-safe publication after the existing experiment finishes."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from . import evaluation, serving
from .config import experiment_signature, load_config
from .files import replace_file
from .data import EpisodeDataset, OOD_CONDITIONS
from .experiment import ResourceBusyError, _file_lock, _pid_alive, experiment_status, gpu_lock
from .metric_definitions import POSITION_RMSE_DEFINITION
from .physics import PhysicsConfig
from .predict import EnsemblePredictor, WorldModelPredictor, constant_velocity_rollout, physics_rollout
from .training import load_artifacts, resolve_device
from .provenance import capture_provenance, verify_provenance


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value, *, immutable=False):
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    if immutable:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)


def _freeze_inputs(config, config_path):
    run = Path(config["outputs"]["root"])
    data = Path(config["data"]["root"])
    tags = ["perception", "decoder", *(f"{kind}_{seed}" for kind in ("learned", "hybrid")
                                        for seed in config["training"]["seeds"])]
    required = [Path(config_path), run / "fitted_physics.json", data / "metadata.json"]
    required.extend(run / "checkpoints" / f"{tag}_{suffix}.pt" for tag in tags for suffix in ("best", "latest"))
    required.extend(data / f"{split}.npz" for split in ("train", "validation", "calibration", "test", *OOD_CONDITIONS))
    required.extend(data.glob("*.npz"))
    files = {}
    for path in sorted(set(required)):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Cannot freeze a missing or symbolic input: {path}")
        files[str(path.resolve())] = _sha256(path)
    return {"format_version": 1, "experiment_signature": experiment_signature(config),
            "config": config, "files": files, "execution_provenance": capture_provenance()}


def _check_frozen(config, config_path, expected):
    actual = _freeze_inputs(config, config_path)
    # Older freezes cannot attest which source code ran; do not backfill them.
    if ({key: value for key, value in actual.items() if key != "execution_provenance"}
            != {key: value for key, value in expected.items() if key != "execution_provenance"}):
        raise ValueError("Frozen research inputs changed; refuse to mix experiments or overwrite this release")
    verify_provenance(expected.get("execution_provenance"))


def _complete_evidence(output, freeze):
    path = output / "evidence.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Existing evaluation is incomplete; use a new output directory instead of rerunning inference over it")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (evidence.get("complete") is not True or evidence.get("format_version") != 1
            or evidence.get("metric_definition") != POSITION_RMSE_DEFINITION):
        raise ValueError("Existing evaluation has no complete, compatible research evidence")
    provenance = evidence.get("provenance", {})
    recorded_execution = provenance.get("execution_provenance")
    frozen_execution = freeze.get("execution_provenance")
    if frozen_execution is not None:
        if not isinstance(recorded_execution, dict) or any(
                recorded_execution.get(key) != frozen_execution.get(key)
                for key in ("source_sha256", "evaluation_protocol_version")):
            raise ValueError("Evaluation source provenance differs from the frozen research inputs")
    verify_provenance(recorded_execution)
    config = freeze["config"]
    run, data = Path(config["outputs"]["root"]), Path(config["data"]["root"])
    for basename, checksum in provenance.get("checkpoints", {}).items():
        if Path(basename).name != basename or freeze["files"].get(str((run / "checkpoints" / basename).resolve())) != checksum:
            raise ValueError("Evaluation checkpoints differ from frozen research inputs")
    expected_checkpoints = {Path(path).name for path in freeze["files"]
                            if Path(path).parent == (run / "checkpoints").resolve() and path.endswith("_best.pt")}
    if not expected_checkpoints.issubset(provenance.get("checkpoints", {})):
        raise ValueError("Evaluation is missing selected checkpoint fingerprints")
    expected_data = {Path(path).name: checksum for path, checksum in freeze["files"].items()
                     if Path(path).parent == data.resolve()}
    if provenance.get("dataset_files") != expected_data:
        raise ValueError("Evaluation dataset fingerprints differ from frozen research inputs")
    if provenance.get("fitted_physics_sha256") != freeze["files"][str((run / "fitted_physics.json").resolve())]:
        raise ValueError("Evaluation fitted physics differs from frozen research inputs")
    files = evidence.get("files", {})
    if not {"metrics.json", "calibration.json"}.issubset(files):
        raise ValueError("Evaluation evidence is missing required measurements")
    for relative, checksum in files.items():
        item = output / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or item.is_symlink()
                or not item.is_file() or _sha256(item) != checksum):
            raise ValueError(f"Evaluation artifact checksum mismatch: {relative}")
    return evidence


@torch.inference_mode()
def verify_serving_parity(config, bundle):
    """Compare every public predictor on one deterministic validation context."""
    source = load_artifacts(config, "cpu")
    published = serving.load_serving(bundle, "cpu")
    cfg = PhysicsConfig(**config["physics"])
    dynamics = {"constant_velocity": partial(constant_velocity_rollout, cfg=cfg),
                "imperfect_physics": partial(physics_rollout, cfg=replace(cfg, drag=0.0, restitution=1.0)),
                "fitted_physics": partial(physics_rollout, cfg=source["fitted_physics"])}
    dynamics.update({f"{kind}_{seed}": model for kind in ("learned", "hybrid") for seed, model in source[kind]})
    predictors = {name: WorldModelPredictor(source["estimator"], source["decoder"], model, "cpu")
                  for name, model in dynamics.items()}
    predictors["hybrid_ensemble"] = EnsemblePredictor(source["estimator"], source["decoder"],
                                                     [model for _, model in source["hybrid"]], "cpu")
    if set(predictors) != set(published["predictors"]):
        raise ValueError("Published predictor keys differ from the evaluated experiment")
    sample = EpisodeDataset(config["data"]["root"], "validation", horizon=50, render_frames=True)[0]
    inputs = tuple(sample[key].unsqueeze(0) for key in ("context_frames", "context_actions", "future_actions"))
    results = {}
    for name, predictor in predictors.items():
        expected, actual = predictor.predict(*inputs), published["predictors"][name].predict(*inputs)
        if set(expected) != set(actual):
            raise ValueError(f"Published prediction outputs differ: {name}")
        results[name] = {}
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=1e-5, atol=1e-6)
            results[name][f"{key}_max_absolute_error"] = float((expected[key] - actual[key]).abs().max())
    return {"device": "cpu", "split": "validation", "episode_seed": sample["episode_seed"],
            "context_start": sample["start"], "horizon": 50, "rtol": 1e-5, "atol": 1e-6,
            "inputs": ["context_frames", "context_actions", "future_actions"], "predictors": results}


def release_research(config_path, output, bundle, device="cuda", wait=False, poll_seconds=30):
    """Wait for the existing pipeline, freeze inputs, evaluate once, and publish.

    No training is started or resumed. Incomplete evaluations are preserved and
    require a new output path; completed evaluations can resume publication.
    """
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    initial_config_sha256 = _sha256(config_path)
    output, bundle = Path(output).resolve(), Path(bundle).resolve()
    if output == bundle or output in bundle.parents or bundle in output.parents:
        raise ValueError("Evaluation and serving bundle directories must not overlap")
    state_root = output.with_name(f".{output.name}.research-release")
    state_root.mkdir(parents=True, exist_ok=True)
    status_path = state_root / "release_status.json"
    status = {"pid": os.getpid(), "config_path": str(config_path), "output": str(output),
              "bundle": str(bundle), "state": "waiting", "started_at": datetime.now(timezone.utc).isoformat()}

    def update(**changes):
        status.update(changes, heartbeat_at=datetime.now(timezone.utc).isoformat())
        _write_json(status_path, status)

    with _file_lock(state_root / "release.lock", "Research release"):
        try:
            while True:
                observed = experiment_status(config_path)
                state = observed.get("state")
                alive = (_pid_alive(observed.get("pid")) or _pid_alive(observed.get("child_pid"))
                         or observed.get("child_alive", False))
                if state == "complete" and not alive:
                    break
                if state not in ("running", "complete"):
                    raise RuntimeError(f"Existing experiment is {state}; release never starts or restarts training")
                update(state="waiting", experiment=observed)
                if not wait:
                    raise ResourceBusyError("Existing experiment has not finished and exited; leave it undisturbed or use --wait")
                time.sleep(poll_seconds)
            if _sha256(config_path) != initial_config_sha256 or load_config(config_path) != config:
                raise ValueError("Experiment configuration changed while waiting; preserve this release and review the new configuration")
            resolved_device = resolve_device(device)
            with gpu_lock() if resolved_device.type == "cuda" else nullcontext():
                freeze_path = state_root / "freeze.json"
                if freeze_path.exists():
                    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
                    _check_frozen(config, config_path, frozen)
                else:
                    frozen = _freeze_inputs(config, config_path)
                    if output.exists():
                        previous_path = output / "evidence.json"
                        if not previous_path.is_file() or previous_path.is_symlink():
                            raise ValueError("Existing evaluation is incomplete; use a new output directory instead of rerunning inference over it")
                        previous = json.loads(previous_path.read_text(encoding="utf-8"))
                        # Publication today cannot attest to an older evaluation's execution.
                        frozen["execution_provenance"] = previous.get("provenance", {}).get("execution_provenance")
                        _complete_evidence(output, frozen)
                    _write_json(freeze_path, frozen, immutable=True)
                update(state="evaluating", freeze_sha256=_sha256(freeze_path), device=str(resolved_device))
                if not output.exists():
                    evaluation.evaluate(config, str(resolved_device), output=output)
                evidence = _complete_evidence(output, frozen)
                _check_frozen(config, config_path, frozen)
                update(state="reporting", evidence_sha256=_sha256(output / "evidence.json"))
                from .research_report import build_report
                build_report(output)
                _check_frozen(config, config_path, frozen)
                update(state="publishing")
                candidate = bundle if bundle.exists() else bundle.with_name(f".{bundle.name}.research-staging")
                if not candidate.exists():
                    serving.export_serving(config, candidate, evaluation=output)
                published = serving.load_serving(candidate, "cpu")
                binding = published["metadata"].get("research_evidence", {})
                if (binding.get("evidence_sha256") != _sha256(output / "evidence.json")
                        or binding.get("metrics_sha256") != _sha256(output / "metrics.json")
                        or published["metadata"].get("model_hashes") != evidence["provenance"].get("model_hashes")):
                    raise ValueError("Existing serving bundle is not bound to this evaluated experiment; preserve it and choose another destination")
                if published["metadata"].get("training_complete") is not True:
                    raise ValueError("The evaluated experiment has incomplete training stages; refuse to publish it as a completed research release")
                parity = verify_serving_parity(config, candidate)
                _check_frozen(config, config_path, frozen)
                _write_json(state_root / "parity.json", parity)
                if candidate != bundle:
                    if bundle.exists():
                        raise FileExistsError("Serving destination appeared during parity verification; preserve both bundles")
                    candidate.rename(bundle)
                update(state="complete", parity_sha256=_sha256(state_root / "parity.json"),
                       bundle_manifest_sha256=_sha256(bundle / "manifest.json"),
                       artifacts={str(path.relative_to(output)): _sha256(path)
                                  for path in sorted(output.rglob("*")) if path.is_file()})
                return output
        except BaseException as error:
            update(state="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed", error=str(error))
            raise
