from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nwm.config import load_config
from nwm.evidence import episode_measurements, render_gallery, source_snapshot, verify_source_snapshot
from nwm.metrics import RolloutMetrics
from nwm.physics import PhysicsConfig
from test_serving import source_experiment


def test_episode_statistics_reconstruct_aggregate_and_truth_strata():
    cfg = PhysicsConfig()
    target = torch.zeros(3, 50, 3, 4)
    target[..., :2] = .5
    states = target.clone()
    states[..., :2] += torch.tensor([3 / 64, 4 / 64])
    initial = target[:, 0]
    contacts = torch.zeros(3, 50, 15, dtype=torch.bool)
    contacts[1, 3, 5] = True
    contacts[2, 4, 0] = contacts[2, 4, 5] = True
    future = torch.zeros(3, 50, 2)
    future[1, 1, 0] = .1
    batch = {"episode_seed": torch.tensor([1, 2, 3]), "future_actions": future,
             "initial_state": initial}
    pixel = torch.full((3, 50), .2)
    arrays = episode_measurements({"states": states}, target, initial, contacts, torch.zeros_like(future),
                                  pixel, batch, initial, cfg, [10, 20, 50], {})
    assert arrays["contact_group"].tolist() == [0, 1, 2]
    assert arrays["impulse_present"].tolist() == [False, True, False]
    assert np.sqrt(arrays["position_sq_px"].mean()) == pytest.approx(5)
    aggregate = RolloutMetrics([10, 20, 50], cfg.radius, cfg.dt)
    aggregate.update(states, target, initial, contacts, actions=torch.zeros_like(future), pixel_mse=pixel)
    for index, horizon in enumerate([10, 20, 50]):
        measured = aggregate.compute()["horizons"][str(horizon)]
        assert measured["position_rmse_px"] == pytest.approx(np.sqrt(arrays["position_sq_px"][:, :horizon].mean()))
        assert [measured[f"collision_{key}"] for key in ("tp", "fp", "fn")] == arrays["contact_counts"][:, index].sum(0).tolist()


def test_source_changes_are_detected(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    (run / "checkpoints").mkdir(parents=True)
    config = load_config("configs/smoke.json")
    config["data"]["root"], config["outputs"]["root"] = str(data), str(run)
    (data / "metadata.json").write_text('{"fingerprint":"fixed"}')
    (data / "test.npz").write_bytes(b"states")
    (run / "fitted_physics.json").write_text("{}")
    (run / "checkpoints/hybrid_0_best.pt").write_bytes(b"weights")
    frozen = source_snapshot(config)
    verify_source_snapshot(config, frozen)
    assert frozen["execution_provenance"]["environment"]["dependencies"]
    incompatible_source = deepcopy(frozen)
    incompatible_source["execution_provenance"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source or evaluation protocol changed"):
        verify_source_snapshot(config, incompatible_source)
    legacy = {key: value for key, value in frozen.items() if key != "execution_provenance"}
    verify_source_snapshot(config, legacy)
    (run / "checkpoints/hybrid_0_best.pt").write_bytes(b"different")
    with pytest.raises(ValueError, match="inputs changed"):
        verify_source_snapshot(config, frozen)


def test_enhanced_evaluation_isolated_and_reconstructible(source_experiment, tmp_path, monkeypatch):
    from nwm.data import generate_dataset
    from nwm.evaluation import evaluate
    from nwm.research_report import build_report
    config = deepcopy(source_experiment)
    config["data"]["root"] = str(tmp_path / "data")
    generate_dataset(config)
    # Only encoding is stubbed; predictors, aggregate metrics, calibration and report all execute.
    def save_placeholder(path, radius, seed):
        video = Path(path).with_suffix(".mp4")
        video.write_bytes(b"gallery encoding is verified by the real smoke run")
        return video, None
    monkeypatch.setattr("nwm.evidence.render_gallery", save_placeholder)
    output = tmp_path / "enhanced"
    before = source_snapshot(config)
    evaluate(config, "cpu", output=output)
    verify_source_snapshot(config, before)
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["provenance"]["execution_provenance"]["source_sha256"] == before["execution_provenance"]["source_sha256"]
    assert evidence["complete"] and len(evidence["groups"]) == 6 * 34
    assert len(evidence["gallery"]) == 18
    groups = {(row["split"], row["initialization"], row["actions"], row["model"], row["seed"]): row for row in evidence["groups"]}
    metrics = json.loads((output / "metrics.json").read_text())
    for record in metrics["records"]:
        group = groups[tuple(record[key] for key in ("split", "initialization", "actions", "model", "seed"))]
        with np.load(output / group["file"], allow_pickle=False) as arrays:
            for horizon in (10, 20, 50):
                assert np.sqrt(arrays["position_sq_px"][:, :horizon].mean()) == pytest.approx(
                    record["horizons"][str(horizon)]["position_rmse_px"], rel=1e-6, abs=1e-6)
    first = (output / "report.html").read_bytes()
    build_report(output)
    assert (output / "report.html").read_bytes() == first
    with pytest.raises(FileExistsError, match="fresh"):
        evaluate(config, "cpu", output=output)
