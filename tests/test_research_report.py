"""The research release must remain reconstructible without model inference."""

import builtins
from hashlib import sha256
import json

import numpy as np
import pytest

from nwm.research_report import METRIC_DEFINITION, build_report


def test_coverage_bounds_do_not_collapse_at_all_successes_or_failures():
    from nwm.research_report import _coverage_interval
    all_covered = _coverage_interval(np.ones(4))
    all_missed = _coverage_interval(np.zeros(4))
    assert all_covered["estimate"] == 1 and 0 < all_covered["ci95"][0] < 1
    assert all_covered["ci95"][1] == 1
    assert all_missed["estimate"] == 0 and 0 < all_missed["ci95"][1] < 1
    assert all_missed["ci95"][0] == 0


@pytest.fixture
def evaluation(tmp_path):
    (tmp_path / "episodes").mkdir()
    groups = []
    for index, model in enumerate(("fitted_physics", "imperfect_physics", "hybrid_ensemble")):
        arrays = {
            "episode_seed": np.array([11, 12, 13, 14], dtype=np.int64),
            "position_sq_px": np.full((4, 50), 25.0),
            "position_distance_px": np.full((4, 50), 5.0),
            "velocity_sq": np.full((4, 50), 4.0),
            "pixel_mse": np.full((4, 50), 0.125),
            "boundary_rate": np.zeros((4, 50)), "overlap_rate": np.zeros((4, 50)),
            "contact_counts": np.tile(np.array([2, 1, 1]), (4, 3, 1)),
            "contact_group": np.array([0, 1, 2, 2], dtype=np.uint8),
            "impulse_present": np.array([False, False, True, True]),
        }
        if model == "hybrid_ensemble":
            arrays["uncertainty_errors_px"] = np.broadcast_to(np.array([0.5, 1, 1.5, 2])[:, None, None], (4, 3, 3)).copy()
            arrays["uncertainty_radii_px"] = np.ones((4, 3, 3))
        relative = f"episodes/g{index:03}.npz"
        np.savez_compressed(tmp_path / relative, **arrays)
        groups.append({"key": f"g{index:03}", "split": "test", "initialization": "estimated", "actions": "recorded",
                       "model": model, "seed": None, "episodes": 4, "file": relative})
    evidence = {"format_version": 1, "complete": True, "metric_definition": METRIC_DEFINITION,
                "horizons": [10, 20, 50], "groups": groups, "gallery": [], "provenance": {"checkpoints": {"x.pt": "fixed"}}}
    (tmp_path / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"calibration": {"split": "calibration"},
        "true_state_decoder_error": [{"split": "test", "pixel_mse_by_horizon": {"50": 0.125}}]}), encoding="utf-8")
    return tmp_path


def _edit_group(root, group_index, transform):
    path = root / f"episodes/g{group_index:03}.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    transform(arrays)
    np.savez_compressed(path, **arrays)


def test_identical_predictors_have_zero_paired_interval_and_euclidean_rmse(evaluation):
    path = build_report(evaluation)
    statistics = json.loads((evaluation / "statistical_comparisons.json").read_text())
    primary = statistics["primary"]
    assert primary["estimate"] == 0
    assert primary["ci95"] == [0, 0]
    assert primary["primary"] and primary["small_sample"]
    assert statistics["summaries"][0]["horizons"]["50"]["position_rmse_px"]["estimate"] == 5
    assert statistics["bootstrap"]["resamples"] == 2000
    assert statistics["bootstrap"]["seed"] == 20261019
    assert path.name == "report.html"
    assert "fitted-physics predictor uses the exact force-law family" in path.read_text(encoding="utf-8")


def test_episode_alignment_mismatch_is_rejected(evaluation):
    _edit_group(evaluation, 2, lambda arrays: arrays.update(episode_seed=arrays["episode_seed"][::-1]))
    with pytest.raises(ValueError, match="alignment mismatch"):
        build_report(evaluation)


@pytest.mark.parametrize("change, message", [
    (lambda arrays: arrays.update(position_sq_px=np.full((4, 50), np.nan)), "Nonfinite"),
    (lambda arrays: arrays.update(uncertainty_radii_px=np.ones((4, 3, 2))), "Invalid uncertainty_radii_px shape"),
    (lambda arrays: arrays.update(episode_seed=np.array([1, 1, 2, 3])), "unique integers"),
])
def test_invalid_episode_evidence_is_rejected(evaluation, change, message):
    _edit_group(evaluation, 2, change)
    with pytest.raises(ValueError, match=message):
        build_report(evaluation)


def test_per_object_coverage_and_interval_width(evaluation):
    build_report(evaluation)
    statistics = json.loads((evaluation / "statistical_comparisons.json").read_text())
    assert len(statistics["uncertainty"]) == 9
    for row in statistics["uncertainty"]:
        assert row["coverage"]["estimate"] == 0.5
        assert row["mean_radius_px"]["estimate"] == 1
        assert row["mean_interval_width_px"]["estimate"] == 2
    assert {row["stratum"] for row in statistics["strata"]} == {
        "free_flight", "wall_only", "ball_collision", "with_future_impulse", "without_future_impulse"}


def test_offline_regeneration_is_deterministic_without_inference(evaluation, monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name in ("torch", "nwm.training", "nwm.models", "nwm.evaluation", "nwm.data"):
            raise AssertionError(f"Offline report attempted model or dataset import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    build_report(evaluation)
    paths = [evaluation / "report.html", evaluation / "statistical_comparisons.json", *sorted((evaluation / "report_plots").glob("*.png"))]
    first = [sha256(path.read_bytes()).hexdigest() for path in paths]
    build_report(evaluation)
    assert first == [sha256(path.read_bytes()).hexdigest() for path in paths]


def test_checksums_and_containment_are_checked(evaluation):
    evidence_path = evaluation / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["files"] = {"episodes/g000.npz": "wrong"}
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        build_report(evaluation)
    evidence["files"] = {}
    evidence["groups"][0]["file"] = "../outside.npz"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        build_report(evaluation)


def test_single_episode_small_sample_is_supported(evaluation):
    evidence_path = evaluation / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    for index, group in enumerate(evidence["groups"]):
        _edit_group(evaluation, index, lambda arrays: arrays.update({name: array[:1] for name, array in arrays.items()}))
        group["episodes"] = 1
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    build_report(evaluation)
    statistics = json.loads((evaluation / "statistical_comparisons.json").read_text())
    assert statistics["primary"]["small_sample"]
    assert any(row["episodes"] == 0 for row in statistics["strata"])
