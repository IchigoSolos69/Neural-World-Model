import pytest
import torch

from nwm.metrics import (RolloutMetrics, collision_counts, conformal_multiplier,
                         infer_contacts, uncertainty_metrics)


def _states():
    initial = torch.tensor([[[0.2, 0.2, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.8, 0.8, 0.0, 0.0]]])
    return initial, initial[:, None].repeat(1, 2, 1, 1)


def test_perfect_rollout_has_zero_errors():
    initial, states = _states()
    metric = RolloutMetrics([1, 2], radius=0.06, dt=1 / 30)
    metric.update(states, states, initial, torch.zeros(1, 2, 15, dtype=torch.bool))
    result = metric.compute()["horizons"]["2"]
    assert result["position_rmse_px"] == 0
    assert result["position_fde_px"] == 0
    assert result["velocity_rmse"] == 0
    assert result["boundary_violation_rate"] == 0
    assert result["overlap_rate"] == 0


def test_state_and_pixel_metrics_use_expected_units():
    initial, states = _states()
    prediction = states.clone()
    prediction[..., 0] += 0.125
    prediction[..., 2] += 0.5
    metric = RolloutMetrics([2], radius=0.06, dt=1 / 30)
    metric.update(prediction, states, initial, torch.zeros(1, 2, 15, dtype=torch.bool),
                  pixel_mse=torch.ones(1, 2))
    result = metric.compute()["horizons"]["2"]
    assert result["position_rmse_px"] == pytest.approx(8)
    assert result["position_ade_px"] == pytest.approx(8)
    assert result["velocity_rmse"] == pytest.approx(0.5)
    assert result["pixel_mse"] == 1


def test_contact_runs_are_deduplicated_and_matched_once():
    predicted = torch.zeros(1, 12, 15, dtype=torch.bool)
    target = torch.zeros_like(predicted)
    predicted[0, [1, 3, 9], 0] = True
    target[0, 2:5, 0] = True
    target[0, 8, 0] = True
    assert collision_counts(predicted, target) == (2, 1, 0)


def test_contact_matching_preserves_two_adjacent_valid_matches():
    predicted = torch.zeros(1, 8, 15, dtype=torch.bool)
    target = torch.zeros_like(predicted)
    predicted[0, [2, 4], 0] = True
    target[0, [1, 3], 0] = True
    assert collision_counts(predicted, target) == (2, 0, 0)


def test_velocity_reversal_identifies_a_between_frame_wall_contact():
    initial, states = _states()
    initial[0, 0] = torch.tensor([0.07, 0.2, -0.6, 0.0])
    states = initial[:, None].clone()
    states[0, 0, 0] = torch.tensor([0.069, 0.2, 0.54, 0.0])
    contacts = infer_contacts(initial, states, radius=0.06, dt=1 / 30)
    assert contacts[0, 0, 3]
    assert contacts.sum() == 1


def test_stationary_touching_objects_are_not_impacts():
    initial, _ = _states()
    initial[0, 0, :2] = torch.tensor([0.06, 0.2])
    initial[0, 1, :2] = torch.tensor([0.18, 0.2])
    contacts = infer_contacts(initial, initial[:, None], radius=0.06, dt=1 / 30)
    assert not contacts.any()


def test_conformal_rank_and_zero_spread_floor():
    errors = torch.arange(1.0, 20.0)
    calibrated = conformal_multiplier(errors, torch.zeros_like(errors), coverage=0.9, floor=0.5)
    assert calibrated["multiplier"] == 36
    assert calibrated["sample_count"] == 19
    result = uncertainty_metrics(errors, torch.zeros_like(errors), calibrated)
    assert result["coverage"] == pytest.approx(18 / 19)
    assert result["mean_radius_px"] == 18
    assert result["mean_diameter_px"] == 36
    assert calibrated["finite_sample_rank_supported"]


def test_insufficient_calibration_count_is_flagged():
    calibrated = conformal_multiplier(torch.ones(4), torch.ones(4))
    assert not calibrated["finite_sample_rank_supported"]
    assert calibrated["maximum_supported_coverage"] == 0.8


def test_separate_object_calibration_reports_each_coverage():
    errors = torch.arange(1.0, 20.0)[:, None].repeat(1, 3)
    spread = torch.ones_like(errors)
    calibration = {"objects": [conformal_multiplier(errors[:, index], spread[:, index]) for index in range(3)]}
    result = uncertainty_metrics(errors, spread, calibration)
    assert len(result["per_object"]) == 3
    assert result["coverage"] == pytest.approx(18 / 19)
    assert result["mean_radius_px"] == 18


def test_invalid_calibration_data_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        conformal_multiplier(torch.tensor([]), torch.tensor([]))
    with pytest.raises(ValueError, match="finite"):
        conformal_multiplier(torch.tensor([float("nan")]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="positive"):
        conformal_multiplier(torch.ones(1), torch.ones(1), floor=0)
