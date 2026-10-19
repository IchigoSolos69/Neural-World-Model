import math

import pytest
import torch
from torch import nn

from nwm.config import experiment_signature, load_config
from nwm.metric_definitions import (POSITION_RMSE_DEFINITION, normalize_checkpoint_metrics,
                                    position_squared_error)
from nwm.metrics import RolloutMetrics
from nwm.serving import _select_default
from nwm.training import _batch_loss, _load_checkpoint


def test_three_four_displacement_is_five_pixels_in_training_and_evaluation():
    target = torch.zeros(1, 20, 3, 4)
    prediction = target.clone()
    prediction[..., :2] += torch.tensor([3 / 64, 4 / 64])
    squared = position_squared_error(prediction, target)
    assert squared.shape == (1, 20, 3)
    torch.testing.assert_close(squared, torch.full_like(squared, 25))

    class FixedDynamics(nn.Module):
        def forward(self, initial, actions):
            return prediction[:, :actions.shape[1]]

    batch = {"estimate": target[:, 0], "future_actions": torch.zeros(1, 20, 2),
             "target_states": target}
    _, training_mse = _batch_loss(FixedDynamics(), batch, "learned", "cpu", 0.06, 20, False, 0)
    metric = RolloutMetrics([20], radius=0.06, dt=1 / 30)
    metric.update(prediction, target, target[:, 0], torch.zeros(1, 20, 15, dtype=torch.bool))
    assert training_mse.sqrt().item() == 5
    assert metric.compute()["horizons"]["20"]["position_rmse_px"] == 5


def test_serving_fallback_normalizes_legacy_metrics_and_preserves_ranking(tmp_path):
    config = {"data": {"root": str(tmp_path / "missing")}}
    candidates = {"learned_0": {"best_metric": 5 / math.sqrt(2)},
                  "hybrid_0": {"best_metric": 4 / math.sqrt(2)}}
    selected, metadata = _select_default(config, {}, candidates)
    assert selected == "hybrid_0"
    assert metadata["candidates"] == pytest.approx({"learned_0": 5, "hybrid_0": 4})
    assert metadata["position_rmse_definition"] == POSITION_RMSE_DEFINITION
    assert "position_rmse_definition" not in candidates["learned_0"]


@pytest.mark.parametrize("stage", ["learned", "hybrid"])
def test_legacy_conversion_is_idempotent_and_does_not_mutate_checkpoint(stage):
    original = {"stage": stage, "model": {"weight": torch.tensor([1.0])},
                "best_model": {"weight": torch.tensor([2.0])}, "optimizer": {"step": 17},
                "rng": {"torch": torch.get_rng_state()}, "best_epoch": 3,
                "best_metric": 2.0, "history": [{"validation": 3.0, "loss": 7.0}, {"validation": 2.0}]}
    converted = normalize_checkpoint_metrics(original)
    assert converted["best_metric"] == pytest.approx(2 * math.sqrt(2))
    assert [row["validation"] for row in converted["history"]] == pytest.approx([3 * math.sqrt(2), 2 * math.sqrt(2)])
    assert converted["best_epoch"] == original["best_epoch"]
    assert converted["history"][0]["loss"] == 7
    for name in ("model", "best_model", "optimizer", "rng"):
        assert converted[name] is original[name]
    assert original["best_metric"] == 2
    assert [row["validation"] for row in original["history"]] == [3, 2]
    assert "position_rmse_definition" not in original
    assert normalize_checkpoint_metrics(converted) is converted


def test_legacy_checkpoint_loading_leaves_source_bytes_unchanged(tmp_path):
    config = load_config("configs/smoke.json")
    path = tmp_path / "learned_0_best.pt"
    torch.save({"signature": experiment_signature(config), "best_metric": 3,
                "model": {"weight": torch.tensor([9.0])}}, path)
    original = path.read_bytes()
    loaded = _load_checkpoint(path, config)
    assert loaded["best_metric"] == pytest.approx(3 * math.sqrt(2))
    assert path.read_bytes() == original
    torch.testing.assert_close(loaded["model"]["weight"], torch.tensor([9.0]), rtol=0, atol=0)


def test_unknown_definition_and_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="Unknown position RMSE"):
        normalize_checkpoint_metrics({"stage": "learned", "position_rmse_definition": "unknown"})
    with pytest.raises(ValueError, match="matching shapes"):
        position_squared_error(torch.zeros(1, 3, 4), torch.zeros(2, 3, 4))
    perception = {"stage": "perception", "best_metric": 2}
    assert normalize_checkpoint_metrics(perception) is perception
