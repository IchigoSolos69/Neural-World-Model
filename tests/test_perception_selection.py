import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from nwm.config import load_config
from nwm.training import _fit_stage, _load_checkpoint, normalized_state_loss


class VelocityWindows(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        state = torch.tensor([[0.25, 0.25, 0.0, 0.0],
                              [0.75, 0.25, 0.0, 0.0],
                              [0.50, 0.75, 0.0, 0.0]])
        return {"context_states": state.repeat(4, 1, 1), "initial_state": state,
                "context_actions": torch.zeros(3, 2), "future_actions": torch.zeros(20, 2)}


class VelocityEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = nn.Parameter(torch.tensor(0.4))

    def forward(self, frames, actions):
        positions = frames.new_tensor([[0.25, 0.25], [0.75, 0.25], [0.50, 0.75]])
        state = torch.cat((positions, self.velocity.expand(3, 2)), dim=-1)
        return state.expand(len(frames), -1, -1)


def perception_config(tmp_path, epochs):
    config = load_config("configs/smoke.json")
    config["outputs"]["root"] = str(tmp_path)
    config["training"].update(perception_epochs=epochs, batch_size=1, learning_rate=0.1, patience=10)
    return config


def test_velocity_only_improvement_selects_latest_perception_checkpoint(tmp_path):
    config = perception_config(tmp_path, epochs=3)
    windows = VelocityWindows()
    selected = _fit_stage(config, VelocityEstimator(), "perception", 0, windows, windows, "cpu", False)
    checkpoint = _load_checkpoint(tmp_path / "checkpoints/perception_best.pt", config)
    assert checkpoint["selection_metric"] == "normalized_state_mse"
    assert checkpoint["best_epoch"] == 2
    metrics = [row["validation"] for row in checkpoint["history"]]
    assert metrics[0] > metrics[1] > metrics[2]
    prediction = selected(torch.zeros(1, 4, 3, 64, 64), torch.zeros(1, 3, 2))
    target = windows[0]["initial_state"].unsqueeze(0)
    torch.testing.assert_close(prediction[..., :2], target[..., :2], rtol=0, atol=0)
    assert checkpoint["best_metric"] == pytest.approx(normalized_state_loss(prediction, target).item())
    # New metric checkpoints retain ordinary epoch-boundary resume support.
    config["training"]["perception_epochs"] = 4
    _fit_stage(config, VelocityEstimator(), "perception", 0, windows, windows, "cpu", True)
    assert _load_checkpoint(tmp_path / "checkpoints/perception_latest.pt", config)["epoch"] == 3


@pytest.mark.parametrize("legacy_metric", [None, "position_rmse_px"])
def test_legacy_perception_resume_is_refused_without_changing_checkpoints(tmp_path, legacy_metric):
    config = perception_config(tmp_path, epochs=1)
    windows = VelocityWindows()
    _fit_stage(config, VelocityEstimator(), "perception", 0, windows, windows, "cpu", False)
    paths = [tmp_path / "checkpoints" / f"perception_{suffix}.pt" for suffix in ("latest", "best")]
    for path in paths:
        checkpoint = _load_checkpoint(path, config)
        if legacy_metric is None:
            del checkpoint["selection_metric"]
        else:
            checkpoint["selection_metric"] = legacy_metric
        torch.save(checkpoint, path)
    original = [path.read_bytes() for path in paths]
    config["training"]["perception_epochs"] = 2
    with pytest.raises(ValueError, match="selection metric.*new outputs.root"):
        _fit_stage(config, VelocityEstimator(), "perception", 0, windows, windows, "cpu", True)
    assert [path.read_bytes() for path in paths] == original
    # Inference remains able to read legacy weights without resuming training.
    model = VelocityEstimator()
    model.load_state_dict(_load_checkpoint(paths[1], config)["model"])
