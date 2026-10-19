import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from nwm.config import experiment_signature, load_config
from nwm.training import (_fit_stage, _rng_state, _restore_rng, curriculum_horizon,
                          device_check, normalized_state_loss, seed_everything)


def test_curriculum_contains_all_stages():
    assert [curriculum_horizon(epoch, 3) for epoch in range(3)] == [1, 10, 20]
    assert [curriculum_horizon(epoch, 6) for epoch in range(6)] == [1, 1, 10, 10, 20, 20]


@pytest.mark.parametrize("locked_attempts", [2, 5])
def test_atomic_checkpoint_replace_retries_without_losing_previous_file(tmp_path, monkeypatch, locked_attempts):
    import nwm.training as training
    path = tmp_path / "checkpoint.pt"
    training._atomic_save({"value": 1}, path)
    original_bytes = path.read_bytes()
    replace = training.os.replace
    attempts, delays = [], []
    def temporarily_locked(source, destination):
        attempts.append(1)
        if len(attempts) <= locked_attempts:
            assert path.read_bytes() == original_bytes
            raise PermissionError("temporary Windows file lock")
        return replace(source, destination)
    monkeypatch.setattr(training.os, "replace", temporarily_locked)
    monkeypatch.setattr("nwm.files.sleep", delays.append)
    if locked_attempts == 5:
        with pytest.raises(PermissionError):
            training._atomic_save({"value": 2}, path)
        assert path.read_bytes() == original_bytes
        assert len(attempts) == 5 and len(delays) == 4
    else:
        training._atomic_save({"value": 2}, path)
        assert torch.load(path, weights_only=True) == {"value": 2}
        assert len(attempts) == 3 and delays == [.02, .04]


def test_config_resolves_paths_and_rejects_short_episodes(tmp_path):
    config = load_config("configs/smoke.json")
    assert Path(config["data"]["root"]).is_absolute()
    changed = copy.deepcopy(config)
    changed["outputs"]["root"] = "elsewhere"
    changed["training"]["dynamics_epochs"] += 1
    assert experiment_signature(changed) == experiment_signature(config)
    changed["data"]["seed"] += 1
    assert experiment_signature(changed) != experiment_signature(config)
    config["data"]["frames"] = 30
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="54 frames"):
        load_config(path)


def test_portable_rng_roundtrip():
    seed_everything(13)
    state = _rng_state()
    expected = (torch.rand(4), np.random.rand(4))
    _restore_rng(state)
    torch.testing.assert_close(torch.rand(4), expected[0])
    np.testing.assert_array_equal(np.random.rand(4), expected[1])
    assert device_check("cpu")["device"] == "cpu"


class TinyWindows(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"estimate": torch.zeros(3, 4), "future_actions": torch.ones(20, 2) * (index + 1),
                "target_states": torch.ones(20, 3, 4) * 0.2}


class TinyDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.randn(()))

    def forward(self, initial, actions):
        return (initial[:, None] + self.value * actions[:, :, :1, None]).expand(-1, -1, 3, 4)


@pytest.mark.parametrize("interrupt_suffix", ["latest.pt", "best.pt"])
@pytest.mark.parametrize("legacy_metrics", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_epoch_boundary_resume_matches_uninterrupted(tmp_path, monkeypatch, interrupt_suffix, legacy_metrics, device):
    config = load_config("configs/smoke.json")
    config["outputs"]["root"] = str(tmp_path / "resumed")
    config["training"].update(dynamics_epochs=3, batch_size=2, patience=20)
    dataset = TinyWindows()
    import nwm.training as training
    real_save = training._atomic_save

    def interrupt_after_first_best(value, path):
        real_save(value, path)
        if value["epoch"] == 0 and str(path).endswith(interrupt_suffix):
            raise KeyboardInterrupt("simulated interruption after epoch checkpoint")

    seed_everything(17)
    model = TinyDynamics()
    with monkeypatch.context() as patch:
        patch.setattr(training, "_atomic_save", interrupt_after_first_best)
        with pytest.raises(KeyboardInterrupt):
            _fit_stage(config, model, "learned", 17, dataset, dataset, device, False)
    if legacy_metrics:
        for path in (tmp_path / "resumed/checkpoints").glob("*.pt"):
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint["position_rmse_definition"]
            checkpoint["best_metric"] /= 2 ** 0.5
            for row in checkpoint["history"]:
                row["validation"] /= 2 ** 0.5
            torch.save(checkpoint, path)
    resumed = _fit_stage(config, TinyDynamics(), "learned", 17, dataset, dataset, device, True)
    seed_everything(17)
    config["outputs"]["root"] = str(tmp_path / "continuous")
    continuous = _fit_stage(config, TinyDynamics(), "learned", 17, dataset, dataset, device, False)
    torch.testing.assert_close(resumed.value, continuous.value, rtol=0, atol=0)
    resume_checkpoint = torch.load(tmp_path / "resumed/checkpoints/learned_17_latest.pt", weights_only=True)
    full_checkpoint = torch.load(tmp_path / "continuous/checkpoints/learned_17_latest.pt", weights_only=True)
    assert [row["loss"] for row in resume_checkpoint["history"]] == [row["loss"] for row in full_checkpoint["history"]]
    assert [row["validation"] for row in resume_checkpoint["history"]] == pytest.approx(
        [row["validation"] for row in full_checkpoint["history"]])
    assert resume_checkpoint["best_epoch"] == full_checkpoint["best_epoch"]
    assert resume_checkpoint["position_rmse_definition"] == "euclidean_position_rmse_px_v1"
    torch.testing.assert_close(resume_checkpoint["rng"]["torch"], full_checkpoint["rng"]["torch"], rtol=0, atol=0)
    for key, value in resume_checkpoint["optimizer"]["state"][0].items():
        torch.testing.assert_close(value, full_checkpoint["optimizer"]["state"][0][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="different perception"):
        _fit_stage(config, TinyDynamics(), "learned", 17, dataset, dataset, device, True,
                   perception_hash="retrained-perception")


def test_normalized_state_loss_checks_both_components():
    prediction = torch.zeros(1, 3, 4, requires_grad=True)
    loss = normalized_state_loss(prediction, torch.ones_like(prediction))
    loss.backward()
    assert (prediction.grad != 0).all()


def test_early_stopping_gives_new_curriculum_stage_its_patience(tmp_path, monkeypatch):
    config = load_config("configs/smoke.json")
    config["outputs"]["root"] = str(tmp_path)
    config["training"].update(dynamics_epochs=9, patience=2)
    import nwm.training as training
    monkeypatch.setattr(training, "_batch_loss", lambda model, *args: (model.value.square(), model.value.new_tensor(1.0)))
    _fit_stage(config, TinyDynamics(), "learned", 0, TinyWindows(), TinyWindows(), "cpu", False)
    checkpoint = torch.load(tmp_path / "checkpoints/learned_0_latest.pt", weights_only=True)
    assert checkpoint["stopped_early"]
    assert sum(row["horizon"] == 20 for row in checkpoint["history"]) == 2
