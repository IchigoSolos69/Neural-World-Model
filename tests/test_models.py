"""Model/inference contracts, causality, and differentiable long rollouts."""

import inspect

import pytest
import torch
from torch import nn

from nwm.models import HybridDynamics, LearnedDynamics, StateDecoder, StateEstimator
from nwm.physics import PhysicsConfig
from nwm.predict import EnsemblePredictor, WorldModelPredictor, constant_velocity_rollout, physics_rollout


def initial_state(batch=2):
    state = torch.tensor([
        [0.20, 0.20, 0.12, -0.06],
        [0.50, 0.60, -0.08, 0.05],
        [0.80, 0.30, 0.02, 0.07],
    ])
    return state.unsqueeze(0).repeat(batch, 1, 1)


class FixedEstimator(nn.Module):
    def forward(self, frames, actions):
        return initial_state(frames.shape[0]).to(frames)


class EmptyDecoder(nn.Module):
    def forward(self, states):
        return states.new_zeros(*states.shape[:-2], 3, 64, 64)


def test_visual_models_shapes_and_backward():
    estimator, decoder = StateEstimator(), StateDecoder()
    state = estimator(torch.rand(2, 4, 3, 64, 64), torch.zeros(2, 3, 2))
    assert state.shape == (2, 3, 4)
    assert (state[..., :2] > 0).all() and (state[..., :2] < 1).all()
    frames = decoder(state[:, None].expand(-1, 2, -1, -1))
    assert frames.shape == (2, 2, 3, 64, 64)
    (frames.square().mean() + state.square().mean()).backward()
    assert estimator.encoder[0].weight.grad is not None
    assert decoder.project[0].weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for m in (estimator, decoder)
               for p in m.parameters() if p.grad is not None)


@pytest.mark.parametrize("model_type", [LearnedDynamics, HybridDynamics])
def test_50_frame_rollouts_are_differentiable_and_causal(model_type):
    torch.manual_seed(7)
    model = model_type()
    initial = initial_state().requires_grad_()
    actions = torch.zeros(2, 50, 2)
    actions[:, 20, 0] = 0.1
    prediction = model(initial, actions)
    assert prediction.shape == (2, 50, 3, 4)
    prediction.square().mean().backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert model.readout.weight.grad.abs().sum() > 0
    with torch.no_grad():
        other_actions = actions.clone()
        other_actions[:, 10:] = 0.75
        other_prediction = model(initial.detach(), other_actions)
    torch.testing.assert_close(prediction[:, :10], other_prediction[:, :10])
    if isinstance(model, HybridDynamics):
        assert model.last_residual.shape == (2, 50, 3, 2)


def test_first_future_impulse_applies_to_first_transition():
    initial = initial_state(1)
    actions = torch.zeros(1, 2, 2)
    actions[:, 0, 0] = 0.3
    cfg = PhysicsConfig(drag=0.0, restitution=1.0)
    for rollout in (constant_velocity_rollout, physics_rollout):
        prediction = rollout(initial, actions, cfg)
        expected_vx = initial[:, 0, 2] + 0.3
        torch.testing.assert_close(prediction[:, 0, 0, 2], expected_vx)
        torch.testing.assert_close(prediction[:, 0, 0, 0], initial[:, 0, 0] + expected_vx * cfg.dt)
        torch.testing.assert_close(prediction[:, 1, 0, 0], initial[:, 0, 0] + expected_vx * 2 * cfg.dt)
    with torch.no_grad():
        hybrid = HybridDynamics(cfg)
        torch.testing.assert_close(hybrid(initial, actions), physics_rollout(initial, actions, cfg))


def test_hybrid_restitution_learns_through_predicted_contact():
    model = HybridDynamics()
    initial = initial_state(1)
    initial[:, 0] = initial.new_tensor([0.061, 0.2, -0.3, 0.0])
    prediction = model(initial, torch.zeros(1, 1, 2))
    assert prediction[0, 0, 0, 2] > 0
    prediction[..., 2:].square().mean().backward()
    assert model.readout.bias.grad[6].abs() > 0
    assert torch.isfinite(model.readout.bias.grad).all()


def test_inference_shapes_no_truth_argument_and_training_modes_restored():
    estimator, decoder, dynamics = StateEstimator(), StateDecoder(), LearnedDynamics()
    estimator.train()
    estimator.encoder[0].eval()
    decoder.eval()
    dynamics.train()
    before = [m.training for root in (estimator, decoder, dynamics) for m in root.modules()]
    predictor = WorldModelPredictor(estimator, decoder, dynamics)
    result = predictor.predict(torch.zeros(1, 4, 3, 64, 64, dtype=torch.uint8), None, torch.zeros(1, 2, 2))
    assert result["states"].shape == (1, 2, 3, 4)
    assert result["frames"].shape == (1, 2, 3, 64, 64)
    assert not result["states"].requires_grad
    after = [m.training for root in (estimator, decoder, dynamics) for m in root.modules()]
    assert before == after
    assert list(inspect.signature(predictor.predict).parameters) == [
        "context_frames", "context_actions", "future_actions"
    ]


def test_ensemble_radial_population_spread():
    def shifted(offset):
        def rollout(initial, actions):
            delta = initial.new_tensor([offset, 0, 0, 0])
            return (initial + delta)[:, None].expand(-1, actions.shape[1], -1, -1)
        return rollout

    predictor = EnsemblePredictor(FixedEstimator(), EmptyDecoder(), [shifted(0.1), shifted(0.3)])
    result = predictor.predict(torch.zeros(2, 4, 3, 64, 64), torch.zeros(2, 3, 2), torch.zeros(2, 3, 2))
    assert result["members"].shape == (2, 2, 3, 3, 4)
    assert result["frames"].shape == (2, 3, 3, 64, 64)
    torch.testing.assert_close(result["position_spread"], torch.full((2, 3, 3), 0.1))
    torch.testing.assert_close(result["states"], result["members"].mean(dim=1))


@pytest.mark.parametrize("horizon", [0, 51])
def test_inference_rejects_unsupported_horizons(horizon):
    predictor = WorldModelPredictor(FixedEstimator(), EmptyDecoder(), LearnedDynamics())
    with pytest.raises(ValueError, match="horizon"):
        predictor.predict(torch.zeros(1, 4, 3, 64, 64), None, torch.zeros(1, horizon, 2))


def test_inference_rejects_nonfinite_and_unnormalized_inputs():
    predictor = WorldModelPredictor(FixedEstimator(), EmptyDecoder(), LearnedDynamics())
    frames = torch.zeros(1, 4, 3, 64, 64)
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(frames, None, torch.full((1, 1, 2), float("nan")))
    with pytest.raises(ValueError, match="RGB"):
        predictor.predict(frames + 255, None, torch.zeros(1, 1, 2))


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(11)
    model = LearnedDynamics()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    actions = torch.zeros(2, 2, 2)
    model(initial_state(), actions).square().mean().backward()
    optimizer.step()
    checkpoint = tmp_path / "resume.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1}, checkpoint)
    restored = LearnedDynamics()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    saved = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(saved["model"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(model(initial_state(), actions), restored(initial_state(), actions))
    assert saved["epoch"] == 1
    assert restored_optimizer.state_dict()["state"]
