"""Inference API and the two parameter-free dynamics baselines."""

from contextlib import contextmanager

import torch
from torch import nn

from .models import _check_rollout
from .physics import PhysicsConfig, step


def physics_rollout(initial, actions, cfg=None):
    """Roll out supplied physics; fitted coefficients use this same function."""
    _check_rollout(initial, actions)
    cfg = cfg or PhysicsConfig()
    states = []
    state = initial
    for action in actions.unbind(dim=1):
        state, _ = step(state, action, cfg)
        states.append(state)
    return torch.stack(states, dim=1)


def constant_velocity_rollout(initial, actions, cfg=None):
    """Apply recorded impulses, then drift without drag or contact response."""
    _check_rollout(initial, actions)
    cfg = cfg or PhysicsConfig()
    state = initial
    states = []
    for action in actions.unbind(dim=1):
        velocity_delta = torch.cat((action[:, None], torch.zeros_like(state[:, 1:, 2:])), dim=1)
        velocity = state[..., 2:] + velocity_delta
        state = torch.cat((state[..., :2] + cfg.dt * velocity, velocity), dim=-1)
        states.append(state)
    return torch.stack(states, dim=1)


@contextmanager
def _evaluation_mode(modules):
    # Restore each nested module's original mode, including selectively frozen parts.
    unique = {id(child): child for module in modules if isinstance(module, nn.Module)
              for child in module.modules()}
    original = [(module, module.training) for module in unique.values()]
    try:
        for module, _ in original:
            module.training = False
        with torch.inference_mode():
            yield
    finally:
        for module, training in original:
            module.training = training


def _inputs(context_frames, context_actions, future_actions, device):
    frames = torch.as_tensor(context_frames, device=device)
    is_byte = frames.dtype == torch.uint8
    frames = frames.to(dtype=torch.float32)
    if is_byte:
        frames = frames / 255.0
    if frames.ndim != 5 or frames.shape[1:] != (4, 3, 64, 64) or frames.shape[0] < 1:
        raise ValueError("context_frames must have shape [batch, 4, 3, 64, 64]")
    batch = frames.shape[0]
    past = (torch.zeros(batch, 3, 2, device=device) if context_actions is None else
            torch.as_tensor(context_actions, dtype=torch.float32, device=device))
    future = torch.as_tensor(future_actions, dtype=torch.float32, device=device)
    if past.shape != (batch, 3, 2):
        raise ValueError("context_actions must have shape [batch, 3, 2]")
    if future.ndim != 3 or future.shape[0] != batch or future.shape[2] != 2:
        raise ValueError("future_actions must have shape [batch, horizon, 2]")
    if not 1 <= future.shape[1] <= 50:
        raise ValueError("prediction horizon must be between 1 and 50")
    if not all(torch.isfinite(value).all().item() for value in (frames, past, future)):
        raise ValueError("prediction inputs must be finite")
    if frames.min().item() < 0 or frames.max().item() > 1:
        raise ValueError("frames must be float RGB in [0, 1], or uint8 RGB")
    return frames, past, future


class WorldModelPredictor:
    """Predict from video and actions; returned tensors remain on ``device``."""

    def __init__(self, estimator, decoder, dynamics, device="cpu"):
        self.device = torch.device(device)
        self.estimator = estimator.to(self.device)
        self.decoder = decoder.to(self.device)
        self.dynamics = dynamics.to(self.device) if isinstance(dynamics, nn.Module) else dynamics

    def predict(self, context_frames, context_actions, future_actions):
        frames, past, future = _inputs(context_frames, context_actions, future_actions, self.device)
        with _evaluation_mode((self.estimator, self.decoder, self.dynamics)):
            initial = self.estimator(frames, past)
            states = self.dynamics(initial, future)
            return {"states": states, "frames": self.decoder(states)}


class EnsemblePredictor:
    """Share perception, then average independent dynamics rollouts.

    ``members`` has shape ``[batch, members, horizon, 3, 4]``.
    ``position_spread`` is the population radial RMS spread, in world units,
    with shape ``[batch, horizon, 3]``. Calibration applies its own spread floor.
    """

    def __init__(self, estimator, decoder, list_dynamics, device="cpu"):
        if not list_dynamics:
            raise ValueError("the ensemble needs at least one dynamics model")
        self.device = torch.device(device)
        self.estimator = estimator.to(self.device)
        self.decoder = decoder.to(self.device)
        self.dynamics = [model.to(self.device) if isinstance(model, nn.Module) else model
                         for model in list_dynamics]

    def predict(self, context_frames, context_actions, future_actions):
        frames, past, future = _inputs(context_frames, context_actions, future_actions, self.device)
        with _evaluation_mode((self.estimator, self.decoder, *self.dynamics)):
            initial = self.estimator(frames, past)
            members = torch.stack([model(initial, future) for model in self.dynamics], dim=1)
            states = members.mean(dim=1)
            squared_radius = (members[..., :2] - states[:, None, ..., :2]).square().sum(dim=-1)
            return {
                "states": states,
                "frames": self.decoder(states),
                "position_spread": squared_radius.mean(dim=1).sqrt(),
                "members": members,
            }
