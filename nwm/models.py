"""Shared visual perception and autoregressive dynamics models.

States have shape ``[..., 3, 4]`` and contain ``(x, y, vx, vy)`` in
world units. Actions are instantaneous changes to the red ball's velocity.
"""

from dataclasses import replace
import math

import torch
from torch import nn

from .physics import PhysicsConfig, step


class StateEstimator(nn.Module):
    """Estimate the last state from four frames and their three transitions."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
        )
        self.temporal = nn.GRU(130, 128, batch_first=True)
        self.readout = nn.Linear(128, 12)

    def forward(self, context_frames, context_actions):
        if context_frames.ndim != 5 or context_frames.shape[1:] != (4, 3, 64, 64):
            raise ValueError("context_frames must have shape [batch, 4, 3, 64, 64]")
        batch = context_frames.shape[0]
        if context_actions.shape != (batch, 3, 2):
            raise ValueError("context_actions must have shape [batch, 3, 2]")
        embeddings = self.encoder(context_frames.reshape(-1, 3, 64, 64))
        embeddings = embeddings.reshape(batch, 4, 128)
        preceding_actions = torch.cat(
            (context_actions.new_zeros(batch, 1, 2), context_actions), dim=1
        )
        sequence, _ = self.temporal(torch.cat((embeddings, preceding_actions), dim=-1))
        raw = self.readout(sequence[:, -1]).reshape(batch, 3, 4)
        return torch.cat((raw[..., :2].sigmoid(), raw[..., 2:]), dim=-1)


class StateDecoder(nn.Module):
    """Reconstruct RGB images while preserving arbitrary leading dimensions."""

    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(12, 128 * 8 * 8), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, states):
        if states.shape[-2:] != (3, 4):
            raise ValueError("states must have trailing shape [3, 4]")
        leading = states.shape[:-2]
        features = self.project(states.reshape(-1, 12)).reshape(-1, 128, 8, 8)
        return self.decoder(features).reshape(*leading, 3, 64, 64)


def _check_rollout(initial, actions):
    if initial.ndim != 3 or initial.shape[1:] != (3, 4):
        raise ValueError("initial must have shape [batch, 3, 4]")
    if actions.ndim != 3 or actions.shape[0] != initial.shape[0] or actions.shape[2] != 2:
        raise ValueError("actions must have shape [batch, horizon, 2]")
    if actions.shape[1] < 1:
        raise ValueError("actions must contain at least one transition")


class LearnedDynamics(nn.Module):
    """A recurrent state-delta predictor with no future-state inputs."""

    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRUCell(14, 128)
        self.readout = nn.Linear(128, 12)
        nn.init.normal_(self.readout.weight, std=0.001)
        nn.init.zeros_(self.readout.bias)

    def rollout(self, initial, actions):
        _check_rollout(initial, actions)
        state = initial
        hidden = initial.new_zeros(initial.shape[0], 128)
        states = []
        for action in actions.unbind(dim=1):
            hidden = self.recurrent(torch.cat((state.flatten(1), action), dim=-1), hidden)
            state = state + self.readout(hidden).reshape_as(state)
            states.append(state)
        return torch.stack(states, dim=1)

    def forward(self, initial, actions):
        return self.rollout(initial, actions)


class HybridDynamics(nn.Module):
    """Correct an elastic, drag-free simulator with learned acceleration/contact e.

    Restitution is bounded by a sigmoid and is consumed only by contacts detected
    inside the shared solver. No recorded future contacts enter this model.
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.physics_cfg = replace(cfg or PhysicsConfig(), drag=0.0, restitution=1.0)
        self.recurrent = nn.GRUCell(14, 128)
        self.readout = nn.Linear(128, 7)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        # A finite sigmoid logit approximates the elastic prior without truth e.
        self.initial_restitution_logit = math.log(0.99 / 0.01)
        self.last_residual = None

    def rollout(self, initial, actions):
        _check_rollout(initial, actions)
        state = initial
        hidden = initial.new_zeros(initial.shape[0], 128)
        states, residuals = [], []
        for action in actions.unbind(dim=1):
            hidden = self.recurrent(torch.cat((state.flatten(1), action), dim=-1), hidden)
            correction = self.readout(hidden)
            acceleration = correction[:, :6].reshape(-1, 3, 2)
            restitution = (correction[:, 6] + self.initial_restitution_logit).sigmoid()
            state, _ = step(
                state, action, self.physics_cfg,
                acceleration=acceleration, restitution=restitution,
            )
            states.append(state)
            residuals.append(acceleration)
        self.last_residual = torch.stack(residuals, dim=1)
        return torch.stack(states, dim=1)

    def forward(self, initial, actions):
        return self.rollout(initial, actions)
