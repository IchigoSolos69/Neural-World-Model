"""Small, differentiable disc simulator shared by truth and hybrid models.

Coordinates span [0, 1], y increases downwards, and actions are instantaneous
velocity changes on the red disc at the beginning of a frame transition.
"""

from dataclasses import dataclass
import math

import torch


PAIRS = ((0, 1), (0, 2), (1, 2))
CONTACT_NAMES = tuple(f"ball_{i}_{j}" for i, j in PAIRS) + tuple(
    f"ball_{i}_{wall}" for i in range(3) for wall in ("left", "right", "top", "bottom")
)
COLORS = ((0.95, 0.18, 0.20), (0.18, 0.78, 0.36), (0.20, 0.45, 0.98))


@dataclass(frozen=True)
class PhysicsConfig:
    dt: float = 1 / 30
    substeps: int = 4
    radius: float = 0.06
    drag: float = 0.2
    restitution: float = 0.9

    def __post_init__(self):
        if self.dt <= 0 or self.substeps < 1 or not isinstance(self.substeps, int):
            raise ValueError("dt must be positive and substeps must be a positive integer")
        if not 0 < self.radius < 0.25 or self.drag < 0 or not 0 <= self.restitution <= 1:
            raise ValueError("invalid radius, drag, or restitution")


def step(state, action, cfg: PhysicsConfig, acceleration=None, restitution=None):
    """Advance [..., 3, 4] states by one frame; return state and 15 impacts.

    The solver is piecewise differentiable, including through the restitution
    coefficient and residual acceleration. Event decisions themselves are hard.
    Contacts mark approaching impacts, not stationary overlaps.
    """
    if state.shape[-2:] != (3, 4) or not state.is_floating_point():
        raise ValueError("state must be a floating tensor of shape [..., 3, 4]")
    action = torch.as_tensor(action, device=state.device, dtype=state.dtype)
    if action.shape[-1:] != (2,):
        raise ValueError("action must have shape [..., 2]")
    action = torch.broadcast_to(action, (*state.shape[:-2], 2))
    pos, vel = state[..., :2], state[..., 2:]
    zero_action = torch.zeros_like(vel[..., 1:, :])
    vel = vel + torch.cat((action.unsqueeze(-2), zero_action), dim=-2)
    acc = torch.zeros_like(vel) if acceleration is None else torch.as_tensor(
        acceleration, device=state.device, dtype=state.dtype
    )
    coeff = torch.as_tensor(
        cfg.restitution if restitution is None else restitution,
        device=state.device,
        dtype=state.dtype,
    ).clamp(0, 1)
    while coeff.ndim > len(state.shape[:-2]) and coeff.shape[-1] == 1:
        coeff = coeff.squeeze(-1)
    coeff = torch.broadcast_to(coeff, state.shape[:-2])
    contacts = torch.zeros((*state.shape[:-2], 15), device=state.device, dtype=torch.bool)
    h = cfg.dt / cfg.substeps
    decay = math.exp(-cfg.drag * h)

    def walls(p, v):
        x, y = p.unbind(-1)
        vx, vy = v.unbind(-1)
        e = coeff.unsqueeze(-1)
        left = (x <= cfg.radius) & (vx < 0)
        right = (x >= 1 - cfg.radius) & (vx > 0)
        top = (y <= cfg.radius) & (vy < 0)
        bottom = (y >= 1 - cfg.radius) & (vy > 0)
        vx = torch.where(left | right, -e * vx, vx)
        vy = torch.where(top | bottom, -e * vy, vy)
        p = torch.stack((x.clamp(cfg.radius, 1 - cfg.radius), y.clamp(cfg.radius, 1 - cfg.radius)), -1)
        return p, torch.stack((vx, vy), -1), torch.stack((left, right, top, bottom), -1).flatten(-2)

    for _ in range(cfg.substeps):
        vel = vel * decay + acc * h
        pos = pos + vel * h
        pos, vel, wall_contacts = walls(pos, vel)
        pair_contacts = []
        for i, j in PAIRS:
            delta = pos[..., j, :] - pos[..., i, :]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            normal = delta / distance.clamp_min(1e-8).unsqueeze(-1)
            fallback = torch.zeros_like(normal) + normal.new_tensor((1.0, 0.0))
            normal = torch.where((distance > 1e-8).unsqueeze(-1), normal, fallback)
            approaching = ((vel[..., j, :] - vel[..., i, :]) * normal).sum(-1)
            hit = (distance <= 2 * cfg.radius + 1e-7) & (approaching < 0)
            impulse = torch.where(hit, -(1 + coeff) * approaching / 2, torch.zeros_like(approaching))
            correction = ((2 * cfg.radius - distance).clamp_min(0) / 2).unsqueeze(-1) * normal
            pos = torch.stack([
                pos[..., k, :] - correction if k == i else
                pos[..., k, :] + correction if k == j else pos[..., k, :]
                for k in range(3)
            ], -2)
            impulse_vector = impulse.unsqueeze(-1) * normal
            vel = torch.stack([
                vel[..., k, :] - impulse_vector if k == i else
                vel[..., k, :] + impulse_vector if k == j else vel[..., k, :]
                for k in range(3)
            ], -2)
            pair_contacts.append(hit)
        # Pair separation at a wall can push a disc outside the box.
        pos, vel, corrected_wall_contacts = walls(pos, vel)
        frame_contacts = torch.cat((torch.stack(pair_contacts, -1), wall_contacts | corrected_wall_contacts), -1)
        contacts = contacts | frame_contacts
    return torch.cat((pos, vel), -1), contacts


def render_states(states, size=64, radius=0.06):
    """Rasterize fixed-color discs; preserve leading dimensions and device."""
    if size < 1 or radius <= 0:
        raise ValueError("size and radius must be positive")
    states = torch.as_tensor(states, dtype=torch.float32)
    if states.shape[-2:] != (3, 4):
        raise ValueError("states must have shape [..., 3, 4]")
    grid = (torch.arange(size, device=states.device, dtype=states.dtype) + 0.5) / size
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    pos = states[..., :2]
    distance_squared = (x - pos[..., 0, None, None]) ** 2 + (y - pos[..., 1, None, None]) ** 2
    masks = distance_squared <= radius ** 2
    colors = states.new_tensor(COLORS)
    # Highest object index paints last if an invalid predicted state overlaps.
    background = torch.full((*states.shape[:-2], 3, size, size), 0.035, device=states.device)
    for i in range(3):
        background = torch.where(masks[..., i, None, :, :], colors[i, :, None, None], background)
    return background
