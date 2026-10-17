"""Compact, seeded synthetic episodes and episode-isolated video windows."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .physics import PhysicsConfig, render_states, step


FORMAT_VERSION = 1
OOD_CONDITIONS = {
    "ood_drag_low": {"drag": 0.05},
    "ood_drag_high": {"drag": 0.4},
    "ood_restitution_low": {"restitution": 0.6},
    "ood_restitution_high": {"restitution": 1.0},
    "ood_speed": {"speed_scale": 1.5},
}


def _initial_conditions(seed, cfg, frames, speed_scale=1.0):
    rng = np.random.default_rng(int(seed))
    positions = []
    for _ in range(3):
        for attempt in range(10_000):
            candidate = rng.uniform(cfg.radius + 0.01, 1 - cfg.radius - 0.01, 2)
            if all(np.linalg.norm(candidate - other) > 2 * cfg.radius + 0.01 for other in positions):
                positions.append(candidate)
                break
        else:
            raise RuntimeError("could not find non-overlapping disc positions")
    angles = rng.uniform(0, 2 * np.pi, 3)
    speeds = rng.uniform(0.15, 0.6, 3) * speed_scale
    velocities = np.stack((np.cos(angles), np.sin(angles)), -1) * speeds[:, None]
    state = np.concatenate((np.asarray(positions), velocities), -1).astype(np.float32)
    actions = np.zeros((frames - 1, 2), dtype=np.float32)
    active = rng.random(frames - 1) < 0.12
    actions[active] = rng.uniform(-0.25, 0.25, (int(active.sum()), 2))
    return state, actions


def generate_episodes(seeds, frames, cfg, speed_scale=1.0):
    """Generate one split in bounded batches, with no RGB data on disk."""
    seeds = np.asarray(seeds, dtype=np.int64)
    if frames < 2:
        raise ValueError("episodes require at least two frames")
    states = np.zeros((len(seeds), frames, 3, 4), dtype=np.float32)
    actions = np.zeros((len(seeds), frames - 1, 2), dtype=np.float32)
    contacts = np.zeros((len(seeds), frames - 1, 15), dtype=np.bool_)
    for i, seed in enumerate(seeds):
        states[i, 0], actions[i] = _initial_conditions(seed, cfg, frames, speed_scale)
    with torch.no_grad():
        for begin in range(0, len(seeds), 512):
            end = min(begin + 512, len(seeds))
            current = torch.from_numpy(states[begin:end, 0])
            batch_actions = torch.from_numpy(actions[begin:end])
            for t in range(frames - 1):
                current, events = step(current, batch_actions[:, t], cfg)
                states[begin:end, t + 1] = current.numpy()
                contacts[begin:end, t] = events.numpy()
    return {"states": states, "actions": actions, "contacts": contacts, "seeds": seeds}


def _dataset_spec(config):
    data = config["data"]
    cfg = PhysicsConfig(**config.get("physics", {}))
    counts = {name: int(data[name]) for name in ("train", "validation", "calibration", "test")}
    counts.update({name: int(data.get("ood_per_condition", 100)) for name in OOD_CONDITIONS})
    if any(count < 0 for count in counts.values()) or int(data["frames"]) < 2:
        raise ValueError("split counts must be nonnegative and frames must be at least two")
    return {"version": FORMAT_VERSION, "physics": asdict(cfg), "seed": int(data["seed"]),
            "frames": int(data["frames"]), "counts": counts, "ood": OOD_CONDITIONS}


def _fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def validate_dataset(config):
    """Read-only validation of generation identity and every compact archive."""
    root = Path(config["data"]["root"]).expanduser().resolve()
    spec = _dataset_spec(config)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}; run python -m nwm generate first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != _fingerprint(spec) or metadata.get("spec") != spec:
        raise ValueError(f"Incompatible dataset at {root}; generate into another data.root")
    offset, frames = spec["seed"], spec["frames"]
    for name, count in spec["counts"].items():
        path = root / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset split {path}; rerun generation")
        expected_shapes = {"states": (count, frames, 3, 4), "actions": (count, frames - 1, 2),
                           "contacts": (count, frames - 1, 15), "seeds": (count,), "parameters": (count, 3)}
        expected_dtypes = {"states": np.float32, "actions": np.float32, "contacts": np.bool_,
                           "seeds": np.int64, "parameters": np.float32}
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in expected_shapes if key in archive}
        for key, shape in expected_shapes.items():
            if key not in arrays or arrays[key].shape != shape or arrays[key].dtype != expected_dtypes[key]:
                raise ValueError(f"Invalid {key} shape or dtype in {path}")
        if not np.array_equal(arrays["seeds"], np.arange(offset, offset + count, dtype=np.int64)):
            raise ValueError(f"Unexpected episode seeds in {path}")
        offset += count
        for key in ("states", "actions", "parameters"):
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"Nonfinite {key} in {path}")
        condition = OOD_CONDITIONS.get(name, {})
        expected_parameters = np.asarray([condition.get("drag", spec["physics"]["drag"]),
                                          condition.get("restitution", spec["physics"]["restitution"]),
                                          condition.get("speed_scale", 1.0)], dtype=np.float32)
        if not np.all(arrays["parameters"] == expected_parameters):
            raise ValueError(f"Unexpected physical parameters in {path}")
    return root


def generate_dataset(config):
    """Create or reuse compatible splits; refuse accidental incompatible reuse."""
    root = Path(config["data"]["root"]).expanduser().resolve()
    spec = _dataset_spec(config)
    cfg = PhysicsConfig(**spec["physics"])
    counts = spec["counts"]
    fingerprint = _fingerprint(spec)
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint or metadata.get("spec") != spec:
            raise ValueError(f"Incompatible dataset at {root}; choose another data.root")
    elif root.exists() and any(root.glob("*.npz")):
        raise ValueError(f"Dataset at {root} lacks metadata; choose another data.root")
    root.mkdir(parents=True, exist_ok=True)
    # Write the specification first so interrupted generation can safely resume.
    metadata = {"fingerprint": fingerprint, "spec": spec, "config": config,
                "contact_order": ["pair_01", "pair_02", "pair_12"] +
                [f"{i}_{wall}" for i in range(3) for wall in ("left", "right", "top", "bottom")]}
    temporary_metadata = metadata_path.with_suffix(".partial.json")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    offset = spec["seed"]
    for name, count in counts.items():
        seeds = np.arange(offset, offset + count, dtype=np.int64)
        offset += count
        destination = root / f"{name}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as existing:
                if existing["states"].shape != (count, spec["frames"], 3, 4) or not np.array_equal(existing["seeds"], seeds):
                    raise ValueError(f"Existing split {destination} is inconsistent with metadata")
            continue
        condition = dict(OOD_CONDITIONS.get(name, {}))
        speed_scale = condition.pop("speed_scale", 1.0)
        episode_cfg = replace(cfg, **condition)
        episodes = generate_episodes(seeds, spec["frames"], episode_cfg, speed_scale)
        episodes["parameters"] = np.tile(
            np.asarray([episode_cfg.drag, episode_cfg.restitution, speed_scale], dtype=np.float32), (count, 1)
        )
        # A temporary archive avoids treating interrupted writes as complete.
        temporary = destination.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **episodes)
        temporary.replace(destination)
    return validate_dataset(config)


class EpisodeDataset(Dataset):
    """Deterministic windows. Split membership is fixed before windowing.

    render_frames=False omits image tensors so state-only dynamics training
    avoids rasterization. context_states is always available for explicit use.
    """

    def __init__(self, root, split, context=4, horizon=20, windows_per_episode=1, render_frames=True):
        root = Path(root)
        with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
            self.states = archive["states"]
            self.actions = archive["actions"]
            self.contacts = archive["contacts"]
            self.seeds = archive["seeds"]
        if context < 1 or horizon < 1 or windows_per_episode < 1:
            raise ValueError("context, horizon, and windows_per_episode must be positive")
        self.context, self.horizon = int(context), int(horizon)
        self.windows_per_episode = int(windows_per_episode)
        self.render_frames = render_frames
        self.window_count = self.states.shape[1] - context - horizon + 1
        if self.window_count < 1:
            raise ValueError("context plus horizon exceeds episode length")
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.radius = metadata.get("spec", {}).get("physics", {}).get("radius", 0.06)

    def __len__(self):
        return len(self.seeds) * self.windows_per_episode

    def window_indices(self, index):
        """Return (episode index, first context frame) for cache alignment."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode, window = divmod(index, self.windows_per_episode)
        start = window * self.window_count // self.windows_per_episode
        return episode, start

    def __getitem__(self, index):
        episode, start = self.window_indices(index)
        last_context = start + self.context - 1
        end = last_context + self.horizon + 1
        context_states = torch.from_numpy(self.states[episode, start:last_context + 1].copy())
        target_states = torch.from_numpy(self.states[episode, last_context + 1:end].copy())
        result = {
            "context_states": context_states,
            "context_actions": torch.from_numpy(self.actions[episode, start:last_context].copy()),
            "future_actions": torch.from_numpy(self.actions[episode, last_context:end - 1].copy()),
            "initial_state": context_states[-1],
            "target_states": target_states,
            "contacts": torch.from_numpy(self.contacts[episode, last_context:end - 1].copy()),
            "episode_seed": int(self.seeds[episode]),
            "episode_index": episode,
            "start": start,
        }
        if self.render_frames:
            result["context_frames"] = render_states(context_states, radius=self.radius)
            result["target_frames"] = render_states(target_states, radius=self.radius)
        return result
