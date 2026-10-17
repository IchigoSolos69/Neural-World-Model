import numpy as np
import pytest
import torch

from nwm.data import EpisodeDataset, generate_dataset, generate_episodes, validate_dataset
from nwm.physics import PhysicsConfig, render_states, step


def state():
    return torch.tensor([[0.3, 0.3, 0.2, 0.0], [0.7, 0.3, -0.2, 0.0], [0.5, 0.8, 0.0, 0.0]])


def test_seeded_generation_and_nonoverlap():
    cfg = PhysicsConfig()
    first = generate_episodes([7, 8], 12, cfg)
    second = generate_episodes([7, 8], 12, cfg)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert np.all(np.linalg.norm(first["states"][:, 0, i, :2] - first["states"][:, 0, j, :2], axis=-1) > 2 * cfg.radius)


def test_drag_and_action_timing():
    cfg = PhysicsConfig()
    source = state()
    target, contacts = step(source, torch.tensor([0.1, -0.05]), cfg)
    expected_velocity = source[:, 2:].clone()
    expected_velocity[0] += torch.tensor([0.1, -0.05])
    expected_velocity *= np.exp(-cfg.drag * cfg.dt)
    torch.testing.assert_close(target[:, 2:], expected_velocity)
    assert not contacts.any()
    assert target[0, 0] > source[0, 0]
    assert target[0, 1] < source[0, 1]
    torch.testing.assert_close(source, state())


def test_elastic_pair_energy_momentum_and_single_contact():
    cfg = PhysicsConfig(drag=0, restitution=1)
    source = state()
    source[0, 0], source[1, 0] = 0.43, 0.57
    source[0, 2], source[1, 2] = 0.4, -0.4
    target, contacts = step(source, torch.zeros(2), cfg)
    assert contacts[0] and contacts.sum() == 1
    torch.testing.assert_close(target[:, 2:].sum(0), source[:, 2:].sum(0))
    torch.testing.assert_close((target[:, 2:] ** 2).sum(), (source[:, 2:] ** 2).sum())
    following, events = step(target, torch.zeros(2), cfg)
    assert not events.any()
    assert following[0, 2] < 0 and following[1, 2] > 0


def test_wall_restitution_and_resting_overlap():
    cfg = PhysicsConfig(drag=0, restitution=0.9)
    source = state()
    source[0, 0], source[0, 2] = 0.061, -0.3
    target, contacts = step(source, torch.zeros(2), cfg)
    assert contacts[3] and contacts.sum() == 1
    assert target[0, 2].item() == pytest.approx(0.27)
    source[0, 0], source[0, 2] = cfg.radius - 0.01, 0
    _, contacts = step(source, torch.zeros(2), cfg)
    assert not contacts.any()


def test_step_gradients_and_batch_restitution():
    source = state().repeat(2, 1, 1).requires_grad_()
    acceleration = torch.zeros((2, 3, 2), requires_grad=True)
    coeff = torch.tensor([0.7, 0.9], requires_grad=True)
    target, contacts = step(source, torch.zeros(2, 2), PhysicsConfig(), acceleration, coeff)
    target.square().sum().backward()
    assert source.grad.isfinite().all()
    assert acceleration.grad.isfinite().all()
    assert contacts.shape == (2, 15)


def test_renderer_shapes_and_geometry():
    frames = render_states(state().repeat(2, 4, 1, 1))
    assert frames.shape == (2, 4, 3, 64, 64)
    assert frames.dtype == torch.float32
    assert frames.min() >= 0 and frames.max() <= 1
    assert frames[0, 0, 0, 19, 19] > frames[0, 0, 1, 19, 19]
    assert frames[0, 0, 1, 19, 44] > frames[0, 0, 0, 19, 44]


@pytest.fixture
def dataset_config(tmp_path):
    return {"data": {"root": str(tmp_path), "seed": 3, "frames": 12, "train": 3,
                     "validation": 2, "calibration": 2, "test": 2, "ood_per_condition": 1},
            "physics": {"drag": 0.2, "restitution": 0.9}}


def test_splits_windows_and_future_alignment(dataset_config):
    root = generate_dataset(dataset_config)
    all_seeds = []
    for archive in root.glob("*.npz"):
        with np.load(archive) as data:
            all_seeds.extend(data["seeds"].tolist())
            assert "frames" not in data
    assert len(all_seeds) == len(set(all_seeds))
    ds = EpisodeDataset(root, "train", context=4, horizon=5, windows_per_episode=2)
    item = ds[0]
    assert item["context_frames"].shape == (4, 3, 64, 64)
    assert item["target_frames"].shape == (5, 3, 64, 64)
    assert item["context_actions"].shape == (3, 2)
    assert item["future_actions"].shape == (5, 2)
    predicted, contacts = step(item["initial_state"], item["future_actions"][0], PhysicsConfig())
    torch.testing.assert_close(predicted, item["target_states"][0])
    torch.testing.assert_close(contacts, item["contacts"][0])
    assert ds[1]["start"] > 0
    assert ds[2]["episode_seed"] != ds[0]["episode_seed"]
    compact = EpisodeDataset(root, "train", 4, 5, render_frames=False)[0]
    assert "context_frames" not in compact and "target_frames" not in compact
    assert compact["context_states"].shape == (4, 3, 4)


def test_existing_dataset_reuse_and_incompatible_refusal(dataset_config):
    root = generate_dataset(dataset_config)
    assert generate_dataset(dataset_config) == root
    dataset_config["physics"]["drag"] = 0.4
    with pytest.raises(ValueError, match="Incompatible"):
        generate_dataset(dataset_config)


def test_read_only_validation_rejects_identity_and_archive_changes(dataset_config):
    root = generate_dataset(dataset_config)
    metadata = (root / "metadata.json").read_bytes()
    assert validate_dataset(dataset_config) == root
    dataset_config["data"]["seed"] += 1
    with pytest.raises(ValueError, match="Incompatible"):
        validate_dataset(dataset_config)
    assert (root / "metadata.json").read_bytes() == metadata
    dataset_config["data"]["seed"] -= 1
    path = root / "train.npz"
    with np.load(path) as archive:
        arrays = dict(archive)
    arrays["contacts"] = arrays["contacts"][:, :, :14]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="contacts"):
        validate_dataset(dataset_config)
