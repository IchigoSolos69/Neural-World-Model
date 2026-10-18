"""Labeled ground-truth and five-predictor rollout videos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .data import EpisodeDataset, validate_dataset
from .evaluation import rollout_predictions
from .physics import PhysicsConfig, render_states


def _panels(videos: list[torch.Tensor], titles: list[str], context: int = 4) -> list[np.ndarray]:
    arrays = [(video.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
              for video in videos]
    width, height, header, footer = 192, 192, 28, 26
    frames = []
    for timestep in range(len(arrays[0])):
        canvas = Image.new("RGB", (3 * width, 2 * (height + header) + footer), (16, 19, 24))
        draw = ImageDraw.Draw(canvas)
        for panel, (array, title) in enumerate(zip(arrays, titles)):
            x, y = (panel % 3) * width, (panel // 3) * (height + header)
            draw.text((x + 7, y + 8), title, fill=(245, 245, 245))
            image = Image.fromarray(array[timestep]).resize((width, height), resample=Image.Resampling.NEAREST)
            canvas.paste(image, (x, y + header))
        label = (f"Observed context {timestep + 1}/{context}" if timestep < context else
                 f"Predicted future frame {timestep - context + 1} | recorded impulses supplied")
        draw.text((8, canvas.height - footer + 7), label, fill=(220, 225, 232))
        frames.append(np.asarray(canvas))
    return frames


def _save_video(frames: list[np.ndarray], destination: Path, fps: int = 30) -> tuple[Path, str | None]:
    import imageio.v2 as imageio

    try:
        imageio.mimsave(destination, frames, fps=fps, codec="libx264", quality=8,
                        macro_block_size=1, output_params=["-movflags", "+faststart"])
        return destination, None
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        # A GIF preserves the demonstration when an ffmpeg binary is unavailable.
        destination.unlink(missing_ok=True)
        fallback = destination.with_suffix(".gif")
        imageio.mimsave(fallback, frames, duration=1000 / fps, loop=0)
        return fallback, f"MP4 unavailable; used GIF: {type(error).__name__}: {error}"


@torch.inference_mode()
def make_demo(config: dict, device: str = "auto") -> Path:
    from .training import load_artifacts

    validate_dataset(config)
    artifacts = load_artifacts(config, device)
    cfg = PhysicsConfig(**config["physics"])
    horizon = max(config["evaluation"]["horizons"])
    dataset = EpisodeDataset(config["data"]["root"], "test", context=4, horizon=horizon, render_frames=True)
    sample = dataset[0]
    target_device = artifacts["device"]
    context = sample["context_frames"].to(target_device)
    past = sample["context_actions"].to(target_device)
    actions = sample["future_actions"].to(target_device)
    initial = artifacts["estimator"](context[None], past[None])
    predictions = rollout_predictions(artifacts, initial, actions[None], cfg)
    learned = next(row for row in predictions if row["model"] == "learned")
    chosen = [next(row for row in predictions if row["model"] == name)
              for name in ("constant_velocity", "imperfect_physics", "fitted_physics")]
    chosen.extend((learned, next(row for row in predictions if row["model"] == "hybrid_ensemble")))
    titles = ["Ground truth", "Constant velocity", "Imperfect physics", "Fitted physics",
              f"Learned (seed {learned['seed']})", "Hybrid ensemble"]
    ground_truth = torch.cat((context, sample["target_frames"].to(target_device)))
    decoded, rendered = [ground_truth], [ground_truth]
    for row in chosen:
        states = row["states"][0]
        # Every model receives the same learned decoder for the main video.
        frames = artifacts["decoder"](states)
        decoded.append(torch.cat((context, frames)))
        rendered.append(torch.cat((context, render_states(states, radius=cfg.radius))))
    output = Path(config["outputs"]["root"]) / "demo"
    output.mkdir(parents=True, exist_ok=True)
    decoded_path, decoded_note = _save_video(_panels(decoded, titles), output / "decoded_rollouts.mp4")
    state_path, state_note = _save_video(_panels(rendered, titles), output / "state_rollouts.mp4")
    Image.fromarray(_panels(rendered, titles)[-1]).save(output / "final_frame.png")
    manifest = {"episode_seed": sample["episode_seed"], "split": "test", "horizon": horizon,
                "context_frames": 4, "initialization": "shared visual estimator",
                "decoded_video": str(decoded_path), "state_video": str(state_path),
                "learned_seed": learned["seed"], "hybrid_seeds": [seed for seed, _ in artifacts["hybrid"]],
                "notes": [note for note in (decoded_note, state_note) if note],
                "interpretation": "Decoded video includes common decoder error. State video uses the known renderer "
                                  "to expose predicted trajectories. Four observed frames precede predictions."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output
