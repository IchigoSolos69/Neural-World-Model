# Neural World Model

A small research prototype that predicts top-down ball motion from four RGB frames and known future impulses. It compares learned recurrent dynamics with an imperfect simulator, fitted physics, and learned residual physics. State labels are used during training; inference receives only video and actions.

## Run locally

On Windows, double-click `run.bat` (or the **Neural World Model** desktop shortcut) to run the smoke experiment and open its results folder. The launcher uses the project virtual environment, resumes saved training, and stops if any step fails. Run `run.bat full` from a terminal for the full experiment, or `run.bat --help` for usage.

Python **3.11 or 3.12** is recommended. From the project folder in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -e ".[dev,notebook]"
.\.venv\Scripts\python -m nwm check-device --device cuda
.\.venv\Scripts\python -m nwm generate --config configs/smoke.json
.\.venv\Scripts\python -m nwm train --config configs/smoke.json --device auto
.\.venv\Scripts\python -m nwm evaluate --config configs/smoke.json --device auto
.\.venv\Scripts\python -m nwm demo --config configs/smoke.json --device auto
.\.venv\Scripts\python -m pytest
```

The CUDA 12.8 wheels support the local RTX 5050 generation. Choose a compatible official wheel for other hardware using the [PyTorch installer](https://pytorch.org/get-started/locally/). CPU mode works for tests and small experiments. `check-device` executes forward and backward kernels, rather than checking availability alone. A system CUDA toolkit is not required for prebuilt wheels.

The tested Windows/Python 3.12 dependency versions are recorded in `requirements/windows-cu128-py312.txt`. The package supports Python 3.11+, and Colab uses its own compatible runtime.

The smoke configuration deliberately trains for only 2 perception epochs, 2 decoder epochs, and 3 dynamics epochs on 32 episodes. It verifies the pipeline and produces **undertrained demonstration artifacts**, not research performance claims. It still runs all three dynamics seeds, the full 50-frame rollout, every OOD condition, action ablations, and independent calibration.

Use the same commands with `--config configs/full.json` for the full experiment: 8,000 training, 500 model-validation, 500 calibration, and 1,000 test episodes, plus 500 episodes in each of five OOD conditions. Data generation is deterministic and resumes compatible partial generation. Training budgets are ceilings with validation-based early stopping. Full training can take substantial time; measure the smoke timings before allocating a run.

## Continue training on another machine

```powershell
python -m nwm train --config configs/full.json --resume
python -m nwm train --config configs/full.json --stage hybrid --seed 0 --resume
python tools/make_colab_bundle.py --config configs/full.json --include-artifacts
```

The bundle excludes the virtual environment and includes source, the notebook, and optionally the selected dataset and run. Upload it using the first cell of [the notebook](notebooks/neural_world_model.ipynb). In Colab, select a GPU runtime and run the notebook. Existing compatible PyTorch is retained. Paths resolve relative to the source tree, so bundled checkpoints work on Windows and Linux. [Free Colab GPU availability and runtime limits vary](https://research.google.com/colaboratory/faq.html).

Each epoch saves optimizer state, selected weights, Python/NumPy/Torch RNG states, and history. `--resume` continues from the last complete epoch. Exact continuation is tested on the same backend and configuration; different GPU architectures can introduce numerical differences. Epoch budgets may be extended, but that changes curriculum scheduling. If the selected perception checkpoint changes, existing dynamics checkpoints are rejected because their inputs have changed. Use a new run directory for a new experiment. Dataset identity and checkpoint compatibility are validated before reuse.

## Models and experiment

- **Perception:** three strided CNN layers (32/64/128 channels), a 128-dimensional embedding, and a 128-unit GRU estimate three `(x, y, vx, vy)` states from four frames and three past actions.
- **Decoder:** a shared state-to-image transposed CNN, trained on state labels with foreground-weighted pixel error. It is frozen for every dynamics comparison.
- **Constant velocity:** applies impulses and drifts without contacts or drag.
- **Imperfect physics:** uses zero drag and elastic contacts.
- **Fitted physics:** estimates global drag from contact-free training transitions and restitution from isolated wall impacts. This strong baseline reveals when a neural correction is merely recovering coefficients.
- **Learned dynamics:** a 128-unit recurrent network predicts state changes.
- **Hybrid dynamics:** a matching recurrent network predicts residual acceleration and bounded contact restitution; the shared physics solver integrates positions. Initial restitution is approximately elastic, rather than initialized to the true coefficient.

Truth uses drag `0.2 s^-1`, restitution `0.9`, radius `0.06`, a unit square, 30 FPS, and four physics substeps per frame. Actions are instantaneous velocity changes on the red ball, applied **before** the corresponding transition. All baselines receive identical actions and the same frozen visual state estimate. An additional true-initial-state evaluation separates perception from dynamics error.

The known renderer supplies training images and ground-truth frames. RGB videos are not stored in the dataset. Episode seeds are disjoint across all splits before extracting windows. Training uses fixed physics; OOD tests change drag to `0.05/0.4`, restitution to `0.6/1.0`, or initial speed to `1.5x`. These conditions are excluded from validation and calibration. The prototype tests robustness, not adaptation to unseen laws.

Residual physics is established prior work; see [APHYNITY](https://arxiv.org/abs/2010.04456). This project tests its behavior in a controlled visual setting, without claiming a novel architecture or guaranteed improvement over fitted physics.

## Inference API

```python
from nwm.config import load_config
from nwm.training import load_artifacts
from nwm.predict import EnsemblePredictor

artifacts = load_artifacts(load_config("configs/smoke.json"))
predictor = EnsemblePredictor(
    artifacts["estimator"], artifacts["decoder"],
    [model for seed, model in artifacts["hybrid"]], artifacts["device"],
)
prediction = predictor.predict(context_frames, context_actions, future_actions)
```

Inputs are `[B,4,3,64,64]` RGB frames (float `[0,1]` or uint8), `[B,3,2]` past impulses, and `[B,H,2]` future impulses for `1 <= H <= 50`. The first future impulse advances the last context frame. Zero impulses represent no intervention. Outputs include states `[B,H,3,4]`, frames `[B,H,3,64,64]`, ensemble members `[B,M,H,3,4]`, and radial position spread `[B,H,3]` in world units. Multiply distances by 64 for pixels. `WorldModelPredictor` exposes the same three-input method for an individual predictor. Future states and contact labels are never inference inputs.

## Results and interpretation

Artifacts live under `runs/<configuration>/`:

| Artifact | Contents |
|---|---|
| `checkpoints/` | Best and latest perception, decoder, and three-seed dynamics weights; optimizer/RNG/history |
| `environment.json` | Actual Python, PyTorch, device, GPU capability and memory |
| `fitted_physics.json` | Coefficients fitted exclusively from training labels |
| `evaluation/metrics.json` and `.csv` | Per-seed and aggregate ID/OOD metrics, estimated/true initialization, action ablations |
| `evaluation/calibration.json` | Separate 90% calibration multipliers for each object and horizon |
| `evaluation/error_vs_horizon.png` | Autoregressive position error through 50 frames |
| `demo/decoded_rollouts.mp4` | Ground truth and five predictors using the common learned decoder |
| `demo/state_rollouts.mp4` | The same predicted states rendered directly to reveal trajectory errors |
| `demo/final_frame.png` | Final state comparison; GIF fallback if MP4 encoding is unavailable |

State metrics are primary: pixel reconstruction can hide trajectory errors. Reports include position RMSE/ADE/FDE in pixels, velocity RMSE, collision precision/recall/F1, boundary violations, overlap, and decoded pixel MSE. True-state decoder error is reported separately. Collisions are inferred with the **same discretized swept-trajectory detector** for every predictor and matched one-to-one to recorded impacts within ±1 frame; they are not direct reads of a predictor's solver events. Runs of contact detections are deduplicated. Consequently even an exact trajectory can have detector error.

Uncertainty comes from three hybrid dynamics seeds conditional on shared perception. Held-out episodes calibrate a 0.5-pixel minimum spread separately for each fixed object at frames 10/20/50. Reports show empirical coverage and interval width, with an explicit flag if there are too few calibration episodes for the finite-sample rank. Intervals are marginal per object and horizon, not simultaneous trajectory guarantees. Distribution shift can reduce coverage; agreement between models does not establish reliability.

The no-action ablation zeros only future impulses and retains the same targets and observed history. Scientific completion means reporting the comparison honestly, including failures; hybrid improvement is a hypothesis.

Real camera footage, unknown object identities or geometry, variable object counts, policy learning, and interactive apps are outside v1.
