# Neural World Model

A small research prototype that predicts top-down ball motion from four RGB frames and known future impulses. It compares learned recurrent dynamics with an imperfect simulator, fitted physics, and learned residual physics. State labels are used during training; inference receives only video and actions.

## Local production beta

The local app provides **Simulation**, **Forecast** and **Backtest** workspaces at [localhost:8000](http://127.0.0.1:8000). Simulation uses the trained synthetic-disc predictors. Forecast is an experimental LTX generated continuation of uploaded video; Backtest withholds the recorded future and scores only after saving the prediction. Generated realism is not evidence of physical accuracy.

After the Python setup below, build and start the app:

```powershell
.\.venv\Scripts\python -m pip install -e ".[app,app-test]"
cd frontend
npm ci
npm test
npm run build
cd ..
python -m nwm export-serving --config configs/smoke.json --output D:/NeuralWorldModel/bundles/smoke
run.bat app
```

Use the project virtual environment for all `python` commands. The **Neural World Model App** desktop shortcut starts the same launcher. `configs/app.json` controls storage and bundle locations. The checked-in configuration uses CPU inference so simulations can run while the full experiment occupies the GPU. The served smoke bundle is explicitly undertrained. Publish the full bundle only after full training, evaluation and calibration finish:

```powershell
python -m nwm experiment-status --config configs/full.json
python -m nwm export-serving --config configs/full.json --output D:/NeuralWorldModel/bundles/full
```

Exports are immutable, checksum-verified, dataset-free bundles. Their manifest records validation selection, training completion and calibration provenance. Publish a new directory and update the app config to change an existing bundle. Live training directories are never served directly.

Jobs survive restart in SQLite. One supervised process owns inference, with four queued jobs, cancellation, a 20-minute timeout and recovery of interrupted jobs. Failed/interrupted jobs can be retried. Results include actual comparison videos, curves, measurements and calibrated synthetic position radii. Forecast has no accuracy score without recorded targets and no calibrated uncertainty estimate. Simulation radii are conditional on shared perception; changed physics can invalidate coverage.

The app binds only to `127.0.0.1`. It has no authentication, remote access or multi-user deployment support. Uploads accept MP4/WebM up to 100 MiB, 30 seconds and 1080p (including portrait). Prefixes are sampled causally at 16 FPS and aspect-preserving letterboxing is recorded. Simulation actions use frame 1 for the transition immediately after the last observed frame. Real-video generation does not support interventions.

Managed files stay inside `D:/NeuralWorldModel`, capped at 27 GiB with 8 GiB free reserved. Deleting the last job referencing an upload also reclaims that upload. Unreferenced uploads can be deleted at `DELETE /api/uploads/{id}`. No automatic model download or cloud execution occurs.

### Experimental video prerequisites

The candidate is [LTX-Video 2B 0.9.8 distilled](https://huggingface.co/Lightricks/LTX-Video), using pinned file hashes, BF16 and sequential CPU offload. No prompt enhancer or upscaler is loaded. A 17-frame observed prefix predicts the next 10/20/50 frames. Advertised presets are 384×256 and a 288×192 fallback. Local generation remains disabled until three consecutive 50-frame development runs pass the real hardware gate. Memory fit, latency and physical accuracy are **not yet verified** on the RTX 5050.

```powershell
python -m nwm video-check --config configs/app.json
python -m pip install -e ".[video]"
python -m nwm video-download --config configs/app.json
python -m nwm video-feasibility --config configs/app.json --prefix dev1/prefix.npz --prefix dev2/prefix.npz --prefix dev3/prefix.npz --output D:/NeuralWorldModel/models/ltx-2b-0.9.8/feasibility.json
```

The assets require about **23.65 GiB**, plus temporary/output space. The last local preflight refused the download because it would violate the 8 GiB free-space reserve. Add free space or choose a larger managed drive before downloading. The preflight report records exact byte counts. Stop training before the GPU feasibility test; its wall-clock watchdog terminates and reaps timed-out processes.

For a separate GPU machine, select **Export notebook job** in Forecast/Backtest and use [the manual video notebook](notebooks/video_job.ipynb). Download the prepared job, run the notebook yourself, then import its result into the same waiting job. The exchange contains only the prefix and job identity; recorded targets remain local. Imports verify job ID, revision, prefix, seed, dimensions and checksum. Runtime selection and billing remain manual; no cloud runtime has been started for this feature. CUDA BF16 is required, so a T4 is unsupported by this adapter.

### Public evaluation mini set

```powershell
python -m nwm prepare-real-data --config configs/app.json
python -m nwm prepare-real-data --config configs/app.json --download
python -m nwm benchmark-video --config configs/app.json --manifest D:/NeuralWorldModel/datasets/physics-iq-mini/manifest.json --output D:/NeuralWorldModel/evaluation/physics-iq-mini --split test
```

This adapter selects three development and eight test scenarios from [Physics-IQ Original](https://github.com/google-deepmind/physics-IQ-benchmark), using SHA256 scenario-name ordering, center view and take 1. GCS object generations and checksums pin the downloads. The first command records metadata; downloading is explicit. Source clips are normalized to at most 30 seconds at 384×256/16 FPS. Only the 17 samples ending at t=3 seconds condition each 50-frame rollout. The development split is for hardware feasibility; test clips never train, select or calibrate the model. Results compare pixel MSE against frame holding with one seed, not an official Physics-IQ score or a best-of-N result. The upstream dataset is CC BY 4.0; retain its manifest attribution when sharing clips.

Run `python -m pytest` and `npm test`/`npm run build` in `frontend` to check the pipeline and app. CI runs Python 3.11 CPU tests and the frontend checks. Windows production validation also covers dataset-free inference, request bounds, queue isolation, crash recovery, notebook identity, causal sampling and hostile array headers. Actual large-model inference and the public mini benchmark remain pending the storage and hardware gate.

## Run locally

On Windows, double-click `run.bat` (or the **Neural World Model** desktop shortcut) to run the smoke experiment and open its results folder. The launcher uses the project virtual environment, resumes saved training, and stops if any step fails. Run `run.bat full` from a terminal for the full experiment, or `run.bat --help` for usage.

Python **3.11 or 3.12** is recommended. From the project folder in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -e ".[dev,notebook,app,app-test]"
.\.venv\Scripts\python -m nwm check-device --device cuda
.\.venv\Scripts\python -m nwm generate --config configs/smoke.json
.\.venv\Scripts\python -m nwm train --config configs/smoke.json --device auto
.\.venv\Scripts\python -m nwm evaluate --config configs/smoke.json --device auto
.\.venv\Scripts\python -m nwm demo --config configs/smoke.json --device auto
.\.venv\Scripts\python -m pytest
```

The CUDA 12.8 wheels support the local RTX 5050 generation. Choose a compatible official wheel for other hardware using the [PyTorch installer](https://pytorch.org/get-started/locally/). CPU mode works for tests and small experiments. `check-device` executes forward and backward kernels, rather than checking availability alone. A system CUDA toolkit is not required for prebuilt wheels.

The tested Windows/Python 3.12 dependency versions are recorded in `requirements/windows-cu128-py312.txt`. The package supports Python 3.11+, and Colab uses its own compatible runtime.

Configuration files may be stored anywhere. Relative `data.root` and `outputs.root` paths always resolve against this source tree; explicit absolute paths are preserved.

The smoke configuration deliberately trains for only 2 perception epochs, 2 decoder epochs, and 3 dynamics epochs on 32 episodes. It verifies the pipeline and produces **undertrained demonstration artifacts**, not research performance claims. It still runs all three dynamics seeds, the full 50-frame rollout, every OOD condition, action ablations, and independent calibration.

Use the same commands with `--config configs/full.json` for the full experiment: 8,000 training, 500 model-validation, 500 calibration, and 1,000 test episodes, plus 500 episodes in each of five OOD conditions. Data generation is deterministic and resumes compatible partial generation. Training budgets are ceilings with validation-based early stopping. Full training can take substantial time; measure the smoke timings before allocating a run.

## Continue training on another machine

```powershell
python -m nwm train --config configs/full.json --resume
python -m nwm train --config configs/full.json --stage hybrid --seed 0 --resume
python tools/make_colab_bundle.py --config configs/full.json --include-artifacts
```

The bundle excludes the virtual environment and includes source, the notebook, and optionally the selected dataset and run. Upload it using the first cell of [the notebook](notebooks/neural_world_model.ipynb). If that picker is unavailable, upload `nwm_colab.zip` through Colab's Files sidebar and rerun the first cell. In Colab, select a GPU runtime and run the notebook. Existing compatible PyTorch is retained. Paths resolve relative to the source tree, so bundled checkpoints work on Windows and Linux. [Free Colab GPU availability and runtime limits vary](https://research.google.com/colaboratory/faq.html).

Each epoch saves optimizer state, selected weights, Python/NumPy/Torch RNG states, and history. `--resume` continues from the last complete epoch. Exact continuation is tested on the same backend and configuration; different GPU architectures can introduce numerical differences. Epoch budgets may be extended, but that changes curriculum scheduling. If the selected perception checkpoint changes, existing dynamics checkpoints are rejected because their inputs have changed. Use a new run directory for a new experiment. Dataset identity and checkpoint compatibility are validated before reuse.

Perception checkpoints now use normalized position-and-velocity error for selection. Older checkpoints selected only by position remain usable for inference, but cannot resume perception training under the new criterion. Preserve the old run and use a new `outputs.root` to retrain all stages; do not mix a new estimator with old dynamics weights.

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

Inference needs the configuration, selected checkpoints, and `fitted_physics.json`; dataset archives are not required. Training, evaluation, and demonstration commands still validate their datasets before use.

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

The synthetic research track does not handle real camera footage, unknown object identities or geometry, variable object counts or policy learning. The separate experimental video adapter generates continuations without claiming those scientific capabilities.

## Evidence-first research release

Position RMSE now has one definition everywhere: `sqrt(mean(dx_px² + dy_px²))`, pooled over objects, episodes and frames through the reported horizon. Historical dynamics checkpoints used coordinate RMSE, smaller by `sqrt(2)`. Loading those checkpoints converts only validation metadata in memory and records the conversion; model weights, optimizer/RNG state and source files stay unchanged. Resumption continues with the same selected epoch and standard units. The current running process keeps its already-loaded code and is left undisturbed.

Generate enhanced evidence in a **fresh** directory using existing checkpoints, without training:

```powershell
python -m nwm evaluate --config configs/smoke.json --device cpu --output runs/smoke/evaluation-evidence-v2
python -m nwm report --evaluation runs/smoke/evaluation-evidence-v2
```

An existing `--output` directory is refused, preserving previous or interrupted evaluations. Without `--output`, evaluation retains its original output location and aggregate behavior. Enhanced evaluation freezes checkpoint/data hashes, saves compact episode archives and truth-defined contact/impulse strata, and generates `report.html`, plots, `statistical_comparisons.json`, and median/p90/worst-case videos. Gallery ranks use hybrid-ensemble 50-frame prefix RMSE, nearest-rank percentiles and episode-ID tie breaking; the learned video panel always uses the first configured seed. They are descriptive examples, never model selection.

The primary comparison is hybrid ensemble minus fitted physics at 50 frames on ordinary test physics, estimated initialization and recorded actions. Negative differences favor hybrid. Confidence intervals use 2,000 paired whole-episode bootstrap resamples with seed `20261019`; each resample recomputes pooled RMSE. Comparisons across training seeds share resampled episode IDs. Seed dispersion is reported separately, and episode intervals are conditional on the trained models. Secondary comparisons are exploratory, without multiplicity correction. Small groups are flagged rather than presented as strong evidence. Per-object coverage uses 95% Wilson intervals across independent held-out episodes, remaining informative for all-covered/all-missed samples; width intervals use the episode bootstrap. Calibration remains separate, per object/horizon; changed-physics coverage is measured without recalibrating on those tests.

`report` reads saved measurements only. It verifies their hashes and can regenerate HTML, JSON and figures without model weights, the source dataset, inference, or a GPU. Keep the HTML beside its JSON, `episodes/`, `gallery/` and `report_plots/` folders when moving it. Smoke reports are explicitly undertrained pipeline demonstrations.

For the existing full experiment, queue one finite release after its train/evaluate/demo pipeline exits successfully:

```powershell
python -m nwm release-research --config configs/full.json --device cuda --output runs/full/evaluation-evidence-v2 --bundle D:/NeuralWorldModel/bundles/full --wait
```

This command never starts or resumes training and never uses paid cloud resources. It waits without holding the GPU lock, then freezes inputs, runs enhanced evaluation once, rebuilds the report, and exports an evidence-bound bundle. Every public predictor is compared with the source on one deterministic 50-frame validation context, including states, decoded frames and ensemble spread. The verified staging directory is renamed to the final bundle only after parity passes. The smoke bundle stays available in the app; Refresh discovers the full bundle.

Release heartbeat/status, immutable `freeze.json` and `parity.json` live in the sibling `.evaluation-evidence-v2.research-release/` directory. Rerunning the same command can finish publication from complete evidence without repeating inference. Incomplete evaluations, changed inputs, unrelated existing bundles and failed original experiments are refused and preserved for inspection. A different output directory is required to intentionally repeat an incomplete evaluation.

To bind an independently completed evaluation explicitly:

```powershell
python -m nwm export-serving --config configs/full.json --evaluation runs/full/evaluation-evidence-v2 --output D:/NeuralWorldModel/bundles/full
```

The fitted baseline uses the exact simulator force-law family. It may win; a hybrid improvement is not an acceptance requirement. No new force laws, architectures, datasets or training seeds are introduced by this evidence release. General-video hardware feasibility remains a separate, unverified gate.
