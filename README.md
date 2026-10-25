# Neural World Model

A reproducible research prototype that observes **four RGB frames** and predicts the next **10, 20, or 50 frames** under supplied actions in a synthetic top-down scene of three equal-mass, equal-radius colored discs. It compares learned recurrent dynamics with constant velocity, an imperfect simulator, fitted physics, and learned residual physics. State labels are used during training; ordinary inference receives only video and actions.

The research question is: **Can learned corrections improve predictions from an imperfect physics model?** Fitted physics uses the simulator's actual force-law family and is a strong baseline. Hybrid improvement is an experimental hypothesis, not a promised result or an acceptance requirement.

The repository also includes a local FastAPI/React app and a separate experimental real-video continuation adapter. Generated video realism does not establish physical understanding.

## Contents

- [Current status and verified evidence](#current-status-and-verified-evidence)
- [Read-only diagnostics](#read-only-diagnostics)
- [Local production beta](#local-production-beta)
- [Run locally](#run-locally)
- [Configurations and dataset](#configurations-and-dataset)
- [Continue training on another machine](#continue-training-on-another-machine)
- [Models and experiment](#models-and-experiment)
- [Training protocol](#training-protocol)
- [Inference API](#inference-api)
- [Results and interpretation](#results-and-interpretation)
- [Evidence-first research release](#evidence-first-research-release)
- [Command reference](#command-reference)
- [Repository layout](#repository-layout)
- [Validation and troubleshooting](#validation-and-troubleshooting)
- [Scope and references](#scope-and-references)

## Current status and verified evidence

**Checked on October 25, 2026.** These observations describe the development machine and saved artifacts. Generated datasets, checkpoints, reports, bundles, and desktop shortcuts are not supplied by a fresh source checkout.

| Area | Verified state |
| --- | --- |
| Synthetic smoke pipeline | Generation, shared perception/decoder training, three learned and three hybrid seeds, evaluation, held-out calibration, and rollout videos completed. Models are deliberately undertrained. |
| Enhanced smoke evidence | 204 episode-measurement groups, 18 gallery videos, and eight report plots. Offline regeneration was byte-identical with model/data/inference imports blocked. |
| Serving parity | All ten public smoke predictors matched source inference exactly on one deterministic 50-frame validation example, including frames and ensemble outputs. This checks export fidelity, not prediction quality. |
| Recorded Python checks | October 19: **134 passed, four CUDA checks skipped** to avoid competing with training. These are historical results, not a new October 25 test run. |
| Recorded frontend checks | October 18: four tests and the production build passed. |
| Full experiment | **Interrupted**: supervisor and child are no longer alive. The last saved training log ends at `hybrid_2` epoch **34/45**. Checkpoints remain; full evaluation and publication have not finished. |
| Release worker | Its saved state says `waiting`, but the process is no longer alive and the heartbeat dates to October 19. No completed full evidence manifest or published full bundle is present. |
| Recorded synthetic GPU environment | RTX 5050 Laptop GPU, 7.93 GiB VRAM, Python 3.12.14, PyTorch 2.11.0+cu128, CUDA runtime 12.8. |
| Colab smoke | A CPU notebook smoke run passed on October 17 and its runtime was stopped. This is not cloud GPU verification. |
| Real-video track | Eleven public clips were prepared; large-model inference, public benchmark results, and local video-model hardware feasibility remain unverified. |

The smoke primary comparison, hybrid ensemble minus fitted physics at 50 frames, was **−0.06524 px**, with a 95% paired bootstrap interval **[−1.05487, 1.00981] px**, across only four ordinary test episodes. The interval does not distinguish the predictors. This demonstrates the reporting pipeline and does not support a research performance claim.

Local records include `runs/research-evidence-verification.json`, `runs/production-verification.json`, `runs/real-data-verification.json`, and `runs/colab-smoke-verification.json`. The saved smoke report is `runs/smoke/evaluation-evidence-v2/report.html`; its export comparison is `runs/smoke/serving-evidence-v2-parity.json`. These artifacts are ignored by Git.

Use observed status rather than trusting an old `state: running` field:

```powershell
python -m nwm experiment-status --config configs/full.json
```

This documentation update does not resume training, restart the release worker, download model assets, or start cloud compute.

## Read-only diagnostics

Use `doctor` before resuming a run or diagnosing the local app:

```powershell
python -m nwm doctor --config configs/full.json --app-config configs/app.json
python -m nwm doctor --config configs/full.json --verify-artifacts --json
```

Defaults are the full experiment and checked-in app config. Text output is intended for humans; `--json` emits a version-1 machine-readable report. Exit `0` means no error checks, although warnings may remain; `1` means an error was found; invalid CLI arguments retain exit `2`.

Quick checks report recorded versus observed training/release state, parent/child liveness, heartbeat age, checkpoint presence, last **log-derived** progress, managed quota/free-space reserve, configured bundles, and CUDA availability. An old heartbeat alone does not mean a process is dead. Incomplete experiments, missing optional bundles, unavailable drives, and unavailable CUDA are warnings rather than claims of corruption.

`--verify-artifacts` additionally loads present checkpoints on CPU, checks configuration/model compatibility, validates available datasets, and verifies present serving/evidence artifacts using the existing validators. It performs no prediction. Corrupt or incompatible existing artifacts are errors; missing stages of an unfinished experiment remain warnings.

Diagnostics do not rewrite status, create managed directories, acquire resource locks, resume work, repair artifacts, download assets, or execute GPU kernels. Suggested commands are printed only. CUDA availability is not a kernel compatibility check: explicitly run `python -m nwm check-device --device cuda` when GPU execution is appropriate.

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
.\run.bat app
```

Use the project virtual environment for all `python` commands. Node.js **24** and npm match the frontend CI environment. The **Neural World Model App** desktop shortcut starts the same launcher; desktop shortcuts are machine-local and are not installed by cloning the repository. `configs/app.json` controls storage and bundle locations. Its CPU inference setting permits simulations to coexist with GPU training. The served smoke bundle is explicitly undertrained. Publish the full bundle only after full training, evaluation and calibration finish; the evidence-bound release workflow is described below:

```powershell
python -m nwm experiment-status --config configs/full.json
python -m nwm export-serving --config configs/full.json --evaluation runs/full/evaluation-evidence-v2 --output D:/NeuralWorldModel/bundles/full
```

Exports are immutable, checksum-verified, dataset-free bundles. Their manifest records validation selection, training completion and calibration provenance. An export with `--evaluation` also binds checkpoint/data fingerprints and saved calibration to the completed enhanced evaluation. Ordinary exports without that option remain useful for demos but are not enhanced-evidence releases. Publish a new directory and update the app config to change an existing bundle. Live training directories are never served directly.

Jobs survive restart in SQLite. One supervised process owns inference, with four queued jobs, cancellation, a 20-minute timeout and recovery of interrupted jobs. Failed/interrupted jobs can be retried. Results include actual comparison videos, curves, measurements and calibrated synthetic position radii. Forecast has no accuracy score without recorded targets and no calibrated uncertainty estimate. Simulation radii are conditional on shared perception; changed physics can invalidate coverage.

The app binds only to `127.0.0.1`. It has no authentication, remote access or multi-user deployment support. Uploads accept MP4/WebM up to 100 MiB, 30 seconds and 1080p (including portrait). Prefixes are sampled causally at 16 FPS and aspect-preserving letterboxing is recorded. Simulation actions use frame 1 for the transition immediately after the last observed frame. Real-video generation does not support interventions.

The application is a **local beta**, not a completed hosted production service. Its main settings and interfaces are:

| Setting | Checked-in default |
| --- | --- |
| Managed root | `D:/NeuralWorldModel` |
| HTTP port / inference device | `8000` / `cpu` |
| Storage limit / free-space reserve | 27 GiB / 8 GiB |
| Queue limit / timeout | Four queued jobs / 1,200 seconds |
| Bundle IDs | `full`, `smoke`; missing manifests are omitted |

Edit absolute storage and bundle paths before using another drive or OS. Build the frontend before normal serving. For development, run the backend separately and `npm run dev` in `frontend`; Vite proxies `/api` to `127.0.0.1:8000`. Node is not needed to serve an already-built frontend.

| HTTP endpoint | Purpose |
| --- | --- |
| `GET /api/health`, `GET /api/status` | Worker readiness, bundles, and capabilities |
| `POST /api/uploads`, `DELETE /api/uploads/{upload_id}` | Video upload and unreferenced-upload deletion |
| `GET /api/jobs`, `POST /api/jobs`, `GET /api/jobs/{job_id}` | List, submit, and inspect jobs |
| `POST /api/jobs/{job_id}/cancel`, `POST /api/jobs/{job_id}/retry` | Cancel or retry work |
| `DELETE /api/jobs/{job_id}` | Remove a job and reclaim eligible files |
| `GET /api/artifacts/{job_id}/{filename}` | Retrieve result files |
| `GET /api/jobs/{job_id}/export`, `POST /api/jobs/{job_id}/import` | Manual notebook exchange |
| `GET /api/video-notebook`, `GET /api/source-bundle` | Download notebook/source |

The running backend exposes its interactive API schema at `/docs`.

Managed files stay inside `D:/NeuralWorldModel`, capped at 27 GiB with 8 GiB free reserved. Deleting the last job referencing an upload also reclaims that upload. Unreferenced uploads can be deleted at `DELETE /api/uploads/{id}`. No automatic model download or cloud execution occurs.

### Experimental video prerequisites

The candidate is [LTX-Video 2B 0.9.8 distilled](https://huggingface.co/Lightricks/LTX-Video), using pinned file hashes, BF16 and sequential CPU offload. No prompt enhancer or upscaler is loaded. A 17-frame observed prefix predicts the next 10/20/50 frames. Advertised presets are 384×256 and a 288×192 fallback. Local generation remains disabled until three consecutive 50-frame development runs pass the real hardware gate. Memory fit, latency and physical accuracy are **not yet verified** on the RTX 5050.

```powershell
python -m nwm video-check --config configs/app.json
python -m pip install -e ".[video]"
python -m nwm video-download --config configs/app.json
python -m nwm video-feasibility --config configs/app.json --prefix dev1/prefix.npz --prefix dev2/prefix.npz --prefix dev3/prefix.npz --output D:/NeuralWorldModel/models/ltx-2b-0.9.8/feasibility.json
```

The assets require about **23.65 GiB**, plus temporary/output space. The last recorded local preflight refused the download because it would violate the 8 GiB free-space reserve. Recheck current space rather than treating that historical shortfall as a current measurement. Add free space or choose a larger managed drive before downloading. The three `--prefix` arguments above are placeholders for distinct development archives. The preflight report records exact byte counts. Finish or stop training before the GPU feasibility test; its wall-clock watchdog terminates and reaps timed-out processes.

For a separate GPU machine, select **Export notebook job** in Forecast/Backtest and use [the manual video notebook](notebooks/video_job.ipynb). Download the prepared job, run the notebook yourself, then import its result into the same waiting job. The exchange contains only the prefix and job identity; recorded targets remain local. Imports verify job ID, revision, prefix, seed, dimensions and checksum. Runtime selection and billing remain manual; no cloud runtime has been started for this feature. CUDA BF16 is required, so a T4 is unsupported by this adapter.

### Public evaluation mini set

```powershell
python -m nwm prepare-real-data --config configs/app.json
python -m nwm prepare-real-data --config configs/app.json --download
python -m nwm benchmark-video --config configs/app.json --manifest D:/NeuralWorldModel/datasets/physics-iq-mini/manifest.json --output D:/NeuralWorldModel/evaluation/physics-iq-mini --split test
```

This adapter selects three development and eight test scenarios from [Physics-IQ Original](https://github.com/google-deepmind/physics-IQ-benchmark), using SHA256 scenario-name ordering, center view and take 1. GCS object generations and checksums pin the downloads. The first command records metadata; downloading is explicit. Source clips are normalized to at most 30 seconds at 384×256/16 FPS. Only the 17 samples ending at t=3 seconds condition each 50-frame rollout. The development split is for hardware feasibility; test clips never train, select or calibrate the model. Results compare pixel MSE against frame holding with one seed, not an official Physics-IQ score or a best-of-N result. The upstream dataset is CC BY 4.0; retain its manifest attribution when sharing clips.

Run `python -m pytest` and `npm test`/`npm run build` in `frontend` to check the pipeline and app. CI is configured for Ubuntu/Python 3.11, Windows/Python 3.12 CPU tests, and Ubuntu frontend checks. Hosted Windows results remain pending until an actual Actions run completes. Windows production validation also covers dataset-free inference, request bounds, queue isolation, crash recovery, notebook identity, causal sampling and hostile array headers. Actual large-model inference and the public mini benchmark remain pending the storage and hardware gate.

## Run locally

On Windows, double-click `run.bat` (or the **Neural World Model** desktop shortcut) to run the smoke experiment and open its results folder. The launcher uses the project virtual environment, resumes saved training, and stops if any step fails. In PowerShell use `.\run.bat full` for the full experiment, `.\run.bat app` for the app, or `.\run.bat --help` for usage. Research modes generate data before invoking the supervised train/evaluate/demo workflow. Keep the app launcher console open; Ctrl+C stops the server.

Python **3.11 or 3.12** is recommended. From the project folder in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -e ".[dev,notebook,app,app-test]"
.\.venv\Scripts\python -m nwm check-device --device cuda
.\.venv\Scripts\python -m nwm generate --config configs/smoke.json
.\.venv\Scripts\python -m nwm run-experiment --config configs/smoke.json --device auto --resume
.\.venv\Scripts\python -m pytest
```

The CUDA 12.8 wheel index above is the one used by the recorded local RTX 5050 environment. Choose a compatible official wheel for other hardware using the [PyTorch installer](https://pytorch.org/get-started/locally/). For a CPU-only installation, replace the torch install line with `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`. `check-device` executes forward and backward kernels, rather than checking availability alone. A system CUDA toolkit is not required for prebuilt wheels.

If PowerShell activation is unavailable, use `.\.venv\Scripts\python.exe` explicitly throughout. On Linux/macOS create the environment with `python3 -m venv .venv`, then `source .venv/bin/activate`. The package CLI is portable; the Windows batch launcher and checked-in `D:` app paths are platform-specific.

The supervised command runs train → evaluate → demo, saves logs/status, coordinates GPU ownership, and rejects duplicate experiment processes. To execute individual stages instead:

```powershell
python -m nwm train --config configs/smoke.json --device auto --resume
python -m nwm evaluate --config configs/smoke.json --device auto
python -m nwm demo --config configs/smoke.json --device auto
```

The tested Windows/Python 3.12 dependency versions are recorded in `requirements/windows-cu128-py312.txt`. The package supports Python 3.11+, and Colab uses its own compatible runtime.

Configuration files may be stored anywhere. Relative `data.root` and `outputs.root` paths always resolve against this source tree; explicit absolute paths are preserved.

The smoke configuration deliberately trains for only 2 perception epochs, 2 decoder epochs, and 3 dynamics epochs on 32 episodes. It verifies the pipeline and produces **undertrained demonstration artifacts**, not research performance claims. It still runs all three dynamics seeds, the full 50-frame rollout, every OOD condition, action ablations, and independent calibration.

Use the same commands with `--config configs/full.json` for the full experiment: 8,000 training, 500 model-validation, 500 calibration, and 1,000 test episodes, plus 500 episodes in each of five OOD conditions. Data generation is deterministic and resumes compatible partial generation. Training budgets are ceilings with validation-based early stopping. Full training can take substantial time; measure the smoke timings before allocating a run.

## Configurations and dataset

Experiment configs share data, physics, training, evaluation, and output sections. See [configs/smoke.json](configs/smoke.json), [configs/full.json](configs/full.json), and the separate machine-specific [configs/app.json](configs/app.json).

| Setting | Smoke | Full |
| --- | --- | --- |
| Training episodes | 32 | 8,000 |
| Model validation | 8 | 500 |
| Uncertainty calibration | 12 | 500 |
| Ordinary testing | 8 | 1,000 |
| Episodes per changed condition | 4 | 500 |
| Total episodes, including five changed conditions | 80 | 12,500 |
| Evaluation cap per condition | 4 | All |
| Perception / decoder epoch ceilings | 2 / 2 | 30 / 20 |
| Dynamics epoch ceiling per seed/family | 3 | 45 |
| Training windows per episode | 2 | 4 |
| Training / evaluation batch size | 32 / 8 | 32 / 32 |
| Dynamics seeds | `0, 1, 2` | `0, 1, 2` |
| Data / output root | `data/smoke`, `runs/smoke` | `data/full`, `runs/full` |

The full ordinary dataset contains 10,000 episodes, with 2,500 additional changed-condition episodes. Each episode stores 64 states, 63 actions, 63 contact-event vectors, physics parameters, and its seed in compressed NPZ files. Contacts distinguish 12 wall-impact channels and three disc-pair channels. Metadata records generation identity and fingerprints; RGB frames are rendered when needed.

Initial positions reject overlaps and leave clearance from walls. Initial speed is sampled from `0.15–0.6` world units/second. Each transition has a `0.12` probability of an impulse, whose active components are sampled from `[-0.25, 0.25]` world units/second. Coordinates use downward-increasing `y`; actions always target the red disc.

Changed-condition split names are `ood_drag_low`, `ood_drag_high`, `ood_restitution_low`, `ood_restitution_high`, and `ood_speed`. Their values and scientific role are described below. Use new data/output roots for a different experiment instead of bypassing compatibility checks.

## Continue training on another machine

```powershell
python -m nwm train --config configs/full.json --resume
python -m nwm train --config configs/full.json --stage hybrid --seed 0 --resume
python tools/make_colab_bundle.py --config configs/full.json --include-artifacts
```

The bundle excludes the virtual environment and includes source, the notebook, and optionally the selected dataset and run. Upload it using the first cell of [the notebook](notebooks/neural_world_model.ipynb). If that picker is unavailable, upload `nwm_colab.zip` through Colab's Files sidebar and rerun the first cell. In Colab, select a GPU runtime and run the notebook. Existing compatible PyTorch is retained. Paths resolve relative to the source tree, so bundled checkpoints work on Windows and Linux. [Free Colab GPU availability and runtime limits vary](https://research.google.com/colaboratory/faq.html).

For local notebook use, run `python -m jupyter lab notebooks/neural_world_model.ipynb`. Set `CONFIG_NAME`, `DEVICE`, and `RUN_TRAINING` in its configuration cell; training is enabled by default. Its evaluation cell generates the standard aggregate report; enhanced evidence uses the separate `evaluate --output` workflow below. The bundle creator accepts `--output` and permits artifact inputs only inside the project tree. Managed app bundles on `D:` are not implicitly included. Preserve data/checkpoints before temporary runtimes end. Cloud runtime selection and any billing remain operator decisions.

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

## Training protocol

```text
4 observed RGB frames + 3 observed-transition actions
                         |
               shared visual estimator
                         |
             3 estimated (x, y, vx, vy) states
                         |
       predictor + supplied future action sequence
                         |
             autoregressive predicted states
                         |
                 shared frozen decoder
                         |
                 predicted RGB frames
```

The encoder uses `4×4`, stride-2 convolutions and projects flattened `128×8×8` features to 128 dimensions. The estimator's GRU has 128 hidden units and combines image embeddings with preceding actions. Its 12 outputs represent three fixed object identities. The decoder projects 12 state values to `128×8×8` features, then upsamples with three transposed convolutions to RGB.

Learned and hybrid dynamics use 128-unit GRUCells with 14 inputs: 12 current-state values plus a two-component action. Learned dynamics output 12 state deltas; hybrid dynamics output six residual acceleration components and one bounded contact restitution value. Restitution begins at `0.99`, near the imperfect elastic prior. Contact decisions come from the predicted trajectory in the shared solver; recorded future contacts are never supplied. The solver is piecewise differentiable with hard contact decisions.

Perception and decoding train first and remain frozen for the dynamics comparisons. Dynamics training initializes from estimated visual states, rather than perfect simulator states. The all-stage training command fits baseline coefficients from training labels after the dynamics stages finish; coefficient fitting does not depend on their predictions.

| Choice | Default / implementation |
| --- | --- |
| Optimizer | AdamW, learning rate `1e-3`, batch size 32 |
| State training loss | Mean squared error after dividing `(x,y,vx,vy)` errors by `(0.25,0.25,0.5,0.5)` |
| Decoder training loss | Foreground-weighted absolute pixel error; checkpoint selection uses ordinary pixel MSE |
| Hybrid regularization | Residual-acceleration squared penalty weighted by `1e-4` |
| Rollout curriculum | 1, 10, and 20 frames across successive epoch thirds |
| Checkpoint selection | Normalized state MSE for perception, pixel MSE for decoding, validation position RMSE for dynamics |
| Early stopping | Patience 5; dynamics get time in all curriculum stages |

Evaluation rolls out to 50 frames without teacher forcing. Test outcomes never choose checkpoints. The hybrid ensemble averages independently rolled-out states and decodes that mean; averaging does not preserve individual members' geometry constraints, so boundary and overlap checks remain relevant.

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

For observed frames at times 0–3, past actions describe 0→1, 1→2, and 2→3; `future_actions[:,0]` describes 3→4 and is applied before that integration step. `context_actions=None` is accepted as zeros. Returned tensors remain on the configured device. The raw `position_spread` is not a calibrated confidence radius; calibration metadata is separate.

Inference needs the configuration, selected checkpoints, and `fitted_physics.json`; dataset archives are not required. Training, evaluation, and demonstration commands still validate their datasets before use. To load an exported dataset-free bundle instead:

```python
from nwm.serving import load_serving

bundle = load_serving("D:/NeuralWorldModel/bundles/smoke", device="cpu")
prediction = bundle["predictors"]["hybrid_ensemble"].predict(
    context_frames, context_actions, future_actions
)
```

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

| Metric | Definition |
| --- | --- |
| Position RMSE | `sqrt(mean(dx_px² + dy_px²))`, pooled over episodes, objects, and prefix frames through the horizon |
| Position ADE | Mean Euclidean position distance over the prefix, in pixels |
| Position FDE | Mean Euclidean position distance at the final horizon frame, in pixels |
| Velocity RMSE | Euclidean velocity RMSE in world units/second |
| Pixel MSE | Common decoder versus true raster frames with RGB scaled to `[0,1]` |
| Boundary / overlap rate | Fractions of object-frames / disc-pair-frames violating geometry |
| Uncertainty width | Diameter, twice the calibrated position-error radius |

The standard aggregate `error_vs_horizon.png` plots ADE curves; enhanced research report position-error plots use pooled RMSE. The true-state decoder diagnostic receives true future states only to measure reconstruction limitations; this is separate from all prediction calls.

Uncertainty comes from three hybrid dynamics seeds conditional on shared perception. Held-out episodes calibrate a 0.5-pixel minimum spread separately for each fixed object at frames 10/20/50. Reports show empirical coverage and interval width, with an explicit flag if there are too few calibration episodes for the finite-sample rank. Intervals are marginal per object and horizon, not simultaneous trajectory guarantees. Distribution shift can reduce coverage; agreement between models does not establish reliability.

The no-action ablation zeros only future impulses and retains the same targets and observed history. Neural seed and ensemble rows include this ablation; standard physics baseline rows use recorded actions. Scientific completion means reporting the comparison honestly, including failures; hybrid improvement is a hypothesis.

The synthetic research track does not handle real camera footage, unknown object identities or geometry, variable object counts or policy learning. The separate experimental video adapter generates continuations without claiming those scientific capabilities.

## Evidence-first research release

Position RMSE now has one definition everywhere: `sqrt(mean(dx_px² + dy_px²))`, pooled over objects, episodes and frames through the reported horizon. New checkpoints and reports record `euclidean_position_rmse_px_v1`. Historical dynamics checkpoints used coordinate RMSE, smaller by `sqrt(2)`. Loading those checkpoints converts only validation metadata in memory and records the conversion; model weights, optimizer/RNG state, checkpoint ordering, and source files stay unchanged. Resumption continues with the same selected epoch and standard units. Already-running processes retain their loaded code; documentation edits do not restart experiments.

Generate enhanced evidence in a **fresh** directory using existing checkpoints, without training:

```powershell
python -m nwm evaluate --config configs/smoke.json --device cpu --output runs/smoke/evaluation-research
python -m nwm report --evaluation runs/smoke/evaluation-research
```

An existing `--output` directory is refused, preserving previous or interrupted evaluations. Without `--output`, evaluation retains its original output location and aggregate behavior. Enhanced evaluation freezes checkpoint/data hashes, saves compact episode archives and truth-defined contact/impulse strata, and generates `report.html`, plots, `statistical_comparisons.json`, and median/p90/worst-case videos. Gallery ranks use hybrid-ensemble 50-frame prefix RMSE, nearest-rank percentiles and episode-ID tie breaking; the learned video panel always uses the first configured seed. They are descriptive examples, never model selection.

Contact strata distinguish free flight, wall-only collisions, and any ball collision, with ball contacts taking precedence. A separate overlapping split distinguishes episodes with/without future impulses. Truth labels define analysis strata only. Per-episode files retain aligned episode identifiers; gallery records retain selected IDs, ranks, errors, and saved trajectories. Enhanced evaluation refuses to run against an active experiment.

The primary comparison is hybrid ensemble minus fitted physics at 50 frames on ordinary test physics, estimated initialization and recorded actions. Negative differences favor hybrid. Confidence intervals use 2,000 paired whole-episode bootstrap resamples with seed `20261019`; each resample recomputes pooled RMSE. Comparisons across training seeds share resampled episode IDs. Seed dispersion is reported separately, and episode intervals are conditional on the trained models. Secondary comparisons are exploratory, without multiplicity correction. Small groups are flagged rather than presented as strong evidence. Per-object coverage uses 95% Wilson intervals across independent held-out episodes, remaining informative for all-covered/all-missed samples; width intervals use the episode bootstrap. Calibration remains separate, per object/horizon; changed-physics coverage is measured without recalibrating on those tests.

`report` reads saved measurements only. It verifies their hashes and can regenerate HTML, JSON and figures without model weights, the source dataset, inference, or a GPU. Keep the HTML beside its JSON, `episodes/`, `gallery/` and `report_plots/` folders when moving it. Smoke reports are explicitly undertrained pipeline demonstrations.

To open the already-generated development-machine smoke report without new inference:

```powershell
python -m nwm report --evaluation runs/smoke/evaluation-evidence-v2
python -m http.server 8766 --bind 127.0.0.1 --directory runs/smoke/evaluation-evidence-v2
```

Visit [the local report](http://127.0.0.1:8766/report.html) while that server is running. The `evaluation-evidence-v2` directory is a local artifact, not part of a source-only checkout.

The original full experiment is currently interrupted. To explicitly resume a compatible full run and complete train/evaluate/demo, the operator can use:

```powershell
python -m nwm generate --config configs/full.json
python -m nwm run-experiment --config configs/full.json --device cuda --resume
```

Check that no experiment/child is active before launching; do not start duplicate trainers. Once that supervised pipeline is running, a **separate terminal** can queue a finite release that waits for successful completion. Alternatively, run release after completion without `--wait`:

```powershell
python -m nwm release-research --config configs/full.json --device cuda --output runs/full/evaluation-evidence-v2 --bundle D:/NeuralWorldModel/bundles/full --wait
```

This command never starts or resumes training and never uses paid cloud resources. It waits without holding the GPU lock, then freezes inputs, runs enhanced evaluation once, rebuilds the report, and exports an evidence-bound bundle. Every public predictor is compared with the source on one deterministic 50-frame validation context, including states, decoded frames and ensemble spread. The verified staging directory is renamed to the final bundle only after parity passes. The smoke bundle stays available in the app; Refresh discovers the full bundle.

`--wait` waits only for a running/successfully completed experiment; it refuses interrupted, failed, unknown, or not-started experiments. The previously launched waiting process is no longer alive. A saved `waiting` status is not an active scheduler, and neither training nor release restarts itself after the machine/process stops.

Release heartbeat/status, immutable `freeze.json` and `parity.json` live in the sibling `.evaluation-evidence-v2.research-release/` directory. Rerunning the same command can finish publication from complete evidence without repeating inference. Incomplete evaluations, changed inputs, unrelated existing bundles and failed original experiments are refused and preserved for inspection. A different output directory is required to intentionally repeat an incomplete evaluation.

To bind an independently completed evaluation explicitly:

```powershell
python -m nwm export-serving --config configs/full.json --evaluation runs/full/evaluation-evidence-v2 --output D:/NeuralWorldModel/bundles/full
```

The fitted baseline uses the exact simulator force-law family. It may win; a hybrid improvement is not an acceptance requirement. No new force laws, architectures, datasets or training seeds are introduced by this evidence release. General-video hardware feasibility remains a separate, unverified gate.

### Source and environment provenance

New checkpoints, enhanced evaluations, release freezes, and serving manifests also record forward-only execution provenance. The shared helper captures Git revision/dirty state when available, a deterministic source-content digest, installed dependency versions, Python/platform information, and evaluation protocol `nwm-evidence-v1`.

The digest covers sorted relative paths and SHA256 contents of `nwm/**/*.py`, `pyproject.toml`, and requirements files, including untracked package sources. It excludes documentation, tests, datasets, runs, caches, and frontend build output. Source/dependency fingerprints still work without Git, including extracted notebook bundles; unavailable Git identity is recorded explicitly.

New checkpoints retain their original training provenance. When resuming legacy checkpoints, their training origin stays unrecorded and the new execution context is recorded as a resumption. Neither current source nor current dependency versions are retroactively assigned to old weights. Existing checkpoints and other historical artifacts remain readable without migration.

Enhanced evidence records `provenance.execution_provenance`; new freezes record `execution_provenance`. New serving manifests distinguish `evaluation_provenance` copied from the evidence from the current `export_provenance`. A missing historical evaluation record stays `null` rather than being inferred from export time. CPU export and GPU evaluation environments can differ.

For newly captured evidence/releases, source/protocol fingerprints are checked through evaluation and publication in addition to existing data/weight hashes. Git metadata and environment differences alone do not fail source verification. Changes to execution sources require a fresh evaluation. Offline reports display saved provenance only and never inspect the current checkout or installed environment. Legacy artifacts lacking provenance are labeled unrecorded rather than backfilled.

## Command reference

Every command uses `python -m nwm COMMAND`. Use `--help` on the root or any subcommand to inspect its options.

| Command | Purpose and important options |
| --- | --- |
| `generate` | Generate compatible synthetic data; `--config` defaults to smoke |
| `train` | `--config`, `--device auto/cpu/cuda`, `--resume`, `--stage all/perception/decoder/learned/hybrid`, repeatable `--seed` |
| `evaluate` | Aggregate evaluation, or enhanced evidence with a fresh `--output`; smoke default |
| `demo` | Decoded and directly rendered rollout comparisons; smoke default |
| `check-device` | Actual forward/backward computation; `--device` |
| `doctor` | Read-only full/app diagnostics; `--config`, `--app-config`, `--verify-artifacts`, `--json` |
| `run-experiment` | Supervised train/evaluate/demo; full config and CUDA default; optional `--resume` |
| `experiment-status` | Observed supervisor/child state; full config default |
| `report` | Offline report regeneration; required `--evaluation` |
| `release-research` | Required `--output` and `--bundle`; full/CUDA defaults; optional `--wait` |
| `export-serving` | Required new `--output`; optional `--evaluation` evidence binding; full config default |
| `serve` | App config default; optional `--port`, `--open-browser` |
| `video-check` | Storage/asset preflight; optional `--output`, default `runs/video-preflight.json` |
| `video-download` | Explicit pinned video-model asset download |
| `video-feasibility` | Repeatable required `--prefix`, optional `--output` for the development hardware gate |
| `prepare-real-data` | Prepare the public mini manifest; `--download` explicitly fetches clips |
| `benchmark-video` | Required `--manifest`, `--output`; `--split development/test` |

Direct CLI `train` and `run-experiment` require `--resume` to request checkpoint continuation; the Windows research launcher always supplies it. Use the supervisor for normal experiments and avoid concurrent direct low-level training commands against the same run or GPU.

## Repository layout

```text
configs/                     Synthetic experiment, local app, and public mini settings
nwm/
  physics.py, data.py         Shared solver, renderer, episode generation and clips
  models.py, predict.py       Perception, decoder, dynamics and prediction APIs
  training.py                Optimization, coefficient fitting, selection and resume
  metrics.py                 Trajectory, contact and uncertainty measurements
  metric_definitions.py      Shared RMSE definition and legacy compatibility
  provenance.py             Forward-only source, Git and environment fingerprints
  diagnostics.py            Read-only project and artifact health checks
  evaluation.py, evidence.py Aggregate and per-episode evidence collection
  research_report.py         Offline bootstrap statistics, plots and HTML
  research_release.py        Frozen-input evaluation and verified publication
  serving.py                 Dataset-free bundle export and load
  experiment.py, files.py    Supervision, locks and atomic file replacement
  app*.py                    HTTP application, durable storage and inference worker
  video*.py                  Video processing, pinned adapter and public benchmark
  demo.py, __main__.py        Demonstrations and command-line entry point
frontend/                    React/TypeScript/Vite app and domain tests
notebooks/                   Research notebook and manual video-job notebook
tests/                       Physics, model, research, serving and app checks
tools/make_colab_bundle.py    Portable source/artifact archive creator
requirements/                Recorded Windows dependency snapshot
.github/workflows/check.yml  Python CPU and frontend CI
run.bat                      Windows experiment/app launcher
data/                        Generated compact episodes; ignored by Git
runs/                        Checkpoints, logs, measurements and reports; ignored by Git
```

Additional output locations:

| Location | Contents |
| --- | --- |
| `data/<experiment>/metadata.json`, `*.npz` | Generation identity, seeds, states, actions, parameters, contacts |
| `runs/<experiment>/process.json` | Supervisor state and process information; use `experiment-status` for liveness |
| `runs/<experiment>/training.log`, `training-errors.log` | Stage output and error diagnostics |
| Enhanced evaluation `episodes/` | Compressed aligned per-episode measurements for offline statistics |
| Enhanced evaluation `gallery/` | Ranked example videos, IDs, and trajectory archives |
| Enhanced evaluation `evidence.json` | Completion marker, metric definition, fingerprints and artifact hashes |
| Enhanced evaluation `statistical_comparisons.json` | Primary/secondary contrasts, confidence intervals, strata and uncertainty |
| Enhanced evaluation `report.html`, `report_plots/` | Portable research report and figures |
| Sibling `.<evaluation-name>.research-release/` | Release lock/status, immutable input freeze, and parity results |
| Managed app `bundles/<id>/` | Immutable inference weights, calibration, fitted coefficients and manifest |
| Managed app root | `jobs.sqlite3`, uploads/jobs, optional video assets and public clips |

A fresh checkout has no generated weights, source bundle, built frontend, or research results. Generate/build/export only the artifacts required for the workflow you intend to run.

## Validation and troubleshooting

Run the Python and frontend checks after installing their dependencies:

```powershell
python -m pytest -q
cd frontend
npm ci
npm test
npm run build
cd ..
```

To avoid touching CUDA while training is active, use a separate PowerShell session with `$env:CUDA_VISIBLE_DEVICES='-1'` before pytest. This intentionally skips CUDA cases; it does not verify GPU execution. The CI workflow retains Ubuntu/Python 3.11 and Node 24 frontend checks and adds a Windows/Python 3.12 CPU job with a 20-minute timeout. It does not run full training or large video-model inference. Configuration is not proof of a successful hosted run; no workflow is dispatched by local implementation checks.

Tests cover deterministic simulation, overlap-free placement, action timing, drag and collisions, elastic conservation, split isolation, prediction shapes and causal inputs, checkpoint resumption/selection, metric compatibility, episode alignment, identical-predictor zero differences, held-out calibration, offline report regeneration, frozen-input release validation, serving parity, job/queue recovery, notebook identity, causal sampling, and hostile array headers.

Reliability checks additionally cover read-only diagnostics, source changes/untracked files/Git-free bundles, separate evaluation/export environments, and legacy provenance. Native Windows tests exercise child liveness before/after exit, resource-lock release after abrupt owner death, real delete-sharing failures during atomic replacement, and the actual batch launcher from a different working directory with spaces. Launcher fixtures use stub commands and stop before any browser, Explorer, or research run; subprocesses have bounded timeouts and closed stdin.

| Symptom | What to check |
| --- | --- |
| Wrong Python environment | Activate `.venv` or use its interpreter explicitly. The batch launcher expects `.venv/Scripts/python.exe`. |
| Unclear current readiness | Run `doctor`; use `--verify-artifacts` for integrity checks and `--json` for automation. Warnings do not imply failed validation. |
| Missing data/checkpoints | Generate the selected dataset and train the required stages. A source checkout contains no trained weights. |
| Incompatible dataset or estimator | Preserve the run; restore the matching configuration or use new data/output roots. Do not bypass identity checks. |
| CUDA unavailable or unsupported kernel | Run `check-device --device cuda`; check the wheel and driver against the official installer. CPU supports small checks. |
| Status JSON says running but the process stopped | Trust observed liveness from `experiment-status`, inspect logs, and explicitly resume the supervised pipeline when intended. |
| Release status stays waiting | Inspect heartbeat/process liveness. A stopped release worker is not a scheduler; restart it only after the experiment is running or complete. |
| Enhanced output or bundle already exists | Preserve it. Regenerate complete reports offline, resume a compatible release, or choose a new output directory. |
| App has no models | Export a bundle, point `configs/app.json` at its manifest, and Refresh. Missing bundles are omitted. |
| Frontend missing or stale | Run `npm ci` and `npm run build` in `frontend`. |
| Video assets cannot download | Inspect `video-check` byte counts, quota, disk space, and reserve before changing managed storage. |
| Plausible video but incorrect motion | Inspect state/contact metrics for the synthetic track. Appearance alone does not establish physical accuracy. |

## Scope and references

V1 assumes three fixed object identities, known geometry, training-only simulator state supervision, and supplied future actions. It evaluates prediction and robustness within that controlled setting. It does not provide arbitrary physical-law inference, unknown-object tracking, variable object counts, planning policies, real-camera physics estimation, or a hosted multi-user service.

Three seeds give limited evidence about training variation. Shared perception leaves perception uncertainty outside the dynamics ensemble. Calibration does not guarantee whole-trajectory or changed-condition coverage. Full research conclusions remain pending completion of the interrupted experiment and frozen enhanced evaluation. Larger models, new force laws, additional training seeds, and general-video research are deferred.

References and upstream resources:

- [APHYNITY](https://arxiv.org/abs/2010.04456): prior work motivating learned physics corrections.
- [PyTorch installation](https://pytorch.org/get-started/locally/): official wheel selection.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): notebook runtime information.
- [LTX-Video](https://huggingface.co/Lightricks/LTX-Video): experimental video checkpoint; exact revisions and file hashes are recorded in `nwm/video_model.py`.
- [Physics-IQ benchmark](https://github.com/google-deepmind/physics-IQ-benchmark): source of the separate public mini set; retain its manifest attribution and upstream terms.

The repository currently has no project `LICENSE` file. Model and dataset terms are separate from the source repository.
