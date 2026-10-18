"""One supervised inference process; the HTTP server never owns ML models."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import threading
import time

from .app_storage import Store, TERMINAL, now


def _write_result(store, job_id, result):
    destination = store.job_dir(job_id) / "artifacts"
    destination.mkdir(exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    store.ensure_space()
    store.update_job(job_id, status="succeeded", progress=1, message="Prediction complete", result=result)


def _simulation(store, job, cache):
    import numpy as np
    import torch
    from .data import generate_episodes
    from .demo import _panels, _save_video
    from .metrics import RolloutMetrics
    from .physics import PhysicsConfig, render_states, step
    from .serving import load_serving, calibrated_radii

    request, job_id = job["request"], job["id"]
    selected = next(item for item in store.config["bundles"] if item["id"] == request["bundle_id"])
    cache_key = (selected["path"], store.config.get("device", "cpu"))
    if cache_key not in cache:
        cache.clear()
        cache[cache_key] = load_serving(*cache_key)
    artifacts = cache[cache_key]
    device = artifacts["device"]
    torch.set_num_threads(2)
    cfg = PhysicsConfig(**artifacts["config"]["physics"])
    truth_cfg = replace(cfg, drag=request["drag"], restitution=request["restitution"])
    horizon = request["horizon"]
    episode = generate_episodes([request["seed"]], 4, truth_cfg)
    context_states = torch.from_numpy(episode["states"]).to(device)
    past = torch.from_numpy(episode["actions"]).to(device)
    context = render_states(context_states, radius=cfg.radius)
    actions = torch.zeros(1, horizon, 2, device=device)
    for action in request["actions"]:
        actions[:, action["frame"] - 1] = actions.new_tensor([action["dx"], action["dy"]])
    # Truth remains outside every predictor: only video and recorded actions enter predict.
    state = context_states[:, -1]
    truth, contacts = [], []
    for action in actions.unbind(1):
        state, events = step(state, action, truth_cfg)
        truth.append(state)
        contacts.append(events)
    truth = torch.stack(truth, 1)
    contacts = torch.stack(contacts, 1)
    truth_frames = render_states(truth, radius=cfg.radius)
    output = store.job_dir(job_id) / "artifacts"
    output.mkdir(exist_ok=True)
    default = artifacts["metadata"]["default_model"]
    learned = default if default.startswith("learned_") else next(name for name in artifacts["predictors"] if name.startswith("learned_"))
    chosen = {"constant_velocity", "imperfect_physics", "fitted_physics", learned, "hybrid_ensemble"}
    state_videos, decoded_videos, names = [torch.cat((context[0], truth_frames[0]))], [torch.cat((context[0], truth_frames[0]))], ["Ground truth"]
    metrics, curves, uncertainty = [], [], []
    uncertainty_note = ""
    horizons = [value for value in (10, 20, 50) if value <= horizon]
    with torch.inference_mode():
        for index, (name, predictor) in enumerate(artifacts["predictors"].items()):
            store.update_job(job_id, progress=0.1 + index / len(artifacts["predictors"]) * 0.6,
                             message=f"Predicting with {name.replace('_', ' ')}")
            prediction = predictor.predict(context, past, actions)
            accumulator = RolloutMetrics(horizons, cfg.radius, cfg.dt)
            pixel = (prediction["frames"] - truth_frames).square().flatten(2).mean(-1)
            accumulator.update(prediction["states"], truth, context_states[:, -1], contacts,
                               actions=actions, pixel_mse=pixel)
            measured = accumulator.compute()
            metrics.extend({"model": name, "horizon": int(key), **values}
                           for key, values in measured["horizons"].items())
            point_error = (prediction["states"][..., :2] - truth[..., :2]).square().sum(-1).mean(-1).sqrt()[0] * 64
            curves.append({"model": name, "horizons": list(range(1, horizon + 1)),
                           "position_rmse_px": point_error.cpu().tolist()})
            if name == "hybrid_ensemble":
                radii = calibrated_radii(prediction["position_spread"], artifacts["calibration"])
                uncertainty = [{"horizon": int(key), "radius_px": float(value.mean()),
                                "interval_width_px": float(value.mean() * 2),
                                "object_radii_px": value[0].cpu().tolist()}
                               for key, value in radii.items() if int(key) <= horizon]
                uncertainty_note = "90% marginal radii per object and horizon, conditional on shared perception; no trajectory or changed-physics coverage guarantee."
                if not artifacts["metadata"].get("calibration_binding", {}).get("verified"):
                    uncertainty_note = "Calibration association with these weights is unverified; radii are diagnostic only."
            if name in chosen:
                state_videos.append(torch.cat((context[0], render_states(prediction["states"][0], radius=cfg.radius))))
                decoded_videos.append(torch.cat((context[0], prediction["frames"][0])))
                names.append(name.replace("_", " "))
            if name == default:
                default_frames = torch.cat((context[0], prediction["frames"][0])).cpu()
    videos = []
    # The comparison panels have exactly six positions; include the common baselines and ensemble.
    for filename, label, frames in (("state_comparison.mp4", "Trajectory comparison", _panels(state_videos[:6], names[:6])),
                                    ("decoded_comparison.mp4", "Common decoder comparison", _panels(decoded_videos[:6], names[:6])),
                                    ("ground_truth.mp4", "Ground truth", None), ("prediction.mp4", "Default prediction", None)):
        if frames is None:
            value = torch.cat((context[0], truth_frames[0])).cpu() if filename == "ground_truth.mp4" else default_frames
            frames = list((value.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8))
        path, note = _save_video(frames, output / filename)
        videos.append({"label": label, "url": f"/api/artifacts/{job_id}/{path.name}", "note": note})
    return {"videos": videos, "metrics": metrics, "curves": curves, "uncertainty": uncertainty,
            "uncertainty_note": uncertainty_note,
            "fps": 30, "context_frames": 4, "horizon": horizon, "default_model": default,
            "model_version": artifacts["metadata"]["experiment_signature"][:12],
            "experiment_size": artifacts["metadata"]["experiment_size"], "execution_origin": "local",
            "interpretation": "Single-scenario measurements; use held-out experiment reports for aggregate model quality."}


def _real_video(store, job, cache):
    import numpy as np
    from .video import prepare_video
    from .video_model import LTXVideoPredictor

    request, job_id = job["request"], job["id"]
    directory = store.job_dir(job_id)
    upload = store.upload(request["upload_id"])
    prepared_file = directory / "prepared.json"
    if prepared_file.is_file():
        prepared = json.loads(prepared_file.read_text(encoding="utf-8"))
    else:
        from .video_model import video_availability
        resolution = video_availability(store.path("models", "ltx-2b-0.9.8"))["resolution"]
        prepared = prepare_video(upload["path"], directory / "work", request["cutoff_seconds"], request["horizon"],
                                 backtest=request["mode"] == "backtest", width=resolution[0], height=resolution[1])
        prepared_file.write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    imported = directory / "work/imported_prediction.npz"
    if imported.is_file():
        with np.load(imported, allow_pickle=False) as archive:
            forecast = archive["frames"]
        return finish_video(store, job, prepared, forecast, "manual_notebook")
    if request["execution"] == "notebook":
        store.update_job(job_id, status="waiting_notebook", progress=0.1,
                         message="Export this job, run it in a notebook, and import the prediction")
        return None
    cache_key = ("video", str(store.path("models", "ltx-2b-0.9.8")))
    if cache_key not in cache:
        cache.clear()
        cache[cache_key] = LTXVideoPredictor(cache_key[1])
    predictor = cache[cache_key]
    with np.load(prepared["prefix_path"], allow_pickle=False) as archive:
        prefix = archive["frames"]
    forecast = predictor.predict(prefix, request["horizon"], request["seed"])
    return finish_video(store, job, prepared, forecast, "local")


def finish_video(store, job, prepared, forecast, origin):
    import numpy as np
    from .demo import _save_video
    from .video import score_video
    job_id = job["id"]
    output = store.job_dir(job_id) / "artifacts"
    output.mkdir(exist_ok=True)
    # Commit the prefix-only prediction before accessing the held-out future.
    prediction_path = output / "prediction.npz"
    np.savez_compressed(prediction_path, frames=forecast)
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    (output / "commit.json").write_text(json.dumps({"prediction_sha256": digest, "committed_at": now()}), encoding="utf-8")
    with np.load(prepared["prefix_path"], allow_pickle=False) as archive:
        prefix = archive["frames"]
    predicted_path, _ = _save_video(list(np.concatenate((prefix, forecast))), output / "prediction.mp4", fps=16)
    videos = [{"label": "Experimental continuation", "url": f"/api/artifacts/{job_id}/{predicted_path.name}"}]
    metrics, curves = [], []
    if prepared.get("target_path"):
        with np.load(prepared["target_path"], allow_pickle=False) as archive:
            target = archive["frames"]
        measured = score_video(forecast, target, prefix[-1])
        for name, key in (("continuation", "pixel_mse"), ("frame_hold", "frame_hold_pixel_mse")):
            if key in measured:
                metrics.append({"model": name, "horizon": len(forecast), "pixel_mse": measured[key]})
        curves = [{"model": model, "horizons": list(range(1, len(forecast) + 1)), "pixel_mse": measured[key]}
                  for model, key in (("continuation", "pixel_mse_curve"), ("frame_hold", "frame_hold_pixel_mse_curve"))]
        (output / "backtest.json").write_text(json.dumps(measured, indent=2), encoding="utf-8")
        truth_path, _ = _save_video(list(np.concatenate((prefix, target))), output / "ground_truth.mp4", fps=16)
        videos.append({"label": "Recorded future", "url": f"/api/artifacts/{job_id}/{truth_path.name}"})
    return {"videos": videos, "metrics": metrics, "curves": curves, "uncertainty": [],
            "fps": 16, "context_frames": len(prefix), "horizon": len(forecast),
            "model_version": "ltxv-2b-0.9.8-distilled", "execution_origin": origin,
            "prediction_sha256": digest, "interpretation": "Experimental generated continuation. Video error does not certify physical accuracy."}


def _worker(config, incoming):
    from .experiment import gpu_lock, ResourceBusyError
    store, cache = Store(config), {}
    while True:
        job_id = incoming.get()
        if job_id is None:
            return
        job = store.get_job(job_id)
        try:
            store.ensure_space(32 * 1024 ** 2)
            needs_gpu = (job["mode"] != "simulation" and job["request"]["execution"] == "local") or config.get("device") == "cuda"
            if needs_gpu:
                from .experiment import experiment_status
                experiment = experiment_status(Path(__file__).resolve().parents[1] / "configs/full.json")
                if experiment.get("state") == "running" or experiment.get("child_alive"):
                    raise ResourceBusyError("Full training is using the GPU; this job will wait")
            with gpu_lock() if needs_gpu else nullcontext():
                result = _simulation(store, job, cache) if job["mode"] == "simulation" else _real_video(store, job, cache)
            if result is not None:
                _write_result(store, job_id, result)
        except ResourceBusyError as error:
            store.update_job(job_id, status="queued", message=str(error), progress=0)
            time.sleep(2)
        except Exception as error:
            store.update_job(job_id, status="failed", error=f"{type(error).__name__}: {error}", message="Prediction failed")


class Supervisor:
    def __init__(self, config):
        self.config, self.store = config, Store(config)
        self.process = self.incoming = None
        self.busy_job_id = None
        self.heartbeat = now()
        self._stop = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        self.store.recover()
        self.thread = threading.Thread(target=self._guarded_loop, name="nwm-supervisor", daemon=True)
        self.thread.start()

    def _spawn(self):
        context = mp.get_context("spawn")
        self.incoming = context.Queue(maxsize=1)
        self.process = context.Process(target=_worker, args=(self.config, self.incoming), daemon=True)
        self.process.start()

    def _terminate(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
            self.process.close()
            self.process = None
        if self.incoming is not None:
            self.incoming.cancel_join_thread()
            self.incoming.close()
            self.incoming = None

    def _loop(self):
        began = time.monotonic()
        while not self._stop.wait(0.25):
            self.heartbeat = now()
            if self.busy_job_id:
                job = self.store.get_job(self.busy_job_id)
                if job["cancel_requested"] or time.monotonic() - began > self.config.get("timeout_seconds", 1200):
                    self._terminate()
                    cancelled = job["cancel_requested"]
                    self.store.update_job(job["id"], status="cancelled" if cancelled else "failed",
                                          error=None if cancelled else "Prediction exceeded the 20-minute job limit",
                                          message="Cancelled" if cancelled else "Prediction timed out")
                    self.busy_job_id = None
                elif self.process is not None and not self.process.is_alive():
                    self.store.update_job(job["id"], status="interrupted", error="The inference process exited unexpectedly")
                    self._terminate()
                    self.busy_job_id = None
                elif job["status"] != "running":
                    self.busy_job_id = None
            if self.busy_job_id is None:
                if self.process is not None and not self.process.is_alive():
                    self._terminate()
                if self.process is None:
                    self._spawn()
                job = self.store.claim()
                if job:
                    self.busy_job_id = job["id"]
                    began = time.monotonic()
                    self.incoming.put(job["id"])

    def _guarded_loop(self):
        try:
            self._loop()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self._terminate()

    def close(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        self._terminate()
        if self.busy_job_id:
            self.store.update_job(self.busy_job_id, status="interrupted", error="The application closed before this job finished")
            self.busy_job_id = None

    def status(self):
        return {"state": "failed" if self.error else "busy" if self.busy_job_id else "idle",
                "busy_job_id": self.busy_job_id, "error": self.error,
                "heartbeat": self.heartbeat, "pid": self.process.pid if self.process else None}
