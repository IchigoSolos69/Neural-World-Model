"""Pinned, offline-only LTX continuation and explicit manual notebook exchange."""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from .video import CONTEXT_FRAMES, FPS, PRESETS, array_sha256, read_frame_archive, validate_frames

MODEL_REPO = "Lightricks/LTX-Video"
MODEL_REVISION = "a71a8c466879b8132b109dee93dbd806e69a47e1"
CONFIG_REPO = "Lightricks/LTX-Video-0.9.5"
CONFIG_REVISION = "e58e28c39631af4d1468ee57a853764e11c1d37e"
CHECKPOINT = "ltxv-2b-0.9.8-distilled.safetensors"
DIFFUSERS_VERSION = "0.35.1"
GIB = 1024**3

# Metadata read from the official repositories at immutable commits, not estimated download sizes.
_MODEL_FILES = [
    (CHECKPOINT, 6340744492, "76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f"),
    ("text_encoder/model-00001-of-00004.safetensors", 4989319680, "7a68b2c8c080696a10109612a649bc69330991ecfea65930ccfdfbdb011f2686"),
    ("text_encoder/model-00002-of-00004.safetensors", 4999830656, "b8ed6556d7507e38af5b428c605fb2a6f2bdb7e80bd481308b865f7a40c551ca"),
    ("text_encoder/model-00003-of-00004.safetensors", 4865612720, "c831635f83041f83faf0024b39c6ecb21b45d70dd38a63ea5bac6c7c6e5e558c"),
    ("text_encoder/model-00004-of-00004.safetensors", 4194506688, "02a5f2d69205be92ad48fe5d712d38c2ff55627969116aeffc58bd75a28da468"),
    ("text_encoder/config.json", 781, "e8425c7fb4b09361c160eba83f89f3b3186d69bb189a8d67ad6bee8a160479a0"),
    ("text_encoder/model.safetensors.index.json", 19886, "a545bb25dc0f423d84be7b577311bba8bb7c6931f1eefcea65fc8b0a61a60a76"),
    ("tokenizer/added_tokens.json", 2593, "ea5a91a3234f66ea642c8e672d67f0f493759a9bee6910ae304ea9b9492118b5"),
    ("tokenizer/special_tokens_map.json", 2543, "7a1985a994c41886db38c719d2a3d2f40606663cc19d7c5d6a85d349320e06d2"),
    ("tokenizer/spiece.model", 791656, "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"),
    ("tokenizer/tokenizer_config.json", 20618, "e8c727fbdfbc495f9ec68fc4b89eda5d7d0a800cc33d1a661ae208b20b50e17a"),
]
_CONFIG_FILES = [
    ("model_index.json", 412, "14d1974241a29026f6dea7bb7e1f89e35b58abc25a994ce3a903f33fa75e75d8"),
    ("scheduler/scheduler_config.json", 455, "195c1b88aa62b95e9c96827ac04e0857334d1bb027abdf06d2ad7684d1c90cf3"),
    ("transformer/config.json", 500, "ce265f6df6494b12121082ce148b2c580f8840921614768fdc1fc921a29685a2"),
    ("vae/config.json", 1251, "fdf9d40b5a5d0ca3b1379a6d3f1753fb8627dee4927bb98e5ebadd14e4a3e4cb"),
]
ASSET_MANIFEST = [{"path": name, "bytes": size, "sha256": digest, "repo": repo, "revision": revision}
                  for repo, revision, rows in ((MODEL_REPO, MODEL_REVISION, _MODEL_FILES),
                                               (CONFIG_REPO, CONFIG_REVISION, _CONFIG_FILES))
                  for name, size, digest in rows]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_preflight(managed_root: str | Path, model_dir: str | Path | None = None,
                    *, quota_bytes: int = 27 * GIB, reserve_bytes: int = 8 * GIB) -> dict:
    root = Path(managed_root).resolve()
    model = Path(model_dir).resolve() if model_dir else root / "models" / "ltx-2b-0.9.8"
    if not model.is_relative_to(root):
        raise ValueError("Model assets must stay inside managed storage")
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    missing = [row for row in ASSET_MANIFEST
               if not (model / row["path"]).is_file() or (model / row["path"]).stat().st_size != row["bytes"]]
    required = sum(row["bytes"] for row in missing)
    free = shutil.disk_usage(ancestor).free
    # Direct streamed downloads rename in place; reserve another 256 MiB for metadata and job output.
    temporary = 256 * 1024**2 if missing else 0
    allowed = used + required + temporary <= quota_bytes and free - required - temporary >= reserve_bytes
    reason = None if allowed else "Insufficient managed quota or disk space after preserving 8 GiB free"
    return {"available": allowed, "reason": reason, "required_download_bytes": required,
            "temporary_bytes": temporary, "disk_free_bytes": free, "managed_used_bytes": used,
            "quota_bytes": quota_bytes, "reserve_bytes": reserve_bytes,
            "total_asset_bytes": sum(row["bytes"] for row in ASSET_MANIFEST),
            "model_dir": str(model), "missing_files": [row["path"] for row in missing]}


def download_assets(managed_root: str | Path, model_dir: str | Path | None = None) -> Path:
    """Explicit operator action only; prediction never calls this function."""
    import urllib.request

    check = asset_preflight(managed_root, model_dir)
    if not check["available"]:
        raise ValueError(check["reason"])
    output = Path(check["model_dir"])
    for row in ASSET_MANIFEST:
        destination = output / row["path"]
        if destination.is_file() and destination.stat().st_size == row["bytes"]:
            if sha256_file(destination) == row["sha256"]:
                continue
            raise ValueError(f"Existing model asset has an invalid checksum: {row['path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            url = f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{row['path']}"
            count = 0
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as stream:
                while chunk := response.read(4 * 1024**2):
                    count += len(chunk)
                    if count > row["bytes"]:
                        raise ValueError("Asset download exceeded its pinned size")
                    stream.write(chunk)
                    digest.update(chunk)
            if count != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError(f"Asset checksum/size mismatch: {row['path']}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    (output / "assets.json").write_text(json.dumps({"model_revision": MODEL_REVISION,
                                                   "files": ASSET_MANIFEST}, indent=2), encoding="utf-8")
    return output


def video_availability(model_dir: str | Path, report_path: str | Path | None = None) -> dict:
    model = Path(model_dir)
    missing = [row["path"] for row in ASSET_MANIFEST if not (model / row["path"]).is_file()
               or (model / row["path"]).stat().st_size != row["bytes"]]
    report = Path(report_path) if report_path else model / "feasibility.json"
    details = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
    verified = (details.get("available") is True and details.get("model_revision") == MODEL_REVISION
                and details.get("consecutive_successes", 0) >= 3
                and tuple(details.get("resolution", [])) in PRESETS.values())
    return {"available": not missing and verified, "model_revision": MODEL_REVISION,
            "reason": "Model assets are missing" if missing else
                      None if verified else details.get("reason", "Three-run hardware feasibility check is required"),
            "resolution": details.get("resolution", list(PRESETS["standard"])), "missing_files": missing,
            "feasibility": details}


class LTXVideoPredictor:
    def __init__(self, model_dir: str | Path, report_path: str | Path | None = None,
                 *, require_verified: bool = True):
        self.model_dir = Path(model_dir).resolve()
        self.status = video_availability(model_dir, report_path)
        if require_verified and not self.status["available"]:
            raise RuntimeError(self.status["reason"])
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import diffusers
            import torch
            from diffusers import LTXConditionPipeline
        except ImportError as error:
            raise RuntimeError('Install the pinned video dependencies: pip install ".[video]"') from error
        if diffusers.__version__ != DIFFUSERS_VERSION:
            raise RuntimeError(f"This adapter requires diffusers=={DIFFUSERS_VERSION}")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The LTX adapter requires a CUDA GPU supporting BF16")
        for row in ASSET_MANIFEST:
            file = self.model_dir / row["path"]
            if not file.is_file() or file.stat().st_size != row["bytes"] or sha256_file(file) != row["sha256"]:
                raise RuntimeError(f"Missing or corrupt pinned model asset: {row['path']}")
        # The root LTX repo's VAE config is for 0.9.0. Pinned 0.9.5 uses the 0.9.8 2B architecture.
        pipe = LTXConditionPipeline.from_single_file(str(self.model_dir / CHECKPOINT),
                                                    config=str(self.model_dir), local_files_only=True,
                                                    torch_dtype=torch.bfloat16)
        pipe.vae.enable_tiling()
        pipe.enable_sequential_cpu_offload()
        self._pipeline = pipe
        return pipe

    def predict(self, prefix: np.ndarray, horizon: int, seed: int = 0) -> np.ndarray:
        validate_frames(prefix, CONTEXT_FRAMES)
        if horizon not in (10, 20, 50) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Invalid video horizon or seed")
        if self.status["available"] and list(prefix.shape[2:0:-1]) != self.status["resolution"]:
            raise ValueError("Input preset differs from the verified hardware preset")
        pipe = self._load()
        import torch
        from PIL import Image
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        total_frames = ((CONTEXT_FRAMES + horizon - 2) // 8 + 1) * 8 + 1
        with torch.inference_mode():
            generated = pipe(conditions=[LTXVideoCondition(video=[Image.fromarray(frame) for frame in prefix],
                                                           frame_index=0, strength=1.0)],
                             prompt="Continue the visible physical scene with natural continuous motion.",
                             negative_prompt="", width=prefix.shape[2], height=prefix.shape[1],
                             num_frames=total_frames, frame_rate=FPS,
                             timesteps=[1000, 993.7, 987.5, 981.2, 975, 909.4, 725, 421.9],
                             guidance_scale=1.0, decode_timestep=0.05, decode_noise_scale=0.025,
                             image_cond_noise_scale=0.0, generator=torch.Generator("cpu").manual_seed(seed),
                             output_type="np").frames[0]
        future = np.rint(np.clip(np.asarray(generated)[CONTEXT_FRAMES:CONTEXT_FRAMES + horizon], 0, 1) * 255).astype(np.uint8)
        validate_frames(future, horizon)
        return future


def _feasibility_attempt(model, path, width, height, seed, connection):
    started = time.monotonic()
    try:
        import psutil
        import torch
        from PIL import Image
        with np.load(path, allow_pickle=False) as archive:
            frames = archive["frames"]
        validate_frames(frames, CONTEXT_FRAMES)
        if frames.shape[1:3] != (height, width):
            frames = np.stack([np.asarray(Image.fromarray(frame).resize((width, height))) for frame in frames])
        torch.cuda.reset_peak_memory_stats()
        LTXVideoPredictor(model, require_verified=False).predict(frames, 50, seed)
        connection.send({"success": True, "elapsed_seconds": time.monotonic() - started,
                         "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                         "process_rss_bytes": psutil.Process().memory_info().rss})
    except Exception as error:
        connection.send({"success": False, "elapsed_seconds": time.monotonic() - started,
                         "error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


def _wait_attempt(process, connection, timeout=1200):
    try:
        if not connection.poll(timeout):
            return {"success": False, "elapsed_seconds": timeout,
                    "error": "Generation exceeded the 20-minute feasibility limit"}
        return connection.recv()
    except EOFError:
        return {"success": False, "error": "Feasibility process exited without a result"}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
        connection.close()


def run_feasibility(managed_root: str | Path, prefixes: list[str | Path], output: str | Path,
                    *, model_dir: str | Path | None = None) -> dict:
    """Three independent development prefixes; no hidden target images are opened."""
    if len(prefixes) != 3:
        raise ValueError("Supply exactly three development prefix archives")
    preflight = asset_preflight(managed_root, model_dir)
    report = {"format": 1, "model_revision": MODEL_REVISION, "available": False,
              "consecutive_successes": 0, "storage": preflight, "attempts": []}
    if not preflight["available"]:
        report["reason"] = preflight["reason"]
    else:
        model = Path(preflight["model_dir"])
        if video_availability(model)["missing_files"]:
            report["reason"] = "Pinned assets must be downloaded explicitly before the feasibility test"
        else:
            from .experiment import gpu_lock, experiment_status, ResourceBusyError
            experiment = experiment_status(Path(__file__).resolve().parents[1] / "configs/full.json")
            if experiment.get("state") == "running" or experiment.get("child_alive"):
                raise ResourceBusyError("Full training is using the GPU; run feasibility after it finishes")
            for width, height in PRESETS.values():
                successes = 0
                for index, path in enumerate(prefixes):
                    row = {"resolution": [width, height], "seed": index}
                    context = mp.get_context("spawn")
                    receiving, sending = context.Pipe(duplex=False)
                    with gpu_lock():
                        process = context.Process(target=_feasibility_attempt,
                                                  args=(str(model), str(path), width, height, index, sending))
                        process.start()
                        sending.close()
                        row.update(_wait_attempt(process, receiving))
                    row.update(fps=FPS, context_timestamps=(np.arange(17) / FPS).tolist(),
                               future_timestamps=(np.arange(17, 67) / FPS).tolist())
                    successes += int(row["success"])
                    report["attempts"].append(row)
                    if not row["success"]:
                        break
                if successes == 3:
                    report.update(available=True, consecutive_successes=3, resolution=[width, height], reason=None)
                    break
            if not report["available"]:
                report["reason"] = "Neither advertised preset passed three consecutive development runs"
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def export_video_job(prefix: np.ndarray, destination: str | Path, *, job_id: str,
                     horizon: int, seed: int = 0) -> dict:
    validate_frames(prefix, CONTEXT_FRAMES)
    if horizon not in (10, 20, 50):
        raise ValueError("Invalid video horizon")
    request = {"format": 1, "job_id": job_id, "seed": seed, "horizon": horizon,
               "model_revision": MODEL_REVISION, "prefix_sha256": array_sha256(prefix),
               "fps": FPS, "expected_shape": [horizon, *prefix.shape[1:]]}
    array = io.BytesIO()
    np.savez_compressed(array, frames=prefix)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("request.json", json.dumps(request, indent=2))
        archive.writestr("prefix.npz", array.getvalue())
    return request


def run_video_job(job_zip: str | Path, result_zip: str | Path, model_dir: str | Path) -> dict:
    """Manual notebook entry point; requests never contain the recorded future."""
    if Path(job_zip).stat().st_size > 20 * 1024**2:
        raise ValueError("Job archive exceeds 20 MiB")
    with zipfile.ZipFile(job_zip) as archive:
        if len(archive.infolist()) != 2 or set(archive.namelist()) != {"request.json", "prefix.npz"}:
            raise ValueError("Job archive must contain only request.json and prefix.npz")
        if sum(item.file_size for item in archive.infolist()) > 20 * 1024**2 or archive.getinfo("request.json").file_size > 65536:
            raise ValueError("Job archive exceeds the permitted uncompressed size")
        request = json.loads(archive.read("request.json"))
        if not isinstance(request, dict):
            raise ValueError("Job request must be an object")
        prefix = read_frame_archive(archive.read("prefix.npz"), CONTEXT_FRAMES)
    validate_frames(prefix, CONTEXT_FRAMES)
    if (request.get("format") != 1 or request.get("model_revision") != MODEL_REVISION
            or request.get("prefix_sha256") != array_sha256(prefix) or request.get("fps") != FPS
            or request.get("horizon") not in (10, 20, 50)
            or request.get("expected_shape") != [request["horizon"], *prefix.shape[1:]]):
        raise ValueError("Job identity, model revision, or prefix checksum is incompatible")
    prediction = LTXVideoPredictor(model_dir, require_verified=False).predict(prefix, request["horizon"], request["seed"])
    response = {**request, "execution_source": "manual_notebook", "prediction_sha256": array_sha256(prediction)}
    array = io.BytesIO()
    np.savez_compressed(array, frames=prediction)
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("response.json", json.dumps(response, indent=2))
        archive.writestr("prediction.npz", array.getvalue())
    return response
