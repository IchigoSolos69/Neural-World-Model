"""Bounded local video preparation. Predictors receive sanitized RGB prefixes only."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

import imageio_ffmpeg
import numpy as np

FPS = 16
CONTEXT_FRAMES = 17
MAX_BYTES = 100 * 1024**2
MAX_SECONDS = 30
PRESETS = {"standard": (384, 256), "small": (288, 192)}


def _ffmpeg(args: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner",
                               "-protocol_whitelist", "file,pipe", *args],
                              capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise ValueError("Video inspection or decoding exceeded its time limit") from error


def probe_video(path: str | Path) -> dict:
    path = Path(path).resolve()
    if path.suffix.lower() not in (".mp4", ".webm") or not path.is_file():
        raise ValueError("Choose an existing MP4 or WebM file")
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Video must be nonempty and at most 100 MiB")
    result = _ffmpeg(["-i", str(path), "-map", "0:v:0", "-frames:v", "0", "-f", "null", "-"], 10)
    text = result.stderr.decode("utf-8", errors="replace")
    duration = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    stream = next((line for line in text.splitlines() if "Stream #" in line and "Video:" in line), "")
    dimensions = re.search(r"\b(\d{2,5})x(\d{2,5})\b", stream)
    if result.returncode or not duration or not dimensions:
        raise ValueError("Video is corrupt, unsupported, or has no finite-duration video stream")
    seconds = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    width, height = map(int, dimensions.groups())
    if not 0 < seconds <= MAX_SECONDS:
        raise ValueError("Video duration must be at most 30 seconds")
    if max(width, height) > 1920 or min(width, height) > 1080:
        raise ValueError("Video dimensions must be at most 1920 by 1080, or the portrait equivalent")
    rotation = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", text)
    return {"duration_seconds": seconds, "width": width, "height": height,
            "rotation_degrees": float(rotation[1]) if rotation else 0.0,
            "bytes": path.stat().st_size}


def _decode_until(path: Path, end_seconds: float, width: int, height: int) -> np.ndarray:
    # Trim before resampling: no frame at/after the withheld boundary can enter the prefix.
    filters = (f"trim=end={end_seconds:.9f},setpts=PTS-STARTPTS,"
               f"fps={FPS}:start_time=0:round=up:eof_action=pass,"
               "tpad=stop_mode=clone:stop_duration=0.125,"
               f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    max_frames = min(math.ceil(end_seconds * FPS) + 2, MAX_SECONDS * FPS + 2)
    result = _ffmpeg(["-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
                      "-vf", filters, "-frames:v", str(max_frames), "-threads", "2",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    size = width * height * 3
    if result.returncode or not result.stdout or len(result.stdout) % size:
        raise ValueError("Video could not be decoded")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3).copy()


def array_sha256(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def validate_frames(frames: np.ndarray, count: int | None = None) -> None:
    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.ndim != 4
            or frames.shape[-1] != 3 or (count is not None and len(frames) != count)
            or tuple(frames.shape[2:0:-1]) not in PRESETS.values()):
        raise ValueError("Frames must be uint8 RGB [time,height,width,3] at an advertised preset")


def read_frame_archive(encoded: bytes, count: int) -> np.ndarray:
    """Check both ZIP expansion and the NPY header before allocating an imported array."""
    maximum = count * 256 * 384 * 3 + 4096
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        if archive.namelist() != ["frames.npy"] or archive.infolist()[0].file_size > maximum:
            raise ValueError("Frame archive exceeds its array size limit")
        raw = archive.read("frames.npy")
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported array header")
    if (dtype != np.dtype("uint8") or len(shape) != 4 or shape[0] != count or shape[-1] != 3
            or tuple(shape[2:0:-1]) not in PRESETS.values()):
        raise ValueError("Invalid frame array shape or dtype")
    frames = np.load(io.BytesIO(raw), allow_pickle=False)
    validate_frames(frames, count)
    return frames


def prepare_video(path: str | Path, output_dir: str | Path, cutoff_seconds: float, horizon: int,
                  *, backtest: bool = False, width: int = 384, height: int = 256) -> dict:
    """Use 17 samples ending at floor(cutoff*16)/16; future begins one step later.

    Source sampling holds the latest available frame. FFmpeg applies rotation metadata
    before aspect-preserving letterboxing. Future targets are never stored in prefix.npz.
    """
    if horizon not in (10, 20, 50) or (width, height) not in PRESETS.values():
        raise ValueError("Choose horizon 10, 20, or 50 and a supported resolution")
    if not math.isfinite(cutoff_seconds) or cutoff_seconds < 1:
        raise ValueError("The observation cutoff must include at least one second of context")
    source = Path(path).resolve()
    info = probe_video(source)
    final_index = math.floor(cutoff_seconds * FPS + 1e-8)
    cutoff = final_index / FPS
    if cutoff >= info["duration_seconds"]:
        raise ValueError("The observation cutoff must be inside the video")
    if backtest and (final_index + horizon) / FPS >= info["duration_seconds"]:
        raise ValueError("Backtest requires recorded footage through the requested horizon")
    observed = _decode_until(source, cutoff + 1e-7, width, height)
    if len(observed) <= final_index:
        raise ValueError("Video contains too few observed frames")
    prefix = observed[final_index - CONTEXT_FRAMES + 1:final_index + 1]
    validate_frames(prefix, CONTEXT_FRAMES)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix_path = output / "prefix.npz"
    context_times = np.arange(final_index - 16, final_index + 1) / FPS
    future_times = np.arange(final_index + 1, final_index + horizon + 1) / FPS
    np.savez_compressed(prefix_path, frames=prefix, timestamps=context_times)
    target_path = None
    if backtest:
        all_frames = _decode_until(source, (final_index + horizon) / FPS + 1e-7, width, height)
        target = all_frames[final_index + 1:final_index + horizon + 1]
        validate_frames(target, horizon)
        target_path = output / "target.npz"
        np.savez_compressed(target_path, frames=target, timestamps=future_times)
    metadata = {"format": 1, "fps": FPS, "context_frames": CONTEXT_FRAMES,
                "horizon": horizon, "cutoff_seconds": cutoff, "requested_cutoff_seconds": cutoff_seconds,
                "context_timestamps": context_times.tolist(), "future_timestamps": future_times.tolist(),
                "expected_shape": [horizon, height, width, 3], "prefix_sha256": array_sha256(prefix),
                "sampling": "latest source frame at or before each 16 FPS sample; letterboxed RGB",
                "source": info, "backtest": backtest}
    metadata_path = output / "prepared.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "prefix_path": str(prefix_path), "target_path": str(target_path) if target_path else None,
            "metadata_path": str(metadata_path)}


def score_video(forecast: np.ndarray, target: np.ndarray, last_observed: np.ndarray) -> dict:
    validate_frames(forecast)
    validate_frames(target, len(forecast))
    if forecast.shape != target.shape or last_observed.shape != target.shape[1:]:
        raise ValueError("Prediction, target, and last observed frame dimensions differ")
    future = target.astype(np.float32) / 255
    error = ((forecast.astype(np.float32) / 255 - future) ** 2).mean(axis=(1, 2, 3))
    hold = ((last_observed.astype(np.float32)[None] / 255 - future) ** 2).mean(axis=(1, 2, 3))
    return {"pixel_mse": float(error.mean()), "pixel_mse_curve": error.tolist(),
            "frame_hold_pixel_mse": float(hold.mean()), "frame_hold_pixel_mse_curve": hold.tolist(),
            "metric_range": "RGB scaled to [0,1], including letterbox pixels",
            "interpretation": "Video reconstruction error; not a physics accuracy or uncertainty score"}


def write_video(frames: np.ndarray, destination: str | Path, fps: int = FPS) -> Path:
    import imageio.v2 as imageio
    destination = Path(destination)
    imageio.mimsave(destination, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1,
                    output_params=["-movflags", "+faststart"])
    return destination
