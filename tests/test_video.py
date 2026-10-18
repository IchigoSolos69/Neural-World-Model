import json
import io
from pathlib import Path
import zipfile

import imageio.v2 as imageio
import numpy as np
import pytest

from nwm.video import prepare_video, probe_video, score_video
from nwm.video_benchmark import select_scenarios
from nwm.video_model import asset_preflight, video_availability, export_video_job


def clip(path, changed_future=False):
    frames = np.zeros((80, 48, 64, 3), dtype=np.uint8)
    for index in range(80):
        frames[index, :, :, 1] = index
        frames[index, 10:20, min(index, 48):min(index, 48) + 8, 0] = 255
    if changed_future:
        frames[18:] = 250
    imageio.mimsave(path, frames, fps=16, codec="libx264", macro_block_size=1,
                    output_params=["-crf", "0", "-g", "1", "-threads", "1"])
    return path


def test_future_changes_do_not_change_prediction_prefix(tmp_path):
    first = prepare_video(clip(tmp_path / "a.mp4"), tmp_path / "a", 1.0625, 20, backtest=True)
    second = prepare_video(clip(tmp_path / "b.mp4", True), tmp_path / "b", 1.0625, 20, backtest=True)
    assert first["prefix_sha256"] == second["prefix_sha256"]
    assert first["future_timestamps"][0] == 1.125
    assert first["context_timestamps"][-1] == 1.0625
    with np.load(first["prefix_path"]) as data:
        assert data["frames"].shape == (17, 256, 384, 3)
    with np.load(first["target_path"]) as a, np.load(second["target_path"]) as b:
        assert not np.array_equal(a["frames"], b["frames"])
        prediction = np.zeros_like(a["frames"])
        scores = [score_video(prediction, value["frames"], prediction[0])["pixel_mse"] for value in (a, b)]
        assert scores[0] != scores[1]
    forecast = prepare_video(tmp_path / "a.mp4", tmp_path / "forecast", 1.0625, 50)
    assert forecast["target_path"] is None and not (tmp_path / "forecast/target.npz").exists()


def test_video_limits_short_context_and_letterbox(tmp_path):
    path = clip(tmp_path / "source.mp4")
    assert probe_video(path)["width"] == 64
    for cutoff, horizon in ((.5, 10), (5, 10), (1, 51)):
        with pytest.raises(ValueError):
            prepare_video(path, tmp_path / "bad", cutoff, horizon)
    with pytest.raises(ValueError, match="recorded footage"):
        prepare_video(path, tmp_path / "bad", 3, 50, backtest=True)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="corrupt"):
        probe_video(bad)
    portrait = tmp_path / "portrait.mp4"
    imageio.mimsave(portrait, np.full((40, 96, 48, 3), 255, dtype=np.uint8), fps=16, macro_block_size=1)
    result = prepare_video(portrait, tmp_path / "portrait", 1, 10)
    with np.load(result["prefix_path"]) as data:
        assert np.all(data["frames"][:, :, :50] == 0)


def test_asset_gate_refuses_insufficient_space_without_downloading(tmp_path, monkeypatch):
    import nwm.video_model as model
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(model.shutil, "disk_usage", lambda path: usage(100, 90, 10))
    check = asset_preflight(tmp_path)
    assert not check["available"] and check["required_download_bytes"] > 23 * 1024**3
    assert not video_availability(tmp_path / "model")["available"]
    with pytest.raises(ValueError, match="Insufficient"):
        model.download_assets(tmp_path)
    assert not list(tmp_path.rglob("*.part"))


def test_benchmark_scenarios_are_isolated_and_order_invariant():
    objects = [{"name": f"full-videos/take-1/16FPS/{i:04}_full-videos_16FPS_perspective-center_take-1_trimmed-scene-{i}.mp4",
                "size": "100", "generation": "1"} for i in range(20)]
    selected = select_scenarios(objects)
    assert selected == select_scenarios(list(reversed(objects)))
    development = {row["scenario_id"] for row in selected if row["split"] == "development"}
    testing = {row["scenario_id"] for row in selected if row["split"] == "test"}
    assert len(development) == 3 and len(testing) == 8 and not development & testing


def test_manual_notebook_runner_is_prefix_only_and_result_is_bound(tmp_path, monkeypatch):
    import nwm.video_model as model
    from nwm.video import array_sha256
    prefix = np.zeros((17, 192, 288, 3), dtype=np.uint8)
    job, result = tmp_path / "job.zip", tmp_path / "result.zip"
    request = export_video_job(prefix, job, job_id="a" * 32, horizon=10, seed=7)
    calls = []

    class FakePredictor:
        def __init__(self, directory, require_verified):
            assert require_verified is False

        def predict(self, observed, horizon, seed):
            calls.append((observed.copy(), horizon, seed))
            return np.repeat(observed[-1:], horizon, axis=0)

    monkeypatch.setattr(model, "LTXVideoPredictor", FakePredictor)
    response = model.run_video_job(job, result, tmp_path / "weights")
    assert len(calls) == 1 and calls[0][1:] == (10, 7)
    np.testing.assert_array_equal(calls[0][0], prefix)
    assert all(response[key] == value for key, value in request.items())
    assert response["prediction_sha256"] == array_sha256(np.zeros((10, 192, 288, 3), dtype=np.uint8))
    with zipfile.ZipFile(result) as archive:
        assert set(archive.namelist()) == {"response.json", "prediction.npz"}


def test_import_rejects_oversized_array_header_before_allocating():
    from nwm.video import read_frame_archive
    raw = io.BytesIO()
    np.lib.format.write_array_header_1_0(raw, {"descr": "|u1", "fortran_order": False,
                                             "shape": (17, 256, 384, 99999999)})
    encoded = io.BytesIO()
    with zipfile.ZipFile(encoded, "w") as archive:
        archive.writestr("frames.npy", raw.getvalue())
    with pytest.raises(ValueError, match="shape"):
        read_frame_archive(encoded.getvalue(), 17)


def test_feasibility_timeout_terminates_and_reaps_process():
    from nwm.video_model import _wait_attempt
    events = []

    class Process:
        def is_alive(self): return not events
        def terminate(self): events.append("terminate")
        def join(self, timeout): events.append("join")
        def close(self): events.append("process_close")

    class Connection:
        def poll(self, timeout): return False
        def close(self): events.append("pipe_close")

    result = _wait_attempt(Process(), Connection(), timeout=.01)
    assert not result["success"] and "limit" in result["error"]
    assert events == ["terminate", "join", "process_close", "pipe_close"]


def test_media_decoder_cannot_follow_network_references():
    from nwm.video import _ffmpeg
    result = _ffmpeg(["-i", "https://127.0.0.1:9/video.mp4", "-frames:v", "0", "-f", "null", "-"], timeout=3)
    assert result.returncode and b"not on whitelist" in result.stderr
