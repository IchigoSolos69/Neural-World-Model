from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nwm.app import create_app
from nwm.app_storage import Store
from nwm.app_worker import _simulation, finish_video
from nwm.serving import export_serving
from nwm.video import array_sha256
from nwm.video_model import export_video_job
from test_serving import source_experiment


@pytest.fixture
def settings(tmp_path):
    bundle = tmp_path / "fake-bundle"
    bundle.mkdir()
    (bundle / "weights.pt").write_bytes(b"test route discovery only")
    (bundle / "manifest.json").write_text(json.dumps({"format_version": 1, "files": {"weights.pt": "unused"},
        "experiment_size": "smoke", "default_model": "fitted_physics", "experiment_signature": "abcdef123456"}))
    return {"root": str(tmp_path / "app"), "limit_gib": 1, "reserve_gib": 0, "device": "cpu",
            "timeout_seconds": 10, "queue_limit": 4, "bundles": [{"id": "smoke", "path": str(bundle)}]}


@pytest.fixture
def client(settings):
    app = create_app(settings, start_worker=False)
    with TestClient(app) as connection:
        yield connection, app.state.store


def send(client, body):
    return client.post("/api/jobs", json=body, headers={"X-NWM-Client": "local"})


def test_local_boundary_and_bounded_request_contract(client):
    connection, store = client
    assert connection.get("/api/status").json()["bundles"][0]["kind"] == "smoke"
    assert connection.get("/api/status", headers={"Host": "evil.example"}).status_code == 403
    assert connection.get("/api/status", headers={"Origin": "https://evil.example"}).status_code == 403
    request = {"mode": "simulation", "bundle_id": "smoke", "horizon": 10}
    assert connection.post("/api/jobs", json=request).status_code == 403
    assert send(connection, {**request, "horizon": 1000}).status_code == 422
    assert send(connection, {**request, "batch_size": 1000}).status_code == 422
    assert send(connection, {**request, "actions": [{"frame": 11, "dx": .1, "dy": 0}]}).status_code == 422
    assert send(connection, {**request, "actions": [{"frame": 1, "dx": .1, "dy": 0}] * 2}).status_code == 422
    assert send(connection, {**request, "bundle_id": "../../secret"}).status_code == 400
    for _ in range(4):
        assert send(connection, request).status_code == 202
    assert send(connection, request).status_code == 400
    first = store.jobs()[0]
    assert connection.get(f"/api/artifacts/{first['id']}/target.npz").status_code == 404
    notebook = connection.get("/api/video-notebook")
    assert notebook.status_code == 200 and json.loads(notebook.content)["nbformat"] == 4


def test_atomic_claim_recovery_cancel_and_owned_deletion(settings):
    store = Store(settings)
    original = store.create_job({"mode": "simulation"})
    with ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(lambda _: Store(settings).claim(), range(2)))
    assert sum(value is not None for value in claims) == 1
    assert store.get_job(original["id"])["status"] == "running"
    Store(settings).recover()
    assert store.get_job(original["id"])["status"] == "interrupted"
    sentinel = store.root.parent / "keep.txt"
    sentinel.write_text("user-owned")
    store.delete_job(original["id"])
    assert sentinel.read_text() == "user-owned"
    with pytest.raises((ValueError, KeyError)):
        store.job_dir("../keep.txt")
    queued = store.create_job({"mode": "simulation"})
    assert store.cancel(queued["id"])["status"] == "cancelled"
    assert store.claim() is None


def test_quota_rejection_precedes_job_creation(client):
    connection, store = client
    store.config["limit_gib"] = 0.001
    assert send(connection, {"mode": "simulation", "bundle_id": "smoke"}).status_code == 400
    assert store.jobs() == []
    assert connection.post("/api/uploads", files={"file": ("bad.mp4", b"not a video")},
                           headers={"X-NWM-Client": "local"}).status_code == 400
    assert not list(store.path("uploads").glob("*.mp4"))


def test_manual_import_is_bound_to_prefix_and_cannot_include_future(client):
    connection, store = client
    prefix = np.zeros((17, 256, 384, 3), dtype=np.uint8)
    job = store.create_job({"mode": "backtest", "horizon": 10, "seed": 7, "execution": "notebook"})
    directory = store.job_dir(job["id"])
    work = directory / "work"
    work.mkdir()
    np.savez_compressed(work / "prefix.npz", frames=prefix)
    np.savez_compressed(work / "target.npz", frames=np.full((10, 256, 384, 3), 255, dtype=np.uint8))
    (directory / "prepared.json").write_text(json.dumps({"prefix_path": str(work / "prefix.npz")}))
    store.update_job(job["id"], status="waiting_notebook")
    downloaded = connection.get(f"/api/jobs/{job['id']}/export")
    assert downloaded.status_code == 200
    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        assert set(archive.namelist()) == {"request.json", "prefix.npz"}
        request = json.loads(archive.read("request.json"))
    prediction = np.zeros((10, 256, 384, 3), dtype=np.uint8)
    array = io.BytesIO()
    np.savez_compressed(array, frames=prediction)

    def response_bytes(identity, extra=False):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("response.json", json.dumps({**identity, "prediction_sha256": array_sha256(prediction)}))
            archive.writestr("prediction.npz", array.getvalue())
            if extra:
                archive.writestr("../escape", "no")
        return data.getvalue()

    url = f"/api/jobs/{job['id']}/import"
    headers = {"X-NWM-Client": "local"}
    for identity, extra in (({**request, "seed": 8}, False), (request, True)):
        assert connection.post(url, files={"file": ("result.zip", response_bytes(identity, extra))}, headers=headers).status_code == 400
    accepted = connection.post(url, files={"file": ("result.zip", response_bytes(request))}, headers=headers)
    assert accepted.status_code == 202 and accepted.json()["status"] == "queued"
    with np.load(work / "imported_prediction.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frames"], prediction)


def test_synthetic_worker_uses_dataset_free_bundle_and_real_metrics(source_experiment, settings, tmp_path):
    bundle = export_serving(source_experiment, tmp_path / "published")
    settings["bundles"][0]["path"] = str(bundle)
    store = Store(settings)
    job = store.create_job({"mode": "simulation", "bundle_id": "smoke", "seed": 1, "horizon": 10,
                            "actions": [{"frame": 1, "dx": .1, "dy": 0}], "drag": .2, "restitution": .9})
    result = _simulation(store, job, {})
    assert len(result["metrics"]) == 10
    assert all(np.isfinite(row["position_rmse_px"]) for row in result["metrics"])
    assert len(result["videos"]) == 4
    assert result["uncertainty"][0]["horizon"] == 10
    assert all(len(curve["horizons"]) == 10 for curve in result["curves"])
    assert result["experiment_size"] == "smoke"


def test_supervisor_reports_failure_instead_of_an_idle_dead_thread(settings, monkeypatch):
    from nwm.app_worker import Supervisor
    supervisor = Supervisor(settings)
    def broken():
        raise OSError("disk unavailable")
    monkeypatch.setattr(supervisor, "_loop", broken)
    supervisor._guarded_loop()
    assert supervisor.status()["state"] == "failed"
    assert "disk unavailable" in supervisor.status()["error"]


def test_deleting_last_video_job_reclaims_only_its_unreferenced_upload(settings):
    store = Store(settings)
    upload_id = "a" * 32
    path = store.path("uploads", upload_id + ".mp4")
    path.parent.mkdir()
    path.write_bytes(b"owned upload")
    store.add_upload(upload_id, {"path": str(path)})
    first = store.create_job({"mode": "forecast", "upload_id": upload_id})
    second = store.create_job({"mode": "forecast", "upload_id": upload_id})
    store.cancel(first["id"])
    store.cancel(second["id"])
    store.delete_job(first["id"])
    assert path.is_file()
    store.delete_job(second["id"])
    assert not path.exists()


def test_source_download_never_serves_experiment_archives(client, tmp_path, monkeypatch):
    import nwm.app as module
    connection, _ = client
    monkeypatch.setattr(module, "PROJECT", tmp_path)
    assert connection.get("/api/source-bundle").status_code == 404
    path = tmp_path / "nwm_colab.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nwm/__init__.py", "")
    assert connection.get("/api/source-bundle").status_code == 200
    for name in ("data/test.npz", "runs/full/checkpoint.pt", "../private", ".env"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(name, "must not leave as source")
        assert connection.get("/api/source-bundle").status_code == 400


@pytest.mark.parametrize("failure,expected", [("cancel", "cancelled"), ("timeout", "failed"), ("crash", "interrupted")])
def test_supervisor_releases_busy_slot_on_cancel_timeout_or_crash(settings, monkeypatch, failure, expected):
    from nwm.app_worker import Supervisor
    supervisor = Supervisor(settings)
    job = supervisor.store.create_job({"mode": "simulation"})
    supervisor.store.claim()
    supervisor.busy_job_id = job["id"]
    if failure == "cancel":
        supervisor.store.cancel(job["id"])
    class Process:
        alive = failure != "crash"
        def is_alive(self): return self.alive
        def terminate(self): self.alive = False
        def join(self, timeout): pass
        def close(self): pass
    supervisor.process = Process()
    steps = iter([False, True])
    monkeypatch.setattr(supervisor._stop, "wait", lambda _: next(steps))
    moments = iter([0, 20 if failure == "timeout" else 0])
    monkeypatch.setattr("nwm.app_worker.time.monotonic", lambda: next(moments))
    monkeypatch.setattr(supervisor, "_spawn", lambda: None)
    supervisor._loop()
    assert supervisor.store.get_job(job["id"])["status"] == expected
    assert supervisor.busy_job_id is None
