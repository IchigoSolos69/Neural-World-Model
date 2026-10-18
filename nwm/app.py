"""Loopback-only HTTP application with durable, bounded prediction jobs."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import io
import json
from pathlib import Path
import re
import uuid
import zipfile

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .app_storage import Store, app_config
from .app_worker import Supervisor

PROJECT = Path(__file__).resolve().parents[1]
LIMIT = 100 * 1024 ** 2


class Impulse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    frame: int = Field(ge=1, le=50)
    dx: float = Field(ge=-1, le=1)
    dy: float = Field(ge=-1, le=1)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: str = Field(pattern="^(simulation|forecast|backtest)$")
    bundle_id: str | None = Field(default=None, max_length=40)
    seed: int = Field(default=0, ge=0, lt=2**31)
    horizon: int = 50
    actions: list[Impulse] = Field(default_factory=list, max_length=50)
    drag: float = Field(default=0.2, ge=0, le=1)
    restitution: float = Field(default=0.9, ge=0, le=1)
    upload_id: str | None = None
    cutoff_seconds: float = Field(default=1, ge=1, lt=30)
    execution: str = Field(default="local", pattern="^(local|notebook)$")

    @model_validator(mode="after")
    def causal_request(self):
        if self.horizon not in (10, 20, 50):
            raise ValueError("Choose horizon 10, 20, or 50")
        frames = [action.frame for action in self.actions]
        if len(set(frames)) != len(frames) or any(frame > self.horizon for frame in frames):
            raise ValueError("Impulse frames must be distinct and inside the forecast horizon")
        if self.mode == "simulation":
            if self.upload_id or self.execution != "local":
                raise ValueError("Simulation runs locally and does not accept a video upload")
        elif not self.upload_id or self.actions:
            raise ValueError("Real-video jobs require an upload and do not support interventions")
        return self


def _bundles(config):
    result = []
    for item in config.get("bundles", []):
        path = Path(item["path"]) / "manifest.json"
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != 1 or not manifest.get("files"):
                continue
            if any(not re.fullmatch(r"[a-zA-Z0-9_.-]+", name) or not (path.parent / name).is_file()
                   for name in manifest["files"]):
                continue
            kind = manifest.get("experiment_size", "smoke")
            result.append({"id": item["id"], "kind": kind,
                           "label": "Full experiment" if kind == "full" else
                                    "Partial full experiment · undertrained" if kind == "full_partial" else
                                    "Smoke demo · undertrained",
                           "default_model": manifest["default_model"],
                           "version": manifest["experiment_signature"][:12]})
        except (KeyError, ValueError, OSError):
            continue
    return result


def _video_status(store):
    from .video_model import asset_preflight, video_availability
    model_dir = store.path("models", "ltx-2b-0.9.8")
    preflight = asset_preflight(store.root, model_dir,
                                quota_bytes=int(store.config.get("limit_gib", 27) * 1024**3),
                                reserve_bytes=int(store.config.get("reserve_gib", 8) * 1024**3))
    value = video_availability(model_dir)
    value["storage"] = preflight
    if not preflight["available"]:
        value.update(available=False, reason=preflight["reason"])
    return value


def create_app(config="configs/app.json", *, start_worker=True):
    config = app_config(config) if not isinstance(config, dict) else config
    store = Store(config)
    supervisor = Supervisor(config)
    uploads_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(application):
        from .experiment import _file_lock
        with _file_lock(store.path("app.lock"), "Local application"):
            if start_worker:
                supervisor.start()
            try:
                yield
            finally:
                if start_worker:
                    supervisor.close()

    application = FastAPI(title="Neural World Model", version="0.2.0", lifespan=lifespan,
                          docs_url=None, redoc_url=None)
    application.state.store = store
    application.state.supervisor = supervisor

    @application.middleware("http")
    async def loopback_boundary(request, call_next):
        host = request.headers.get("host", "").split(":", 1)[0]
        allowed = {"127.0.0.1", "localhost"} | ({"testserver"} if not start_worker else set())
        if host not in allowed:
            return JSONResponse({"detail": "This application accepts localhost requests only"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin not in {f"http://{request.headers.get('host')}", f"https://{request.headers.get('host')}"}:
            return JSONResponse({"detail": "Cross-origin access is disabled"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("x-nwm-client") != "local":
            return JSONResponse({"detail": "Missing local application request header"}, status_code=403)
        length = request.headers.get("content-length")
        try:
            if length and int(length) > LIMIT + 1024 * 1024:
                return JSONResponse({"detail": "Upload exceeds 100 MiB"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid content length"}, status_code=400)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'"
        return response

    @application.exception_handler(ValueError)
    async def invalid_request(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @application.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({"detail": "Requested record does not exist"}, status_code=404)

    @application.get("/api/status")
    def status():
        return {"storage": store.capacity(), "bundles": _bundles(config), "worker": supervisor.status(),
                "video": _video_status(store), "jobs": store.jobs()}

    @application.get("/api/health")
    def health():
        worker = supervisor.status()
        return {"status": "error" if worker["state"] == "failed" else "ok",
                "ready": bool(_bundles(config)) and worker["state"] != "failed", "worker": worker}

    @application.get("/api/video-notebook")
    def video_notebook():
        return FileResponse(PROJECT / "notebooks/video_job.ipynb", filename="video_job.ipynb")

    @application.get("/api/source-bundle")
    def source_bundle():
        path = PROJECT / "nwm_colab.zip"
        if not path.is_file():
            raise HTTPException(404, "Create the source-only bundle with tools/make_colab_bundle.py first")
        with zipfile.ZipFile(path) as archive:
            source_folders = {"nwm", "configs", "notebooks", "tools", "tests", "requirements", "frontend"}
            source_files = {"README.md", "pyproject.toml", ".gitignore", "run.bat"}
            if sum(row.file_size for row in archive.infolist()) > 200 * 1024**2 or any(
                (Path(name).parts[0] not in source_folders and name not in source_files)
                or ".." in Path(name).parts or "node_modules" in Path(name).parts
                or "\\" in name or name.startswith("/") for name in archive.namelist()
            ):
                raise ValueError("Rebuild a source-only bundle without --include-artifacts")
        return FileResponse(path, filename="nwm_colab.zip")

    @application.get("/api/jobs")
    def jobs():
        return store.jobs()

    @application.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return store.get_job(job_id)

    @application.post("/api/jobs", status_code=202)
    def submit(request: JobRequest):
        store.ensure_space(64 * 1024**2)
        values = request.model_dump()
        if request.mode == "simulation":
            if request.bundle_id not in {item["id"] for item in _bundles(config)}:
                raise ValueError("Select an available serving bundle")
        else:
            upload = store.upload(request.upload_id)
            if request.cutoff_seconds >= upload["duration_seconds"]:
                raise ValueError("Observation cutoff must be inside the video")
            if request.mode == "backtest" and request.cutoff_seconds + request.horizon / 16 >= upload["duration_seconds"]:
                raise ValueError("The video must contain the entire held-out future")
            if request.execution == "local" and not _video_status(store)["available"]:
                raise ValueError("Local video generation is unavailable; choose notebook export or resolve the readiness check")
        return store.create_job(values)

    @application.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return store.cancel(job_id)

    @application.post("/api/jobs/{job_id}/retry", status_code=202)
    def retry(job_id: str):
        previous = store.get_job(job_id)
        if previous["status"] not in {"failed", "interrupted", "cancelled"}:
            raise ValueError("Only failed, interrupted, or cancelled jobs can be retried")
        return submit(JobRequest(**previous["request"]))

    @application.delete("/api/jobs/{job_id}")
    def delete(job_id: str):
        store.delete_job(job_id)
        return {"deleted": job_id}

    @application.post("/api/uploads", status_code=201)
    async def upload(file: UploadFile):
        from .video import probe_video
        from starlette.concurrency import run_in_threadpool
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp4", ".webm"}:
            raise ValueError("Choose an MP4 or WebM file")
        upload_id = uuid.uuid4().hex
        destination = store.path("uploads", upload_id + suffix)
        destination.parent.mkdir(exist_ok=True)
        count = 0
        async with uploads_lock:
            try:
                with destination.open("xb") as stream:
                    while chunk := await file.read(1024 * 1024):
                        count += len(chunk)
                        if count > LIMIT:
                            raise ValueError("Video exceeds 100 MiB")
                        with store.lock:
                            store.ensure_space(len(chunk))
                            stream.write(chunk)
                metadata = await run_in_threadpool(probe_video, destination)
                metadata.update(id=upload_id, name=Path(file.filename).name, path=str(destination), duration=metadata["duration_seconds"])
                store.add_upload(upload_id, metadata)
                return {key: value for key, value in metadata.items() if key != "path"}
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            finally:
                await file.close()

    @application.delete("/api/uploads/{upload_id}")
    def delete_upload(upload_id: str):
        upload = store.upload(upload_id)
        if store.upload_referenced(upload_id):
            raise ValueError("Delete jobs that reference this upload first")
        path = Path(upload["path"]).resolve()
        if not path.is_relative_to(store.path("uploads")):
            raise ValueError("Invalid managed upload path")
        path.unlink(missing_ok=True)
        with store.connect() as connection:
            connection.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
        return {"deleted": upload_id}

    @application.get("/api/jobs/{job_id}/export")
    def export_job(job_id: str):
        import numpy as np
        from .video_model import export_video_job
        current = store.get_job(job_id)
        directory = store.job_dir(job_id)
        if current["status"] == "succeeded":
            destination = directory / "results.zip"
            store.ensure_space(sum(path.stat().st_size for path in (directory / "artifacts").iterdir()
                                   if path.is_file()) + 65536)
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("job.json", json.dumps(current, indent=2))
                for path in (directory / "artifacts").iterdir():
                    if path.is_file() and not path.is_symlink():
                        archive.write(path, path.name)
            store.ensure_space()
            return FileResponse(destination, filename=f"nwm-results-{job_id}.zip")
        if current["status"] != "waiting_notebook":
            raise ValueError("This job is not ready for notebook export")
        prepared = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
        with np.load(prepared["prefix_path"], allow_pickle=False) as archive:
            prefix = archive["frames"]
        destination = directory / "notebook_job.zip"
        identity = export_video_job(prefix, destination, job_id=job_id,
                                    horizon=current["request"]["horizon"], seed=current["request"]["seed"])
        (directory / "notebook_request.json").write_text(json.dumps(identity), encoding="utf-8")
        return FileResponse(destination, filename=f"nwm-job-{job_id}.zip")

    @application.post("/api/jobs/{job_id}/import", status_code=202)
    async def import_job(job_id: str, file: UploadFile):
        import numpy as np
        from .video import array_sha256, read_frame_archive
        current = store.get_job(job_id)
        directory = store.job_dir(job_id)
        if current["status"] != "waiting_notebook" or not (directory / "notebook_request.json").is_file():
            raise ValueError("Export the waiting notebook job before importing a result")
        expected = json.loads((directory / "notebook_request.json").read_text(encoding="utf-8"))
        data = await file.read(20 * 1024**2 + 1)
        await file.close()
        if len(data) > 20 * 1024**2:
            raise ValueError("Notebook result exceeds 20 MiB")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if len(archive.infolist()) != 2 or set(archive.namelist()) != {"response.json", "prediction.npz"}:
                    raise ValueError("Notebook result must contain only response.json and prediction.npz")
                if sum(row.file_size for row in archive.infolist()) > 20 * 1024**2 or archive.getinfo("response.json").file_size > 65536:
                    raise ValueError("Notebook result exceeds the uncompressed size limit")
                response = json.loads(archive.read("response.json"))
                if not isinstance(response, dict):
                    raise ValueError("Notebook response must be an object")
                if any(response.get(key) != value for key, value in expected.items()):
                    raise ValueError("Notebook result does not match this job, prefix, model, or seed")
                encoded = archive.read("prediction.npz")
            frames = read_frame_archive(encoded, expected["horizon"])
            if list(frames.shape) != expected["expected_shape"] or array_sha256(frames) != response.get("prediction_sha256"):
                raise ValueError("Notebook prediction shape or checksum does not match")
        except (zipfile.BadZipFile, KeyError, OSError) as error:
            raise ValueError("Invalid notebook result archive") from error
        store.ensure_space(len(encoded) + 32 * 1024**2)
        work = directory / "work"
        work.mkdir(exist_ok=True)
        np.savez_compressed(work / "imported_prediction.npz", frames=frames)
        store.update_job(job_id, status="queued", message="Notebook prediction imported; waiting for local scoring", progress=0.2)
        return store.get_job(job_id)

    @application.get("/api/artifacts/{job_id}/{filename}")
    def artifact(job_id: str, filename: str):
        current = store.get_job(job_id)
        allowed = {"metrics.json", "prediction.npz", "commit.json", "backtest.json"}
        allowed.update(Path(item["url"]).name for item in (current.get("result") or {}).get("videos", []))
        if filename not in allowed:
            raise HTTPException(404, "Unknown public artifact")
        path = store.job_dir(job_id) / "artifacts" / filename
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "Artifact is not available")
        return FileResponse(path)

    assets = PROJECT / "frontend/dist/assets"
    if assets.is_dir():
        application.mount("/assets", StaticFiles(directory=assets), name="assets")

    @application.get("/")
    def index():
        path = PROJECT / "frontend/dist/index.html"
        if not path.is_file():
            return JSONResponse({"detail": "Build the interface with npm ci and npm run build inside frontend/"}, status_code=503)
        return FileResponse(path)

    return application
