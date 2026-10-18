"""Bounded local files and a small durable single-worker job ledger."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import uuid

GIB = 1024 ** 3
IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def now():
    return datetime.now(timezone.utc).isoformat()


def app_config(path="configs/app.json"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config["root"] = str(Path(config["root"]).expanduser().resolve())
    if config.get("limit_gib", 27) <= 0 or config.get("reserve_gib", 8) < 0:
        raise ValueError("Storage limit must be positive and reserve nonnegative")
    if not 1 <= config.get("queue_limit", 4) <= 4:
        raise ValueError("Queue limit must be between one and four")
    if not 1 <= config.get("timeout_seconds", 1200) <= 1200:
        raise ValueError("Job timeout must be between one and 1200 seconds")
    ids = [item["id"] for item in config.get("bundles", [])]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9_-]{1,40}", key) for key in ids):
        raise ValueError("Bundle IDs must be unique simple names")
    for item in config.get("bundles", []):
        item["path"] = str(Path(item["path"]).resolve())
    return config


class Store:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["root"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "jobs.sqlite3"
        self.lock = threading.RLock()
        with self.connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, mode TEXT NOT NULL, status TEXT NOT NULL,
                    request TEXT NOT NULL, result TEXT, progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '', error TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, metadata TEXT NOT NULL, created_at TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def path(self, *parts):
        path = self.root.joinpath(*parts).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Path leaves managed storage")
        return path

    def job_dir(self, job_id):
        if not IDENTIFIER.fullmatch(job_id):
            raise KeyError("Unknown job")
        folder = self.path("jobs", job_id)
        if folder != self.root / "jobs" / job_id:
            raise ValueError("Managed job directories cannot redirect through links")
        return folder

    def capacity(self):
        used = sum(path.stat().st_size for path in self.root.rglob("*")
                   if path.is_file() and not path.is_symlink())
        return {"used_bytes": used, "limit_bytes": int(self.config.get("limit_gib", 27) * GIB),
                "free_bytes": shutil.disk_usage(self.root).free,
                "reserve_bytes": int(self.config.get("reserve_gib", 8) * GIB)}

    def ensure_space(self, additional=0):
        data = self.capacity()
        if data["used_bytes"] + additional > data["limit_bytes"]:
            raise ValueError("Managed storage quota exceeded; delete unwanted jobs or add storage")
        if data["free_bytes"] - additional < data["reserve_bytes"]:
            raise ValueError("Insufficient disk space to preserve the configured free-space reserve")
        return data

    @staticmethod
    def _job(row):
        if row is None:
            raise KeyError("Unknown job")
        value = dict(row)
        value["request"] = json.loads(value["request"])
        value["result"] = json.loads(value["result"]) if value["result"] else None
        value["cancel_requested"] = bool(value["cancel_requested"])
        return value

    def create_job(self, request):
        job_id, timestamp = uuid.uuid4().hex, now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = connection.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','waiting_notebook')").fetchone()[0]
            if count >= self.config.get("queue_limit", 4):
                raise ValueError("The queue is full; wait for a job to finish or cancel one")
            connection.execute("INSERT INTO jobs(id,mode,status,request,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                               (job_id, request["mode"], "queued", json.dumps(request), timestamp, timestamp))
        self.job_dir(job_id).mkdir(parents=True)
        return self.get_job(job_id)

    def get_job(self, job_id):
        self.job_dir(job_id)
        with self.connect() as connection:
            return self._job(connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def jobs(self):
        with self.connect() as connection:
            return [self._job(row) for row in connection.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 200")]

    def update_job(self, job_id, **fields):
        if not fields or not set(fields) <= {"status", "result", "progress", "message", "error", "cancel_requested"}:
            raise ValueError("Invalid job update")
        if "result" in fields:
            fields["result"] = json.dumps(fields["result"])
        fields["updated_at"] = now()
        with self.connect() as connection:
            connection.execute("UPDATE jobs SET " + ",".join(f"{key}=?" for key in fields) + " WHERE id=?",
                               (*fields.values(), job_id))

    def claim(self):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            connection.execute("UPDATE jobs SET status='running',updated_at=?,message=? WHERE id=?",
                               (now(), "Starting prediction", row["id"]))
            value = self._job(row)
            value["status"] = "running"
            return value

    def recover(self):
        with self.connect() as connection:
            connection.execute("UPDATE jobs SET status='interrupted',error=?,updated_at=? WHERE status='running'",
                               ("The worker stopped before completing this job. Retry to run it again.", now()))

    def cancel(self, job_id):
        job = self.get_job(job_id)
        if job["status"] not in TERMINAL:
            self.update_job(job_id, cancel_requested=True,
                            **({} if job["status"] == "running" else {"status": "cancelled", "message": "Cancelled"}))
        return self.get_job(job_id)

    def add_upload(self, upload_id, metadata):
        with self.connect() as connection:
            connection.execute("INSERT INTO uploads VALUES(?,?,?)", (upload_id, json.dumps(metadata), now()))

    def upload(self, upload_id):
        if not IDENTIFIER.fullmatch(upload_id):
            raise KeyError("Unknown upload")
        with self.connect() as connection:
            row = connection.execute("SELECT metadata FROM uploads WHERE id=?", (upload_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown upload")
        return json.loads(row[0])

    def upload_referenced(self, upload_id):
        with self.connect() as connection:
            return any(json.loads(row[0]).get("upload_id") == upload_id
                       for row in connection.execute("SELECT request FROM jobs"))

    def delete_job(self, job_id):
        job = self.get_job(job_id)
        if job["status"] not in TERMINAL:
            raise ValueError("Cancel the job before deleting its files")
        folder = self.job_dir(job_id)
        if folder.is_dir():
            shutil.rmtree(folder)
        with self.connect() as connection:
            connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        upload_id = job["request"].get("upload_id")
        if upload_id and not self.upload_referenced(upload_id):
            upload = self.upload(upload_id)
            path = Path(upload["path"]).resolve()
            if not path.is_relative_to(self.path("uploads")):
                raise ValueError("Invalid managed upload path")
            path.unlink(missing_ok=True)
            with self.connect() as connection:
                connection.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
