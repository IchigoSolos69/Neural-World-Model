"""A declared, scenario-isolated Physics-IQ Original mini evaluation, never training data."""

from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

from .app_storage import Store
from .video import _ffmpeg, prepare_video, score_video

PREFIX = "full-videos/take-1/16FPS/"
SOURCE = "https://github.com/google-deepmind/physics-IQ-benchmark"


def select_scenarios(objects, development=3, testing=8):
    """Select by scenario names, without inspecting frames, targets or outcomes."""
    chosen = {}
    for item in sorted(objects, key=lambda item: item["name"]):
        name = item["name"]
        match = re.search(r"trimmed-([^/]+)\.mp4$", name)
        if (not name.startswith(PREFIX) or "perspective-center" not in name or not match
                or ".." in Path(name).parts or not str(item.get("generation", "")).isdigit()):
            continue
        scenario = match[1]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scenario):
            continue
        chosen.setdefault(scenario, {**item, "scenario_id": scenario})
    ordered = sorted(chosen, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    if len(ordered) < development + testing:
        raise ValueError("Catalog has too few distinct center-view scenarios")
    return [{**chosen[name], "split": "development" if index < development else "test"}
            for index, name in enumerate(ordered[:development + testing])]


def public_catalog():
    items, token = [], None
    for _ in range(10):
        params = {"prefix": PREFIX, "maxResults": 1000,
                  "fields": "items(name,size,generation,md5Hash),nextPageToken"}
        if token:
            params["pageToken"] = token
        url = "https://storage.googleapis.com/storage/v1/b/physics-iq-benchmark/o?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as response:
            page = json.loads(response.read(2 * 1024**2))
        items.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return items
    raise ValueError("Catalog pagination exceeded its bounded limit")


def prepare_mini(config, *, download=False):
    store = Store(config)
    folder = store.path("datasets", "physics-iq-mini")
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        declared = Path(__file__).resolve().parents[1] / "configs/physics_iq_mini.json"
        manifest = json.loads(declared.read_text(encoding="utf-8"))
        entries = manifest["entries"]
        total = sum(int(item["size"]) for item in entries)
        if total > 512 * 1024**2:
            raise ValueError("Selected mini dataset exceeds the 512 MiB download cap")
        store.ensure_space(2 * total + 16 * 1024**2)
        folder.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not download:
        return manifest_path
    entries = manifest.get("entries", [])
    if (len(entries) != 11 or len({item["scenario_id"] for item in entries}) != 11
            or sum(item["split"] == "development" for item in entries) != 3
            or sum(item["split"] == "test" for item in entries) != 8
            or sum(int(item["size"]) for item in entries) > 512 * 1024**2):
        raise ValueError("Invalid benchmark split manifest")
    for item in entries:
        if (not re.fullmatch(r"[A-Za-z0-9_-]+", item["scenario_id"])
                or not item["name"].startswith(PREFIX) or ".." in Path(item["name"]).parts
                or "\\" in item["name"] or not str(item["generation"]).isdigit()
                or not 0 < int(item["size"]) <= 100 * 1024**2):
            raise ValueError("Invalid benchmark file identity")
    for item in manifest["entries"]:
        raw = folder / Path(item["name"]).name
        processed = folder / f"{item['scenario_id']}.mp4"
        if not raw.is_file():
            store.ensure_space(int(item["size"]) + 16 * 1024**2)
            partial = raw.with_suffix(".partial")
            url = "https://storage.googleapis.com/physics-iq-benchmark/" + urllib.parse.quote(item["name"], safe="/")
            url += "?generation=" + item["generation"]
            count = 0
            try:
                with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as output:
                    while chunk := response.read(1024**2):
                        count += len(chunk)
                        if count > int(item["size"]):
                            raise ValueError("Dataset download exceeds its pinned size")
                        store.ensure_space(len(chunk))
                        output.write(chunk)
                if count != int(item["size"]):
                    raise ValueError("Incomplete dataset download")
                partial.replace(raw)
            finally:
                partial.unlink(missing_ok=True)
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        if raw.stat().st_size != int(item["size"]):
            raise ValueError("Dataset source size mismatch")
        if item.get("md5Hash") and base64.b64encode(hashlib.md5(raw.read_bytes()).digest()).decode() != item["md5Hash"]:
            raise ValueError("Dataset source MD5 mismatch")
        if item.get("source_sha256") and item["source_sha256"] != digest:
            raise ValueError("Dataset source checksum mismatch")
        item["source_sha256"] = digest
        if not processed.is_file():
            store.ensure_space(16 * 1024**2)
            result = _ffmpeg(["-y", "-i", str(raw), "-map", "0:v:0", "-an", "-t", "30", "-vf",
                              "fps=16,scale=384:256:force_original_aspect_ratio=decrease,pad=384:256:(ow-iw)/2:(oh-ih)/2",
                              "-c:v", "libx264", "-crf", "18", "-threads", "2", str(processed)], timeout=60)
            if result.returncode:
                processed.unlink(missing_ok=True)
                raise ValueError("Dataset normalization failed")
        item["video"] = processed.name
        item["normalized_sha256"] = hashlib.sha256(processed.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def evaluate_mini(manifest_path, model_dir, output_path, split="test"):
    import numpy as np
    from .video_model import LTXVideoPredictor
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split not in {"development", "test"} or manifest.get("dataset") != "Physics-IQ Original mini":
        raise ValueError("Choose a declared mini benchmark and split")
    predictor = LTXVideoPredictor(model_dir)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in manifest["entries"]:
        if item["split"] != split:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", item["scenario_id"]):
            raise ValueError("Invalid scenario directory")
        source = (manifest_path.parent / item.get("video", "missing")).resolve()
        if not source.is_relative_to(manifest_path.parent.resolve()) or not source.is_file():
            raise ValueError("Download and normalize the declared benchmark first")
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["normalized_sha256"]:
            raise ValueError("Normalized benchmark checksum mismatch")
        width, height = predictor.status["resolution"]
        prepared = prepare_video(source, output / item["scenario_id"], 3, 50, backtest=True,
                                 width=width, height=height)
        with np.load(prepared["prefix_path"], allow_pickle=False) as archive:
            prefix = archive["frames"]
        prediction = predictor.predict(prefix, 50, seed=0)
        np.savez_compressed(output / item["scenario_id"] / "prediction.npz", frames=prediction)
        with np.load(prepared["target_path"], allow_pickle=False) as archive:
            target = archive["frames"]
        records.append({"scenario_id": item["scenario_id"], **score_video(prediction, target, prefix[-1])})
    report = {"dataset": manifest["dataset"], "split": split, "seed": 0, "records": records,
              "interpretation": manifest["interpretation"]}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
