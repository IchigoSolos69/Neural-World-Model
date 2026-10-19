"""Rebuild research statistics and a portable report from saved measurements only."""

from __future__ import annotations

from hashlib import sha256
from html import escape
import json
import math
import os
from pathlib import Path

import numpy as np


METRIC_DEFINITION = "euclidean_position_rmse_px_v1"
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20_261_019
OBJECTS = ("red", "green", "blue")
SMALL_SAMPLE = 30


def _contained(root: Path, relative: str) -> Path:
    path = root / relative
    if not relative or Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Evidence path escapes evaluation directory: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing evidence file: {relative}")
    return path


def _read_evidence(root: Path) -> tuple[dict, dict, list[tuple[dict, dict]]]:
    evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if (evidence.get("format_version") != 1 or evidence.get("complete") is not True
            or evidence.get("metric_definition") != METRIC_DEFINITION
            or evidence.get("horizons") != [10, 20, 50]):
        raise ValueError("A complete format-1 evidence archive with the standard metric and horizons is required")
    if metrics.get("calibration", {}).get("split") != "calibration":
        raise ValueError("Uncertainty calibration must identify the held-out calibration split")
    for relative, expected in evidence.get("files", {}).items():
        if sha256(_contained(root, relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evidence checksum mismatch: {relative}")
    groups, keys, identities, split_ids, split_strata = [], set(), set(), {}, {}
    for metadata in evidence.get("groups", []):
        key = metadata["key"]
        identity = tuple(metadata[field] for field in ("split", "initialization", "actions", "model", "seed"))
        if key in keys or identity in identities:
            raise ValueError("Duplicate evidence group")
        keys.add(key)
        identities.add(identity)
        if metadata["initialization"] not in ("estimated", "true") or metadata["actions"] not in ("recorded", "removed"):
            raise ValueError("Invalid initialization or action condition")
        with np.load(_contained(root, metadata["file"]), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        count = metadata["episodes"]
        if not isinstance(count, int) or count < 1:
            raise ValueError("Every evidence group needs at least one episode")
        shapes = {"episode_seed": (count,), "position_sq_px": (count, 50),
                  "position_distance_px": (count, 50), "velocity_sq": (count, 50),
                  "pixel_mse": (count, 50), "boundary_rate": (count, 50),
                  "overlap_rate": (count, 50), "contact_counts": (count, 3, 3),
                  "contact_group": (count,), "impulse_present": (count,)}
        has_uncertainty = "uncertainty_errors_px" in arrays or "uncertainty_radii_px" in arrays
        if has_uncertainty:
            shapes.update(uncertainty_errors_px=(count, 3, 3), uncertainty_radii_px=(count, 3, 3))
            if metadata["model"] != "hybrid_ensemble" or metadata["initialization"] != "estimated" or metadata["actions"] != "recorded":
                raise ValueError("Uncertainty evidence must use the estimated, recorded-action hybrid ensemble")
        for name, shape in shapes.items():
            if name not in arrays or arrays[name].shape != shape:
                raise ValueError(f"Invalid {name} shape in {key}; expected {shape}")
            if not np.issubdtype(arrays[name].dtype, np.number) and arrays[name].dtype != np.bool_:
                raise ValueError(f"Non-numeric evidence: {name}")
            if not np.isfinite(arrays[name]).all() or (arrays[name] < 0).any():
                raise ValueError(f"Nonfinite or negative evidence: {name}")
        if not np.issubdtype(arrays["episode_seed"].dtype, np.integer) or len(np.unique(arrays["episode_seed"])) != count:
            raise ValueError("Episode identifiers must be unique integers")
        if not np.issubdtype(arrays["contact_counts"].dtype, np.integer):
            raise ValueError("Contact counts must be integers")
        if not np.isin(arrays["contact_group"], [0, 1, 2]).all() or not np.isin(arrays["impulse_present"], [0, 1]).all():
            raise ValueError("Invalid contact or impulse strata")
        if (arrays["boundary_rate"] > 1).any() or (arrays["overlap_rate"] > 1).any():
            raise ValueError("Violation rates must lie in [0, 1]")
        split = metadata["split"]
        if split in split_ids:
            if not np.array_equal(split_ids[split], arrays["episode_seed"]):
                raise ValueError(f"Episode alignment mismatch for {split}; pairing by row is unsafe")
            if any(not np.array_equal(split_strata[split][name], arrays[name])
                   for name in ("contact_group", "impulse_present")):
                raise ValueError(f"Ground-truth strata differ between predictors in {split}")
        else:
            split_ids[split] = arrays["episode_seed"]
            split_strata[split] = {name: arrays[name] for name in ("contact_group", "impulse_present")}
        groups.append((metadata, arrays))
    if not groups:
        raise ValueError("No per-episode evidence groups")
    for entry in evidence.get("gallery", []):
        for field in ("path", "state_archive"):
            _contained(root, entry[field])
    return evidence, metrics, groups


def _interval(value: float, samples: np.ndarray) -> dict:
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"estimate": float(value), "ci95": [float(low), float(high)]}


def _coverage_interval(covered: np.ndarray) -> dict:
    """Wilson bounds stay informative when every sampled episode is covered (or missed)."""
    count = len(covered)
    value = float(covered.mean())
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (value + z * z / (2 * count)) / denominator
    width = z * math.sqrt(value * (1 - value) / count + z * z / (4 * count * count)) / denominator
    return {"estimate": value, "ci95": [max(0.0, center - width), min(1.0, center + width)],
            "method": "95% Wilson score interval across independent held-out episodes"}


def _weights(count: int, rng: np.random.Generator) -> np.ndarray:
    # Counts preserve whole-episode pairing while avoiding [resample, episode, frame] arrays.
    result = np.empty((BOOTSTRAP_RESAMPLES, count), dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        result[index] = np.bincount(rng.integers(count, size=count), minlength=count) / count
    return result


def _episode_values(arrays: dict, horizon: int, horizon_index: int) -> dict:
    counts = arrays["contact_counts"][:, horizon_index].astype(np.float64)
    return {"position_rmse_px": arrays["position_sq_px"][:, :horizon].mean(1, dtype=np.float64),
            "position_ade_px": arrays["position_distance_px"][:, :horizon].mean(1, dtype=np.float64),
            "position_fde_px": arrays["position_distance_px"][:, horizon - 1],
            "velocity_rmse": arrays["velocity_sq"][:, :horizon].mean(1, dtype=np.float64),
            "pixel_mse": arrays["pixel_mse"][:, :horizon].mean(1, dtype=np.float64),
            "boundary_violation_rate": arrays["boundary_rate"][:, :horizon].mean(1, dtype=np.float64),
            "overlap_rate": arrays["overlap_rate"][:, :horizon].mean(1, dtype=np.float64),
            "collision_counts": counts}


def _summary(arrays: dict, horizon: int, index: int, weights: np.ndarray) -> tuple[dict, dict]:
    result, draws = {}, {}
    for name, values in _episode_values(arrays, horizon, index).items():
        if name == "collision_counts":
            tp, fp, fn = values.mean(0)
            sample_tp, sample_fp, sample_fn = (weights @ values).T
            for label, numerator, denominator, sample_numerator, sample_denominator in (
                ("collision_precision", tp, tp + fp, sample_tp, sample_tp + sample_fp),
                ("collision_recall", tp, tp + fn, sample_tp, sample_tp + sample_fn),
                ("collision_f1", 2 * tp, 2 * tp + fp + fn, 2 * sample_tp, 2 * sample_tp + sample_fp + sample_fn),
            ):
                samples = np.divide(sample_numerator, sample_denominator,
                                    out=np.zeros_like(sample_numerator), where=sample_denominator != 0)
                result[label] = _interval(numerator / denominator if denominator else 0.0, samples)
            continue
        estimate, samples = values.mean(), weights @ values
        if name in ("position_rmse_px", "velocity_rmse"):
            estimate, samples = np.sqrt(estimate), np.sqrt(samples)
        result[name] = _interval(estimate, samples)
        draws[name] = samples
    return result, draws


def _identity(metadata: dict) -> dict:
    return {key: metadata[key] for key in ("split", "initialization", "actions", "model", "seed")}


def _pair(label: str, model: dict, baseline: dict, horizon: int, estimate: float,
          samples: np.ndarray, count: int, primary: bool = False) -> dict:
    return {"comparison": label, "primary": primary, "condition": {
        key: model[key] for key in ("split", "initialization", "actions")},
        "model": model["model"], "seed": model["seed"], "baseline": baseline["model"],
        "baseline_seed": baseline["seed"], "horizon": horizon, "episodes": count,
        "small_sample": count < SMALL_SAMPLE, "metric": "position_rmse_px",
        "difference_direction": "model_minus_baseline; negative favors model",
        **_interval(estimate, samples)}


def _statistics(evidence: dict, groups: list[tuple[dict, dict]]) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries, comparisons, strata, uncertainty, seed_dispersion = [], [], [], [], []
    by_split = {}
    for metadata, arrays in groups:
        by_split.setdefault(metadata["split"], []).append((metadata, arrays))
    for split, members in sorted(by_split.items()):
        count = len(members[0][1]["episode_seed"])
        weights = _weights(count, rng)
        calculated = {}
        lookup = {}
        for metadata, arrays in members:
            lookup[(metadata["initialization"], metadata["actions"], metadata["model"], metadata["seed"])] = metadata
            horizons, draws = {}, {}
            for index, horizon in enumerate(evidence["horizons"]):
                horizons[str(horizon)], draws[horizon] = _summary(arrays, horizon, index, weights)
            row = {**_identity(metadata), "episodes": count, "small_sample": count < SMALL_SAMPLE, "horizons": horizons}
            summaries.append(row)
            calculated[metadata["key"]] = row, draws
            if "uncertainty_errors_px" in arrays:
                for index, horizon in enumerate(evidence["horizons"]):
                    for object_index, object_name in enumerate(OBJECTS):
                        errors = arrays["uncertainty_errors_px"][:, index, object_index]
                        radii = arrays["uncertainty_radii_px"][:, index, object_index]
                        covered = (errors <= radii).astype(float)
                        uncertainty.append({"split": split, "horizon": horizon, "object": object_name,
                                            "episodes": count, "small_sample": count < SMALL_SAMPLE,
                                            "coverage_target": 0.9,
                                            "coverage": _coverage_interval(covered),
                                            "mean_radius_px": _interval(radii.mean(), weights @ radii),
                                            "mean_interval_width_px": _interval(2 * radii.mean(), weights @ (2 * radii))})
        for metadata, arrays in members:
            row, draws = calculated[metadata["key"]]
            for baseline_name in ("fitted_physics", "imperfect_physics"):
                baseline = lookup.get((metadata["initialization"], metadata["actions"], baseline_name, None))
                if baseline is None or metadata["key"] == baseline["key"]:
                    continue
                baseline_row, baseline_draws = calculated[baseline["key"]]
                for horizon in evidence["horizons"]:
                    primary = (split == "test" and metadata["initialization"] == "estimated"
                               and metadata["actions"] == "recorded" and metadata["model"] == "hybrid_ensemble"
                               and baseline_name == "fitted_physics" and horizon == 50)
                    comparisons.append(_pair("predictor_vs_physics", metadata, baseline, horizon,
                        row["horizons"][str(horizon)]["position_rmse_px"]["estimate"]
                        - baseline_row["horizons"][str(horizon)]["position_rmse_px"]["estimate"],
                        draws[horizon]["position_rmse_px"] - baseline_draws[horizon]["position_rmse_px"], count, primary))
            for condition, label, desired in (("actions", "recorded_vs_removed", "removed"),
                                              ("initialization", "estimated_vs_true", "true")):
                if metadata[condition] != {"actions": "recorded", "initialization": "estimated"}[condition]:
                    continue
                partner_key = [metadata[key] for key in ("initialization", "actions", "model", "seed")]
                partner_key[1 if condition == "actions" else 0] = desired
                partner = lookup.get(tuple(partner_key))
                if partner is None:
                    continue
                partner_row, partner_draws = calculated[partner["key"]]
                for horizon in evidence["horizons"]:
                    comparison = _pair(label, metadata, partner, horizon,
                        row["horizons"][str(horizon)]["position_rmse_px"]["estimate"]
                        - partner_row["horizons"][str(horizon)]["position_rmse_px"]["estimate"],
                        draws[horizon]["position_rmse_px"] - partner_draws[horizon]["position_rmse_px"], count)
                    comparison["difference_direction"] = ("recorded_minus_removed; negative favors recorded actions"
                        if condition == "actions" else "estimated_minus_true; positive measures perception penalty")
                    comparisons.append(comparison)
        for initialization in ("estimated", "true"):
            for actions in ("recorded", "removed"):
                for family in ("learned", "hybrid"):
                    selected = [metadata for metadata, _ in members if metadata["initialization"] == initialization
                                and metadata["actions"] == actions and metadata["model"] == family]
                    if not selected:
                        continue
                    selected.sort(key=lambda item: item["seed"])
                    for horizon in evidence["horizons"]:
                        estimates = np.array([calculated[item["key"]][0]["horizons"][str(horizon)]["position_rmse_px"]["estimate"]
                                              for item in selected])
                        samples = np.mean([calculated[item["key"]][1][horizon]["position_rmse_px"] for item in selected], axis=0)
                        entry = {"split": split, "initialization": initialization, "actions": actions,
                                 "model": family, "horizon": horizon, "seeds": [item["seed"] for item in selected],
                                 "episodes": count, "small_sample": count < SMALL_SAMPLE,
                                 "individual_seed_rmse_px": estimates.tolist(),
                                 "seed_std_px": float(estimates.std(ddof=1)) if len(estimates) > 1 else None,
                                 "seed_range_px": [float(estimates.min()), float(estimates.max())],
                                 "conditional_on_trained_seeds": True, **_interval(estimates.mean(), samples)}
                        seed_dispersion.append(entry)
                        for baseline_name in ("fitted_physics", "imperfect_physics"):
                            baseline = lookup.get((initialization, actions, baseline_name, None))
                            if baseline is not None:
                                baseline_row, baseline_draws = calculated[baseline["key"]]
                                comparison = _pair("family_mean_vs_physics", selected[0] | {"seed": None}, baseline, horizon,
                                    estimates.mean() - baseline_row["horizons"][str(horizon)]["position_rmse_px"]["estimate"],
                                    samples - baseline_draws[horizon]["position_rmse_px"], count)
                                comparison.update(seeds=entry["seeds"], conditional_on_trained_seeds=True)
                                comparisons.append(comparison)
        # Strata use truth-defined events; never predicted contacts or model performance.
        base_arrays = members[0][1]
        masks = [("free_flight", base_arrays["contact_group"] == 0),
                 ("wall_only", base_arrays["contact_group"] == 1),
                 ("ball_collision", base_arrays["contact_group"] == 2),
                 ("with_future_impulse", base_arrays["impulse_present"].astype(bool)),
                 ("without_future_impulse", ~base_arrays["impulse_present"].astype(bool))]
        for name, mask in masks:
            size = int(mask.sum())
            if not size:
                strata.append({"split": split, "stratum": name, "episodes": 0, "small_sample": True, "summaries": []})
                continue
            stratum_weights = _weights(size, rng)
            rows = []
            for metadata, arrays in members:
                values = arrays["position_sq_px"][mask]
                horizons = {}
                for horizon in evidence["horizons"]:
                    per_episode = values[:, :horizon].mean(1, dtype=np.float64)
                    horizons[str(horizon)] = _interval(np.sqrt(per_episode.mean()), np.sqrt(stratum_weights @ per_episode))
                rows.append({**_identity(metadata), "horizons": horizons})
            strata.append({"split": split, "stratum": name, "episodes": size,
                           "small_sample": size < SMALL_SAMPLE, "summaries": rows})
    primary = next((row for row in comparisons if row["primary"]), None)
    if primary is None:
        raise ValueError("Evidence lacks the predeclared hybrid-ensemble versus fitted-physics primary comparison")
    return {"format_version": 1, "metric_definition": METRIC_DEFINITION,
            "bootstrap": {"method": "paired percentile bootstrap of whole episodes within condition",
                          "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "confidence": 0.95,
                          "small_sample_threshold": SMALL_SAMPLE,
                          "interpretation": "Intervals condition on the fixed trained checkpoints. Secondary comparisons are exploratory; no multiplicity correction."},
            "primary": primary, "primary_comparison": primary,
            "summaries": summaries, "comparisons": comparisons, "seed_dispersion": seed_dispersion,
            "strata": strata, "uncertainty": uncertainty}


def _plots(root: Path, statistics: dict, groups: list[tuple[dict, dict]]) -> list[tuple[str, str]]:
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = root / "report_plots"
    destination.mkdir(exist_ok=True)
    plots = []
    for split in sorted({metadata["split"] for metadata, _ in groups}):
        # Split names are data labels, never output paths.
        number = len(plots)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for axis, initialization in zip(axes, ("estimated", "true")):
            families = {}
            for metadata, arrays in groups:
                if metadata["split"] == split and metadata["initialization"] == initialization and metadata["actions"] == "recorded":
                    curves = np.sqrt(np.cumsum(arrays["position_sq_px"].mean(0)) / np.arange(1, 51))
                    families.setdefault(metadata["model"], []).append(curves)
            for model, curves in families.items():
                mean = np.mean(curves, axis=0)
                axis.plot(np.arange(1, 51), mean, label=model.replace("_", " "))
            axis.set(title=f"{split}: {initialization} initialization", xlabel="Future horizon (frames)")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Prefix position RMSE (pixels)")
        if axes[1].lines:
            axes[1].legend(fontsize=7)
        fig.tight_layout()
        relative = f"report_plots/position_{number:02d}.png"
        fig.savefig(root / relative, dpi=140, metadata={"Software": "Neural World Model research report"})
        plt.close(fig)
        plots.append((f"{split} position error", relative))
    contrasts = [row for row in statistics["comparisons"] if row["comparison"] == "predictor_vs_physics"
                 and row["model"] == "hybrid_ensemble" and row["baseline"] == "fitted_physics"
                 and row["horizon"] == 50 and row["condition"]["initialization"] == "estimated"
                 and row["condition"]["actions"] == "recorded"]
    if contrasts:
        fig, axis = plt.subplots(figsize=(9, 4))
        for i, row in enumerate(contrasts):
            axis.plot(row["ci95"], [i, i], color="#235a87", linewidth=2)
            axis.plot(row["estimate"], i, "o", color="#235a87")
        axis.axvline(0, color="black", linewidth=1, linestyle="--")
        axis.set(yticks=range(len(contrasts)), yticklabels=[row["condition"]["split"] for row in contrasts],
                 xlabel="Hybrid ensemble minus fitted physics RMSE (px); negative favors hybrid",
                 title="50-frame paired differences with 95% episode-bootstrap intervals")
        fig.tight_layout()
        relative = "report_plots/paired_differences.png"
        fig.savefig(root / relative, dpi=140, metadata={"Software": "Neural World Model research report"})
        plt.close(fig)
        plots.append(("Paired hybrid versus fitted-physics differences", relative))
    if statistics["uncertainty"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for split in sorted({row["split"] for row in statistics["uncertainty"]}):
            for object_name in OBJECTS:
                rows = sorted([row for row in statistics["uncertainty"] if row["split"] == split and row["object"] == object_name],
                              key=lambda row: row["horizon"])
                axes[0].plot([row["horizon"] for row in rows], [row["coverage"]["estimate"] for row in rows],
                             marker="o", label=f"{split}: {object_name}")
                axes[1].plot([row["horizon"] for row in rows], [row["mean_interval_width_px"]["estimate"] for row in rows], marker="o")
        axes[0].axhline(0.9, linestyle="--", color="black", linewidth=1)
        axes[0].set(xlabel="Horizon (frames)", ylabel="Marginal coverage", ylim=(0, 1.03))
        axes[1].set(xlabel="Horizon (frames)", ylabel="Mean radial interval diameter (pixels)")
        axes[0].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        relative = "report_plots/uncertainty.png"
        fig.savefig(root / relative, dpi=140, metadata={"Software": "Neural World Model research report"})
        plt.close(fig)
        plots.append(("Per-object uncertainty", relative))
    return plots


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.4g}"


def _ci(value: dict) -> str:
    return f"{_number(value['estimate'])} [{_number(value['ci95'][0])}, {_number(value['ci95'][1])}]"


def _table(headers: list[str], rows: list[list]) -> str:
    return "<div class='table'><table><thead><tr>" + "".join(f"<th>{escape(str(item))}</th>" for item in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(str(item))}</td>" for item in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def _html(root: Path, evidence: dict, metrics: dict, statistics: dict, plots: list[tuple[str, str]]) -> str:
    primary = statistics["primary_comparison"]
    if primary is None:
        headline = "Primary comparison unavailable in this evidence archive."
    else:
        estimate, (low, high) = primary["estimate"], primary["ci95"]
        conclusion = "favors the hybrid ensemble" if high < 0 else "favors fitted physics" if low > 0 else "does not distinguish the predictors at the stated confidence level"
        headline = (f"Hybrid ensemble minus fitted physics at 50 frames: {_ci(primary)} pixels. "
                    f"The paired interval {conclusion}. This comparison uses ordinary test physics, estimated initialization, and recorded actions.")
    smoke = evidence.get("config", {}).get("data", {}).get("train", 0) < 8000
    status = "Smoke pipeline check — undertrained models; not final research results" if smoke else "Full experiment — fixed, predeclared evaluation protocol"
    sections = ["<h1>Neural World Model: research evidence</h1>", f"<p><strong>{escape(status)}</strong></p>", f"<p class='lead'>{escape(headline)}</p>",
        "<p>All statistics below are regenerated from frozen per-episode measurements. No model, dataset, or GPU is loaded to build this document. "
        "Negative model-minus-baseline differences favor the model. Intervals use 2,000 paired whole-episode bootstrap resamples, seed 20261019, and percentile 95% bounds. "
        "Intervals are conditional on the trained checkpoints; training-seed dispersion is reported separately. Secondary comparisons are exploratory and have no multiplicity correction. "
        "Groups with fewer than 30 episodes are flagged as small samples; bootstrap intervals can be unstable or degenerate. "
        "Per-object coverage uses 95% Wilson intervals across episodes so all-covered or all-missed samples do not imply exact certainty.</p>",
        "<h2>Measurements and baseline comparisons</h2>",
        "<p>Position RMSE is √mean(dx² + dy²), in pixels, pooled over episodes, three objects, and all prefix frames through the stated horizon. "
        "The square root is taken after pooling, including within each bootstrap resample. Endpoint FDE is mean Euclidean distance at the final frame. "
        "Velocity RMSE uses Euclidean velocity distance in world units/second. Collision events use the common trajectory-based detector and ±1-frame matching. "
        "Pixel MSE compares common-decoder RGB output with rendered truth on [0,1]. Boundary and overlap rates are object-frame and pair-frame fractions.</p>"]
    rows, primary_rows = [], []
    for row in statistics["summaries"]:
        if row["actions"] == "recorded":
            h = row["horizons"]["50"]
            values = [row["split"], row["initialization"], row["model"], row["seed"],
                         f"{row['episodes']}{' (small)' if row['small_sample'] else ''}", _ci(h["position_rmse_px"]),
                         _ci(h["velocity_rmse"]), _ci(h["collision_f1"]), _ci(h["pixel_mse"]),
                         _ci(h["boundary_violation_rate"]), _ci(h["overlap_rate"])]
            rows.append(values)
            if row["split"] == "test" and row["initialization"] == "estimated":
                primary_rows.append(values)
    headers = ["Condition", "Initialization", "Predictor", "Seed", "Episodes", "RMSE px [95% CI]", "Velocity RMSE", "Collision F1", "Pixel MSE", "Boundary rate", "Overlap rate"]
    sections.append(_table(headers, primary_rows))
    sections.append("<details><summary>All conditions and initialization measurements at 50 frames</summary>" + _table(headers, rows) + "</details>")
    sections.append("<h2>Paired comparisons</h2><details><summary>All primary and exploratory comparisons at 10/20/50 frames</summary>" + _table(
        ["Comparison", "Condition", "Initialization", "Actions", "Predictor / seed", "Reference", "Horizon", "Difference px [95% CI]"],
        [[row["comparison"] + (" (PRIMARY)" if row["primary"] else ""), row["condition"]["split"], row["condition"]["initialization"], row["condition"]["actions"],
          f"{row['model']} / {row['seed']}", row["baseline"], row["horizon"], _ci(row)] for row in statistics["comparisons"]]) + "</details>")
    sections.append("<p>For action removal, the difference is recorded minus removed future actions; targets and observed context are identical. "
                    "For initialization, the difference is estimated minus true state, so positive values quantify a perception penalty.</p>")
    sections.append("<h2>Training-seed variation</h2><p>Each family estimate averages the separately pooled RMSE of its trained seeds. "
        "Its bootstrap resamples the same episodes for every seed. The sample standard deviation across seed estimates measures training variation and is not the episode-bootstrap confidence interval.</p><details><summary>Family means and variation across training seeds</summary>" + _table(
        ["Condition", "Initialization", "Actions", "Family", "Horizon", "Seeds", "Family mean px [95% CI]", "Seed sample SD px"],
        [[row["split"], row["initialization"], row["actions"], row["model"], row["horizon"], row["seeds"], _ci(row), _number(row["seed_std_px"])]
         for row in statistics["seed_dispersion"]]) + "</details>")
    sections.append("<h2>Error versus horizon</h2>")
    for title, path in plots:
        sections.append(f"<figure><img src='{escape(path, quote=True)}' alt='{escape(title, quote=True)}'><figcaption>{escape(title)}. Family curves average seed-level RMSE; uncertainty bounds are in the tables.</figcaption></figure>")
    sections.append("<h2>Where failures occur</h2><p>Contact strata are mutually exclusive: free flight, wall contacts without ball contacts, and any ball collision. "
                    "The impulse split is separate and overlaps those strata. Labels use recorded ground-truth events over the full 50-frame target, exclusively for analysis.</p>")
    strata_rows = []
    for stratum in statistics["strata"]:
        for row in stratum["summaries"]:
            if row["initialization"] == "estimated" and row["actions"] == "recorded":
                strata_rows.append([stratum["split"], stratum["stratum"], f"{stratum['episodes']}{' (small)' if stratum['small_sample'] else ''}",
                                    row["model"], row["seed"], _ci(row["horizons"]["50"])])
    sections.append("<details><summary>Contact and impulse strata</summary>" + _table(["Condition", "Stratum", "Episodes", "Predictor", "Seed", "50-frame RMSE px [95% CI]"], strata_rows) + "</details>")
    sections.append("<h2>Uncertainty</h2><p>Radii are calibrated separately for each object and horizon on the calibration split. "
                    "Width below means diameter, twice the radius. Coverage is marginal, conditional on shared perception; it is not a simultaneous trajectory guarantee and has no out-of-distribution guarantee.</p>" + _table(
        ["Condition", "Horizon", "Object", "Episodes", "Coverage [95% CI]", "Mean width px [95% CI]"],
        [[row["split"], row["horizon"], row["object"], row["episodes"], _ci(row["coverage"]), _ci(row["mean_interval_width_px"])] for row in statistics["uncertainty"]]))
    decoder_rows = [[row["split"], horizon, _number(value)] for row in metrics.get("true_state_decoder_error", [])
                    for horizon, value in row.get("pixel_mse_by_horizon", {}).items()]
    sections.append("<h2>Decoder diagnostic</h2><p>The common decoder receives true future states for this diagnostic only. "
                    "Its reconstruction error separates rendering limitations from rollout-state error; true future states are never predictor inputs.</p>" +
                    _table(["Condition", "Horizon", "True-state decoder pixel MSE"], decoder_rows))
    contact_rows = [[row["split"], horizon, _number(value["collision_precision"]),
                     _number(value["collision_recall"]), _number(value["collision_f1"])]
                    for row in metrics.get("true_trajectory_contact_reference", [])
                    for horizon, value in row["horizons"].items()]
    if contact_rows:
        sections.append("<h2>Collision-detector reference</h2><p>The shared detector is also evaluated on the exact ground-truth trajectory. "
                        "Its discretization can miss or duplicate recorded simulator contacts, so imperfect contact scores need not imply incorrect dynamics.</p>"
                        "<details><summary>True-trajectory contact detection at each horizon</summary>" +
                        _table(["Condition", "Horizon", "Precision", "Recall", "F1"], contact_rows) + "</details>")
    sections.append("<h2>Reproducible failure galleries</h2><p>Episodes are selected by the hybrid ensemble’s 50-frame prefix RMSE under estimated initialization and recorded actions. "
                    "Sort by error, then episode identifier to break ties; select nearest-rank median (ceil(0.5N)), p90 (ceil(0.9N)), and maximum (N). "
                    "Ranks are one-based and selection is descriptive, never checkpoint selection.</p>")
    for entry in evidence.get("gallery", []):
        label = f"{entry['split']}: {entry['selection']}, episode {entry['episode_seed']}; RMSE {_number(entry['position_rmse_px'])} px; rank {entry['rank']}/{entry['population']}"
        sections.append(f"<figure><video controls preload='metadata' src='{escape(entry['path'], quote=True)}'></video><figcaption>{escape(label)} "
                        f"<a href='{escape(entry['state_archive'], quote=True)}'>Saved trajectory archive</a></figcaption></figure>")
    sections.append("<h2>Provenance and reproduction</h2><p>Keep this HTML beside its report_plots, gallery, JSON, and episode archives. "
                    "Rebuild with <code>python -m nwm report --evaluation PATH</code>. "
                    "<a href='statistical_comparisons.json'>All statistics (JSON)</a> · <a href='evidence.json'>Evidence manifest</a> · <a href='metrics.json'>Original aggregate metrics</a></p>" +
                    f"<details><summary>Frozen checkpoint and dataset fingerprints</summary><pre>{escape(json.dumps(evidence.get('provenance', {}), indent=2, sort_keys=True))}</pre></details>")
    histories = evidence.get("training_history", {})
    if histories:
        sections.append("<h2>Training selection and compatibility</h2><p>Historical dynamics validation metrics averaged squared error across x/y. "
                        "Their displayed validation values are converted by √2 into the shared Euclidean definition, without changing source files or checkpoint rankings. "
                        "Training losses, optimizer state, and random-number state are unchanged.</p>" + _table(
            ["Stage", "Selected epoch", "Selection metric", "Selected value", "Compatibility conversion"],
            [[tag, value.get("best_epoch", -1) + 1, value.get("selection_metric", "not recorded"),
              _number(value.get("best_metric")), "×√2 from coordinate RMSE" if value.get("position_rmse_compatibility") else "None"]
             for tag, value in histories.items()]))
    sections.append("<h2>Limits</h2><p>The fitted-physics predictor uses the exact force-law family of this simulator. Strong fitted-physics results are expected; "
                    "a hybrid improvement is an experimental hypothesis, not an acceptance requirement. The prototype uses fixed identities, geometry, training-only state supervision, "
                    "and supplied future actions. It does not infer arbitrary unseen laws or establish real-video physics capability. "
                    "Changed-condition tests are held out from checkpoint selection and uncertainty calibration. Limited training seeds do not establish a population-level training distribution. "
                    "Bootstrap confidence bounds describe sampled-episode uncertainty under the stated protocol, not every possible environment.</p>")
    return "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Neural World Model research evidence</title><style>" + \
        "body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui,sans-serif;color:#172a38;background:#fff}h1{font-size:32px}h2{margin-top:42px;border-bottom:1px solid #b9c8d2;padding-bottom:6px}.lead{font-size:20px;max-width:1000px}p{max-width:1050px}.table{overflow-x:auto}table{border-collapse:collapse;font-size:13px;width:100%;font-variant-numeric:tabular-nums}th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #dce4e9;white-space:nowrap}th{background:#eef3f6}figure{margin:24px 0}img{max-width:100%;height:auto}video{width:min(100%,960px)}figcaption{font-size:13px;color:#435c6e}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#f0f4f7;font-size:12px}a{color:#155784}code{font-size:14px}@media print{body{margin:0}.table{overflow:visible}table{font-size:8px}video{display:none}}" + \
        "</style><body>" + "\n".join(sections) + "</body></html>"


def build_report(evaluation_path: str | Path) -> Path:
    """Write deterministic statistics, plots, and report.html without loading models."""
    root = Path(evaluation_path).resolve()
    evidence, metrics, groups = _read_evidence(root)
    statistics = _statistics(evidence, groups)
    statistics["provenance"] = evidence.get("provenance", {})
    (root / "statistical_comparisons.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plots = _plots(root, statistics, groups)
    path = root / "report.html"
    path.write_text(_html(root, evidence, metrics, statistics, plots), encoding="utf-8")
    return path
