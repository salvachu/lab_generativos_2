"""Model-independent geometry checks calibrated exclusively on TRAIN.

Validation scores describe geometry, not artistic quality. Only generated suffixes
are statistically assessed; an observed prefix is checked for shape and finiteness
and is never moved, smoothed, clipped, or penalized for its drawing style.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import sha256_file

DEFAULT_CONFIG = {
    "soft_quantile": 0.995,
    "tail_quantile": 0.999,
    "mad_sigma_factor": 1.4826,
    "severe_mad_multiplier": 6.0,
    "severe_tail_multiplier": 1.5,
    "coordinate_margin_fraction": 0.10,
    "deduplicate_adjacent": True,
    "smoothing_alpha": 0.0,
    "require_nonempty_suffix": True,
}


@dataclass
class ValidationResult:
    valid: bool
    score: float
    warnings: list[str] = field(default_factory=list)
    penalties: dict[str, float] = field(default_factory=dict)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    severe_errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _distribution(values: Sequence[float], config: dict) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        values = np.zeros(1)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    soft = float(np.quantile(values, config["soft_quantile"]))
    tail = float(np.quantile(values, config["tail_quantile"]))
    sigma = config["mad_sigma_factor"] * mad
    # Robust high quantile plus spread, with an explicit multiplicative fallback
    # for distributions whose MAD is zero (counts, repeated/straight strokes).
    severe = max(tail + config["severe_mad_multiplier"] * sigma,
                 tail * config["severe_tail_multiplier"], soft)
    return {"count": int(len(values)), "min": float(values.min()), "median": median,
            "max": float(values.max()), "mean": float(values.mean()), "mad": mad,
            "robust_sigma": sigma, "q01": float(np.quantile(values, .01)),
            "q99": float(np.quantile(values, .99)), "soft_upper": soft,
            "severe_upper": severe}


def _stroke_features(stroke: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment lengths, turning angle / pi, normalized spatial acceleration."""
    delta = np.diff(np.asarray(stroke, dtype=np.float64), axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if len(delta) < 2:
        return lengths, np.empty(0), np.empty(0)
    denominator = lengths[:-1] * lengths[1:]
    active = denominator > 0
    cosine = np.sum(delta[:-1] * delta[1:], axis=1)[active] / denominator[active]
    turns = np.arccos(np.clip(cosine, -1, 1)) / np.pi
    adjacent_length = lengths[:-1] + lengths[1:]
    nonzero = adjacent_length > 0
    acceleration = np.linalg.norm(np.diff(delta, axis=0), axis=1)[nonzero] / adjacent_length[nonzero]
    return lengths, turns, acceleration


def fit_geometry(samples: Sequence[dict], *, config: dict | None = None,
                 source_path: str | Path | None = None) -> dict:
    """Fit once using TRAIN samples, then persist with save_geometry.

    Validation data must never be passed here; source identity is recorded and a
    source path explicitly containing a validation filename is rejected.
    """
    if source_path is not None and Path(source_path).stem.lower() in {"val", "valid", "validation", "test"}:
        raise ValueError("Geometry calibration accepts TRAIN only")
    if not samples:
        raise ValueError("Cannot fit geometry without TRAIN samples")
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if not 0 < cfg["soft_quantile"] <= cfg["tail_quantile"] < 1:
        raise ValueError("Geometry quantiles must satisfy 0 < soft <= tail < 1")
    for key in ("mad_sigma_factor", "severe_mad_multiplier", "severe_tail_multiplier", "coordinate_margin_fraction"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    features = {name: [] for name in ("segment_length", "stroke_arc_length", "turning", "acceleration",
                "stroke_jaggedness", "stroke_count", "point_count", "points_per_stroke", "bbox_diagonal", "zero_arc_fraction")}
    all_points = []
    for sample in samples:
        strokes = sample["strokes"]
        if not strokes:
            raise ValueError("TRAIN sample has no strokes")
        arrays = [np.asarray(s, dtype=np.float64) for s in strokes]
        if any(s.ndim != 2 or s.shape[1] != 2 or not len(s) or not np.isfinite(s).all() for s in arrays):
            raise ValueError("TRAIN geometry must contain finite nonempty (N,2) strokes")
        points = np.concatenate(arrays)
        all_points.append(points)
        features["stroke_count"].append(len(arrays))
        features["point_count"].append(len(points))
        features["bbox_diagonal"].append(float(np.linalg.norm(np.ptp(points, axis=0))))
        degenerate = 0
        for stroke in arrays:
            lengths, turns, acceleration = _stroke_features(stroke)
            features["segment_length"].extend(lengths)
            features["stroke_arc_length"].append(float(lengths.sum()))
            features["points_per_stroke"].append(len(stroke))
            features["turning"].extend(turns)
            features["acceleration"].extend(acceleration)
            features["stroke_jaggedness"].append(float(turns.mean()) if turns.size else 0.0)
            degenerate += int(lengths.sum() == 0)
        features["zero_arc_fraction"].append(degenerate / len(arrays))
    points = np.concatenate(all_points)
    low, high = points.min(axis=0), points.max(axis=0)
    central_span = np.quantile(points, .99, axis=0) - np.quantile(points, .01, axis=0)
    # Span fallback applies only to synthetic fixtures with constant coordinates.
    margin = cfg["coordinate_margin_fraction"] * np.maximum(central_span, high - low)
    stats = {"version": 1, "fit_split": "train", "sample_count": len(samples),
             "source_path": str(source_path) if source_path is not None else None,
             "source_sha256": sha256_file(source_path) if source_path is not None else None,
             "representation_version": "absolute-stroke5-v1", "normalization_scale": 256.0,
             "config": cfg,
             "coordinates": {"min": low.tolist(), "max": high.tolist(),
                             "central_span": central_span.tolist(),
                             "severe_min": (low - margin).tolist(), "severe_max": (high + margin).tolist()},
             "distributions": {name: _distribution(values, cfg) for name, values in features.items()},
             "intra_segment_p995": float(np.quantile(features["segment_length"], .995)),
             "policy": {"score": "1 / (1 + maximum dimensionless feature exceedance); invalid => 0",
                        "threshold": "soft=TRAIN q.995; severe=max(TRAIN q.999 + 6*1.4826*MAD, 1.5*q.999)",
                        "coordinates": "soft=observed TRAIN bounds; severe=soft bounds plus 0.10*max(q99-q01, full span)",
                        "prefix": "shape/finiteness only; no statistical penalties or corrections",
                        "inter_stroke_distance": "never included in continuity penalties"}}
    return stats


def save_geometry(stats: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, allow_nan=False), encoding="utf-8")
    return path


def load_geometry(path: str | Path = "data/processed/geometry.json") -> dict:
    stats = json.loads(Path(path).read_text(encoding="utf-8"))
    if stats.get("fit_split") != "train" or stats.get("version") != 1:
        raise ValueError("Unsupported or non-TRAIN geometry calibration")
    return stats


def _termination_errors(termination, warnings: list, errors: list) -> None:
    if termination is None:
        warnings.append("termination_unreported")
        return
    if isinstance(termination, str):
        reason = termination
        ended = reason == "eos"
        capped = reason in {"max_points", "max_strokes", "cap"}
    elif isinstance(termination, dict):
        reason = termination.get("termination", termination.get("reason"))
        ended = termination.get("ended_by_eos", termination.get("has_eos", termination.get("eos", reason == "eos")))
        capped = bool(termination.get("capped", reason in {"max_points", "max_strokes", "cap"}))
    else:
        errors.append("invalid_termination_metadata")
        return
    if capped:
        errors.append("sequence_cap_reached")
    if not ended:
        errors.append("missing_eos")
    if reason in {"nonfinite_prediction", "invalid_prediction"}:
        errors.append(str(reason))


def validate_candidate(strokes: Sequence, stats: dict, prefix_count: int = 0,
                       termination=None) -> ValidationResult:
    """Assess generated suffix; structural corruption anywhere is severe."""
    warnings: list[str] = []
    severe: list[str] = []
    penalties: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    if type(prefix_count) is not int or not 0 <= prefix_count <= len(strokes):
        return ValidationResult(False, 0.0, severe_errors=["invalid_prefix_count"])
    arrays = []
    for index, stroke in enumerate(strokes):
        try:
            array = np.asarray(stroke, dtype=np.float64)
        except (TypeError, ValueError):
            severe.append(f"malformed_stroke:{index}")
            continue
        if array.ndim != 2 or array.shape[1] != 2 or not len(array):
            severe.append(f"malformed_stroke:{index}")
        elif not np.isfinite(array).all():
            severe.append(f"nonfinite_stroke:{index}")
        arrays.append(array)
    _termination_errors(termination, warnings, severe)
    if any(message.startswith(("malformed_stroke", "nonfinite_stroke")) for message in severe):
        return ValidationResult(False, 0.0, warnings=warnings, severe_errors=severe)
    suffix = arrays[prefix_count:]
    metrics.update(prefix_count=prefix_count, generated_strokes=len(suffix))
    if not suffix:
        if stats["config"].get("require_nonempty_suffix", True):
            severe.append("empty_generated_suffix")
        return ValidationResult(not severe, 0.0 if severe else 1.0, warnings=warnings, severe_errors=severe, metrics=metrics)
    points = np.concatenate(suffix)
    metrics["generated_points"] = len(points)
    low = np.asarray(stats["coordinates"]["min"])
    high = np.asarray(stats["coordinates"]["max"])
    severe_low = np.asarray(stats["coordinates"]["severe_min"])
    severe_high = np.asarray(stats["coordinates"]["severe_max"])
    outside = ((points < low) | (points > high)).any(axis=1)
    extreme = ((points < severe_low) | (points > severe_high)).any(axis=1)
    metrics["outside_train_bounds_fraction"] = float(outside.mean())
    metrics["extreme_outside_fraction"] = float(extreme.mean())
    if outside.any():
        warnings.append("outside_train_coordinate_bounds")
        scale = np.maximum(severe_high - high, np.finfo(np.float64).eps)
        with np.errstate(over="ignore", invalid="ignore"):
            coordinate_excess = np.maximum(low - points, points - high) / scale
        penalties["coordinates"] = float(np.nan_to_num(np.maximum(coordinate_excess, 0), nan=0,
                                                        posinf=np.finfo(np.float64).max).max())
    if extreme.any():
        severe.append("extreme_coordinates")
        # Reject before subtracting huge finite coordinates can overflow norms.
        return ValidationResult(False, 0.0, warnings, penalties, severe_errors=severe, metrics=metrics)

    def upper_penalty(name: str, value: float, *, reject: bool = True):
        dist = stats["distributions"][name]
        soft, hard = dist["soft_upper"], dist["severe_upper"]
        metrics[name] = float(value)
        if value > soft:
            span = max(hard-soft, abs(soft)*np.finfo(np.float64).eps, np.finfo(np.float64).eps)
            penalties[name] = float((value-soft)/span)
            warnings.append(f"high_{name}")
            if reject and value > hard:
                severe.append(f"extreme_{name}")

    lengths_all = []; turns_all = []; accelerations_all = []; jaggedness = []; arcs = []
    for stroke in suffix:
        lengths, turns, accelerations = _stroke_features(stroke)
        lengths_all.extend(lengths); turns_all.extend(turns); accelerations_all.extend(accelerations)
        jaggedness.append(float(turns.mean()) if turns.size else 0.0)
        arcs.append(float(lengths.sum()))
    upper_penalty("stroke_count", len(suffix))
    upper_penalty("point_count", len(points))
    upper_penalty("points_per_stroke", max(map(len, suffix)))
    upper_penalty("segment_length", max(lengths_all, default=0))
    upper_penalty("stroke_arc_length", max(arcs, default=0))
    upper_penalty("bbox_diagonal", float(np.linalg.norm(np.ptp(points, axis=0))))
    # Tiny full sketches are flagged against TRAIN's lower tail. A short
    # completion may legitimately add only a tiny detail, so this warning does
    # not apply when a prefix was observed and never rejects by itself.
    lower_scale = stats["distributions"]["bbox_diagonal"]["q01"]
    if prefix_count == 0 and lower_scale > 0 and metrics["bbox_diagonal"] < lower_scale:
        warnings.append("small_sketch_scale")
        penalties["small_sketch_scale"] = float((lower_scale-metrics["bbox_diagonal"])/lower_scale)
    # Angular reversals can be legitimate details: flag/score them, never reject
    # a sketch for angle alone. Extreme repeated oscillations affect arc/count.
    upper_penalty("stroke_jaggedness", max(jaggedness, default=0), reject=False)
    upper_penalty("acceleration", max(accelerations_all, default=0), reject=False)
    zero_fraction = sum(arc == 0 for arc in arcs) / len(arcs)
    upper_penalty("zero_arc_fraction", zero_fraction, reject=False)
    metrics["single_point_strokes"] = sum(len(stroke) == 1 for stroke in suffix)
    metrics["zero_length_segments"] = sum(length == 0 for length in lengths_all)
    if zero_fraction:
        warnings.append("degenerate_generated_strokes")
    if metrics["zero_length_segments"]:
        warnings.append("duplicate_adjacent_points")
    valid = not severe
    score = 1.0 / (1.0 + max(penalties.values(), default=0.0)) if valid else 0.0
    return ValidationResult(valid, score, list(dict.fromkeys(warnings)), penalties,
                            severe_errors=list(dict.fromkeys(severe)), metrics=metrics)


def postprocess_candidate(raw_output: Sequence, stats: dict, prefix_count: int = 0,
                          termination=None) -> dict:
    """Keep both variants; only remove exact duplicates in generated strokes.

    Optional smoothing is off by default, convex, endpoint-preserving, and only
    applies to mildly flagged generated strokes. Any severe input is retained
    unchanged and rejected, never made plausible by clipping.
    """
    raw = deepcopy(list(raw_output))
    output = deepcopy(raw)
    before = validate_candidate(raw, stats, prefix_count, termination)
    corrections = []
    if before.valid:
        cfg = stats["config"]
        alpha = float(cfg.get("smoothing_alpha", 0.0))
        if not 0 <= alpha <= 1:
            raise ValueError("smoothing_alpha must be between zero and one")
        for index in range(prefix_count, len(output)):
            stroke = np.asarray(output[index]).copy()
            if cfg.get("deduplicate_adjacent", True) and len(stroke) > 1:
                keep = np.r_[True, np.any(stroke[1:] != stroke[:-1], axis=1)]
                removed = int((~keep).sum())
                if removed:
                    stroke = stroke[keep]
                    corrections.append({"stroke_index": index, "kind": "remove_exact_adjacent_duplicates", "removed_points": removed})
            if alpha and len(stroke) > 2 and "high_stroke_jaggedness" in before.warnings:
                changed = stroke.astype(np.float64)
                changed[1:-1] = (1-alpha)*changed[1:-1] + alpha*(changed[:-2]+changed[2:])/2
                stroke = changed
                corrections.append({"stroke_index": index, "kind": "convex_smoothing", "alpha": alpha})
            output[index] = stroke
    after = validate_candidate(output, stats, prefix_count, termination)
    after.corrections = corrections
    return {"raw_output": raw, "postprocessed_output": output, "validation": after,
            "raw_validation": before, "prefix_count": prefix_count,
            "termination": deepcopy(termination)}
