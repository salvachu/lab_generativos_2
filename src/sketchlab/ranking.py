"""Inspectable valid-first ranking and suffix-only diversity filtering.

No undocumented sum of likelihood, geometry, and diversity is used: ordering is
lexicographic, followed by a minimum-distance diversity constraint. Invalid
candidates are reported, never selected to fill the requested grid.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from .geometry import validate_candidate
from .evaluation import polyline_points

DEFAULT_RANKING_CONFIG = {
    "ordering": "geometry_first",
    "min_chamfer_fraction": 0.05,
    "max_metric_points": 256,
}


def _points(strokes: Sequence, prefix_count: int, max_points: int) -> np.ndarray:
    suffix = strokes[prefix_count:]
    if not suffix:
        return np.empty((0, 2), dtype=np.float64)
    return polyline_points(suffix, max_points=max_points)


def suffix_chamfer(strokes_a: Sequence, strokes_b: Sequence,
                    prefix_count_a: int = 0, prefix_count_b: int | None = None,
                    *, max_points: int = 256) -> float:
    """Symmetric mean nearest-neighbor distance in original coordinate units.

    Deterministic arc-length sampling affects this auxiliary metric only. Prefix
    points are excluded; distance cannot be diluted by a shared long prefix.
    """
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    prefix_count_b = prefix_count_a if prefix_count_b is None else prefix_count_b
    a = _points(strokes_a, prefix_count_a, max_points)
    b = _points(strokes_b, prefix_count_b, max_points)
    if not len(a) and not len(b):
        return 0.0
    if not len(a) or not len(b):
        return float("inf")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float((distances.min(axis=1).mean()+distances.min(axis=0).mean())/2)


def diversity_metrics(candidate_records: Sequence[dict], *, max_points: int = 256) -> dict:
    distances = []
    endpoint_distances = []
    counts = []
    for record in candidate_records:
        strokes = record["postprocessed_output"]
        counts.append(len(strokes) - record.get("prefix_count", 0))
    for a, b in combinations(candidate_records, 2):
        prefix_a, prefix_b = a.get("prefix_count", 0), b.get("prefix_count", 0)
        distance = suffix_chamfer(a["postprocessed_output"], b["postprocessed_output"], prefix_a, prefix_b, max_points=max_points)
        if np.isfinite(distance):
            distances.append(distance)
        strokes_a = a["postprocessed_output"][prefix_a:]
        strokes_b = b["postprocessed_output"][prefix_b:]
        if strokes_a and strokes_b:
            endpoint_distances.append(float(np.linalg.norm(np.asarray(strokes_a[-1])[-1] - np.asarray(strokes_b[-1])[-1])))
    return {"pair_count": len(distances), "suffix_chamfer_mean": float(np.mean(distances)) if distances else None,
            "suffix_chamfer_min": float(min(distances)) if distances else None,
            "endpoint_distance_mean": float(np.mean(endpoint_distances)) if endpoint_distances else None,
            "generated_stroke_count_std": float(np.std(counts)) if counts else None,
            "distinct_generated_stroke_counts": len(set(counts)),
            "interpretation": "Diversity is measured only among geometrically accepted generated suffixes; not a semantic plausibility measure."}


def rank_candidates(candidate_records: Sequence[dict], top_k: int, stats: dict,
                    config: dict | None = None) -> dict:
    cfg = {**DEFAULT_RANKING_CONFIG, **(config or {})}
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if cfg["ordering"] not in {"geometry_first", "model_first"}:
        raise ValueError("ordering must be geometry_first or model_first")
    if (not np.isfinite(cfg["min_chamfer_fraction"]) or cfg["min_chamfer_fraction"] < 0
            or type(cfg["max_metric_points"]) is not int or not 2 <= cfg["max_metric_points"] <= 4096):
        raise ValueError("Invalid diversity settings")
    threshold = float(stats["distributions"]["segment_length"]["median"] * cfg["min_chamfer_fraction"])
    records = []
    invalid = []
    for index, original in enumerate(candidate_records):
        record = dict(original)
        record.setdefault("id", index)
        if "postprocessed_output" not in record:
            raise ValueError("Candidate records need postprocessed_output")
        validation = record.get("validation")
        if validation is None:
            validation = validate_candidate(record["postprocessed_output"], stats,
                                            record.get("prefix_count", 0), record.get("termination"))
        if is_dataclass(validation):
            validation = asdict(validation)
        record["validation"] = validation
        score = float(validation.get("score", 0))
        if not validation.get("valid") or validation.get("severe_errors") or not np.isfinite(score):
            invalid.append({"index": index, "id": record["id"], "errors": validation.get("severe_errors", [])})
        else:
            records.append((index, record))

    def key(item):
        index, record = item
        validation = record["validation"]
        model_score = float(record.get("model_score", 0.0) or 0.0)
        if not np.isfinite(model_score):
            model_score = -float("inf")
        geometry = (-float(validation["score"]), len(validation.get("warnings", [])))
        return (*geometry, -model_score, index) if cfg["ordering"] == "geometry_first" else (-model_score, *geometry, index)

    records.sort(key=key)
    selected = []
    duplicate = []
    for index, record in records:
        if len(selected) >= top_k:
            break
        distances = [(other_index, suffix_chamfer(record["postprocessed_output"], other["postprocessed_output"],
                                                  record.get("prefix_count", 0), other.get("prefix_count", 0),
                                                  max_points=int(cfg["max_metric_points"])))
                     for other_index, other in selected]
        too_close = [(other_index, distance) for other_index, distance in distances if distance <= threshold]
        if too_close:
            nearest = min(too_close, key=lambda item: item[1])
            duplicate.append({"index": index, "near_selected_index": nearest[0], "suffix_chamfer": nearest[1]})
            continue
        selected.append((index, record))
    selected_records = [record for _, record in selected]
    return {"selected": selected_records, "selected_indices": [index for index, _ in selected],
            "report": {"candidate_count": len(candidate_records), "valid_count": len(records),
                       "invalid_count": len(invalid), "requested_top_k": top_k, "selected_count": len(selected),
                       "invalid": invalid, "diversity_filtered": duplicate,
                       "quality_order": [index for index, _ in records],
                       "config": cfg, "min_suffix_chamfer": threshold,
                       "threshold_rule": "TRAIN median intra-stroke segment length * min_chamfer_fraction",
                       "ranking_rule": "Valid-only lexicographic ranking, then greedy minimum suffix Chamfer distance; no weighted sum or invalid fallback.",
                       "diversity": diversity_metrics(selected_records, max_points=int(cfg["max_metric_points"]))}}
