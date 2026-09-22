"""Geometry and conditional diversity, explicitly separate from likelihood.

These metrics are diagnostics, not a test of semantic plausibility. Ground truth
is one possible continuation; a larger distance does not prove an invalid sample.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist


def polyline_points(strokes, max_points=256):
    """Sample along vector segments, never connect different strokes."""
    segments = []
    singles = []
    for s in strokes:
        s = np.asarray(s, dtype=np.float64)
        if len(s) == 1:
            singles.append(s[0])
        if len(s) > 1:
            lengths = np.linalg.norm(np.diff(s, axis=0), axis=1)
            for a, b, length in zip(s[:-1], s[1:], lengths):
                if length > 0:
                    segments.append((a, b, length))
            if not lengths.any():
                singles.append(s[0])
    if not segments:
        return np.asarray(singles, dtype=np.float64).reshape(-1, 2)[:max_points]
    lengths = np.asarray([v[2] for v in segments])
    cumulative = np.cumsum(lengths)
    positions = (np.arange(max_points) + .5) / max_points * cumulative[-1]
    indices = np.searchsorted(cumulative, positions)
    previous = np.r_[0., cumulative[:-1]]
    fraction = (positions - previous[indices]) / lengths[indices]
    a = np.asarray([v[0] for v in segments])[indices]
    b = np.asarray([v[1] for v in segments])[indices]
    return a + fraction[:, None] * (b - a)


def distances(strokes_a, strokes_b):
    """Symmetric mean Chamfer and Hausdorff in source coordinate units."""
    a, b = polyline_points(strokes_a), polyline_points(strokes_b)
    if not len(a) or not len(b):
        return {"chamfer": None, "hausdorff": None, "endpoint_chamfer": None}
    d = cdist(a, b)
    ae = np.concatenate([np.asarray(s)[[0, -1]] for s in strokes_a if len(s)])
    be = np.concatenate([np.asarray(s)[[0, -1]] for s in strokes_b if len(s)])
    ed = cdist(ae, be)
    return {
        "chamfer": float((d.min(0).mean() + d.min(1).mean()) / 2),
        "hausdorff": float(max(d.min(0).max(), d.min(1).max())),
        "endpoint_chamfer": float((ed.min(0).mean() + ed.min(1).mean()) / 2),
    }


def geometry_metrics(strokes, jump_threshold=150., canvas_bounds=(-4., 508.)):
    nonempty = [np.asarray(s, dtype=np.float64) for s in strokes if len(s)]
    if not nonempty:
        return {"strokes": 0, "points": 0, "empty": 1., "canvas_validity": None,
                "nominal_canvas_validity": None, "degenerate_fraction": None,
                "mean_stroke_points": None, "intra_jump_fraction": None,
                "sharp_turn_fraction": None, "bbox_diagonal": None}
    points = np.concatenate(nonempty)
    segment_lengths, turns = [], []
    for s in nonempty:
        d = np.diff(s, axis=0)
        lens = np.linalg.norm(d, axis=1)
        segment_lengths.extend(lens)
        if len(d) > 1:
            valid = (lens[:-1] > 1e-6) & (lens[1:] > 1e-6)
            cosine = (d[:-1] * d[1:]).sum(1) / np.maximum(lens[:-1] * lens[1:], 1e-12)
            turns.extend(np.arccos(np.clip(cosine[valid], -1, 1)))
    low, high = canvas_bounds
    return {
        "strokes": len(strokes), "points": len(points), "empty": 0.,
        "canvas_validity": float(((points >= low) & (points <= high)).all(1).mean()),
        "nominal_canvas_validity": float(((points >= 0) & (points <= 504)).all(1).mean()),
        "degenerate_fraction": float(np.mean([len(s) < 2 or np.ptp(s, axis=0).max() < 1e-6 for s in nonempty])),
        "mean_stroke_points": float(np.mean([len(s) for s in nonempty])),
        "intra_jump_fraction": float(np.mean(np.asarray(segment_lengths) > jump_threshold)) if segment_lengths else 0.,
        "sharp_turn_fraction": float(np.mean(np.asarray(turns) > 2.6)) if turns else 0.,
        "bbox_diagonal": float(np.linalg.norm(np.ptp(points, axis=0))),
    }


def prefix_preserved(prefix, completed):
    return len(completed) >= len(prefix) and all(
        np.array_equal(a, b) for a, b in zip(prefix, completed[:len(prefix)])
    )


def finite_mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def prefix_count(n_strokes, fraction):
    if fraction == "one":
        return 1
    if fraction == "two":
        return min(2, n_strokes - 1)
    return max(1, min(n_strokes - 1, int(round(n_strokes * float(fraction)))))


def evaluate_generation(model, samples, output_dir, *, seed=173, n_samples=3,
                        max_points=384, max_strokes=64, temperature=.6,
                        jump_threshold=150.):
    """Same latent seeds + fixed decoder RNG isolate the effect of z.

    Common-random-number coupling does not make sample paths identical: divergent
    stop decisions can change consumption. It removes independent decoder noise
    as the immediate explanation for different outputs.
    """
    import json
    from sketchlab.generation import sample
    from sketchlab.rendering import render_grid

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, serialized, grids = [], [], {}
    conditions = ["one", "two", .25, .5, .75]
    for fraction in conditions:
        panels, labels, counts = [], [], []
        for index, raw in enumerate(samples):
            strokes = raw["strokes"]
            count = prefix_count(len(strokes), fraction)
            prefix, truth = strokes[:count], strokes[count:]
            completions = []
            panels.append(strokes); labels.append(f"val {raw['id']} | observed {count} | GT"); counts.append(count)
            for k in range(n_samples):
                generated, info = sample(
                    model, prefix=prefix, seed=seed + index * 100 + k,
                    decoder_seed=seed + index * 10000,
                    temperature=temperature, max_points=max_points,
                    max_strokes=max_strokes, return_info=True,
                )
                if not prefix_preserved(prefix, generated):
                    raise AssertionError("Completion modified observed strokes")
                suffix = generated[count:]
                completions.append(suffix)
                record = {"id": raw["id"], "condition": str(fraction), "prefix_strokes": count,
                          "sample": k, "prefix_exact": True, **distances(suffix, truth),
                          **geometry_metrics(suffix, jump_threshold), "termination": info}
                records.append(record)
                serialized.append({"id": raw["id"], "condition": str(fraction), "sample": k,
                                   "prefix_count": count,
                                   "strokes": [np.asarray(s).tolist() for s in generated], "info": info})
                if index < 4:
                    panels.append(generated); labels.append(f"z{k+1} | prefix {count}"); counts.append(count)
            pairwise = [distances(a, b)["chamfer"] for a, b in itertools.combinations(completions, 2)]
            diversity = finite_mean(pairwise)
            for r in records[-n_samples:]:
                r["latent_diversity_chamfer"] = diversity
                r["distinct_suffixes"] = len({json.dumps([np.asarray(s).tolist() for s in c]) for c in completions})
        key = str(fraction).replace(".", "p")
        grid_path = output_dir / f"completion_{key}.png"
        render_grid(panels, grid_path, titles=labels, prefix_counts=counts)
        grids[str(fraction)] = str(grid_path)
    random_panels, random_info = [], []
    for k in range(9):
        generated, info = sample(model, seed=seed + 9000 + k, temperature=temperature,
                                 max_points=max_points, max_strokes=max_strokes, return_info=True)
        random_panels.append(generated)
        random_info.append({**geometry_metrics(generated, jump_threshold), "termination": info})
    render_grid(random_panels, output_dir / "random.png", titles=[f"z{i+1}" for i in range(9)])
    summary: dict[str, Any] = {}
    keys = ["chamfer", "hausdorff", "endpoint_chamfer", "canvas_validity", "nominal_canvas_validity",
            "degenerate_fraction", "intra_jump_fraction", "sharp_turn_fraction", "bbox_diagonal",
            "mean_stroke_points", "strokes", "points", "empty", "latent_diversity_chamfer", "distinct_suffixes"]
    summary["overall"] = {k: finite_mean(r[k] for r in records) for k in keys}
    summary["by_prefix"] = {
        str(f): {k: finite_mean(r[k] for r in records if r["condition"] == str(f)) for k in keys}
        for f in conditions
    }
    summary["prefix_exact_fraction"] = float(np.mean([r["prefix_exact"] for r in records]))
    summary["random"] = random_info
    summary["grids"] = grids
    summary["conditions"] = {"validation_sketches": len(samples), "samples_per_prefix": n_samples,
                              "max_generated_points": max_points, "max_generated_strokes": max_strokes,
                              "temperature": temperature, "decoder_rng": "fixed across latent samples",
                              "seed": seed, "jump_threshold_train_p995": jump_threshold}
    summary["records"] = records
    (output_dir / "generation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "generated_samples.json").write_text(json.dumps(serialized), encoding="utf-8")
    return summary
