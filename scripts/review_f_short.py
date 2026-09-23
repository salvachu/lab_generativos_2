"""Evaluate the first compositional Model-F checkpoint on fixed VAL completions."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

from sketchlab.data import load_raw
from sketchlab.evaluation import distances, geometry_metrics, prefix_count, prefix_preserved
from sketchlab.generation import load_model, sample
from sketchlab.geometry import load_geometry, validate_candidate
from sketchlab.training import training_threshold, validate


CONDITIONS = [("1STROKE", "one"), ("2STROKES", "two"), ("25PCT", .25),
              ("50PCT", .5), ("75PCT", .75)]
FILENAMES = {
    "1STROKE": "01_F_COMPLETION_1STROKE.png",
    "2STROKES": "02_F_COMPLETION_2STROKES.png",
    "25PCT": "03_F_COMPLETION_25PCT.png",
    "50PCT": "04_F_COMPLETION_50PCT.png",
    "75PCT": "05_F_COMPLETION_75PCT.png",
}
BOUNDS = (-4.0, 508.0)


def mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def prefix_hash(strokes):
    digest = hashlib.sha256()
    for stroke in strokes:
        array = np.asarray(stroke, dtype="<f4")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes()); digest.update(array.tobytes())
    return digest.hexdigest()


def plot_sketch(ax, strokes, prefix_count_value, title, *, gt=False, annotation=""):
    lo, hi = BOUNDS
    for index, stroke in enumerate(strokes):
        points = np.asarray(stroke, dtype=np.float64)
        color = "#2378d4" if index < prefix_count_value else ("#666666" if gt else "#e76427")
        ax.plot(points[:, 0], points[:, 1], color=color, lw=.85, solid_capstyle="round")
    ax.set(xlim=(lo, hi), ylim=(hi, lo), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel(annotation, fontsize=5.5, linespacing=1.15)
    for spine in ax.spines.values(): spine.set_color("#cccccc")


def measure(generated, count, truth, info, stats, jump_threshold_value):
    suffix = generated[count:]
    geom = geometry_metrics(suffix, jump_threshold_value)
    completion = validate_candidate(generated, stats, prefix_count=count, termination=info)
    geometry_info = {**info, "termination": "eos", "ended_by_eos": True, "capped": False}
    geometry = validate_candidate(generated, stats, prefix_count=count, termination=geometry_info)
    points = sum(len(stroke) for stroke in suffix)
    truth_points = sum(len(stroke) for stroke in truth)
    matched = min(len(suffix), len(truth))
    anchor_error = mean(np.linalg.norm(np.asarray(suffix[i])[0] - np.asarray(truth[i])[0]) for i in range(matched))
    return {
        "eos": bool(info["ended_by_eos"]), "cap": bool(info["capped"]), "termination": info["termination"],
        "generated_strokes": len(suffix), "generated_points": points,
        "canvas_validity": geom["canvas_validity"] or 0.0,
        "fully_canvas_valid": bool(geom["canvas_validity"] == 1.0),
        "geometry_valid": bool(geometry.valid), "completion_valid": bool(completion.valid),
        "premature_termination": bool(info["ended_by_eos"] and points < max(5, .1 * truth_points)),
        "anchor_error": anchor_error, "prefix_exact": True,
        "intra_jump_fraction": geom["intra_jump_fraction"],
        "sharp_turn_fraction": geom["sharp_turn_fraction"],
        "validation_errors": completion.severe_errors, **distances(suffix, truth),
    }


def aggregate(records):
    metrics = [record["metrics"] for record in records]
    return {
        "samples": len(metrics), "eos_rate": mean(m["eos"] for m in metrics),
        "sequence_cap_rate": mean(m["cap"] for m in metrics),
        "mean_generated_strokes": mean(m["generated_strokes"] for m in metrics),
        "mean_generated_points": mean(m["generated_points"] for m in metrics),
        "canvas_validity": mean(m["canvas_validity"] for m in metrics),
        "fully_canvas_valid_rate": mean(m["fully_canvas_valid"] for m in metrics),
        "geometry_valid_rate": mean(m["geometry_valid"] for m in metrics),
        "completion_valid_rate": mean(m["completion_valid"] for m in metrics),
        "premature_termination_rate": mean(m["premature_termination"] for m in metrics),
        "anchor_error": mean(m["anchor_error"] for m in metrics),
        "chamfer": mean(m["chamfer"] for m in metrics),
        "diversity_chamfer": mean(m.get("diversity_chamfer") for m in metrics),
        "prefix_exact_fraction": mean(m["prefix_exact"] for m in metrics),
    }


def render_condition(cases, condition, grouped, path):
    figure, axes = plt.subplots(len(cases), 4, figsize=(12.8, 3.15 * len(cases)), constrained_layout=True)
    for row, case in enumerate(cases):
        count = grouped[(condition, row)][0]["prefix_count"]
        plot_sketch(axes[row, 0], case["strokes"], count, f"case {row+1:02d} · GT", gt=True,
                    annotation=f"prefix {count} · real suffix {len(case['strokes'])-count} strokes")
        for latent, record in enumerate(grouped[(condition, row)]):
            metric = record["metrics"]
            plot_sketch(axes[row, latent + 1], record["strokes"], count, f"F · z{latent+1}",
                        annotation=(f"{metric['generated_strokes']} strokes · {metric['generated_points']} points · "
                                    f"{metric['termination']}\ncanvas {100*metric['canvas_validity']:.0f}% · "
                                    f"{'VALID' if metric['completion_valid'] else 'INVALID'}"))
    figure.suptitle(f"Model F short · completion {condition} · blue prefix / orange continuation / gray GT", fontsize=13)
    figure.savefig(path, dpi=150, facecolor="white"); plt.close(figure)


def render_random(records, path):
    figure, axes = plt.subplots(3, 3, figsize=(9.6, 9.8), constrained_layout=True)
    for index, (ax, record) in enumerate(zip(axes.ravel(), records)):
        metric = record["metrics"]
        plot_sketch(ax, record["strokes"], 0, f"F random · z{index+1}",
                    annotation=(f"{metric['generated_strokes']} strokes · {metric['generated_points']} points · "
                                f"{metric['termination']} · canvas {100*metric['canvas_validity']:.0f}%"))
    figure.suptitle("Model F short · random generation", fontsize=13)
    figure.savefig(path, dpi=150, facecolor="white"); plt.close(figure)


def render_e_comparison(cases, grouped, persisted_root, path):
    selections = [("1STROKE", 0), ("1STROKE", 4), ("25PCT", 0), ("25PCT", 4)]
    figure, axes = plt.subplots(len(selections), 3, figsize=(10.2, 3.3 * len(selections)), constrained_layout=True)
    for row, (condition, case_index) in enumerate(selections):
        folder = "1stroke" if condition == "1STROKE" else "25pct"
        page = persisted_root / "completion" / folder / f"overview_{folder}_page{case_index+1:02d}.png"
        image = mpimg.imread(page); height, width = image.shape[:2]
        lower = image[int(.52 * height):height]
        axes[row, 0].imshow(lower[:, :int(.25 * width)]); axes[row, 0].axis("off")
        axes[row, 1].imshow(lower[:, int(.25 * width):int(.5 * width)]); axes[row, 1].axis("off")
        record = grouped[(condition, case_index)][0]
        plot_sketch(axes[row, 2], record["strokes"], record["prefix_count"],
                    f"F short · z1 · {condition}", annotation=(f"{record['metrics']['generated_strokes']} strokes · "
                    f"{record['metrics']['termination']} · canvas {100*record['metrics']['canvas_validity']:.0f}%"))
    figure.suptitle("Persisted E Round 2 vs F short · same VAL cases and prefixes", fontsize=13)
    figure.savefig(path, dpi=155, facecolor="white"); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("checkpoint")
    parser.add_argument("--completion-output", default="runs/F_short_completion_review")
    parser.add_argument("--review-output", default="runs/F_short_review")
    args = parser.parse_args(); torch.set_num_threads(1)
    completion_output, review_output = Path(args.completion_output), Path(args.review_output)
    if any(path.exists() and any(path.iterdir()) for path in (completion_output, review_output)):
        raise FileExistsError("Review output already populated")
    completion_output.mkdir(parents=True, exist_ok=True)
    grid_output = review_output / "visual_grids"; grid_output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint); validation = load_raw("data/raw/val.pkl")
    fixed_path = Path("runs/visual_benchmark_round2/selected_validation_cases.json")
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    cases = [validation[int(item["validation_index"])] for item in fixed["cases"]]
    for item, case in zip(fixed["cases"], cases):
        if case["id"] != item["id"]: raise RuntimeError("Fixed VAL case mismatch")
    stats, threshold = load_geometry(), training_threshold(None)
    base_seed, temperature, max_points, max_strokes = 43100, .6, 384, 64
    records, grouped = [], {}
    for condition_index, (condition, fraction) in enumerate(CONDITIONS):
        for case_index, case in enumerate(cases):
            count = prefix_count(len(case["strokes"]), fraction)
            prefix, truth = case["strokes"][:count], case["strokes"][count:]
            group = []
            decoder_seed = base_seed + condition_index * 10000 + case_index * 100
            for latent in range(3):
                generated, info = sample(model, prefix=prefix, seed=decoder_seed + latent,
                                         decoder_seed=decoder_seed + 900000, temperature=temperature,
                                         max_points=max_points, max_strokes=max_strokes, return_info=True)
                if not prefix_preserved(prefix, generated): raise AssertionError("F modified the prefix")
                metric = measure(generated, count, truth, info, stats, threshold)
                record = {"condition": condition, "case": case_index + 1, "validation_id": case["id"],
                          "latent": latent + 1, "prefix_count": count, "prefix_hash": prefix_hash(prefix),
                          "metrics": metric, "strokes": generated}
                group.append(record); records.append(record)
            suffixes = [record["strokes"][count:] for record in group]
            pairwise = [distances(a, b)["chamfer"] for a, b in itertools.combinations(suffixes, 2)]
            diversity = mean(pairwise)
            for record in group: record["metrics"]["diversity_chamfer"] = diversity
            grouped[(condition, case_index)] = group
    for condition, _ in CONDITIONS:
        destination = completion_output / FILENAMES[condition]
        render_condition(cases, condition, grouped, destination)
        shutil.copyfile(destination, grid_output / destination.name)

    random_records = []
    for index in range(9):
        seed = base_seed + 80000 + index
        generated, info = sample(model, prefix=[], seed=seed, decoder_seed=seed + 900000,
                                 temperature=temperature, max_points=max_points,
                                 max_strokes=max_strokes, return_info=True)
        metric = measure(generated, 0, [], info, stats, threshold)
        random_records.append({"seed": seed, "metrics": metric, "strokes": generated})
    random_path = completion_output / "06_F_RANDOM_GENERATION.png"
    render_random(random_records, random_path); shutil.copyfile(random_path, grid_output / random_path.name)

    persisted_e = Path("runs/visual_benchmark_round2")
    comparison_path = review_output / "COMPARISON_WITH_E.png"
    render_e_comparison(cases, grouped, persisted_e, comparison_path)
    shutil.copyfile(comparison_path, completion_output / "COMPARISON_WITH_E.png")

    compact = [{key: value for key, value in record.items() if key != "strokes"} for record in records]
    overall = aggregate(records)
    by_prefix = {condition: aggregate([record for record in records if record["condition"] == condition])
                 for condition, _ in CONDITIONS}
    train_summary = json.loads((Path(args.checkpoint).parent / "summary.json").read_text(encoding="utf-8"))
    likelihood = validate(model, cases, 4, beta=.05, free_bits=.0)
    f_metrics = {"checkpoint": str(Path(args.checkpoint).resolve()), "updates": train_summary["steps"],
                 "best_step": train_summary["best_step"], "validation": likelihood,
                 "collapse": train_summary["collapse"], "completion_overall": overall}
    (review_output / "F_metrics.json").write_text(json.dumps(f_metrics, indent=2), encoding="utf-8")
    (review_output / "completion_metrics.json").write_text(
        json.dumps({"overall": overall, "by_prefix": by_prefix, "records": compact}, indent=2), encoding="utf-8")
    random_compact = [{key: value for key, value in record.items() if key != "strokes"} for record in random_records]
    (review_output / "random_metrics.json").write_text(
        json.dumps({"overall": aggregate(random_records), "records": random_compact}, indent=2), encoding="utf-8")
    before = json.loads(Path("runs/F_stroke_diagnostic/SUMMARY.json").read_text(encoding="utf-8"))
    after = json.loads(Path("runs/F_stroke_stage1_diagnostic/SUMMARY.json").read_text(encoding="utf-8"))
    (review_output / "stroke_ae_metrics.json").write_text(
        json.dumps({"updates_300": before, "updates_1500": after}, indent=2), encoding="utf-8")
    report = f"""# Model F short review

Stroke AE passed the visual gate at 1500 updates with 16 samples per stroke. Compositional F ran for {train_summary['steps']} updates; best step was {train_summary['best_step']}.

Completion summary: EOS {overall['eos_rate']:.3f}, cap {overall['sequence_cap_rate']:.3f}, completion-valid {overall['completion_valid_rate']:.3f}, geometry-valid {overall['geometry_valid_rate']:.3f}, diversity Chamfer {overall['diversity_chamfer']:.3f}. Prefix preservation was {overall['prefix_exact_fraction']:.3f}.

The E comparison uses only persisted Round 2 overview images. It does not retrain or resample E.
"""
    (review_output / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {"stroke_ae_before": before, "stroke_ae_after": after, "F": f_metrics,
               "completion": {"overall": overall, "by_prefix": by_prefix},
               "random": aggregate(random_records), "images": [str(path) for path in sorted(completion_output.glob("*.png"))]}
    (review_output / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "overall": overall, "by_prefix": by_prefix,
                      "random": aggregate(random_records)}, indent=2))


if __name__ == "__main__":
    main()
