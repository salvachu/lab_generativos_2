"""Summarize persisted Round 2 evidence and compare it with persisted Round 1."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from sketchlab.data import load_raw
from sketchlab.diagnostics import latent_sensitivity
from sketchlab.evaluation import prefix_count
from sketchlab.generation import load_model
from sketchlab.orchestration import json_ready


ROOT = Path("runs/tournament_round2")
VIS1 = Path("runs/visual_benchmark_round1")
VIS2 = Path("runs/visual_benchmark_round2")
MODELS = "CE"


def aggregate(records, model, *, exclude_75=False):
    completion = list(records["completion"][model])
    if exclude_75:
        completion = [r for r in completion if r.get("condition") != "75pct"]
    random = list(records["random"][model])
    all_records = completion + random
    points = [r["metrics"]["generated_points"] for r in all_records]
    strokes = [r["metrics"]["generated_strokes"] for r in all_records]
    stroke_total = sum(strokes)
    return {
        "median_generated_strokes": float(np.median(strokes)),
        "median_generated_points": float(np.median(points)),
        "p25_generated_points": float(np.quantile(points, .25)),
        "p75_generated_points": float(np.quantile(points, .75)),
        "sequence_cap_rate": float(np.mean([r["metrics"]["termination"] == "max_points" for r in all_records])),
        "eos_rate": float(np.mean([r["metrics"]["eos"] for r in all_records])),
        "completion_valid_rate": float(np.mean([r["metrics"]["valid"] for r in completion])),
        "mean_points_inside_canvas": float(np.mean([r["metrics"]["inside_point_fraction"] for r in all_records])),
        "fully_canvas_valid_rate": float(np.mean([r["metrics"]["all_points_inside"] for r in all_records])),
        "degenerate_rate": (sum(r["metrics"]["degenerate_strokes"] for r in all_records) / stroke_total
                            if stroke_total else 0.0),
        "median_random_strokes": float(np.median([r["metrics"]["generated_strokes"] for r in random])),
        "median_random_points": float(np.median([r["metrics"]["generated_points"] for r in random])),
    }


def finite_mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def status(metric, r1, r2, real_strokes, real_points):
    if r1 is None or r2 is None:
        return "SIMILAR"
    if metric in {"sequence_cap_rate", "degenerate_rate", "mean_chamfer", "mean_hausdorff",
                  "mean_intra_stroke_jump_rate"}:
        tolerance = .01 if "rate" in metric else max(1e-9, abs(float(r1)) * .05)
        return "IMPROVED" if r2 < r1 - tolerance else "WORSE" if r2 > r1 + tolerance else "SIMILAR"
    if metric in {"eos_rate", "completion_valid_rate", "mean_points_inside_canvas",
                  "fully_canvas_valid_rate", "latent_decoder_mean_shift", "latent_stop_logit_shift"}:
        tolerance = .01 if "rate" in metric or "canvas" in metric else max(1e-9, abs(float(r1)) * .05)
        return "IMPROVED" if r2 > r1 + tolerance else "WORSE" if r2 < r1 - tolerance else "SIMILAR"
    if metric in {"median_generated_strokes", "median_random_strokes"}:
        a, b = abs(r1 - real_strokes), abs(r2 - real_strokes)
        return "IMPROVED" if b < a - .5 else "WORSE" if b > a + .5 else "SIMILAR"
    if metric in {"median_generated_points", "median_random_points", "p25_generated_points", "p75_generated_points"}:
        a, b = abs(r1 - real_points), abs(r2 - real_points)
        return "IMPROVED" if b < a - 2 else "WORSE" if b > a + 2 else "SIMILAR"
    if metric == "val_reconstruction":
        tolerance = max(1e-9, abs(float(r1)) * .02)
        return "IMPROVED" if r2 < r1 - tolerance else "WORSE" if r2 > r1 + tolerance else "SIMILAR"
    return "SIMILAR"


def main():
    r1_records = json.loads((VIS1 / "benchmark_metrics.json").read_text(encoding="utf-8"))
    r2_records = json.loads((VIS2 / "benchmark_metrics.json").read_text(encoding="utf-8"))
    r1_score = {r["model"]: r for r in csv.DictReader((Path("runs/tournament_round1") / "scoreboard.csv").open(encoding="utf-8"))}
    r1_generation = {m: json.loads((Path("runs/tournament_round1") / m / "generation_metrics.json").read_text(encoding="utf-8")) for m in MODELS}
    val = load_raw("data/raw/val.pkl")
    fixed = json.loads((VIS1 / "selected_validation_cases.json").read_text(encoding="utf-8"))["cases"]
    first = val[int(fixed[0]["validation_index"])]
    count = prefix_count(len(first["strokes"]), "one")
    prefix = first["strokes"][:count]
    real_strokes = float(np.median([len(s["strokes"]) for s in val]))
    real_points = float(np.median([sum(len(x) for x in s["strokes"]) for s in val]))

    round2_rows = []
    progress_rows = []
    details = {"comparison_protocol": {
        "length_metrics": "same fixed Round 1 visual cases, same four common prefixes plus random; 75% excluded from deltas",
        "distance_metrics": "Round 1 persisted tournament aggregate and Round 2 fixed visual-case aggregate; case sets differ",
        "no_total_score": True,
    }, "models": {}}
    length_fields = ["median_generated_strokes", "median_generated_points", "p25_generated_points",
                     "p75_generated_points", "sequence_cap_rate", "eos_rate", "completion_valid_rate",
                     "mean_points_inside_canvas", "fully_canvas_valid_rate", "degenerate_rate",
                     "median_random_strokes", "median_random_points"]
    for model in MODELS:
        r1_len = aggregate(r1_records, model)
        r2_len_common = aggregate(r2_records, model, exclude_75=True)
        r2_len_all = aggregate(r2_records, model)
        summary = json.loads((ROOT / model / "summary.json").read_text(encoding="utf-8"))
        validation = summary["validation"]
        sensitivity = latent_sensitivity(load_model(ROOT / model / "best.pt"), prefix,
                                         seed=2026, max_points=64, max_strokes=16)
        completions = r2_records["completion"][model]
        detailed = {
            "mean_chamfer": finite_mean(r["metrics"].get("chamfer") for r in completions),
            "mean_hausdorff": finite_mean(r["metrics"].get("hausdorff") for r in completions),
            "mean_intra_stroke_jump_rate": finite_mean(r["metrics"].get("geometry_intra_jump_fraction") for r in completions),
            "mean_latent_diversity_chamfer": finite_mean(r["metrics"].get("latent_diversity_chamfer") for r in completions),
            "valid_only_diversity_chamfer": None,
            "valid_only_diversity_note": "Not estimated: too few strictly valid groups and vectors were intentionally not duplicated in the persisted metrics JSON.",
        }
        row = {"model": model, "round": 2, **r2_len_all,
               "median_real_strokes": real_strokes, "median_real_points": real_points,
               **detailed,
               "accumulated_updates": 800,
               "additional_updates": summary["steps"],
               "training_time_seconds": summary["training_time_seconds"],
               "total_time_seconds": summary["total_time_seconds"],
               "peak_vram_mib": summary["peak_vram_mib"],
               "val_reconstruction": validation["reconstruction"],
               "val_kl_global": validation["global_kl"],
               "val_kl_local": validation["stroke_kl"],
               "pen_accuracy": validation["pen_accuracy"],
               "collapse_warning": summary["collapse"]["persistent_near_zero_kl"],
               "latent_decoder_mean_shift": sensitivity["decoder_mixture_mean_l2"],
               "latent_stop_logit_shift": sensitivity["decoder_stop_logit_l2"],
               "latent_ignored_alert": sensitivity["latent_ignored_alert"],
               "prefix_exact": summary["prefix_exact_fraction"] == 1.0,
               "best_checkpoint": str(ROOT / model / "best.pt"),
               "last_checkpoint": str(ROOT / model / "last.pt")}
        round2_rows.append(row)
        for metric in length_fields:
            r1, r2 = r1_len[metric], r2_len_common[metric]
            progress_rows.append({"model": model, "metric": metric, "round1": r1, "round2": r2,
                                  "delta": r2-r1, "status": status(metric, r1, r2, real_strokes, real_points),
                                  "comparison": "same fixed cases and common prefixes"})
        auxiliary = {
            "val_reconstruction": (float(r1_score[model]["val_reconstruction"]), row["val_reconstruction"]),
            "val_kl_global": (float(r1_score[model]["val_kl_global"]), row["val_kl_global"]),
            "val_kl_local": (float(r1_score[model]["val_kl_local"]), row["val_kl_local"]),
            "mean_chamfer": (float(r1_score[model]["completion_chamfer"]), detailed["mean_chamfer"]),
            "mean_hausdorff": (r1_generation[model]["overall"]["hausdorff"], detailed["mean_hausdorff"]),
            "mean_intra_stroke_jump_rate": (float(r1_score[model]["intra_stroke_jump_rate"]), detailed["mean_intra_stroke_jump_rate"]),
            "mean_latent_diversity_chamfer": (float(r1_score[model]["diversity_chamfer"]), detailed["mean_latent_diversity_chamfer"]),
            "latent_decoder_mean_shift": (float(r1_score[model]["latent_decoder_mean_shift"]), row["latent_decoder_mean_shift"]),
            "latent_stop_logit_shift": (float(r1_score[model]["latent_stop_logit_shift"]), row["latent_stop_logit_shift"]),
        }
        for metric, (r1, r2) in auxiliary.items():
            progress_rows.append({"model": model, "metric": metric, "round1": r1, "round2": r2,
                                  "delta": r2-r1, "status": status(metric, r1, r2, real_strokes, real_points),
                                  "comparison": "persisted Round 1 aggregate vs Round 2; distance case sets differ" if metric.startswith("mean_") else "same validation/training protocol"})
        details["models"][model] = {"round1_common": r1_len, "round2_common": r2_len_common,
                                      "round2_all_prefixes": row, "latent_sensitivity": sensitivity}

    with (ROOT / "round2_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(round2_rows[0])); writer.writeheader(); writer.writerows(round2_rows)
    (ROOT / "round2_metrics.json").write_text(json.dumps(json_ready(round2_rows), indent=2), encoding="utf-8")
    with (ROOT / "progress.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(progress_rows[0])); writer.writeheader(); writer.writerows(progress_rows)
    details["progress"] = progress_rows
    (ROOT / "progress.json").write_text(json.dumps(json_ready(details), indent=2), encoding="utf-8")
    (ROOT / "checkpoints.txt").write_text(
        "Round 2 checkpoints (paths only; .pt files are excluded from the review ZIP)\n" +
        "\n".join(f"{m}: best={ROOT / m / 'best.pt'} | last={ROOT / m / 'last.pt'}" for m in MODELS) + "\n",
        encoding="utf-8")
    print(json.dumps({"metrics": len(round2_rows), "progress_rows": len(progress_rows)}, indent=2))


if __name__ == "__main__":
    main()
