"""Create the fixed-case visual benchmark for Round 2 checkpoints C and E.

No training occurs. Validation cases are selected before loading any model, using
only quantiles of real stroke/point counts. All panels use fixed canvas bounds.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sketchlab.data import load_raw, sha256_file
from sketchlab.evaluation import distances, geometry_metrics, prefix_count
from sketchlab.generation import load_model, sample
from sketchlab.geometry import load_geometry, postprocess_candidate, validate_candidate
from sketchlab.orchestration import json_ready


OUTPUT = Path("runs/visual_benchmark_round2")
BOUNDS = (-4.0, 508.0)
MODELS = "CE"
CHECKPOINTS = {m: Path(f"runs/tournament_round2/{m}/best.pt") for m in MODELS}
CONDITIONS = [("1stroke", "one"), ("2strokes", "two"), ("25pct", .25),
              ("50pct", .5), ("75pct", .75)]
N_LATENTS = 3
TEMPERATURE = .6
MAX_POINTS = 384
MAX_STROKES = 64
BASE_SEED = 43100


def choose_cases(validation, count=8):
    strokes = np.asarray([len(s["strokes"]) for s in validation], dtype=np.float64)
    points = np.asarray([sum(len(x) for x in s["strokes"]) for s in validation], dtype=np.float64)
    # Complexity is model-independent and balances both hierarchical dimensions.
    complexity = (np.argsort(np.argsort(strokes)) + np.argsort(np.argsort(points))) / 2
    order = np.argsort(complexity, kind="stable")
    quantiles = np.linspace(.05, .95, count)
    selected = []
    used = set()
    for q in quantiles:
        target = int(round(q * (len(order) - 1)))
        for delta in range(len(order)):
            options = [target + delta, target - delta]
            match = next((int(order[i]) for i in options if 0 <= i < len(order) and int(order[i]) not in used), None)
            if match is not None:
                selected.append((match, float(q)))
                used.add(match)
                break
    return selected


def prefix_hash(strokes):
    digest = hashlib.sha256()
    for stroke in strokes:
        array = np.asarray(stroke, dtype="<f4")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def metrics(strokes, prefix_strokes, info, stats, truth=None, jump_threshold=150.):
    suffix = [np.asarray(s, dtype=np.float64) for s in strokes[prefix_strokes:]]
    points = np.concatenate(suffix) if suffix else np.empty((0, 2))
    lo, hi = BOUNDS
    inside = ((points >= lo) & (points <= hi)).all(1) if len(points) else np.empty(0, dtype=bool)
    validation = validate_candidate(strokes, stats, prefix_count=prefix_strokes, termination=info)
    reason = validation.severe_errors[0] if validation.severe_errors else (
        validation.warnings[0] if validation.warnings else "ok")
    degenerate = sum(len(s) < 2 or np.ptp(s, axis=0).max() <= 1e-9 for s in suffix)
    result = {
        "generated_strokes": len(suffix),
        "generated_points": int(len(points)),
        "eos": bool(info.get("ended_by_eos")),
        "inside_point_fraction": float(inside.mean()) if len(inside) else 0.0,
        "all_points_inside": bool(inside.all()) if len(inside) else False,
        "outside_points": int((~inside).sum()) if len(inside) else 0,
        "valid": validation.valid,
        "reason": reason,
        "degenerate_strokes": degenerate,
        "termination": info.get("termination"),
        "corrections": [],
    }
    result.update({f"geometry_{k}": v for k, v in geometry_metrics(suffix, jump_threshold).items()})
    if truth is not None:
        result.update(distances(suffix, truth))
    return result


def plot_sketch(ax, strokes, prefix_strokes, title, annotation, *, continuation="#e76427"):
    lo, hi = BOUNDS
    outside = []
    for index, stroke in enumerate(strokes):
        a = np.asarray(stroke, dtype=np.float64)
        color = "#2378d4" if index < prefix_strokes else continuation
        if len(a) == 1:
            ax.scatter(a[:, 0], a[:, 1], s=8, color=color, zorder=2)
        else:
            ax.plot(a[:, 0], a[:, 1], color=color, lw=.9, solid_capstyle="round")
        mask = ((a < lo) | (a > hi)).any(1)
        if mask.any():
            outside.append(a[mask])
    if outside:
        points = np.concatenate(outside)
        # Boundary crosses disclose clipped geometry without pretending its true
        # location lies on the canvas; the caption gives the exact count.
        shown = np.clip(points, lo + 2, hi - 2)
        ax.scatter(shown[:, 0], shown[:, 1], marker="x", s=10, lw=.6,
                   color="#bd1f2d", zorder=5)
    ax.set(xlim=(lo, hi), ylim=(hi, lo), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel(annotation, fontsize=6, linespacing=1.2)
    for spine in ax.spines.values():
        spine.set_color("#c8c8c8")


def annotation(value):
    return (f"gen {value['generated_strokes']} strokes / {value['generated_points']} points | "
            f"EOS {'sí' if value['eos'] else 'no'}\n"
            f"canvas {100*value['inside_point_fraction']:.1f}% | fuera {value['outside_points']} | "
            f"{'VALID' if value['valid'] else 'INVALID'}: {value['reason']}")


def gt_annotation(sample, prefix_strokes):
    generated = sample["strokes"][prefix_strokes:]
    return (f"GT total {len(sample['strokes'])} strokes / {sum(len(s) for s in sample['strokes'])} points\n"
            f"prefix {prefix_strokes} | suffix real {len(generated)} strokes / {sum(len(s) for s in generated)} points")


def save_completion_page(condition_name, case_number, sample_data, prefix_strokes, outputs):
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.8), constrained_layout=True)
    for row, model_name in enumerate(MODELS):
        plot_sketch(axes[row, 0], sample_data["strokes"], prefix_strokes,
                    f"GT id {sample_data['id']} · prefix {condition_name}",
                    gt_annotation(sample_data, prefix_strokes), continuation="#777777")
        for latent_index in range(N_LATENTS):
            record = outputs[model_name][latent_index]
            plot_sketch(axes[row, latent_index + 1], record["strokes"], prefix_strokes,
                        f"{model_name} · z{latent_index+1}", annotation(record["metrics"]))
    fig.suptitle(f"Round 2 · caso {case_number:02d} · completion {condition_name} · canvas fijo [-4, 508]", fontsize=13)
    path = OUTPUT / "completion" / condition_name / f"overview_{condition_name}_page{case_number:02d}.png"
    fig.savefig(path, dpi=155)
    plt.close(fig)


def save_random_pages(records):
    for page in range(2):
        fig, axes = plt.subplots(4, 2, figsize=(6.6, 12.7), constrained_layout=True)
        for row in range(4):
            sample_index = page * 4 + row
            for col, model_name in enumerate(MODELS):
                record = records[model_name][sample_index]
                plot_sketch(axes[row, col], record["strokes"], 0,
                            f"{model_name} · seed {record['seed']}", annotation(record["metrics"]))
        fig.suptitle(f"Round 2 · random generation · página {page+1}/2 · canvas fijo [-4, 508]", fontsize=13)
        fig.savefig(OUTPUT / "random_generation" / f"random_generation_page{page+1:02d}.png", dpi=155)
        plt.close(fig)


def save_raw_vs_post(records, sample_data, prefix_strokes, name):
    fig, axes = plt.subplots(2, 3, figsize=(9.7, 6.8), constrained_layout=True)
    for row, model_name in enumerate(MODELS):
        record = records[model_name]
        plot_sketch(axes[row, 0], sample_data["strokes"], prefix_strokes,
                    f"GT id {sample_data['id']}", gt_annotation(sample_data, prefix_strokes), continuation="#777777")
        plot_sketch(axes[row, 1], record["raw"], prefix_strokes,
                    f"{model_name} · RAW", annotation(record["raw_metrics"]))
        post_metrics = dict(record["post_metrics"])
        correction_count = len(record["corrections"])
        post_label = annotation(post_metrics) + f"\ncorrecciones: {correction_count}"
        plot_sketch(axes[row, 2], record["post"], prefix_strokes,
                    f"{model_name} · POST", post_label)
    fig.suptitle(f"Round 2 · RAW vs POST · {name} · canvas fijo [-4, 508]", fontsize=13)
    fig.savefig(OUTPUT / "raw_vs_postprocessed" / f"raw_vs_postprocessed_{name}.png", dpi=155)
    plt.close(fig)


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated benchmark: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for condition_name, _ in CONDITIONS:
        (OUTPUT / "completion" / condition_name).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "random_generation").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "raw_vs_postprocessed").mkdir(parents=True, exist_ok=True)
    stats = load_geometry()
    validation = load_raw("data/raw/val.pkl")
    round1_manifest = json.loads(Path("runs/visual_benchmark_round1/selected_validation_cases.json").read_text(encoding="utf-8"))
    chosen = [(int(c["validation_index"]), float(c["complexity_quantile_target"]))
              for c in round1_manifest["cases"]]
    for source, (index, _) in zip(round1_manifest["cases"], chosen):
        if validation[index]["id"] != source["id"]:
            raise RuntimeError("Round 1 fixed validation case no longer resolves to the same ID")
    jump_threshold = json.loads(Path("runs/tournament_round2/manifest_C.json").read_text(encoding="utf-8"))["config"]["jump_threshold"]
    cases = []
    for rank, (index, quantile) in enumerate(chosen, start=1):
        sample_data = validation[index]
        condition_counts = {name: prefix_count(len(sample_data["strokes"]), fraction)
                            for name, fraction in CONDITIONS}
        cases.append({"case": rank, "validation_index": index, "id": sample_data["id"],
                      "complexity_quantile_target": quantile,
                      "strokes": len(sample_data["strokes"]),
                      "points": sum(len(s) for s in sample_data["strokes"]),
                      "prefix_counts": condition_counts,
                      "prefix_hashes": {name: prefix_hash(sample_data["strokes"][:count])
                                        for name, count in condition_counts.items()}})
    manifest = {
        "selection": "fixed quantiles of model-independent average rank of VAL stroke and point counts",
        "validation_sha256": sha256_file("data/raw/val.pkl"),
        "canvas_bounds": BOUNDS, "temperature": TEMPERATURE,
        "max_generated_points": MAX_POINTS, "max_generated_strokes": MAX_STROKES,
        "decoder_rng": "fixed across z for each case and shared across models",
        "checkpoints": {m: {"path": str(p), "sha256": sha256_file(p)} for m, p in CHECKPOINTS.items()},
        "cases": cases,
    }
    (OUTPUT / "selected_validation_cases.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")

    models = {name: load_model(path) for name, path in CHECKPOINTS.items()}
    completion_records = {name: [] for name in MODELS}
    raw_post_representative = None
    for condition_index, (condition_name, fraction) in enumerate(CONDITIONS):
        for case_index, (validation_index, _) in enumerate(chosen):
            sample_data = validation[validation_index]
            count = prefix_count(len(sample_data["strokes"]), fraction)
            prefix = [stroke.copy() for stroke in sample_data["strokes"][:count]]
            outputs = {name: [] for name in MODELS}
            decoder_seed = BASE_SEED + condition_index * 10000 + case_index * 100
            for model_name, model in models.items():
                for latent_index in range(N_LATENTS):
                    latent_seed = decoder_seed + latent_index
                    strokes, info = sample(model, prefix=prefix, seed=latent_seed,
                                           decoder_seed=decoder_seed + 900000,
                                           temperature=TEMPERATURE, max_points=MAX_POINTS,
                                           max_strokes=MAX_STROKES, return_info=True)
                    measured = metrics(strokes, count, info, stats,
                                       truth=sample_data["strokes"][count:],
                                       jump_threshold=jump_threshold)
                    record = {"model": model_name, "case": case_index + 1,
                              "validation_id": sample_data["id"], "condition": condition_name,
                              "latent": latent_index + 1, "seed": latent_seed,
                              "prefix_count": count, "info": info, "metrics": measured,
                              "strokes": strokes}
                    outputs[model_name].append(record)
                    completion_records[model_name].append(record)
            for model_name in MODELS:
                suffixes = [r["strokes"][count:] for r in outputs[model_name]]
                pairwise = [distances(suffixes[i], suffixes[j])["chamfer"]
                            for i in range(N_LATENTS) for j in range(i + 1, N_LATENTS)]
                diversity = float(np.mean([v for v in pairwise if v is not None])) if any(v is not None for v in pairwise) else None
                for record in outputs[model_name]:
                    record["metrics"]["latent_diversity_chamfer"] = diversity
            save_completion_page(condition_name, case_index + 1, sample_data, count, outputs)
            if condition_name == "25pct" and case_index == len(chosen) // 2:
                raw_post_representative = (sample_data, count, {m: outputs[m][0] for m in MODELS})

    random_records = {name: [] for name in MODELS}
    for sample_index in range(8):
        seed = BASE_SEED + 80000 + sample_index
        for model_name, model in models.items():
            strokes, info = sample(model, prefix=[], seed=seed, decoder_seed=seed + 900000,
                                   temperature=TEMPERATURE, max_points=MAX_POINTS,
                                   max_strokes=MAX_STROKES, return_info=True)
            random_records[model_name].append({"model": model_name, "seed": seed, "info": info,
                                               "metrics": metrics(strokes, 0, info, stats,
                                                                  jump_threshold=jump_threshold),
                                               "strokes": strokes})
    save_random_pages(random_records)

    if raw_post_representative is not None:
        sample_data, count, representatives = raw_post_representative
        processed = {}
        for model_name, record in representatives.items():
            result = postprocess_candidate(record["strokes"], stats, prefix_count=count,
                                           termination=record["info"])
            post = result["postprocessed_output"]
            processed[model_name] = {
                "raw": record["strokes"], "post": post,
                "raw_metrics": record["metrics"],
                "post_metrics": metrics(post, count, record["info"], stats,
                                        truth=sample_data["strokes"][count:],
                                        jump_threshold=jump_threshold),
                "corrections": result["validation"].corrections,
            }
        save_raw_vs_post(processed, sample_data, count, "completion")

    # Random RAW/POST uses the first fixed seed for every model.
    random_processed = {}
    empty_gt = {"id": "random", "strokes": []}
    for model_name in MODELS:
        record = random_records[model_name][0]
        result = postprocess_candidate(record["strokes"], stats, prefix_count=0,
                                       termination=record["info"])
        post = result["postprocessed_output"]
        random_processed[model_name] = {
            "raw": record["strokes"], "post": post, "raw_metrics": record["metrics"],
            "post_metrics": metrics(post, 0, record["info"], stats,
                                    jump_threshold=jump_threshold),
            "corrections": result["validation"].corrections,
        }
    save_raw_vs_post(random_processed, empty_gt, 0, "random")

    real_strokes = [len(s["strokes"]) for s in validation]
    real_points = [sum(len(x) for x in s["strokes"]) for s in validation]
    summary_rows = []
    compact_metrics = {"completion": {}, "random": {}}
    for model_name in MODELS:
        completion = completion_records[model_name]
        random = random_records[model_name]
        all_records = completion + random
        generated_points = [r["metrics"]["generated_points"] for r in all_records]
        generated_strokes = [r["metrics"]["generated_strokes"] for r in all_records]
        degenerate = sum(r["metrics"]["degenerate_strokes"] for r in all_records)
        stroke_total = sum(generated_strokes)
        row = {
            "model": model_name,
            "round": 2,
            "median_generated_strokes": float(np.median(generated_strokes)),
            "median_generated_points": float(np.median(generated_points)),
            "p25_generated_points": float(np.quantile(generated_points, .25)),
            "p75_generated_points": float(np.quantile(generated_points, .75)),
            "sequence_cap_rate": float(np.mean([r["metrics"]["termination"] == "max_points" for r in all_records])),
            "eos_rate": float(np.mean([r["metrics"]["eos"] for r in all_records])),
            "completion_valid_rate": float(np.mean([r["metrics"]["valid"] for r in completion])),
            "mean_points_inside_canvas": float(np.mean([r["metrics"]["inside_point_fraction"] for r in all_records])),
            "fully_canvas_valid_rate": float(np.mean([r["metrics"]["all_points_inside"] for r in all_records])),
            "degenerate_rate": degenerate / stroke_total if stroke_total else 0.0,
            "median_random_strokes": float(np.median([r["metrics"]["generated_strokes"] for r in random])),
            "median_random_points": float(np.median([r["metrics"]["generated_points"] for r in random])),
            "median_real_strokes": float(np.median(real_strokes)),
            "median_real_points": float(np.median(real_points)),
            "mean_chamfer": float(np.mean([r["metrics"]["chamfer"] for r in completion if r["metrics"].get("chamfer") is not None])),
            "mean_hausdorff": float(np.mean([r["metrics"]["hausdorff"] for r in completion if r["metrics"].get("hausdorff") is not None])),
            "mean_intra_stroke_jump_rate": float(np.mean([r["metrics"]["geometry_intra_jump_fraction"] for r in completion if r["metrics"].get("geometry_intra_jump_fraction") is not None])),
            "mean_latent_diversity_chamfer": float(np.mean([r["metrics"]["latent_diversity_chamfer"] for r in completion if r["metrics"].get("latent_diversity_chamfer") is not None])),
        }
        summary_rows.append(row)
        compact_metrics["completion"][model_name] = [
            {k: json_ready(v) for k, v in r.items() if k != "strokes"} for r in completion]
        compact_metrics["random"][model_name] = [
            {k: json_ready(v) for k, v in r.items() if k != "strokes"} for r in random]
    with (OUTPUT / "generation_length_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    (OUTPUT / "benchmark_metrics.json").write_text(
        json.dumps(json_ready(compact_metrics), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(OUTPUT),
                      "cases": [c["id"] for c in cases],
                      "completion_samples_per_model": len(completion_records["C"]),
                      "random_samples_per_model": len(random_records["C"]),
                      "pages": len(list(OUTPUT.rglob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
