"""Evaluate the short Model-E state-class weighting experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.diagnose_round2 import hierarchical_teacher
from sketchlab.data import load_raw
from sketchlab.evaluation import distances, geometry_metrics, prefix_count
from sketchlab.generation import load_model, sample
from sketchlab.geometry import load_geometry, validate_candidate
from sketchlab.orchestration import json_ready
from sketchlab.training import training_threshold

OUT = Path("runs/eos_weighting_experiment")
CHECKPOINTS = {
    "control": OUT / "training/control/E/last.pt",
    "weighted_mild": OUT / "training/weighted_mild/E/last.pt",
    "weighted_sqrt": OUT / "training/weighted_sqrt/E/last.pt",
}
WEIGHTS = {
    "control": {"stroke_end": 1.0, "eos": 1.0},
    "weighted_mild": {"stroke_end": 1.75, "eos": 3.0},
    "weighted_sqrt": {"stroke_end": 2.417, "eos": 4.801},
}
CONDITIONS = [("1stroke", "one"), ("2strokes", "two"), ("25pct", .25),
              ("50pct", .5), ("75pct", .75)]
TEMPERATURES = (.9, 1.0)
SEED = 73100
MAX_POINTS = 384
MAX_STROKES = 64


def mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def classification(rows, positive, event):
    rows = [r for r in rows if r["event"] in event]
    y = np.asarray([r["true_state"] == positive for r in rows])
    p = np.asarray([r["selected_state"] == positive for r in rows])
    tp, fp, fn, tn = int((y&p).sum()), int((~y&p).sum()), int((y&~p).sum()), int((~y&~p).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


def teacher_metrics(model, cases):
    rows = []
    for _, fraction in CONDITIONS:
        for case in cases:
            rows.extend(hierarchical_teacher(model, case, prefix_count(len(case["strokes"]), fraction)))
    draw = classification(rows, "draw", {"point"})
    stroke = classification(rows, "stroke_end", {"point"})
    eos = classification(rows, "eos", {"anchor", "sketch_eos"})
    final = [r["eos_probability"] for r in rows if r["event"] == "sketch_eos"]
    nonfinal = [r["eos_probability"] for r in rows if r["event"] == "anchor"]
    return {"events": len(rows), "draw_recall": draw["recall"],
            "stroke_end": stroke, "eos": eos,
            "eos_probability_final": mean(final),
            "eos_probability_nonfinal": mean(nonfinal),
            "eos_probability_gap": mean(final) - mean(nonfinal)}


def generation_metrics(strokes, count, info, stats, truth):
    suffix = strokes[count:]
    geom = geometry_metrics(suffix, training_threshold(None))
    validation = validate_candidate(strokes, stats, prefix_count=count, termination=info)
    point_count = sum(len(s) for s in suffix)
    gt_points = sum(len(s) for s in truth)
    flat = np.concatenate(suffix) if suffix else np.empty((0, 2))
    flat_truth = np.concatenate(truth) if truth else np.empty((0, 2))
    aligned = min(len(flat), len(flat_truth))
    drift = float(np.linalg.norm(flat[:aligned]-flat_truth[:aligned], axis=1).mean()) if aligned else None
    d = distances(suffix, truth)
    return {"eos": bool(info["ended_by_eos"]), "cap": bool(info["capped"]),
            "termination": info["termination"], "points": point_count, "strokes": len(suffix),
            "empty": point_count == 0, "premature": point_count < max(5, .1*gt_points),
            "canvas": geom["canvas_validity"] if geom["canvas_validity"] is not None else 0.0,
            "completion_valid": bool(validation.valid), "drift": drift,
            "chamfer": d["chamfer"], "hausdorff": d["hausdorff"],
            "intra_jump_fraction": geom["intra_jump_fraction"]}


def generate(models, cases, stats):
    records = []
    for temp in TEMPERATURES:
        for variant, model in models.items():
            for ci, (condition, fraction) in enumerate(CONDITIONS):
                for vi, case in enumerate(cases):
                    count = prefix_count(len(case["strokes"]), fraction)
                    prefix, truth = case["strokes"][:count], case["strokes"][count:]
                    group = []
                    for k in range(2):
                        seed = SEED + ci*10000 + vi*100 + k
                        strokes, info = sample(model, prefix=prefix, seed=seed,
                            decoder_seed=seed+900000, temperature=temp,
                            max_points=MAX_POINTS, max_strokes=MAX_STROKES, return_info=True)
                        metric = generation_metrics(strokes, count, info, stats, truth)
                        records.append({"variant": variant, "temperature": temp,
                            "condition": condition, "case": vi+1, "validation_id": case["id"],
                            "sample": k, "prefix_count": count, "metrics": metric, "strokes": strokes})
                        group.append(strokes[count:])
                    diversity = distances(group[0], group[1])["chamfer"]
                    for rec in records[-2:]: rec["metrics"]["diversity_chamfer"] = diversity
            for k in range(8):
                seed = SEED + 90000 + k
                strokes, info = sample(model, prefix=[], seed=seed, decoder_seed=seed+900000,
                    temperature=temp, max_points=MAX_POINTS, max_strokes=MAX_STROKES, return_info=True)
                geom = geometry_metrics(strokes, training_threshold(None))
                records.append({"variant": variant, "temperature": temp, "condition": "random",
                    "case": k+1, "validation_id": None, "sample": 0, "prefix_count": 0,
                    "metrics": {"eos": bool(info["ended_by_eos"]), "cap": bool(info["capped"]),
                        "termination": info["termination"], "points": sum(len(s) for s in strokes),
                        "strokes": len(strokes), "empty": not strokes, "premature": sum(len(s) for s in strokes)<5,
                        "canvas": geom["canvas_validity"] or 0.0,
                        "intra_jump_fraction": geom["intra_jump_fraction"]}, "strokes": strokes})
    return records


def summarize(records):
    rows = []
    for variant in CHECKPOINTS:
        for temp in TEMPERATURES:
            selected = [r for r in records if r["variant"] == variant and r["temperature"] == temp and r["condition"] != "random"]
            m = [r["metrics"] for r in selected]
            rows.append({"variant": variant, "temperature": temp, "samples": len(m),
                "eos_rate": mean(x["eos"] for x in m), "cap_rate": mean(x["cap"] for x in m),
                "median_generated_points": float(np.median([x["points"] for x in m])),
                "median_generated_strokes": float(np.median([x["strokes"] for x in m])),
                "canvas_validity": mean(x["canvas"] for x in m),
                "completion_valid_rate": mean(x["completion_valid"] for x in m),
                "premature_rate": mean(x["premature"] for x in m), "empty_rate": mean(x["empty"] for x in m),
                "mean_drift": mean(x["drift"] for x in m), "mean_chamfer": mean(x["chamfer"] for x in m),
                "mean_hausdorff": mean(x["hausdorff"] for x in m),
                "intra_jump_fraction": mean(x["intra_jump_fraction"] for x in m),
                "diversity_chamfer": mean(x.get("diversity_chamfer") for x in m)})
    return rows


def render_grids(records, cases):
    grid_dir = OUT / "visual_grids"; grid_dir.mkdir(exist_ok=True)
    for temp in TEMPERATURES:
        for condition, fraction in [("1stroke", "one"), ("25pct", .25), ("50pct", .5)]:
            case = cases[0]; count = prefix_count(len(case["strokes"]), fraction)
            panels = [("GT", case["strokes"], count)]
            for variant in CHECKPOINTS:
                rec = next(r for r in records if r["variant"] == variant and r["temperature"] == temp
                           and r["condition"] == condition and r["case"] == 1 and r["sample"] == 0)
                m = rec["metrics"]
                panels.append((f"{variant}\n{m['points']} pts · {m['termination']} · valid {int(m['completion_valid'])}",
                               rec["strokes"], count))
            plot_panels(panels, grid_dir/f"{condition}_T{str(temp).replace('.', 'p')}.png")
        panels = []
        for variant in CHECKPOINTS:
            rec = next(r for r in records if r["variant"] == variant and r["temperature"] == temp
                       and r["condition"] == "random" and r["case"] == 1)
            m = rec["metrics"]
            panels.append((f"{variant}\n{m['points']} pts · {m['termination']}", rec["strokes"], 0))
        plot_panels(panels, grid_dir/f"random_T{str(temp).replace('.', 'p')}.png")


def plot_panels(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2*len(panels), 3.5), constrained_layout=True)
    for ax, (title, strokes, count) in zip(np.atleast_1d(axes), panels):
        for i, stroke in enumerate(strokes):
            a = np.asarray(stroke); ax.plot(a[:,0], a[:,1], color="#1976d2" if i<count else "#e66924", lw=1)
        ax.set(xlim=(-4,508), ylim=(508,-4), aspect="equal", title=title); ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)


def main():
    torch.set_num_threads(1)
    val = load_raw("data/raw/val.pkl")
    manifest = json.loads(Path("runs/visual_benchmark_round1/selected_validation_cases.json").read_text(encoding="utf-8"))
    cases = [val[int(item["validation_index"])] for item in manifest["cases"]]
    models = {name: load_model(path) for name, path in CHECKPOINTS.items()}
    teacher = {name: teacher_metrics(model, cases) for name, model in models.items()}
    records = generate(models, cases, load_geometry())
    free = summarize(records)
    training = {name: json.loads((OUT/f"training/{name}/E/summary.json").read_text(encoding="utf-8")) for name in CHECKPOINTS}
    scoreboard = []
    for row in free:
        tf = teacher[row["variant"]]; tr = training[row["variant"]]
        scoreboard.append({**row, "draw_recall": tf["draw_recall"],
            "stroke_end_precision": tf["stroke_end"]["precision"], "stroke_end_recall": tf["stroke_end"]["recall"],
            "eos_precision": tf["eos"]["precision"], "eos_recall": tf["eos"]["recall"],
            "eos_probability_final": tf["eos_probability_final"],
            "eos_probability_nonfinal": tf["eos_probability_nonfinal"], "eos_probability_gap": tf["eos_probability_gap"],
            "validation_coordinate_nll": tr["validation"]["coordinate_nll"],
            "validation_global_kl": tr["validation"]["global_kl"],
            "validation_stroke_kl": tr["validation"]["stroke_kl"],
            "training_time_seconds": tr["training_time_seconds"], "peak_vram_mib": tr["peak_vram_mib"],
            "checkpoint": str(CHECKPOINTS[row["variant"]])})
    with (OUT/"scoreboard.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(scoreboard[0])); w.writeheader(); w.writerows(scoreboard)
    (OUT/"scoreboard.json").write_text(json.dumps(json_ready(scoreboard), indent=2), encoding="utf-8")
    (OUT/"teacher_forced_metrics.json").write_text(json.dumps(json_ready(teacher), indent=2), encoding="utf-8")
    (OUT/"free_running_metrics.json").write_text(json.dumps(json_ready({"summary": free,
        "protocol": {"case_ids": [c["id"] for c in cases], "conditions": [c[0] for c in CONDITIONS],
        "samples_per_case": 2, "random_samples": 8, "temperatures": TEMPERATURES,
        "max_points": MAX_POINTS, "max_strokes": MAX_STROKES, "seed": SEED},
        "records": [{k:v for k,v in r.items() if k != "strokes"} for r in records]}), indent=2), encoding="utf-8")
    counts = {"point_draw": 1102797, "stroke_end": 188702, "stroke_continue": 188702, "eos": 8187}
    weight_payload = {"train_counts": counts,
        "formula": "minority weight = sqrt(majority_count/minority_count), majority weight = 1",
        "computed_sqrt": {"stroke_end": float(np.sqrt(counts["point_draw"]/counts["stroke_end"])),
                          "eos": float(np.sqrt(counts["stroke_continue"]/counts["eos"]))},
        "variants": WEIGHTS, "scope": "training cross-entropies only"}
    (OUT/"class_weights.json").write_text(json.dumps(weight_payload, indent=2), encoding="utf-8")
    render_grids(records, cases)
    print(json.dumps({"status":"completed", "scoreboard_rows":len(scoreboard), "grids":len(list((OUT/'visual_grids').glob('*.png')))}, indent=2))


if __name__ == "__main__": main()
