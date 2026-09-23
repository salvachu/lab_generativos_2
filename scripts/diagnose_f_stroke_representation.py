"""Separate Model F stroke-view resampling error from Stroke-AE error."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from sketchlab.checkpointing import load_checkpoint
from sketchlab.data import load_raw
from sketchlab.evaluation import distances
from sketchlab.stroke_view import stroke_view


def resample(stroke, samples, scale=256.0):
    view = stroke_view(stroke, samples=samples, scale=scale)
    return view["relative"] * scale + view["anchor"] * scale


def dense(stroke, samples=128):
    return resample(stroke, samples=samples, scale=256.0)


def features(stroke):
    points = np.asarray(stroke, dtype=np.float64)
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    arc = float(lengths.sum())
    chord = float(np.linalg.norm(points[-1] - points[0]))
    bbox = float(np.linalg.norm(np.ptp(points, axis=0)))
    valid = lengths > 1e-6
    directions = np.unwrap(np.arctan2(delta[valid, 1], delta[valid, 0]))
    turning = float(np.abs(np.diff(directions)).sum()) if len(directions) > 1 else 0.0
    straightness = chord / max(arc, 1e-9)
    closure = chord / max(bbox, 1e-9)
    detail = turning * np.log1p(len(points)) / max(bbox, 1.0)
    return {"arc_length": arc, "bbox_diagonal": bbox, "straightness": straightness,
            "closure": closure, "turning": turning, "detail_score": detail}


def select_examples(validation, count=32):
    candidates = []
    for validation_index, sketch in enumerate(validation):
        for stroke_index, stroke in enumerate(sketch["strokes"]):
            if len(stroke) >= 3:
                candidates.append({"stroke": stroke, "validation_index": validation_index,
                                   "stroke_index": stroke_index, **features(stroke)})
    rankings = [
        ("short", lambda x: (x["bbox_diagonal"], x["arc_length"]), False),
        ("straight", lambda x: (x["straightness"], x["arc_length"]), True),
        ("curved", lambda x: (x["turning"], x["bbox_diagonal"]), True),
        ("loop", lambda x: (x["closure"], -x["turning"]), False),
        ("long", lambda x: x["arc_length"], True),
        ("small-detail", lambda x: x["detail_score"], True),
    ]
    selected, used = [], set()
    base, remainder = divmod(count, len(rankings))
    for group_index, (label, key, reverse) in enumerate(rankings):
        needed = base + (group_index < remainder)
        added = 0
        for item in sorted(candidates, key=key, reverse=reverse):
            identity = (item["validation_index"], item["stroke_index"])
            if identity in used:
                continue
            selected.append({**item, "category": label})
            used.add(identity); added += 1
            if added == needed:
                break
    return selected


def curvature_error(first, second, scale=256.0):
    a, b = dense(first), dense(second)
    return float(np.sqrt(np.mean((np.diff(a, n=2, axis=0) - np.diff(b, n=2, axis=0)) ** 2)) / scale)


def pair_metrics(first, second):
    a, b = dense(first), dense(second)
    return distances([a], [b])


def render_comparison(rows, path, modes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples_per_row = 2
    columns = examples_per_row * len(modes)
    nrows = int(np.ceil(len(rows) / examples_per_row))
    figure, axes = plt.subplots(nrows, columns, figsize=(3.0 * columns, 3.1 * nrows), squeeze=False)
    colors = {"RAW": "#1f77b4", "RESAMPLED-16": "#9467bd", "AE": "#ff7f0e",
              "16 points": "#9467bd", "24 points": "#2ca02c", "32 points": "#d62728"}
    for slot, ax in enumerate(axes.ravel()):
        example = slot // len(modes)
        mode_index = slot % len(modes)
        if example >= len(rows):
            ax.set_visible(False); continue
        row = rows[example]
        label, field = modes[mode_index]
        points = np.asarray(row[field])
        ax.plot(points[:, 0], points[:, 1], color=colors[label], lw=1.45)
        ax.scatter(points[0, 0], points[0, 1], s=14, color="#111111", zorder=3)
        ax.set(xlim=(0, 504), ylim=(504, 0), aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{example + 1:02d} {row['category']} | {label}", fontsize=7)
    figure.tight_layout(pad=.55)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


@torch.inference_mode()
def diagnose(checkpoint, output, count):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model, payload = load_checkpoint(checkpoint); model.eval()
    if model.model_name != "F":
        raise ValueError("F checkpoint required")
    selected = select_examples(load_raw("data/raw/val.pkl", validate=False), count)
    strokes = [item["stroke"] for item in selected]
    view = model.view([strokes])
    embeddings = model.encode_view(view)[0]
    recon_relative = model.stroke_decoder(embeddings, view["t"]).cpu().numpy() * model.scale
    anchors = view["anchors"][0].cpu().numpy() * model.scale
    rows = []
    for index, (item, relative, anchor) in enumerate(zip(selected, recon_relative, anchors)):
        raw = np.asarray(item["stroke"], dtype=np.float64)
        sampled16 = resample(raw, 16); sampled24 = resample(raw, 24); sampled32 = resample(raw, 32)
        reconstruction = relative + anchor
        representation = pair_metrics(raw, sampled16)
        representation24 = pair_metrics(raw, sampled24)
        representation32 = pair_metrics(raw, sampled32)
        model_error = pair_metrics(sampled16, reconstruction)
        total = pair_metrics(raw, reconstruction)
        rows.append({
            "stroke": index, "validation_index": item["validation_index"],
            "stroke_index": item["stroke_index"], "category": item["category"],
            "raw_points": len(raw), "arc_length": item["arc_length"],
            "representation_chamfer": representation["chamfer"],
            "representation_endpoint_error": float(np.linalg.norm(raw[-1] - sampled16[-1])),
            "representation_curvature_error": curvature_error(raw, sampled16),
            "representation_chamfer_24": representation24["chamfer"],
            "representation_curvature_error_24": curvature_error(raw, sampled24),
            "representation_chamfer_32": representation32["chamfer"],
            "representation_curvature_error_32": curvature_error(raw, sampled32),
            "model_rmse_normalized": float(np.sqrt(np.mean(((sampled16 - reconstruction) / model.scale) ** 2))),
            "model_rmse_pixels": float(np.sqrt(np.mean((sampled16 - reconstruction) ** 2))),
            "model_chamfer": model_error["chamfer"],
            "model_endpoint_error": float(np.linalg.norm(sampled16[-1] - reconstruction[-1])),
            "raw_vs_ae_chamfer": total["chamfer"],
            "raw_vs_ae_endpoint_error": float(np.linalg.norm(raw[-1] - reconstruction[-1])),
            "raw": raw.tolist(), "sampled16": sampled16.tolist(), "sampled24": sampled24.tolist(),
            "sampled32": sampled32.tolist(), "reconstruction": reconstruction.tolist(),
        })
    metric_names = [key for key in rows[0] if key.startswith(("representation_", "model_", "raw_vs_"))]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in metric_names}
    summary["count"] = len(rows)
    summary["checkpoint"] = str(Path(checkpoint).resolve())
    summary["validation_reconstruction"] = payload.get("validation", {}).get("stroke_reconstruction")
    summary["representation_share_of_separated_chamfer"] = (
        summary["representation_chamfer"] /
        (summary["representation_chamfer"] + summary["model_chamfer"]))
    render_comparison(rows, output / "RAW_VS_RESAMPLED_VS_AE.png",
                      [("RAW", "raw"), ("RESAMPLED-16", "sampled16"), ("AE", "reconstruction")])
    render_comparison(rows, output / "RESAMPLING_16_24_32.png",
                      [("16 points", "sampled16"), ("24 points", "sampled24"), ("32 points", "sampled32")])
    serializable = [{key: value for key, value in row.items()
                     if key not in {"raw", "sampled16", "sampled24", "sampled32", "reconstruction"}}
                    for row in rows]
    with (output / "stroke_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=serializable[0].keys()); writer.writeheader(); writer.writerows(serializable)
    (output / "diagnostic.json").write_text(json.dumps({"summary": summary, "strokes": serializable}, indent=2), encoding="utf-8")
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("checkpoint")
    parser.add_argument("--output", default="runs/F_stroke_diagnostic")
    parser.add_argument("--count", type=int, default=32); args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(diagnose(args.checkpoint, args.output, args.count), indent=2))


if __name__ == "__main__":
    main()
