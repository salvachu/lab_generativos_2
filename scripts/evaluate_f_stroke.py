"""Evaluate the F representation stage on diverse raw held-out strokes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sketchlab.checkpointing import load_checkpoint
from sketchlab.data import load_raw
from sketchlab.evaluation import distances


def _features(stroke):
    points = np.asarray(stroke, dtype=np.float64)
    segments = np.diff(points, axis=0)
    arc = float(np.linalg.norm(segments, axis=1).sum())
    chord = float(np.linalg.norm(points[-1] - points[0]))
    bbox = float(np.linalg.norm(np.ptp(points, axis=0)))
    return {"arc_length": arc, "straightness": chord / max(arc, 1e-9), "bbox_diagonal": bbox}


def select_diverse_strokes(validation, count):
    candidates = []
    for sketch_index, sketch in enumerate(validation):
        for stroke_index, stroke in enumerate(sketch["strokes"]):
            if len(stroke) >= 2:
                candidates.append({"stroke": stroke, "sketch_index": sketch_index,
                                   "stroke_index": stroke_index, **_features(stroke)})
    if not candidates:
        raise ValueError("validation contains no nonempty strokes")
    groups = [
        ("short/small", sorted(candidates, key=lambda x: (x["bbox_diagonal"], x["arc_length"]))),
        ("long", sorted(candidates, key=lambda x: x["arc_length"], reverse=True)),
        ("curved", sorted(candidates, key=lambda x: (x["straightness"], -x["arc_length"]))),
        ("straight", sorted(candidates, key=lambda x: (x["straightness"], x["arc_length"]), reverse=True)),
    ]
    selected, used = [], set()
    target_per_group = max(1, int(np.ceil(count / len(groups))))
    for label, ranked in groups:
        added = 0
        for item in ranked:
            key = (item["sketch_index"], item["stroke_index"])
            if key in used:
                continue
            selected.append({**item, "category": label})
            used.add(key); added += 1
            if added >= target_per_group or len(selected) >= count:
                break
    if len(selected) < count:
        for item in candidates:
            key = (item["sketch_index"], item["stroke_index"])
            if key not in used:
                selected.append({**item, "category": "additional"}); used.add(key)
            if len(selected) >= count:
                break
    return selected[:count]


def render_visual_review(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs_per_row = 4
    nrows = int(np.ceil(len(rows) / pairs_per_row))
    figure, axes = plt.subplots(nrows, pairs_per_row * 2,
                                figsize=(3.0 * pairs_per_row * 2, 3.15 * nrows), squeeze=False)
    for index, ax in enumerate(axes.ravel()):
        pair = index // 2
        if pair >= len(rows):
            ax.set_visible(False); continue
        row = rows[pair]
        points = np.asarray(row["raw"] if index % 2 == 0 else row["prediction"])
        ax.plot(points[:, 0], points[:, 1], color="#1f77b4" if index % 2 == 0 else "#ff7f0e", lw=1.5)
        ax.scatter(points[0, 0], points[0, 1], s=14, color="#2ca02c", zorder=3)
        ax.set(xlim=(0, 504), ylim=(504, 0), aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        side = "GT" if index % 2 == 0 else "Reconstruction"
        ax.set_title(f"{pair + 1:02d} {row['category']} | {side}", fontsize=7)
    figure.suptitle("Stroke AE validation review — green dot is the shared anchor", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .985), pad=.55)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


@torch.inference_mode()
def evaluate(model, selected, output, checkpoint, checkpoint_payload):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    strokes = [item["stroke"] for item in selected]
    model.eval(); view = model.view([strokes]); embeddings = model.encode_view(view)[0]
    prediction_normalized = model.stroke_decoder(embeddings, view["t"]).cpu().numpy()
    target_normalized = view["relative"][0].cpu().numpy()
    anchors = view["anchors"][0].cpu().numpy() * model.scale
    rows = []
    for i, (item, relative_norm, target_norm, anchor) in enumerate(
            zip(selected, prediction_normalized, target_normalized, anchors)):
        raw = np.asarray(item["stroke"], dtype=np.float64)
        prediction = relative_norm * model.scale + anchor
        delta = np.diff(relative_norm, n=2, axis=0) - np.diff(target_norm, n=2, axis=0)
        rows.append({
            "stroke": i, "validation_index": item["sketch_index"], "stroke_index": item["stroke_index"],
            "category": item["category"], "raw_points": len(raw), "arc_length": item["arc_length"],
            "straightness": item["straightness"], "bbox_diagonal": item["bbox_diagonal"],
            "rmse_normalized": float(np.sqrt(np.mean((relative_norm - target_norm) ** 2))),
            "rmse_pixels": float(np.sqrt(np.mean(((relative_norm - target_norm) * model.scale) ** 2))),
            "endpoint_error_pixels": float(np.linalg.norm(prediction[-1] - raw[-1])),
            "second_difference_rmse_normalized": float(np.sqrt(np.mean(delta ** 2))) if len(delta) else 0.0,
            **distances([raw], [prediction]), "raw": raw.tolist(), "prediction": prediction.tolist(),
        })
    numeric = ["rmse_normalized", "rmse_pixels", "endpoint_error_pixels",
               "second_difference_rmse_normalized", "chamfer", "hausdorff", "endpoint_chamfer"]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in numeric}
    summary["validation_reconstruction"] = checkpoint_payload.get("validation", {}).get("stroke_reconstruction")
    payload = {"stage": "F_STROKE_AE", "count": len(rows),
               "coordinate_units": {"rmse_normalized": "canvas/256", "other_distances": "canvas pixels"},
               "metrics": summary, "strokes": rows}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    review_path = output.parent / "VISUAL_REVIEW.png"
    render_visual_review(rows, review_path)

    train_summary = json.loads((Path(checkpoint).parent / "summary.json").read_text(encoding="utf-8"))
    finite = all(np.isfinite(value) for row in rows for value in row.values() if isinstance(value, (int, float)))
    stage_summary = {
        "updates": train_summary["steps"], "best_step": train_summary["best_step"],
        "best_checkpoint": str(Path(checkpoint).resolve()), **summary,
        "visual_review": str(review_path.resolve()),
        "training_time_seconds": train_summary["training_time_seconds"],
        "total_time_seconds": train_summary["total_time_seconds"],
        "peak_vram_mib": train_summary["peak_vram_mib"], "device": train_summary["device"],
        "status": train_summary["status"], "finite_metrics": bool(finite),
        "nan_or_instability_observed": not finite or train_summary["status"] not in {"complete", "completed"},
    }
    (output.parent / "STAGE1_SUMMARY.json").write_text(json.dumps(stage_summary, indent=2), encoding="utf-8")
    return stage_summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True); parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    if args.count < 1: parser.error("count must be positive")
    torch.set_num_threads(1); model, payload = load_checkpoint(args.checkpoint)
    if model.model_name != "F": parser.error("F checkpoint required")
    selected = select_diverse_strokes(load_raw("data/raw/val.pkl", validate=False), args.count)
    print(json.dumps(evaluate(model, selected, args.output, args.checkpoint, payload), indent=2))


if __name__ == "__main__":
    main()
