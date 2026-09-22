"""Consolidate persisted Round 1 results; performs no training or dataset reads."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from sketchlab.diagnostics import latent_sensitivity
from sketchlab.generation import load_model
from sketchlab.geometry import load_geometry, validate_candidate
from sketchlab.orchestration import json_ready


ROOT = Path("runs/tournament_round1")
VISUAL = {
    "A": "poor: long out-of-canvas trajectories; many empty/capped random samples",
    "B": "poor: fragmented out-of-canvas trajectories despite preserved KL",
    "C": "poor: conditional diversity but large, incoherent continuations outside canvas",
    "D": "mixed: centered anchors and plausible scale; dense scribbles and some empty/capped outputs",
    "E": "best of Round 1: most spatially contained; still scribbly and not yet creature-like",
}
RANK = {"E": 1, "D": 2, "C": 3, "B": 4, "A": 5}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_history(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    scalars = [float(v) for row in rows for v in row.values() if isinstance(v, (int, float))]
    return rows, all(math.isfinite(v) for v in scalars)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    stats = load_geometry()
    rows = []
    details = {}
    for model_name in "ABCDE":
        run_dir = Path(f"runs/comparison_{model_name}/{model_name}")
        eval_dir = ROOT / model_name
        summary = read(run_dir / "summary.json")
        validation = read(eval_dir / "validation_likelihood.json")
        generation = read(eval_dir / "generation_metrics.json")
        generated = read(eval_dir / "generated_samples.json")
        history, stable = finite_history(run_dir / "history.jsonl")
        validations = []
        exact_groups = {}
        for candidate in generated:
            strokes = [np.asarray(stroke, dtype=np.float64) for stroke in candidate["strokes"]]
            prefix_count = int(candidate["prefix_count"])
            result = validate_candidate(strokes, stats, prefix_count=prefix_count,
                                        termination=candidate["info"])
            validations.append(result.to_dict())
            key = (candidate["id"], candidate["condition"])
            prefix = [stroke.tolist() for stroke in strokes[:prefix_count]]
            exact_groups.setdefault(key, []).append(prefix)
        prefix_exact = all(all(value == values[0] for value in values)
                           for values in exact_groups.values())
        first = generated[0]
        first_strokes = [np.asarray(stroke, dtype=np.float64) for stroke in first["strokes"]]
        prefix = first_strokes[:int(first["prefix_count"])]
        model = load_model(run_dir / "best.pt")
        sensitivity = latent_sensitivity(model, prefix, seed=2026, max_points=64, max_strokes=16)
        completion_eos = np.mean([item["info"]["ended_by_eos"] for item in generated])
        random_eos = np.mean([item["termination"]["ended_by_eos"] for item in generation["random"]])
        valid_rate = np.mean([item["valid"] for item in validations])
        collapse = summary["collapse"]
        row = {
            "model": model_name,
            "round1_rank": RANK[model_name],
            "parameters": summary["params"],
            "steps": summary["steps"],
            "training_time_seconds": summary["training_time_seconds"],
            "peak_vram_mib": summary["peak_vram_mib"],
            "stable_finite_history": stable,
            "train_loss_first": history[0]["loss"],
            "train_loss_last": history[-1]["loss"],
            "val_reconstruction": validation["reconstruction"],
            "val_kl_global": validation["global_kl"],
            "val_kl_local": validation["stroke_kl"],
            "negative_elbo_diagnostic": validation["negative_elbo"],
            "pen_accuracy": validation["pen_accuracy"],
            "valid_output_rate": valid_rate,
            "completion_valid_rate": valid_rate,
            "completion_eos_rate": completion_eos,
            "random_eos_rate": random_eos,
            "degenerate_rate": generation["overall"]["degenerate_fraction"],
            "intra_stroke_jump_rate": generation["overall"]["intra_jump_fraction"],
            "canvas_validity": generation["overall"]["canvas_validity"],
            "completion_chamfer": generation["overall"]["chamfer"],
            "diversity_chamfer": generation["overall"]["latent_diversity_chamfer"],
            "distinct_suffixes_mean": generation["overall"]["distinct_suffixes"],
            "latent_decoder_mean_shift": sensitivity["decoder_mixture_mean_l2"],
            "latent_stop_logit_shift": sensitivity["decoder_stop_logit_l2"],
            "latent_ignored_alert": sensitivity["latent_ignored_alert"],
            "collapse_warning": collapse["persistent_near_zero_kl"],
            "prefix_exact": prefix_exact and generation["prefix_exact_fraction"] == 1.0,
            "visual_assessment": VISUAL[model_name],
            "checkpoint": str(run_dir / "best.pt"),
            "status": summary["status"],
        }
        rows.append(row)
        details[model_name] = {
            "scoreboard": row,
            "collapse": collapse,
            "latent_sensitivity": sensitivity,
            "completion_validations": validations,
            "by_prefix": generation["by_prefix"],
            "random": generation["random"],
            "grids": {
                "random": str(eval_dir / "random.png"),
                "one_stroke": str(eval_dir / "completion_one.png"),
                "two_strokes": str(eval_dir / "completion_two.png"),
                "25_percent": str(eval_dir / "completion_0p25.png"),
                "50_percent": str(eval_dir / "completion_0p5.png"),
                "75_percent": str(eval_dir / "completion_0p75.png"),
            },
        }
    rows.sort(key=lambda row: row["round1_rank"])
    with (ROOT / "scoreboard.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "round": 1,
        "evaluation": {"validation_sketches": 8, "samples_per_prefix": 3,
                       "prefixes": ["one", "two", 0.25, 0.5, 0.75], "seed": 2026},
        "scoreboard": rows,
        "top2": ["E", "D"],
        "selection_basis": [
            "E and D are the only models with contained generation/completion geometry (canvas validity 0.946 and 0.854).",
            "Both have zero measured anomalous intra-stroke jumps and preserve non-zero global/local KL.",
            "E leads D in completion Chamfer, Hausdorff, endpoint error and canvas containment; D retains greater conditional Chamfer diversity.",
            "A collapses globally; A/B/C are visually dominated by long out-of-canvas trajectories. Likelihoods across flat and hierarchical factorizations are not used as a common ranking scale.",
        ],
        "details": details,
    }
    payload = json_ready(payload)
    (ROOT / "scoreboard.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"top2": payload["top2"], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
