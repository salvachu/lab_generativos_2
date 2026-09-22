"""Minimal real-checkpoint integration, not a quality benchmark or tournament."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sketchlab.generation import load_model, sample, sample_multiple, encode
from sketchlab.diagnostics import latent_sensitivity
from sketchlab.evaluation import prefix_preserved
from sketchlab.orchestration import json_ready
from sketchlab.rendering import render_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/smoke")
    p.add_argument("--output", default="runs/smoke_inference")
    args = p.parse_args()
    output = Path(args.output)
    if (output / "summary.json").exists():
        p.error("Choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    prefix = [np.array([[180.123456789, 220.], [200., 195.], [230., 195.],
                        [250., 220.], [230., 245.], [200., 245.], [180., 220.]], dtype=np.float64)]
    results, panels, titles, counts = [], [], [], []
    for name in "ABCDE":
        model = load_model(Path(args.runs) / name / "best.pt")
        posterior = encode(model, prefix)
        assert np.isfinite(posterior["mu"]).all()
        random_sketch, random_info = sample(model, max_points=48, max_strokes=8, return_info=True)
        completed, completion_info = sample(model, prefix, max_points=48, max_strokes=8, return_info=True)
        assert prefix_preserved(prefix, completed)
        assert all(np.isfinite(s).all() for sketch in (random_sketch, completed) for s in sketch)
        candidates = sample_multiple(model, prefix, n_candidates=3, top_k=2,
                                     max_points=48, max_strokes=8)
        assert len(candidates["candidates"]) == 3
        assert all(prefix_preserved(prefix, c["postprocessed_output"]) for c in candidates["candidates"])
        sensitivity = latent_sensitivity(model, prefix, max_points=32, max_strokes=8)
        results.append({"model": name, "params": sum(p.numel() for p in model.parameters()),
                        "prefix_exact": True, "random": random_info, "completion": completion_info,
                        "pipeline": candidates["report"], "latent_sensitivity": sensitivity,
                        "status": "passed", "checkpoint": str(Path(args.runs)/name/"best.pt")})
        panels.extend([random_sketch, completed]); titles.extend([f"{name} tiny | random", f"{name} tiny | completion"]); counts.extend([0,len(prefix)])
        (output / f"{name}_candidates.json").write_text(json.dumps(json_ready(candidates), indent=2, allow_nan=False), encoding="utf-8")
    render_grid(panels, output/"tiny_grid.png", titles=titles, prefix_counts=counts, ncols=2)
    (output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"models_passed": len(results), "prefix_exact": True, "output": str(output)}))


if __name__ == "__main__":
    main()
