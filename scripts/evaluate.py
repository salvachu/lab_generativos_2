"""Evaluate a checkpoint on a fixed held-out slice and save vector render grids."""
import argparse
from pathlib import Path
import numpy as np
import torch
from sketchlab.data import load_raw
from sketchlab.evaluation import evaluate_generation
from sketchlab.generation import load_model
from sketchlab.training import training_threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--temperature", type=float, default=.6)
    p.add_argument("--max-points", type=int, default=384)
    args = p.parse_args()
    if args.count < 1 or args.samples < 2:
        p.error("count>=1 and samples>=2 are required for conditional diversity")
    if (Path(args.output) / "generation_metrics.json").exists():
        p.error("Output already has evaluation results; choose a new directory")
    torch.set_num_threads(1)
    import json
    from sketchlab.training import validate, write_json
    val = load_raw("data/raw/val.pkl")
    rng = np.random.default_rng(args.seed)
    manifest_path = Path(args.checkpoint).parent.parent / "manifest.json"
    indices = json.loads(manifest_path.read_text(encoding="utf-8"))["val_indices"] if manifest_path.exists() else rng.permutation(len(val)).tolist()
    selected = [val[int(i)] for i in indices[:args.count]]
    model = load_model(args.checkpoint)
    evaluate_generation(model, selected, args.output, n_samples=args.samples,
                        max_points=args.max_points, temperature=args.temperature,
                        jump_threshold=training_threshold(None))
    metrics = validate(model, selected, 4, beta=.05, free_bits=0.)
    write_json(Path(args.output) / "validation_likelihood.json", metrics)
    print(f"Evaluation saved in {args.output}")


if __name__ == "__main__":
    main()
