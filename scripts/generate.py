"""Generate or complete via the same candidate pipeline used by the local UI."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from sketchlab.generation import load_model, sample_multiple
from sketchlab.orchestration import json_ready
from sketchlab.rendering import render_grid, render_svg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--prefix", help="JSON file: list of strokes or object with strokes")
    p.add_argument("--output", required=True)
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=.6)
    p.add_argument("--max-points", type=int, default=384)
    p.add_argument("--max-strokes", type=int, default=64)
    p.add_argument("--raw", action="store_true", help="Inspect unfiltered diagnostic samples (not certified valid)")
    p.add_argument("--no-postprocess", action="store_true")
    args = p.parse_args()
    output = Path(args.output)
    if (output / "candidates.json").exists():
        p.error("Output already contains candidates. Choose a fresh directory.")
    output.mkdir(parents=True, exist_ok=True)
    prefix = []
    if args.prefix:
        data = json.loads(Path(args.prefix).read_text(encoding="utf-8"))
        data = data["strokes"] if isinstance(data, dict) else data
        prefix = [np.asarray(s, dtype=np.float64) for s in data]
    torch.set_num_threads(1)
    handle = load_model(args.checkpoint)
    result = sample_multiple(handle, prefix=prefix, n_candidates=args.candidates, top_k=args.top_k,
                             seed=args.seed, temperature=args.temperature, max_points=args.max_points,
                             max_strokes=args.max_strokes, validate=not args.raw,
                             postprocess=not args.no_postprocess, rerank=not args.raw)
    (output / "candidates.json").write_text(json.dumps(json_ready(result), indent=2, allow_nan=False), encoding="utf-8")
    panels, titles, counts = [], [], []
    for c in result["selected"]:
        for name in ("raw_output", "postprocessed_output"):
            strokes = c[name]
            panels.append(strokes); titles.append(f"z seed {c['seed']} | {name}"); counts.append(len(prefix))
            (output / f"candidate_{c['id']}_{name}.svg").write_text(render_svg(strokes, prefix_count=len(prefix)), encoding="utf-8")
    if panels:
        render_grid(panels, output / "raw_vs_processed.png", titles=titles, prefix_counts=counts)
    print(json.dumps(result["report"], indent=2))


if __name__ == "__main__":
    main()
