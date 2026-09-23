"""Run a bounded suite: python -m scripts.train --config configs/short.json."""
import argparse
import json
from pathlib import Path
from sketchlab.training import run_suite
from sketchlab.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/short.json")
    parser.add_argument("--models", default="A,B,C,D,E")
    parser.add_argument("--output", default="runs/short")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", help="Resume optimizer, RNG, sampler and schedule exactly; write a new output")
    mode.add_argument("--warm-start", help="Load weights with a fresh optimizer, sampler and RNG")
    parser.add_argument("--stroke-ae", help="Initialize only F stroke encoder/decoder for composition")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stroke_ae:
        if args.resume or args.warm_start:
            parser.error("--stroke-ae cannot be combined with resume/warm-start")
        config["stroke_ae_checkpoint"] = args.stroke_ae
    models = [m.strip().upper() for m in args.models.split(",")]
    if any(m not in "ABCDEF" or len(m) != 1 for m in models):
        parser.error("models must be comma-separated A,B,C,D,E,F")
    if (Path(args.output) / "manifest.json").exists():
        parser.error("Output already contains an experiment. Choose a new --output to preserve evidence.")
    results = run_suite(config, models, args.output, resume=args.resume, warm_start=args.warm_start)
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
