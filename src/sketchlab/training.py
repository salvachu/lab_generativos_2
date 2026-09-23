"""Bounded, reproducible experiments with structured logs and resumable weights."""
from __future__ import annotations

import csv
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from sketchlab.data import load_raw, sha256_file
from sketchlab.evaluation import evaluate_generation, geometry_metrics, prefix_count
from sketchlab.models import create_model
from sketchlab.checkpointing import save_checkpoint, load_checkpoint
from sketchlab.diagnostics import kl_schedule, collapse_report


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def scalar_metrics(result):
    output = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            output[key] = float(value.detach().cpu())
        elif isinstance(value, (float, int)):
            output[key] = float(value)
    return output


def artificial_prefixes(sketches, rng, fractions=None):
    # Include an empty observation so C/E can also generate unconditionally.
    fractions = [0, "one", "two", .25, .5, .75] if fractions is None else fractions
    if not fractions:
        raise ValueError("prefix_fractions cannot be empty")
    choices = rng.integers(0, len(fractions), len(sketches))
    return [0 if fractions[c] == 0 else prefix_count(len(s), fractions[c]) for s, c in zip(sketches, choices)]


def beta_at(step, config, model_name):
    if model_name == "A":
        return 1.
    return kl_schedule(step, config)


def teacher_forcing_at(update_index, total_steps, config):
    """Return the teacher-fed probability for a bounded local training run."""
    schedule = config.get("teacher_forcing_schedule")
    if not schedule:
        return float(config.get("architecture", {}).get("teacher_forcing", 1.0))
    if schedule.get("type") != "linear_self_feed":
        raise ValueError("teacher_forcing_schedule.type must be linear_self_feed")
    warmup_fraction = float(schedule.get("warmup_fraction", .2))
    final_self_feed = float(schedule.get("final_self_feed", .25))
    if not 0 <= warmup_fraction < 1 or not 0 <= final_self_feed <= .3:
        raise ValueError("scheduled sampling requires warmup_fraction in [0,1) and final_self_feed in [0,.3]")
    progress = update_index / max(1, total_steps - 1)
    self_feed = 0.0 if progress <= warmup_fraction else final_self_feed * (
        progress - warmup_fraction) / (1.0 - warmup_fraction)
    return 1.0 - self_feed


@torch.no_grad()
def validate(model, samples, batch_size, beta, free_bits):
    model.eval()
    if model.model_name == "F":
        metrics = model.validate_samples(samples, batch_size, beta, free_bits)
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite F validation")
        return metrics
    totals, count = {}, 0
    event_totals = {"coordinate": 0., "pen": 0., "stroke": 0.}
    event_sums = {"coordinate_nll": 0., "pen_ce": 0., "pen_accuracy": 0.,
                  "stroke_kl": 0., "stroke_kl_objective": 0.}
    for start in range(0, len(samples), batch_size):
        sketches = [s["strokes"] for s in samples[start:start + batch_size]]
        # Deterministic mixed short/25/50/75% contexts, identical across candidates.
        contexts = [prefix_count(len(s), ["one", "two", .25, .5, .75][(start + j) % 5])
                    for j, s in enumerate(sketches)]
        result = scalar_metrics(model.batch_loss(sketches, contexts, beta=beta,
                                                free_bits=free_bits, deterministic=True))
        for key, value in result.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite validation {key}: {value}")
            totals[key] = totals.get(key, 0.) + value * len(sketches)
        for event in event_totals:
            event_totals[event] += result[f"{event}_events"]
        for key, event in (("coordinate_nll", "coordinate"), ("pen_ce", "pen"),
                           ("pen_accuracy", "pen"), ("stroke_kl", "stroke"),
                           ("stroke_kl_objective", "stroke")):
            event_sums[key] += result[key] * result[f"{event}_events"]
        count += len(sketches)
    if not count:
        raise ValueError("Validation requires at least one sample")
    metrics = {k: v / count for k, v in totals.items()}
    for key, event in (("coordinate_nll", "coordinate"), ("pen_ce", "pen"),
                       ("pen_accuracy", "pen"), ("stroke_kl", "stroke"),
                       ("stroke_kl_objective", "stroke")):
        metrics[key] = event_sums[key] / max(1., event_totals[event])
    metrics.update({f"{k}_events": v for k, v in event_totals.items()})
    metrics["reconstruction"] = metrics["reconstruction_loss"] = metrics["coordinate_nll"] + metrics["pen_ce"]
    metrics["kl"] = metrics["KL_loss"] = metrics["global_kl"] + metrics["stroke_kl"]
    metrics["kl_objective"] = metrics["global_kl_objective"] + metrics["stroke_kl_objective"]
    metrics["loss"] = metrics["total_loss"] = (metrics["reconstruction"] + metrics["beta_effective"] * metrics["kl_objective"] + metrics["auxiliary_loss"])
    return metrics


def training_threshold(train):
    """Compatibility helper reads persisted TRAIN statistics; never refits."""
    path = Path("data/processed/geometry.json")
    if not path.exists():
        raise FileNotFoundError("Run python -m scripts.fit_geometry once before training/evaluation")
    stats = json.loads(path.read_text(encoding="utf-8"))
    return float(stats["intra_segment_p995"])


class BatchStream:
    """Explicit sampler state avoids replaying or skipping data during resume."""
    def __init__(self, size, batch_size, seed, state=None):
        if not 1 <= batch_size <= size:
            raise ValueError("batch_size must be between 1 and training subset size")
        self.size, self.batch_size = size, batch_size
        self.rng = np.random.default_rng(seed + 11)
        self.prefix_rng = np.random.default_rng(seed + 23)
        self.permutation = self.rng.permutation(size)
        self.cursor = 0
        self.examples_seen = 0
        if state:
            if state["size"] != size or state["batch_size"] != batch_size:
                raise ValueError("Resume sampler size/batch size changed")
            self.rng.bit_generator.state = state["rng"]
            self.prefix_rng.bit_generator.state = state["prefix_rng"]
            self.permutation = np.asarray(state["permutation"], dtype=np.int64)
            self.cursor = state["cursor"]
            self.examples_seen = state["examples_seen"]

    def next(self, train, fractions=None):
        if self.cursor >= self.size:
            self.permutation = self.rng.permutation(self.size)
            self.cursor = 0
        indices = self.permutation[self.cursor:self.cursor+self.batch_size]
        self.cursor += len(indices)
        self.examples_seen += len(indices)
        sketches = [train[int(i)]["strokes"] for i in indices]
        return sketches, artificial_prefixes(sketches, self.prefix_rng, fractions)

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size,
                "rng": self.rng.bit_generator.state, "prefix_rng": self.prefix_rng.bit_generator.state,
                "permutation": self.permutation.tolist(), "cursor": self.cursor,
                "examples_seen": self.examples_seen}


def run_experiment(config, model_name, train, val, run_dir, *, resume=None, warm_start=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "last.pt").exists() or (run_dir / "history.jsonl").exists():
        raise FileExistsError("Use a fresh run directory; existing evidence is never overwritten")
    if resume and warm_start:
        raise ValueError("resume and warm_start are mutually exclusive")
    seed = config.get("seed", 2026)
    seed_all(seed)
    torch.set_num_threads(config.get("cpu_threads", 2))
    requested_device = config.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    model_config = {**config.get("architecture", {}), "model": model_name}
    model = create_model(model_config).to(device)
    model_config = dict(model.config)
    # cuDNN keeps an opaque RNN dropout state outside state_dict and Torch RNG.
    # Native GRU uses the captured Torch RNG, enabling exact resume with dropout.
    native_rnn = (torch.device(device).type == "cuda" and model_config.get("layers", 1) > 1
                  and model_config.get("dropout", 0.) > 0)
    torch.backends.cudnn.enabled = not native_rnn
    if model_name == "F":
        model.configure_training(config.get("training_stage", "composition"), config.get("freeze_stroke_ae", True))
        ae_path = config.get("stroke_ae_checkpoint")
        if ae_path and not (resume or warm_start):
            ae, _ = load_checkpoint(ae_path, device=device)
            if ae.model_name != "F" or ae.representation_metadata != model.representation_metadata:
                raise ValueError("Stroke AE representation does not match F config")
            model.stroke_encoder.load_state_dict(ae.stroke_encoder.state_dict(), strict=True)
            model.stroke_decoder.load_state_dict(ae.stroke_decoder.state_dict(), strict=True)
        local, composition = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (local if name.startswith(("stroke_encoder.", "stroke_decoder.")) else composition).append(parameter)
        groups = []
        if composition: groups.append({"params":composition})
        if local: groups.append({"params":local, "lr":config.get("learning_rate", .001)*(
            1. if model.training_stage == "stroke_ae" else config.get("stroke_ae_lr_factor", .1))})
        optimizer = torch.optim.Adam(groups, lr=config.get("learning_rate", .001))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))
    start_step = 0
    saved_state = None
    if resume:
        _, saved = load_checkpoint(resume, model=model, optimizer=optimizer, mode="resume", device=device)
        start_step = saved["step"]
        saved_state = saved["training_state"]
        # Budgets/evaluation may change; training distribution and objective may not.
        keys = ["seed", "batch_size", "learning_rate", "grad_clip", "beta", "beta_start", "kl_schedule",
                "warmup_steps", "cycle_steps", "free_bits", "prefix_fractions", "source_hashes"]
        if model_name == "F":
            keys += ["training_stage", "freeze_stroke_ae", "stroke_ae_lr_factor"]
        for key in keys:
            if saved["config"].get(key) != config.get(key):
                raise ValueError(f"Exact resume changed {key}; use warm_start")
        if saved_state.get("train_ids") != [s["id"] for s in train] or saved_state.get("val_ids") != [s["id"] for s in val]:
            raise ValueError("Exact resume requires the original dataset selections and order")
    elif warm_start:
        load_checkpoint(warm_start, model=model, mode="warm_start", device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    params = sum(p.numel() for p in model.parameters())
    free_bits = 0. if model_name == "A" else config.get("free_bits", .02)
    batch_size = config.get("batch_size", 16)
    batches = BatchStream(len(train), batch_size, seed, saved_state.get("sampler") if saved_state else None)
    total_steps = config.get("steps", 160)
    max_seconds = config.get("max_train_seconds", 180.)
    eval_every = config.get("eval_every", 80)
    history = []
    best_reconstruction = saved_state.get("best_reconstruction", float("inf")) if saved_state else float("inf")
    best_step = saved_state.get("best_step") if saved_state else None
    if resume:
        import shutil
        prior_best = Path(resume).parent / "best.pt"
        if prior_best.exists():
            shutil.copyfile(prior_best, run_dir / "best.pt")
        else:
            best_reconstruction, best_step = float("inf"), None
    validation_time = 0.
    elapsed_train = 0.
    status = "completed"
    seen = 0
    wall_start = time.perf_counter()
    write_json(run_dir / "config.json", {**config, "model_config": model_config})
    initial = validate(model, val, batch_size, beta_at(start_step, config, model_name), free_bits)
    write_json(run_dir / "initial_validation.json", initial)

    def state():
        return {"sampler": batches.state_dict(), "train_ids": [s["id"] for s in train],
                "val_ids": [s["id"] for s in val], "best_reconstruction": best_reconstruction,
                "best_step": best_step, "kl_schedule_step": history[-1]["step"] if history else start_step}

    def save(name, step, validation):
        save_checkpoint(run_dir / name, model, optimizer=optimizer, step=step,
                        epoch=batches.examples_seen/len(train), config=config,
                        training_state=state(), validation=validation, overwrite=True)
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(start_step, start_step + total_steps):
            if elapsed_train >= max_seconds:
                status = "time_budget_reached"
                break
            sketches, contexts = batches.next(train, config.get("prefix_fractions"))
            model.train()
            beta = beta_at(step, config, model_name)
            if device == "cuda": torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            teacher_forcing = teacher_forcing_at(step - start_step, total_steps, config)
            result = model.batch_loss(sketches, contexts, beta=beta, free_bits=free_bits,
                                      teacher_forcing=teacher_forcing)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            result["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Nonfinite gradients at step {step}")
            optimizer.step()
            if device == "cuda": torch.cuda.synchronize()
            elapsed_train += time.perf_counter() - tick
            seen += len(sketches)
            record = {"step": step + 1, "beta": beta, "teacher_forcing": teacher_forcing,
                      "self_feed_probability": 1.0 - teacher_forcing, "train_seconds": elapsed_train,
                      "grad_norm": float(grad_norm.detach().cpu()), **scalar_metrics(result)}
            history.append(record)
            stream.write(json.dumps(record) + "\n")
            if (step + 1) % eval_every == 0 or step + 1 == start_step + total_steps:
                tick = time.perf_counter()
                current = validate(model, val, batch_size, beta, free_bits)
                validation_time += time.perf_counter() - tick
                write_json(run_dir / f"validation_{step+1}.json", current)
                if current["reconstruction"] < best_reconstruction:
                    best_reconstruction = current["reconstruction"]
                    best_step = step + 1
                    save("best.pt", step + 1, current)
                save("last.pt", step + 1, current)
                stream.flush()
    if not history:
        raise RuntimeError("No optimizer updates completed")
    final_step = history[-1]["step"]
    final_validation = validate(model, val, batch_size, beta_at(final_step-1, config, model_name), free_bits)
    if best_step is None or final_validation["reconstruction"] < best_reconstruction:
        best_step = final_step
        best_reconstruction = final_validation["reconstruction"]
        save("best.pt", final_step, final_validation)
    save("last.pt", final_step, final_validation)
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    peak_vram = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.
    # Small autoregressive batches run faster on CPU; evaluation device is explicit.
    model.to("cpu").eval()
    torch.set_num_threads(1)
    generation = None
    tick = time.perf_counter()
    if config.get("generation_eval", True) and not (model_name == "F" and model.training_stage == "stroke_ae"):
        generation = evaluate_generation(
            model, val[:config.get("generation_val_size", 4)], run_dir / "samples",
            seed=config.get("eval_seed", 173), n_samples=config.get("samples_per_prefix", 3),
            max_points=config.get("max_generation_points", 384),
            max_strokes=config.get("max_generation_strokes", 64),
            temperature=config.get("temperature", .6),
            jump_threshold=config["jump_threshold"],
        )
    generation_seconds = time.perf_counter() - tick
    summary = {
        "model": model_name, "params": params, "latent_dim": model_config.get("latent_dim", 32),
        "training_stage": config.get("training_stage"),
        "model_representation": getattr(model, "representation_metadata", None),
        "strategy": "fixed beta=1" if model_name == "A" else f"beta={config.get('beta', .05)}, warmup={config.get('warmup_steps', 100)}, free_bits={free_bits}",
        "steps": len(history), "best_step": best_step, "train_examples_seen": seen,
        "train_loss": float(np.mean([h["loss"] for h in history[-min(20,len(history)):]])),
        "initial_validation": initial, "validation": best["validation"],
        "training_time_seconds": elapsed_train, "validation_time_seconds": validation_time,
        "generation_time_seconds": generation_seconds, "total_time_seconds": time.perf_counter() - wall_start,
        "peak_vram_mib": peak_vram, "device": device, "generation_device": "cpu",
        "checkpoint": str(run_dir / "best.pt"), "last_checkpoint": str(run_dir / "last.pt"), "status": status,
        "resume_from": str(resume) if resume else None, "warm_start_from": str(warm_start) if warm_start else None,
        "collapse": collapse_report(history),
        "completion": generation["overall"] if generation else None,
        "prefix_exact_fraction": generation["prefix_exact_fraction"] if generation else None,
        "notes": "Diagnostic pilot; likelihood scales differ between flat and hierarchical factorizations."
    }
    write_json(run_dir / "summary.json", summary)
    # CSV mirrors the machine-readable JSONL without console step-by-step chatter.
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    write_json(run_dir / "metrics.json", {"validation": best["validation"], "collapse": summary["collapse"], "summary": summary})
    return summary


def run_suite(config, models, output_dir, *, resume=None, warm_start=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(config.get("train_path", "data/raw/train.pkl"))
    val_path = Path(config.get("val_path", "data/raw/val.pkl"))
    raw_train, raw_val = load_raw(train_path), load_raw(val_path)
    seed = config.get("seed", 2026)
    rng = np.random.default_rng(seed)
    train_indices = rng.permutation(len(raw_train))[:config.get("train_size", len(raw_train))]
    val_indices = rng.permutation(len(raw_val))[:config.get("val_size", len(raw_val))]
    train = [raw_train[int(i)] for i in train_indices]
    val = [raw_val[int(i)] for i in val_indices]
    if (resume or warm_start) and len(models) != 1:
        raise ValueError("resume/warm_start requires exactly one model")
    hashes = {str(p): sha256_file(p) for p in (train_path, val_path)}
    config = {**config, "jump_threshold": training_threshold(None), "source_hashes": hashes}
    manifest = {"config": config, "train_indices": train_indices.tolist(), "val_indices": val_indices.tolist(),
                "train_ids": [s["id"] for s in train], "val_ids": [s["id"] for s in val],
                "hashes": hashes,
                "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    write_json(output_dir / "manifest.json", manifest)
    reference = [geometry_metrics(s["strokes"], config["jump_threshold"]) for s in val]
    write_json(output_dir / "validation_geometry.json", reference)
    results = []
    for name in models:
        print(f"Starting {name}: at most {config.get('steps',160)} updates / {config.get('max_train_seconds',180)} training seconds", flush=True)
        try:
            result = run_experiment(config, name, train, val, output_dir / name, resume=resume, warm_start=warm_start)
        except Exception as exc:
            import traceback
            error_dir = output_dir / name
            error_dir.mkdir(exist_ok=True)
            (error_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            result = {"model": name, "status": "failed", "error": str(exc)}
        results.append(result)
        write_json(output_dir / "scoreboard.json", results)
        write_scoreboard(output_dir, results)
        print(f"Finished {name}: {result['status']}", flush=True)
    after = {str(p): sha256_file(p) for p in (train_path, val_path)}
    if after != manifest["hashes"]:
        raise RuntimeError("Raw data hashes changed")
    return results


def write_scoreboard(directory, results):
    rows = []
    for s in results:
        v, c = s.get("validation", {}), s.get("completion") or {}
        rows.append({k: value for k, value in {
            "model": s["model"], "params": s.get("params"), "latent_dim": s.get("latent_dim"),
            "strategy": s.get("strategy"), "steps": s.get("steps"), "train_loss": s.get("train_loss"),
            "val_reconstruction": v.get("reconstruction"), "val_coordinate_nll": v.get("coordinate_nll"),
            "val_kl": v.get("kl"), "val_stroke_kl": v.get("stroke_kl"), "pen_accuracy": v.get("pen_accuracy"),
            "completion_chamfer": c.get("chamfer"), "completion_hausdorff": c.get("hausdorff"),
            "canvas_validity": c.get("canvas_validity"), "empty_fraction": c.get("empty"),
            "diversity_chamfer": c.get("latent_diversity_chamfer"),
            "training_seconds": s.get("training_time_seconds"), "peak_vram_mib": s.get("peak_vram_mib"),
            "checkpoint": s.get("checkpoint"), "status": s["status"],
        }.items()})
    with (Path(directory) / "scoreboard.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
