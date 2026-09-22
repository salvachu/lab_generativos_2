"""Versioned checkpoints: inference, exact continuation, and warm start differ.

Only state dictionaries and primitive metadata are serialized. Atomic replacement
prevents interrupted saves from destroying a valid file. Unrequested overwrites
are refused by default. Exact continuation is supported on the same backend and
software versions; it is tested bit-for-bit on CPU.
"""
from __future__ import annotations

import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1
REPRESENTATION_VERSION = "absolute-stroke5-v1;model-delta-v1"


def capture_rng():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": np_state[0], "keys": np_state[1].tolist(), "pos": np_state[2],
                  "has_gauss": np_state[3], "cached_gaussian": np_state[4]},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["pos"],
                         n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def save_checkpoint(path, model, *, optimizer=None, scheduler=None, step=0, epoch=0.,
                    config=None, training_state=None, validation=None, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    payload = {
        "format_version": FORMAT_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "normalization": {"scale": model.scale, "origin": [0., 0.],
                          "canonical": "absolute", "model": "relative_deltas", "fit_split": "train"},
        "model_config": dict(model.config), "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "step": int(step), "epoch": float(epoch), "config": config or {},
        "training_state": training_state or {}, "validation": validation or {},
        "rng_state": capture_rng(),
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                    "device": str(next(model.parameters()).device), "cuda": torch.version.cuda,
                    "cudnn_enabled": torch.backends.cudnn.enabled},
    }
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def load_checkpoint(path, *, model=None, optimizer=None, scheduler=None,
                    mode="inference", device="cpu"):
    """inference: weights; warm_start: weights+new optimizer/RNG; resume: all state."""
    if mode not in {"inference", "warm_start", "resume"}:
        raise ValueError("mode must be inference, warm_start, or resume")
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not {"model_config", "model_state"} <= payload.keys():
        raise ValueError("Invalid sketchlab checkpoint")
    if payload.get("format_version", FORMAT_VERSION) != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    if mode == "resume" and payload.get("representation_version") != REPRESENTATION_VERSION:
        raise ValueError("Exact resume requires matching representation version")
    if model is None:
        from sketchlab.models import create_model
        model = create_model(payload["model_config"]).to(device)
    elif dict(model.config) != payload["model_config"]:
        raise ValueError("Architecture differs from checkpoint; choose a matching config")
    model.load_state_dict(payload["model_state"], strict=True)
    if mode == "resume":
        if optimizer is None or payload.get("optimizer_state") is None:
            raise ValueError("Exact resume requires saved and current optimizer")
        runtime = payload.get("runtime", {})
        current_device = str(next(model.parameters()).device)
        if runtime.get("torch") != str(torch.__version__) or runtime.get("device") != current_device:
            raise ValueError("Exact resume requires the original Torch version and device; use warm_start")
        if runtime.get("cudnn_enabled", True) != torch.backends.cudnn.enabled:
            raise ValueError("Exact resume requires the original recurrent backend")
        optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scheduler_state") is not None:
            if scheduler is None:
                raise ValueError("A saved scheduler must be supplied for exact resume")
            scheduler.load_state_dict(payload["scheduler_state"])
        restore_rng(payload["rng_state"])
    model.eval() if mode == "inference" else model.train()
    return model, payload
