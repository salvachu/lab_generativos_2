"""Factory for A-E and the compositional conditional VAE F."""
from __future__ import annotations

import math

from torch import nn

DEFAULT_CONFIG = {"model": "A", "hidden_dim": 96, "latent_dim": 32,
                  "mixtures": 5, "stroke_latent_dim": 16, "scale": 256.0,
                  "layers": 1, "dropout": 0.0, "teacher_forcing": 1.0}


def create_model(config: dict | str) -> nn.Module:
    if isinstance(config, str):
        config = {"model": config}
    normalized = {**DEFAULT_CONFIG, **config}
    normalized["model"] = str(normalized["model"]).upper()
    if normalized["model"] not in "ABCDEF" or len(normalized["model"]) != 1:
        raise ValueError("model must be one of A, B, C, D, E, F")
    for key in ("hidden_dim", "latent_dim", "mixtures", "stroke_latent_dim", "layers"):
        if not isinstance(normalized[key], int) or normalized[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not math.isfinite(float(normalized["scale"])) or float(normalized["scale"]) <= 0:
        raise ValueError("scale must be finite and positive")
    if not 0 <= float(normalized["dropout"]) < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 <= float(normalized["teacher_forcing"]) <= 1:
        raise ValueError("teacher_forcing must be in [0, 1]")
    if normalized["model"] in "ABC":
        from .sequence import SequenceVAE
        return SequenceVAE(normalized)
    if normalized["model"] == "F":
        from .model_f import ModelF
        return ModelF(normalized)
    from .hierarchical import HierarchicalVAE
    return HierarchicalVAE(normalized)
