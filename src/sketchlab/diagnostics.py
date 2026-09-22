"""Collapse indicators are evidence, not a diagnosis from one scalar."""
from __future__ import annotations

import numpy as np
import torch

from sketchlab.evaluation import distances, finite_mean, prefix_preserved


def kl_schedule(step, config):
    """Constant, linear warmup, cosine warmup, or repeated linear cycles."""
    kind = config.get("kl_schedule", "linear")
    maximum = float(config.get("beta", .05))
    minimum = float(config.get("beta_start", 0.))
    duration = max(1, int(config.get("warmup_steps", 100)))
    if kind == "constant":
        return maximum
    if kind == "cyclical":
        cycle = max(1, int(config.get("cycle_steps", duration * 2)))
        progress = min(1., (step % cycle + 1) / duration)
    else:
        progress = min(1., (step + 1) / duration)
    if kind == "cosine":
        progress = (1 - np.cos(progress * np.pi)) / 2
    elif kind not in {"linear", "cyclical"}:
        raise ValueError("Unknown KL schedule")
    return minimum + (maximum - minimum) * progress


def collapse_report(history, *, window=20, kl_threshold=1e-3):
    rows = history[-window:]
    raw_kl = [r.get("global_kl", r["kl"]) for r in rows if "kl" in r]
    local_kl = [r["stroke_kl"] for r in rows if "stroke_kl" in r and r.get("stroke_events", 0) > 0]
    return {
        "steps_observed": len(rows), "required_window": window,
        "global_kl_mean": finite_mean(raw_kl), "kl_threshold": kl_threshold,
        "local_stroke_kl_mean": finite_mean(local_kl),
        "persistent_near_zero_kl": len(raw_kl) >= window and all(k < kl_threshold for k in raw_kl),
        "interpretation": "Low KL is an alert. Check latent sensitivity, posterior variance and valid-output diversity; it does not alone prove a strong decoder or collapse.",
    }


@torch.inference_mode()
def latent_sensitivity(model, prefix, *, seed=42, max_points=64, max_strokes=12):
    from sketchlab.generation import sample
    from sketchlab.losses import mdn_parameters
    from sketchlab.models.common import tokens

    model.eval()
    device = model.device
    context = model.context([prefix])
    mu, logvar = model.prior(context)
    z0 = mu
    direction = torch.ones_like(mu) / max(1, model.latent_dim) ** .5
    z1 = mu + direction * (.5 * logvar).exp()
    if model.model_name in "ABC":
        observed = tokens(prefix, model.scale)[:-1]
        rows = np.concatenate([[[0., 0., 1., 0., 0.]], observed])
        inputs = torch.as_tensor(rows, device=device, dtype=torch.float32)[None]
        raw0, stop0, _ = model.decode(inputs, z0, context)
        raw1, stop1, _ = model.decode(inputs, z1, context)
    else:
        em = model.embed_strokes(prefix)
        previous = torch.cat([torch.zeros(1, model.hidden_dim, device=device), em])[None]
        state0, _ = model.decode_strokes(previous, z0, context)
        state1, _ = model.decode_strokes(previous, z1, context)
        raw0, raw1 = model.anchor_head(state0), model.anchor_head(state1)
        stop0, stop1 = model.sketch_end_head(state0), model.sketch_end_head(state1)
    # Compare distributions even if both bounded samples happen to stop immediately.
    p0 = mdn_parameters(raw0[:, -1]); p1 = mdn_parameters(raw1[:, -1])
    mean_shift = float(torch.linalg.vector_norm(p0[1] - p1[1]).cpu())
    logit_shift = float(torch.linalg.vector_norm(stop0[:, -1] - stop1[:, -1]).cpu())
    generated = [sample(model, prefix=prefix, seed=seed, decoder_seed=seed+1,
                        z=z, temperature=.6, max_points=max_points, max_strokes=max_strokes,
                        return_info=True) for z in (z0, z1)]
    suffixes = [g[len(prefix):] for g, _ in generated]
    return {
        "decoder_mixture_mean_l2": mean_shift, "decoder_stop_logit_l2": logit_shift,
        "suffix_chamfer": distances(*suffixes)["chamfer"],
        "stroke_count_difference": abs(len(suffixes[0]) - len(suffixes[1])),
        "prefix_exact": all(prefix_preserved(prefix, g) for g, _ in generated),
        "termination": [info["termination"] for _, info in generated],
        "latent_ignored_alert": mean_shift < 1e-6 and logit_shift < 1e-6,
        "caveat": "Sensitivity proves numerical dependence, not semantic quality. Local latent RNG and decoder RNG are fixed.",
    }
