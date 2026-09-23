"""Reproducible inference shared by evaluation, CLI and local demo.

The returned prefix is copied verbatim. Limits apply only to generated points and
strokes. A cap is reported as a cap, never as learned EOS termination.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from sketchlab.losses import mdn_parameters
from sketchlab.models import create_model
from sketchlab.models.common import tokens, validate_sketch


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> nn.Module:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {"model_config", "model_state"} <= checkpoint.keys():
        raise ValueError("Checkpoint must contain model_config and model_state")
    if checkpoint["model_config"].get("model") == "F" and checkpoint.get("config", {}).get("training_stage") == "stroke_ae":
        raise ValueError("F_STROKE_AE is a representation checkpoint; use a composition checkpoint for generation")
    model = create_model(checkpoint["model_config"]).to(device)
    if model.model_name == "F" and checkpoint.get("model_representation") != model.representation_metadata:
        raise ValueError("F stroke-view metadata differs from checkpoint configuration")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def encode(handle: nn.Module, strokes: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Encode a full sketch; C/E use empty inference context unless sampling a prefix.

    Returns the global posterior parameters, not a random draw. Feed ``mu`` to
    sample(z=...) to inspect reconstruction or interpolate latent vectors.
    """
    strokes = validate_sketch(strokes)
    handle.eval()
    context = handle.context([[]])
    mu, logvar = handle.posterior([strokes], context)
    return {"mu": mu[0].cpu().numpy().copy(), "logvar": logvar[0].cpu().numpy().copy()}


def _normal(mu: Tensor, logvar: Tensor, rng: torch.Generator) -> Tensor:
    noise = torch.randn(mu.shape, generator=rng, dtype=torch.float32).to(mu.device)
    return mu + noise * (0.5 * logvar).exp()


@torch.inference_mode()
def sample_latent(handle: nn.Module, prefix: list[np.ndarray] | None = None,
                  seed: int = 42) -> np.ndarray:
    """Sample the global prior; C/E condition exclusively on the supplied prefix."""
    handle.eval()
    context = handle.context([validate_sketch([] if prefix is None else prefix)])
    mu, logvar = handle.prior(context)
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    return _normal(mu, logvar, rng)[0].cpu().numpy().copy()


def _category(logits: Tensor, temperature: float, rng: torch.Generator) -> int:
    logits = logits.detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite distribution parameters")
    if temperature == 0:
        return int(logits.argmax())
    # Center before division so even tiny positive temperatures cannot create
    # positive infinity and an undefined inf-inf softmax.
    probabilities = torch.softmax((logits - logits.max()).double() / temperature, -1)
    return int(torch.multinomial(probabilities, 1, generator=rng))


def _point(raw: Tensor, temperature: float, rng: torch.Generator) -> np.ndarray:
    raw = raw.detach().float().cpu()
    if not torch.isfinite(raw).all():
        raise FloatingPointError("nonfinite distribution parameters")
    log_weights, means, log_scales, correlations = mdn_parameters(raw)
    index = _category(log_weights, temperature, rng)
    if temperature == 0:
        point = means[index]
    else:
        noise = torch.randn(2, generator=rng)
        rho = correlations[index]
        correlated = torch.stack([noise[0], rho * noise[0] + (1 - rho.square()).sqrt() * noise[1]])
        point = means[index] + log_scales[index].exp() * temperature ** 0.5 * correlated
    if not torch.isfinite(point).all():
        raise FloatingPointError("nonfinite coordinate")
    return point.numpy().astype(np.float64)


def _finish(info: dict[str, Any], continuation: list[np.ndarray], termination: str,
            *, partial_stroke: bool = False) -> tuple[list[np.ndarray], dict[str, Any]]:
    info.update(termination=termination, ended_by_eos=termination == "eos",
                capped=termination in {"max_points", "max_strokes"},
                partial_stroke=partial_stroke, generated_points=sum(len(s) for s in continuation),
                generated_strokes=len(continuation),
                finite=all(np.isfinite(s).all() for s in continuation))
    return continuation, info


def _sequence_sample(model: nn.Module, prefix: list[np.ndarray], z: Tensor, context: Tensor,
                     decoder_rng: torch.Generator, temperature: float, max_points: int,
                     max_strokes: int, info: dict[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]:
    prefix_tokens = tokens(prefix, model.scale)[:-1]
    bos = np.asarray([[0, 0, 1, 0, 0]], dtype=np.float64)
    inputs = torch.as_tensor(np.concatenate([bos, prefix_tokens]), dtype=torch.float32,
                             device=model.device).unsqueeze(0)
    raw, pen, hidden = model.decode(inputs, z, context)
    raw, pen = raw[0, -1], pen[0, -1]
    cursor = np.asarray(prefix[-1][-1], dtype=np.float64).copy() if prefix else np.zeros(2)
    continuation: list[np.ndarray] = []
    current: list[np.ndarray] = []
    termination = "max_points"
    for index in range(max_points):
        try:
            state = _category(pen, temperature, decoder_rng)
            if state == 2:
                termination = "eos"
                break
            delta = _point(raw, temperature, decoder_rng)
            next_cursor = cursor + model.scale * delta
            if not np.isfinite(next_cursor).all():
                raise FloatingPointError("nonfinite cumulative coordinate")
        except FloatingPointError:
            termination = "nonfinite_prediction"
            break
        cursor = next_cursor
        current.append(cursor.copy())
        token = np.zeros(5, dtype=np.float32)
        token[:2], token[2 + state] = delta, 1
        if state == 1:
            continuation.append(np.asarray(current, dtype=np.float64))
            current = []
            if len(continuation) >= max_strokes:
                termination = "max_strokes"
                break
        raw, pen, hidden = model.decode(torch.as_tensor(token, device=model.device)[None, None],
                                        z, context, hidden)
        raw, pen = raw[0, 0], pen[0, 0]
    partial = bool(current) and termination != "eos"
    if current:
        continuation.append(np.asarray(current, dtype=np.float64))
    return _finish(info, continuation, termination, partial_stroke=partial)


def _hierarchical_sample(model: nn.Module, prefix: list[np.ndarray], z: Tensor, context: Tensor,
                         latent_rng: torch.Generator, decoder_rng: torch.Generator,
                         temperature: float, max_points: int, max_strokes: int,
                         info: dict[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]:
    embeddings = model.embed_strokes(prefix)
    bos = torch.zeros((1, model.hidden_dim), device=model.device)
    previous = torch.cat([bos, embeddings], 0).unsqueeze(0)
    states, stroke_hidden = model.decode_strokes(previous, z, context)
    stroke_state = states[:, -1]
    continuation: list[np.ndarray] = []
    points_used = 0
    for stroke_index in range(max_strokes):
        try:
            if _category(model.sketch_end_head(stroke_state)[0], temperature, decoder_rng) == 1:
                return _finish(info, continuation, "eos")
            anchor = _point(model.anchor_head(stroke_state)[0], temperature, decoder_rng)
            anchor_absolute = anchor * model.scale
            if not np.isfinite(anchor_absolute).all():
                raise FloatingPointError("nonfinite absolute anchor")
        except FloatingPointError:
            return _finish(info, continuation, "nonfinite_prediction")
        local_mu, local_logvar = model.local_prior(stroke_state)
        local_z = _normal(local_mu, local_logvar, latent_rng)
        anchor_tensor = torch.as_tensor(anchor, device=model.device, dtype=torch.float32)[None]
        condition = model.point_condition(stroke_state, local_z, anchor_tensor)
        point_previous = torch.zeros((1, 1, 3), device=model.device)
        point_hidden = None
        current = [anchor_absolute]
        points_used += 1
        stroke_ended = False
        while points_used < max_points:
            raw, pen, point_hidden = model.decode_points(point_previous, condition, point_hidden)
            try:
                if _category(pen[0, 0], temperature, decoder_rng) == 1:
                    stroke_ended = True
                    break
                delta = _point(raw[0, 0], temperature, decoder_rng)
                new_point = current[-1] + delta * model.scale
                if not np.isfinite(new_point).all():
                    raise FloatingPointError("nonfinite cumulative coordinate")
            except FloatingPointError:
                continuation.append(np.asarray(current, dtype=np.float64))
                return _finish(info, continuation, "nonfinite_prediction", partial_stroke=True)
            current.append(new_point)
            points_used += 1
            point_previous = torch.as_tensor([[[*delta, 0.0]]], device=model.device, dtype=torch.float32)
        continuation.append(np.asarray(current, dtype=np.float64))
        if points_used >= max_points:
            return _finish(info, continuation, "max_points", partial_stroke=not stroke_ended)
        # The next high-level prediction sees only strokes already generated.
        previous = model.embed_strokes([continuation[-1]]).unsqueeze(0)
        states, stroke_hidden = model.decode_strokes(previous, z, context, stroke_hidden)
        stroke_state = states[:, -1]
    return _finish(info, continuation, "max_strokes")


@torch.inference_mode()
def sample(handle: nn.Module, prefix: list[np.ndarray] | None = None, seed: int = 42,
           temperature: float = 0.6, max_points: int = 384, max_strokes: int = 64,
           z: np.ndarray | Tensor | None = None, decoder_seed: int | None = None,
           return_info: bool = False) -> list[np.ndarray] | tuple[list[np.ndarray], dict[str, Any]]:
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not isinstance(max_points, int) or max_points < 1 or not isinstance(max_strokes, int) or max_strokes < 1:
        raise ValueError("Generation limits must be positive integers")
    prefix = [s.copy() for s in validate_sketch([] if prefix is None else prefix)]
    handle.eval()
    decoder_seed = int(seed) + 1_000_003 if decoder_seed is None else int(decoder_seed)
    latent_rng = torch.Generator(device="cpu").manual_seed(int(seed))
    local_latent_seed = int(seed) + 2_000_003
    local_latent_rng = torch.Generator(device="cpu").manual_seed(local_latent_seed)
    decoder_rng = torch.Generator(device="cpu").manual_seed(decoder_seed)
    context = handle.context([prefix])
    if z is None:
        mu, logvar = handle.prior(context)
        latent = _normal(mu, logvar, latent_rng)
    else:
        latent = torch.as_tensor(z, device=handle.device, dtype=torch.float32).reshape(1, -1)
        if latent.shape[1] != handle.latent_dim or not torch.isfinite(latent).all():
            raise ValueError(f"z must contain {handle.latent_dim} finite values")
    info = {"model": handle.model_name, "seed": int(seed), "decoder_seed": decoder_seed,
            "temperature": float(temperature), "prefix_strokes": len(prefix),
            "stroke_latent_seed": local_latent_seed if handle.model_name in "DE" else None,
            "prefix_points": sum(len(s) for s in prefix), "max_points": max_points,
            "max_strokes": max_strokes, "explicit_z": z is not None}
    if handle.model_name == "F":
        continuation, info = _compositional_sample(handle, prefix, latent, context, decoder_rng,
                                                   temperature, max_points, max_strokes, info)
    elif handle.model_name in "ABC":
        continuation, info = _sequence_sample(handle, prefix, latent, context, decoder_rng,
                                              temperature, max_points, max_strokes, info)
    else:
        continuation, info = _hierarchical_sample(handle, prefix, latent, context, local_latent_rng,
                                                  decoder_rng, temperature, max_points, max_strokes, info)
    result = prefix + continuation
    return (result, info) if return_info else result


def _compositional_sample(model, prefix, z, context, rng, temperature, max_points, max_strokes, info):
    view = model.view([prefix])
    embeddings = model.encode_view(view)[:, :len(prefix)]
    tokens = model.stroke_tokens(embeddings, view["anchors"][:, :len(prefix)])
    continuation = []
    points_used = 0
    info.update(stroke_samples=model.stroke_samples, point_autoregression=False)
    for _ in range(max_strokes):
        state = model.next_state(tokens, z, context)
        try:
            if _category(model.sketch_end_head(state)[0], temperature, rng) == 1:
                return _finish(info, continuation, "eos")
            # Do not silently truncate a stroke or call a cap learned EOS.
            if points_used + model.stroke_samples > max_points:
                return _finish(info, continuation, "max_points")
            anchor = _point(model.anchor_head(state)[0], temperature, rng)
            anchor_tensor = torch.as_tensor(anchor, device=model.device, dtype=torch.float32)[None]
            mu, logvar = model.embedding_distribution(state, anchor_tensor)
            if temperature == 0:
                embedding = mu
            else:
                noise = torch.randn(mu.shape, generator=rng).to(model.device)
                embedding = mu + noise*(.5*logvar).exp()*temperature**.5
            if not torch.isfinite(embedding).all():
                raise FloatingPointError("nonfinite stroke embedding")
            t = torch.linspace(0., 1., model.stroke_samples, device=model.device)
            relative = model.stroke_decoder(embedding, t)[0].cpu().numpy().astype(np.float64)
            stroke = (relative + anchor)*model.scale
            if not np.isfinite(stroke).all():
                raise FloatingPointError("nonfinite stroke coordinates")
            continuation.append(stroke); points_used += len(stroke)
            # Feedback is the sampled stroke code and absolute anchor. There is
            # no previous-point input or cumulative coordinate integration.
            token = model.stroke_tokens(embedding, anchor_tensor).unsqueeze(1)
            tokens = torch.cat([tokens, token], 1)
        except FloatingPointError:
            return _finish(info, continuation, "nonfinite_prediction")
    return _finish(info, continuation, "max_strokes")


def complete_sketch(handle: nn.Module, prefix: list[np.ndarray], n_samples: int = 4,
                    seed: int = 42, temperature: float = 0.6, max_points: int = 512,
                    max_strokes: int = 48, z: np.ndarray | Tensor | None = None) -> list[list[np.ndarray]]:
    if not isinstance(n_samples, int) or not 1 <= n_samples <= 64:
        raise ValueError("n_samples must be between 1 and 64")
    # Fixed decoder noise isolates the effect of global/local latent samples.
    return [sample(handle, prefix, seed + i, temperature, max_points, max_strokes,
                   z=z, decoder_seed=seed + 1_000_003) for i in range(n_samples)]


def generate_random(handle: nn.Module, n_samples: int = 4, seed: int = 42,
                    temperature: float = 0.6, max_points: int = 512,
                    max_strokes: int = 48) -> list[list[np.ndarray]]:
    return complete_sketch(handle, [], n_samples, seed, temperature, max_points, max_strokes)


def render(strokes: list[np.ndarray], prefix_count: int = 0) -> str:
    from sketchlab.rendering import render_svg
    return render_svg(strokes, prefix_count=prefix_count)


def sample_multiple(*args, **kwargs):
    """Optional geometry/ranking orchestration, kept out of model inference."""
    from sketchlab.orchestration import sample_multiple as run
    return run(*args, **kwargs)
