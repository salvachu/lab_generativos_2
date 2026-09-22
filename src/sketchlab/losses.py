"""Masked, numerically stable likelihoods for vector-valued VAE decoders."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over genuine events; masked padding never enters the reduction."""
    selected = torch.where(mask.bool(), values, torch.zeros_like(values))
    return selected.sum() / mask.to(values.dtype).sum().clamp_min(1)


class MDNHead(nn.Module):
    """Correlated bivariate Gaussian mixture, six numbers per component."""

    def __init__(self, input_dim: int, mixtures: int, *, anchor: bool = False):
        super().__init__()
        self.mixtures = mixtures
        self.projection = nn.Linear(input_dim, mixtures * 6)
        with torch.no_grad():
            bias = self.projection.bias.view(mixtures, 6)
            bias.zero_()
            bias[:, 3:5] = -1.5
            if anchor:
                bias[:, 1:3] = 0.9

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs).reshape(*inputs.shape[:-1], self.mixtures, 6)


def mdn_parameters(raw: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Bound *distribution parameters*, never predicted point coordinates."""
    log_weights = F.log_softmax(raw[..., 0], dim=-1)
    means = raw[..., 1:3]
    log_scales = raw[..., 3:5].clamp(-7.0, 3.0)
    correlations = 0.95 * torch.tanh(raw[..., 5])
    return log_weights, means, log_scales, correlations


def bivariate_mixture_nll(raw: Tensor, target: Tensor) -> Tensor:
    """One NLL per coordinate event, computed with logsumexp in float32."""
    raw, target = raw.float(), target.float()
    log_weights, means, log_scales, rho = mdn_parameters(raw)
    standardized = (target.unsqueeze(-2) - means) * torch.exp(-log_scales)
    x, y = standardized.unbind(-1)
    residual = (x.square() + y.square() - 2 * rho * x * y) / (1 - rho.square())
    log_density = (-math.log(2 * math.pi) - log_scales.sum(-1)
                   - 0.5 * torch.log1p(-rho.square()) - 0.5 * residual)
    return -torch.logsumexp(log_weights + log_density, dim=-1)


def gaussian_kl(q_mu: Tensor, q_logvar: Tensor, p_mu: Tensor | None = None,
                p_logvar: Tensor | None = None, *, free_bits: float = 0.0) -> Tensor:
    """KL(q||p), sum over latent dimensions, optionally floor each dimension."""
    if not math.isfinite(free_bits) or free_bits < 0:
        raise ValueError("free_bits must be finite and non-negative")
    if p_mu is None:
        p_mu = torch.zeros_like(q_mu)
    if p_logvar is None:
        p_logvar = torch.zeros_like(q_logvar)
    terms = 0.5 * (p_logvar - q_logvar +
                   (q_logvar.exp() + (q_mu - p_mu).square()) * (-p_logvar).exp() - 1)
    return terms.clamp_min(free_bits).sum(-1)


def gaussian_parameters(raw: Tensor) -> tuple[Tensor, Tensor]:
    mu, logvar = raw.chunk(2, dim=-1)
    return mu, logvar.clamp(-10.0, 6.0)


def reparameterize(mu: Tensor, logvar: Tensor, deterministic: bool = False) -> Tensor:
    return mu if deterministic else mu + torch.randn_like(mu) * (0.5 * logvar).exp()
