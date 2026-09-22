"""Models A/B (Sketch-RNN family) and C (conditional completion VAE)."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sketchlab.losses import (MDNHead, bivariate_mixture_nll, gaussian_kl,
                              gaussian_parameters, masked_mean, reparameterize)
from .common import batch_tokens, encode_packed, feedback_token, prefix_counts_checked


class SequenceVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.model_name = config["model"]
        self.conditional = self.model_name == "C"
        self.scale = float(config["scale"])
        h, z, k = (int(config[key]) for key in ("hidden_dim", "latent_dim", "mixtures"))
        self.latent_dim, self.hidden_dim = z, h
        self.layers = int(config["layers"])
        recurrent = {"num_layers": self.layers, "dropout": float(config["dropout"]) if self.layers > 1 else 0.0}
        self.input_dropout = nn.Dropout(float(config["dropout"]))
        self.encoder = nn.GRU(5, h, batch_first=True, bidirectional=True, **recurrent)
        if self.conditional:
            self.context_encoder = nn.GRU(5, h, batch_first=True, **recurrent)
            self.prior_head = nn.Linear(h, 2 * z)
        context_dim = h if self.conditional else 0
        self.posterior_head = nn.Linear(2 * h + context_dim, 2 * z)
        self.decoder_init = nn.Linear(z + context_dim, h * self.layers)
        self.decoder = nn.GRU(5 + z + context_dim, h, batch_first=True, **recurrent)
        self.coordinate_head = MDNHead(h, k)
        self.pen_head = nn.Linear(h, 3)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def context(self, prefixes: list[list[np.ndarray]]) -> Tensor:
        if not self.conditional:
            return torch.empty((len(prefixes), 0), device=self.device)
        data, lengths, _ = batch_tokens(prefixes, self.device, self.scale)
        # Prefix EOS means "end of observed context" only; no target point leaks.
        return encode_packed(self.context_encoder, self.input_dropout(data), lengths)

    def prior(self, context: Tensor) -> tuple[Tensor, Tensor]:
        if self.conditional:
            return gaussian_parameters(self.prior_head(context))
        zeros = torch.zeros((len(context), self.latent_dim), device=self.device)
        return zeros, zeros

    def posterior(self, sketches: list[list[np.ndarray]], context: Tensor) -> tuple[Tensor, Tensor]:
        data, lengths, _ = batch_tokens(sketches, self.device, self.scale)
        full = encode_packed(self.encoder, self.input_dropout(data), lengths)
        return gaussian_parameters(self.posterior_head(torch.cat([full, context], -1)))

    def initial_state(self, z: Tensor, context: Tensor) -> Tensor:
        state = torch.tanh(self.decoder_init(torch.cat([z, context], -1)))
        return state.reshape(len(z), self.layers, self.hidden_dim).transpose(0, 1).contiguous()

    def decode(self, inputs: Tensor, z: Tensor, context: Tensor,
               hidden: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        conditioning = torch.cat([z, context], -1).unsqueeze(1).expand(-1, inputs.shape[1], -1)
        output, hidden = self.decoder(torch.cat([self.input_dropout(inputs), conditioning], -1),
                                      self.initial_state(z, context) if hidden is None else hidden)
        return self.coordinate_head(output), self.pen_head(output), hidden

    def batch_loss(self, sketches: list[list[np.ndarray]], prefix_counts: list[int],
                   beta: float, free_bits: float = 0.0, deterministic: bool = False,
                   teacher_forcing: float | None = None) -> dict[str, Tensor]:
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        counts = prefix_counts_checked(sketches, prefix_counts)
        prefixes = [s[:c] for s, c in zip(sketches, counts)] if self.conditional else [[] for s in sketches]
        context = self.context(prefixes)
        p_mu, p_logvar = self.prior(context)
        q_mu, q_logvar = self.posterior(sketches, context)
        z = reparameterize(q_mu, q_logvar, deterministic)
        target, _, valid = batch_tokens(sketches, self.device, self.scale)
        inputs = torch.zeros_like(target)
        inputs[:, 0, 2] = 1
        inputs[:, 1:] = target[:, :-1]
        ratio = float(self.config["teacher_forcing"] if teacher_forcing is None else teacher_forcing)
        if not 0 <= ratio <= 1:
            raise ValueError("teacher_forcing must be in [0, 1]")
        # Validation always evaluates likelihood with its observed causal history.
        if not self.training or ratio == 1:
            raw, pen_logits, _ = self.decode(inputs, z, context)
        else:
            point_offsets = torch.tensor([sum(len(s) for s in sketch[:c]) for sketch, c in zip(sketches, counts)],
                                         device=self.device) if self.conditional else torch.zeros(len(sketches), device=self.device)
            raw_steps, pen_steps, hidden = [], [], None
            previous = inputs[:, :1]
            for t in range(target.shape[1]):
                step_raw, step_pen, hidden = self.decode(previous, z, context, hidden)
                raw_steps.append(step_raw)
                pen_steps.append(step_pen)
                if t + 1 < target.shape[1]:
                    predicted = feedback_token(step_raw[:, 0], step_pen[:, 0])
                    force = (torch.rand(len(sketches), device=self.device) < ratio) | (t < point_offsets)
                    previous = torch.where(force[:, None], target[:, t], predicted).unsqueeze(1)
            raw, pen_logits = torch.cat(raw_steps, 1), torch.cat(pen_steps, 1)
        if self.conditional:
            offsets = torch.tensor([sum(len(s) for s in sketch[:c]) for sketch, c in zip(sketches, counts)],
                                   device=self.device)
            valid = valid & (torch.arange(target.shape[1], device=self.device)[None] >= offsets[:, None])
        coordinate_mask = valid & (target[..., 4] == 0)
        nll_values = bivariate_mixture_nll(raw, target[..., :2])
        nll = masked_mean(nll_values, coordinate_mask)
        labels = target[..., 2:5].argmax(-1)
        ce_values = F.cross_entropy(pen_logits.transpose(1, 2), labels, reduction="none")
        ce = masked_mean(ce_values, valid)
        accuracy = masked_mean((pen_logits.argmax(-1) == labels).float(), valid)
        raw_kl = gaussian_kl(q_mu, q_logvar, p_mu, p_logvar).mean()
        controlled_kl = gaussian_kl(q_mu, q_logvar, p_mu, p_logvar, free_bits=free_bits).mean()
        effective_beta = 1.0 if self.model_name == "A" else float(beta)
        reconstruction = nll + ce
        loss = reconstruction + effective_beta * controlled_kl
        joint_reconstruction = (torch.where(coordinate_mask, nll_values, 0).sum() +
                                torch.where(valid, ce_values, 0).sum()) / len(sketches)
        negative_elbo = joint_reconstruction + raw_kl
        return {"loss": loss, "total_loss": loss, "reconstruction_loss": reconstruction,
                "KL_loss": raw_kl, "auxiliary_loss": reconstruction * 0,
                "beta_effective": loss.new_tensor(effective_beta),
                "elbo": -negative_elbo, "negative_elbo": negative_elbo,
                "elbo_is_mc_estimate": loss.new_tensor(float(not deterministic and (not self.training or ratio == 1))),
                "joint_reconstruction_per_sketch": joint_reconstruction,
                "joint_kl_per_sketch": raw_kl,
                "reconstruction": reconstruction, "coordinate_nll": nll, "pen_ce": ce,
                "kl": raw_kl, "pen_accuracy": accuracy, "kl_objective": controlled_kl,
                "global_kl": raw_kl, "stroke_kl": raw_kl * 0,
                "global_kl_objective": controlled_kl, "stroke_kl_objective": controlled_kl * 0,
                "stroke_events": raw_kl.new_zeros(()),
                "coordinate_events": coordinate_mask.sum().float(), "pen_events": valid.sum().float()}
