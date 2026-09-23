"""D/E: ordered sketch VAE, absolute anchors and local autoregressive strokes."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from sketchlab.losses import (MDNHead, bivariate_mixture_nll, gaussian_kl,
                              gaussian_parameters, masked_mean, reparameterize)
from .common import encode_packed, feedback_token, prefix_counts_checked, validate_sketch


class HierarchicalVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.model_name = config["model"]
        self.conditional = self.model_name == "E"
        self.scale = float(config["scale"])
        h, z, u, k = (int(config[key]) for key in
                      ("hidden_dim", "latent_dim", "stroke_latent_dim", "mixtures"))
        self.hidden_dim, self.latent_dim, self.stroke_latent_dim = h, z, u
        self.layers = int(config["layers"])
        class_weights = config.get("state_class_weights", {})
        self.point_state_weights = self._state_weights(class_weights.get("point", [1.0, 1.0]), "point")
        self.sketch_state_weights = self._state_weights(class_weights.get("sketch", [1.0, 1.0]), "sketch")
        recurrent = {"num_layers": self.layers, "dropout": float(config["dropout"]) if self.layers > 1 else 0.0}
        self.input_dropout = nn.Dropout(float(config["dropout"]))
        point_hidden = max(1, h // 2)
        self.point_encoder = nn.GRU(5, point_hidden, batch_first=True, bidirectional=True, **recurrent)
        self.stroke_projection = nn.Linear(2 * point_hidden + 2, h)
        self.sketch_encoder = nn.GRU(h, h, batch_first=True, bidirectional=True, **recurrent)
        if self.conditional:
            self.context_encoder = nn.GRU(h, h, batch_first=True, **recurrent)
            self.prior_head = nn.Linear(h, 2 * z)
        c = h if self.conditional else 0
        self.posterior_head = nn.Linear(2 * h + c, 2 * z)
        self.stroke_init = nn.Linear(z + c, h * self.layers)
        self.stroke_decoder = nn.GRU(h + z + c, h, batch_first=True, **recurrent)
        self.sketch_end_head = nn.Linear(h, 2)
        self.anchor_head = MDNHead(h, k, anchor=True)
        self.local_prior_head = nn.Linear(h, 2 * u)
        self.local_posterior_head = nn.Linear(2 * h, 2 * u)
        point_condition_dim = h + u + 2
        self.point_init = nn.Linear(point_condition_dim, h * self.layers)
        self.point_decoder = nn.GRU(3 + point_condition_dim, h, batch_first=True, **recurrent)
        self.point_coordinate_head = MDNHead(h, k)
        self.point_end_head = nn.Linear(h, 2)

    @staticmethod
    def _state_weights(values, name: str) -> tuple[float, float]:
        values = tuple(float(value) for value in values)
        if len(values) != 2 or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"{name} state_class_weights must contain two finite positive values")
        return values

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def embed_strokes(self, strokes: list[np.ndarray]) -> Tensor:
        if not strokes:
            return torch.empty((0, self.hidden_dim), device=self.device)
        rows, anchors = [], []
        for stroke in validate_sketch(strokes):
            a = np.asarray(stroke, dtype=np.float64) / self.scale
            features = np.zeros((len(a), 5), dtype=np.float32)
            features[:, :2] = a - a[0]
            features[1:, 2:4] = np.diff(a, axis=0)
            features[-1, 4] = 1
            rows.append(torch.as_tensor(features, device=self.device))
            anchors.append(a[0])
        lengths = torch.tensor([len(r) for r in rows], device=self.device)
        encoded = encode_packed(self.point_encoder, self.input_dropout(pad_sequence(rows, batch_first=True)), lengths)
        anchor_tensor = torch.as_tensor(np.asarray(anchors), device=self.device, dtype=torch.float32)
        return torch.tanh(self.stroke_projection(torch.cat([encoded, anchor_tensor], -1)))

    def sketch_embeddings(self, sketches: list[list[np.ndarray]]) -> tuple[Tensor, Tensor]:
        if not sketches:
            raise ValueError("A batch must contain at least one sketch")
        flat = self.embed_strokes([s for sketch in sketches for s in sketch])
        rows, offset = [], 0
        for sketch in sketches:
            count = len(sketch)
            rows.append(flat[offset:offset + count] if count else torch.zeros((1, self.hidden_dim), device=self.device))
            offset += count
        return pad_sequence(rows, batch_first=True), torch.tensor([len(s) for s in sketches], device=self.device)

    def context_from_embeddings(self, embeddings: Tensor, counts: Tensor) -> Tensor:
        if not self.conditional:
            return torch.empty((len(counts), 0), device=self.device)
        # Packing reads only observed stroke embeddings. Each embedding is local
        # to that stroke, so the future cannot enter prefix context through it.
        encoded = encode_packed(self.context_encoder, self.input_dropout(embeddings), counts.clamp_min(1))
        return torch.where((counts > 0)[:, None], encoded, torch.zeros_like(encoded))

    def context(self, prefixes: list[list[np.ndarray]]) -> Tensor:
        embeddings, counts = self.sketch_embeddings(prefixes)
        return self.context_from_embeddings(embeddings, counts)

    def prior(self, context: Tensor) -> tuple[Tensor, Tensor]:
        if self.conditional:
            return gaussian_parameters(self.prior_head(context))
        zeros = torch.zeros((len(context), self.latent_dim), device=self.device)
        return zeros, zeros

    def posterior_from_embeddings(self, embeddings: Tensor, counts: Tensor,
                                  context: Tensor) -> tuple[Tensor, Tensor]:
        encoded = encode_packed(self.sketch_encoder, self.input_dropout(embeddings), counts.clamp_min(1))
        return gaussian_parameters(self.posterior_head(torch.cat([encoded, context], -1)))

    def posterior(self, sketches: list[list[np.ndarray]], context: Tensor) -> tuple[Tensor, Tensor]:
        embeddings, counts = self.sketch_embeddings(sketches)
        return self.posterior_from_embeddings(embeddings, counts, context)

    def initial_state(self, z: Tensor, context: Tensor) -> Tensor:
        state = torch.tanh(self.stroke_init(torch.cat([z, context], -1)))
        return state.reshape(len(z), self.layers, self.hidden_dim).transpose(0, 1).contiguous()

    def decode_strokes(self, previous: Tensor, z: Tensor, context: Tensor,
                       hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        condition = torch.cat([z, context], -1).unsqueeze(1).expand(-1, previous.shape[1], -1)
        return self.stroke_decoder(torch.cat([self.input_dropout(previous), condition], -1),
                                   self.initial_state(z, context) if hidden is None else hidden)

    def local_prior(self, stroke_state: Tensor) -> tuple[Tensor, Tensor]:
        return gaussian_parameters(self.local_prior_head(stroke_state))

    def point_condition(self, stroke_state: Tensor, local_z: Tensor, anchor: Tensor) -> Tensor:
        return torch.cat([stroke_state, local_z, anchor], -1)

    def decode_points(self, previous: Tensor, condition: Tensor,
                      hidden: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        repeated = condition.unsqueeze(1).expand(-1, previous.shape[1], -1)
        initial = (torch.tanh(self.point_init(condition)).reshape(len(condition), self.layers, self.hidden_dim)
                   .transpose(0, 1).contiguous()) if hidden is None else hidden
        output, hidden = self.point_decoder(torch.cat([self.input_dropout(previous), repeated], -1), initial)
        return self.point_coordinate_head(output), self.point_end_head(output), hidden

    def batch_loss(self, sketches: list[list[np.ndarray]], prefix_counts: list[int],
                   beta: float, free_bits: float = 0.0, deterministic: bool = False,
                   teacher_forcing: float | None = None) -> dict[str, Tensor]:
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        ratio = float(self.config["teacher_forcing"] if teacher_forcing is None else teacher_forcing)
        if not 0 <= ratio <= 1:
            raise ValueError("teacher_forcing must be in [0, 1]")
        prefix_counts = prefix_counts_checked(sketches, prefix_counts)
        embeddings, counts = self.sketch_embeddings(sketches)
        prefixes = torch.tensor(prefix_counts if self.conditional else [0] * len(sketches), device=self.device)
        context = self.context_from_embeddings(embeddings, prefixes)
        q_mu, q_logvar = self.posterior_from_embeddings(embeddings, counts, context)
        p_mu, p_logvar = self.prior(context)
        z = reparameterize(q_mu, q_logvar, deterministic)
        # One extra slot predicts sketch EOS, after all real strokes.
        previous = F.pad(embeddings, (0, 0, 1, 0))
        states, _ = self.decode_strokes(previous, z, context)
        positions = torch.arange(states.shape[1], device=self.device)[None]
        slot_mask = (positions <= counts[:, None]) & (positions >= prefixes[:, None])
        slot_labels = (positions == counts[:, None]).long()
        slot_logits = self.sketch_end_head(states)
        sketch_weights = slot_logits.new_tensor(self.sketch_state_weights)
        slot_ce_values = F.cross_entropy(slot_logits.transpose(1, 2), slot_labels,
                                         weight=sketch_weights, reduction="none")
        ce_sum = torch.where(slot_mask, slot_ce_values, torch.zeros_like(slot_ce_values)).sum()
        correct_sum = ((slot_logits.argmax(-1) == slot_labels) & slot_mask).float().sum()
        event_count = slot_mask.sum().float()
        selected_states, selected_embeddings, selected_strokes = [], [], []
        for b, sketch in enumerate(sketches):
            first = prefix_counts[b] if self.conditional else 0
            for s, stroke in enumerate(sketch[first:], start=first):
                selected_states.append(states[b, s])
                selected_embeddings.append(embeddings[b, s])
                selected_strokes.append(stroke)
        zero = states.sum() * 0
        coordinate_nll, local_raw_kl, local_controlled_kl = zero, zero, zero
        coord_count = zero
        if selected_strokes:
            stroke_states = torch.stack(selected_states)
            target_embeddings = torch.stack(selected_embeddings)
            local_p_mu, local_p_logvar = self.local_prior(stroke_states)
            local_q_mu, local_q_logvar = gaussian_parameters(
                self.local_posterior_head(torch.cat([stroke_states, target_embeddings], -1)))
            local_z = reparameterize(local_q_mu, local_q_logvar, deterministic)
            anchors = torch.as_tensor(np.asarray([s[0] for s in selected_strokes]) / self.scale,
                                     device=self.device, dtype=torch.float32)
            anchor_nll = bivariate_mixture_nll(self.anchor_head(stroke_states), anchors)
            targets = []
            for stroke in selected_strokes:
                target = np.zeros((len(stroke), 3), dtype=np.float32)
                target[:-1, :2] = np.diff(np.asarray(stroke, dtype=np.float64), axis=0) / self.scale
                target[-1, 2] = 1
                targets.append(torch.as_tensor(target, device=self.device))
            target = pad_sequence(targets, batch_first=True)
            lengths = torch.tensor([len(s) for s in targets], device=self.device)
            point_mask = torch.arange(target.shape[1], device=self.device)[None] < lengths[:, None]
            coord_mask = point_mask & (target[..., 2] == 0)
            point_previous = torch.zeros_like(target)
            point_previous[:, 1:] = target[:, :-1]
            condition = self.point_condition(stroke_states, local_z, anchors)
            if not self.training or ratio == 1:
                point_raw, point_logits, _ = self.decode_points(point_previous, condition)
            else:
                raw_steps, pen_steps, hidden = [], [], None
                previous_step = point_previous[:, :1]
                for t in range(target.shape[1]):
                    step_raw, step_pen, hidden = self.decode_points(previous_step, condition, hidden)
                    raw_steps.append(step_raw)
                    pen_steps.append(step_pen)
                    if t + 1 < target.shape[1]:
                        predicted = feedback_token(step_raw[:, 0], step_pen[:, 0])
                        predicted = torch.cat([predicted[:, :2], predicted[:, -1:]], -1)
                        force = torch.rand(len(selected_strokes), device=self.device) < ratio
                        previous_step = torch.where(force[:, None], target[:, t], predicted).unsqueeze(1)
                point_raw, point_logits = torch.cat(raw_steps, 1), torch.cat(pen_steps, 1)
            point_nll = bivariate_mixture_nll(point_raw, target[..., :2])
            coord_count = coord_mask.sum() + len(selected_strokes)
            coordinate_nll = (anchor_nll.sum() + torch.where(coord_mask, point_nll, 0).sum()) / coord_count
            point_labels = target[..., 2].long()
            point_weights = point_logits.new_tensor(self.point_state_weights)
            point_ce = F.cross_entropy(point_logits.transpose(1, 2), point_labels,
                                       weight=point_weights, reduction="none")
            ce_sum = ce_sum + torch.where(point_mask, point_ce, 0).sum()
            correct_sum = correct_sum + ((point_logits.argmax(-1) == point_labels) & point_mask).float().sum()
            event_count = event_count + point_mask.sum()
            local_raw_kl = gaussian_kl(local_q_mu, local_q_logvar, local_p_mu, local_p_logvar).mean()
            local_controlled_kl = gaussian_kl(local_q_mu, local_q_logvar, local_p_mu,
                                             local_p_logvar, free_bits=free_bits).mean()
        global_raw_kl = gaussian_kl(q_mu, q_logvar, p_mu, p_logvar).mean()
        global_controlled_kl = gaussian_kl(q_mu, q_logvar, p_mu, p_logvar, free_bits=free_bits).mean()
        pen_ce = ce_sum / event_count.clamp_min(1)
        reconstruction = coordinate_nll + pen_ce
        controlled_kl = global_controlled_kl + local_controlled_kl
        loss = reconstruction + float(beta) * controlled_kl
        joint_reconstruction = (coordinate_nll * coord_count + ce_sum) / len(sketches)
        joint_kl = global_raw_kl + local_raw_kl * len(selected_strokes) / len(sketches)
        negative_elbo = joint_reconstruction + joint_kl
        return {"loss": loss, "total_loss": loss, "reconstruction_loss": reconstruction,
                "KL_loss": global_raw_kl + local_raw_kl, "auxiliary_loss": reconstruction * 0,
                "beta_effective": loss.new_tensor(float(beta)),
                "elbo": -negative_elbo, "negative_elbo": negative_elbo,
                "elbo_is_mc_estimate": loss.new_tensor(float(not deterministic and (not self.training or ratio == 1))),
                "joint_reconstruction_per_sketch": joint_reconstruction,
                "joint_kl_per_sketch": joint_kl,
                "reconstruction": reconstruction, "coordinate_nll": coordinate_nll,
                "pen_ce": pen_ce, "kl": global_raw_kl + local_raw_kl,
                "global_kl": global_raw_kl, "stroke_kl": local_raw_kl,
                "global_kl_objective": global_controlled_kl,
                "stroke_kl_objective": local_controlled_kl,
                "stroke_events": global_raw_kl.new_tensor(float(len(selected_strokes))),
                "pen_accuracy": correct_sum / event_count.clamp_min(1),
                "kl_objective": controlled_kl, "coordinate_events": coord_count.float(),
                "pen_events": event_count}
