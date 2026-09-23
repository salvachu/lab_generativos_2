"""CoSE-inspired Hierarchical Conditional VAE, independently implemented.

Only stroke tokens are autoregressive. The local curve is evaluated in parallel.
See docs/MODEL_F.md for the distinction between paper ideas and this VAE.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from sketchlab.losses import MDNHead, bivariate_mixture_nll, gaussian_kl, gaussian_parameters, masked_mean, reparameterize
from sketchlab.stroke_view import VIEW_VERSION, sketch_view
from .common import prefix_counts_checked


def transformer(width, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(width, heads, 4*width, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)


def position_encoding(length, width, device, dtype):
    t = torch.arange(length, device=device, dtype=dtype)[:, None]
    f = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)*(-math.log(10000)/width))
    out = torch.zeros((length, width), device=device, dtype=dtype)
    out[:, 0::2] = torch.sin(t*f)
    out[:, 1::2] = torch.cos(t*f[:out[:, 1::2].shape[1]])
    return out


class StrokeEncoder(nn.Module):
    """Sees only t and anchor-relative coordinates, never absolute position."""
    def __init__(self, width, embedding_dim, heads=4, layers=2, dropout=0.):
        super().__init__()
        self.input = nn.Linear(3, width)
        self.attention = transformer(width, heads, layers, dropout)
        self.output = nn.Linear(width, embedding_dim)

    def forward(self, relative, t, mask=None):
        t = t.expand(*relative.shape[:-1], 1) if t.ndim == relative.ndim else t[None, :, None].expand(len(relative), -1, -1)
        if mask is None:
            mask = torch.ones(relative.shape[:2], device=relative.device, dtype=torch.bool)
        if not mask.any(-1).all():
            raise ValueError("each encoded stroke needs a valid point")
        x = self.attention(self.input(torch.cat([t, relative], -1)), src_key_padding_mask=~mask)
        pooled = torch.where(mask[..., None], x, 0.).sum(1)/mask.sum(1, keepdim=True)
        return self.output(pooled)


class StrokeDecoder(nn.Module):
    """Continuous anchored MLP. No previous point or recurrent hidden state."""
    def __init__(self, embedding_dim, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_dim+9, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))

    def features(self, t):
        frequencies = t.new_tensor([1., 2., 4., 8.]) * math.pi
        return torch.cat([t, torch.sin(t*frequencies), torch.cos(t*frequencies)], -1)

    def forward(self, embedding, t):
        t = t.reshape(1, -1, 1).expand(len(embedding), -1, -1)
        z = embedding[:, None].expand(-1, t.shape[1], -1)
        # Exact shape(0)=0, independent of evaluation grid and point count.
        return self.net(torch.cat([z, self.features(t)], -1)) - self.net(torch.cat([z, self.features(torch.zeros_like(t))], -1))


class ModelF(nn.Module):
    def __init__(self, config):
        super().__init__()
        defaults = {"stroke_samples":16, "stroke_embedding_dim":32, "attention_heads":4,
                    "stroke_encoder_layers":2, "composition_layers":2, "eos_weight":3.0,
                    "lambda_anchor":1.0, "lambda_embedding":1.0, "lambda_eos":1.0, "lambda_stroke":1.0}
        self.config = {**defaults, **config}
        self.model_name = "F"
        self.scale = float(config["scale"])
        self.hidden_dim, self.latent_dim = config["hidden_dim"], config["latent_dim"]
        self.stroke_samples = int(self.config["stroke_samples"])
        self.embedding_dim = int(self.config["stroke_embedding_dim"])
        h, d, heads = self.hidden_dim, self.embedding_dim, self.config["attention_heads"]
        if h % heads or self.stroke_samples < 2 or d < 1:
            raise ValueError("F requires hidden_dim divisible by heads, samples>=2 and embedding_dim>0")
        for key in ("eos_weight", "lambda_anchor", "lambda_embedding", "lambda_eos", "lambda_stroke"):
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        self.stroke_encoder = StrokeEncoder(h, d, heads, self.config["stroke_encoder_layers"], config["dropout"])
        self.stroke_decoder = StrokeDecoder(d, h)
        self.token_projection = nn.Linear(d+2, h)
        self.context_encoder = transformer(h, heads, self.config["composition_layers"], config["dropout"])
        self.prior_head = nn.Linear(h+1, 2*self.latent_dim)
        self.posterior_head = nn.Linear(2*(h+1), 2*self.latent_dim)
        self.condition_projection = nn.Linear(h+1+self.latent_dim, h)
        self.relational_decoder = transformer(h, heads, self.config["composition_layers"], config["dropout"])
        self.anchor_head = MDNHead(h, config["mixtures"], anchor=True)
        self.embedding_head = nn.Sequential(nn.Linear(h+2, h), nn.GELU(), nn.Linear(h, 2*d))
        self.sketch_end_head = nn.Linear(h, 2)
        self.training_stage = "composition"

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def representation_metadata(self):
        return {"view_version":VIEW_VERSION, "stroke_samples":self.stroke_samples,
                "parameterization":"normalized_arc_length", "scale":self.scale,
                "canonical_unchanged":True, "generation_points_per_stroke":self.stroke_samples}

    def configure_training(self, stage="composition", freeze_stroke_ae=True):
        if stage not in {"stroke_ae", "composition"}:
            raise ValueError("F training_stage must be stroke_ae or composition")
        self.training_stage = stage
        for name, p in self.named_parameters():
            local = name.startswith(("stroke_encoder.", "stroke_decoder."))
            p.requires_grad_(local if stage == "stroke_ae" else (not local or not freeze_stroke_ae))

    def train(self, mode=True):
        super().train(mode)
        if not next(self.stroke_encoder.parameters()).requires_grad:
            self.stroke_encoder.eval(); self.stroke_decoder.eval()
        return self

    def view(self, sketches):
        return sketch_view(sketches, self.stroke_samples, self.scale, self.device)

    def encode_view(self, view):
        mask = view["stroke_mask"]
        embeddings = view["relative"].new_zeros((*mask.shape, self.embedding_dim))
        if mask.any():
            embeddings[mask] = self.stroke_encoder(view["relative"][mask], view["t"])
        return embeddings

    def stroke_tokens(self, embeddings, anchors):
        return self.token_projection(torch.cat([embeddings, anchors], -1))

    def summarize_tokens(self, tokens, counts):
        # BOS at position 0 is always valid, including empty prefixes.
        inputs = F.pad(tokens, (0, 0, 1, 0))
        positions = torch.arange(inputs.shape[1], device=self.device)
        padding = positions[None] > counts[:, None]
        causal = positions[None, :] > positions[:, None]
        inputs = inputs + position_encoding(inputs.shape[1], self.hidden_dim, self.device, inputs.dtype)
        states = self.context_encoder(inputs, mask=causal, src_key_padding_mask=padding)
        summary = states[torch.arange(len(counts), device=self.device), counts]
        return torch.cat([summary, (counts > 0).to(summary.dtype)[:, None]], -1)

    def context(self, prefixes):
        view = self.view(prefixes)
        return self.summarize_tokens(self.stroke_tokens(self.encode_view(view), view["anchors"]), view["stroke_mask"].sum(-1))

    def prior(self, context):
        mu, logvar = gaussian_parameters(self.prior_head(context))
        nonempty = context[:, -1:] > 0
        return torch.where(nonempty, mu, 0.), torch.where(nonempty, logvar, 0.)

    def posterior(self, sketches, context):
        full = self.context(sketches)
        return gaussian_parameters(self.posterior_head(torch.cat([full, context], -1)))

    def decode_tokens(self, tokens, counts, z, context):
        inputs = F.pad(tokens, (0, 0, 1, 0))
        positions = torch.arange(inputs.shape[1], device=self.device)
        causal = positions[None, :] > positions[:, None]
        padding = positions[None] > counts[:, None]
        condition = self.condition_projection(torch.cat([context, z], -1))
        inputs = inputs + condition[:, None] + position_encoding(inputs.shape[1], self.hidden_dim, self.device, inputs.dtype)
        return self.relational_decoder(inputs, mask=causal, src_key_padding_mask=padding)

    def embedding_distribution(self, state, anchor):
        mu, logvar = gaussian_parameters(self.embedding_head(torch.cat([state, anchor], -1)))
        return mu, logvar.clamp(-6., 2.)

    def next_state(self, tokens, z, context):
        counts = torch.full((len(tokens),), tokens.shape[1], device=self.device, dtype=torch.long)
        return self.decode_tokens(tokens, counts, z, context)[:, -1]

    def batch_loss(self, sketches, prefix_counts, beta, free_bits=0., deterministic=False, teacher_forcing=None):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if teacher_forcing is not None and teacher_forcing != 1.:
            raise ValueError("F v1 trains at stroke-level teacher forcing=1")
        prefix_counts = prefix_counts_checked(sketches, prefix_counts)
        view = self.view(sketches); mask = view["stroke_mask"]; counts = mask.sum(-1)
        embedding = self.encode_view(view)
        reconstructed = self.stroke_decoder(embedding[mask], view["t"])
        ae = (reconstructed - view["relative"][mask]).square().mean() if mask.any() else embedding.sum()*0
        zero = ae*0
        anchor_nll = embedding_nll = eos_ce = raw_kl = controlled_kl = accuracy = zero
        slots = suffix_count = zero
        tp = fp = fn = tn = final_prob = nonfinal_prob = zero
        if self.training_stage != "stroke_ae":
            tokens = self.stroke_tokens(embedding, view["anchors"])
            prefixes = torch.tensor(prefix_counts, device=self.device)
            # Prefix summary is masked causally, never pooled across future tokens.
            context = self.summarize_tokens(tokens, prefixes)
            full = self.summarize_tokens(tokens, counts)
            q_mu, q_logvar = gaussian_parameters(self.posterior_head(torch.cat([full, context], -1)))
            p_mu, p_logvar = self.prior(context)
            z = reparameterize(q_mu, q_logvar, deterministic)
            states = self.decode_tokens(tokens, counts, z, context)
            pos = torch.arange(states.shape[1], device=self.device)[None]
            slot_mask = (pos >= prefixes[:, None]) & (pos <= counts[:, None])
            labels = (pos == counts[:, None]).long()
            logits = self.sketch_end_head(states)
            eos_ce = masked_mean(F.cross_entropy(logits.transpose(1, 2), labels,
                weight=logits.new_tensor([1., self.config["eos_weight"]]), reduction="none"), slot_mask)
            predicted = logits.argmax(-1)
            accuracy = masked_mean((predicted == labels).float(), slot_mask)
            positive = (labels == 1) & slot_mask; negative = (labels == 0) & slot_mask
            tp = ((predicted == 1) & positive).sum().float(); fn = ((predicted == 0) & positive).sum().float()
            fp = ((predicted == 1) & negative).sum().float(); tn = ((predicted == 0) & negative).sum().float()
            probabilities = logits.softmax(-1)[..., 1]
            final_prob = masked_mean(probabilities, positive); nonfinal_prob = masked_mean(probabilities, negative)
            suffix_mask = mask & (torch.arange(mask.shape[1], device=self.device)[None] >= prefixes[:, None])
            suffix_count = suffix_mask.sum().float(); slots = slot_mask.sum().float()
            if suffix_mask.any():
                target_anchor = view["anchors"][suffix_mask]
                selected = states[:, :-1][suffix_mask]
                anchor_nll = bivariate_mixture_nll(self.anchor_head(selected), target_anchor).mean()
                mu, logvar = self.embedding_distribution(selected, target_anchor)
                # A fixed target prevents collapse by moving the code toward its predictor.
                target = embedding[suffix_mask].detach()
                embedding_nll = .5 * (math.log(2*math.pi) + logvar + (target-mu).square()*(-logvar).exp()).mean()
            raw_kl = gaussian_kl(q_mu, q_logvar, p_mu, p_logvar).mean()
            controlled_kl = gaussian_kl(q_mu, q_logvar, p_mu, p_logvar, free_bits=free_bits).mean()
        coordinate = self.config["lambda_anchor"]*anchor_nll + self.config["lambda_embedding"]*embedding_nll
        rec = coordinate + self.config["lambda_eos"]*eos_ce
        auxiliary = self.config["lambda_stroke"]*ae
        loss = rec + auxiliary + beta*controlled_kl
        return {"loss":loss, "total_loss":loss, "reconstruction":rec+auxiliary, "reconstruction_loss":rec+auxiliary,
                "coordinate_nll":coordinate, "pen_ce":self.config["lambda_eos"]*eos_ce, "auxiliary_loss":auxiliary,
                "stroke_reconstruction":ae, "anchor_nll":anchor_nll, "embedding_nll":embedding_nll, "eos_ce":eos_ce,
                "kl":raw_kl, "KL_loss":raw_kl, "global_kl":raw_kl, "stroke_kl":zero,
                "global_kl_objective":controlled_kl, "stroke_kl_objective":zero, "kl_objective":controlled_kl,
                "beta_effective":loss.new_tensor(beta), "pen_accuracy":accuracy,
                "coordinate_events":suffix_count, "stroke_events":mask.sum().float(), "pen_events":slots,
                "eos_tp":tp, "eos_fp":fp, "eos_fn":fn, "eos_tn":tn,
                "eos_probability_final":final_prob, "eos_probability_nonfinal":nonfinal_prob,
                "eos_precision":tp/(tp+fp).clamp_min(1), "eos_recall":tp/(tp+fn).clamp_min(1)}

    @torch.no_grad()
    def validate_samples(self, samples, batch_size, beta, free_bits):
        # Event-weight components separately; padding and batch partitions must
        # not change the objective reported on the same held-out set.
        sums = {}; weights = {}; confusion = {k:0. for k in ("eos_tp", "eos_fp", "eos_fn", "eos_tn")}
        for start in range(0, len(samples), batch_size):
            sketches = [s["strokes"] for s in samples[start:start+batch_size]]
            from sketchlab.evaluation import prefix_count
            prefixes = [prefix_count(len(s), ["one","two",.25,.5,.75][(start+j)%5]) for j,s in enumerate(sketches)]
            result = self.batch_loss(sketches, prefixes, beta, free_bits, deterministic=True)
            for key, event in (("stroke_reconstruction","stroke_events"), ("anchor_nll","coordinate_events"),
                ("embedding_nll","coordinate_events"), ("eos_ce","pen_events"), ("pen_accuracy","pen_events"),
                ("global_kl",None), ("global_kl_objective",None), ("eos_probability_final",None),
                ("eos_probability_nonfinal","coordinate_events")):
                w = len(sketches) if event is None else float(result[event])
                sums[key] = sums.get(key,0.) + float(result[key])*w; weights[key] = weights.get(key,0.)+w
            for key in confusion: confusion[key] += float(result[key])
        if not samples: raise ValueError("Validation requires samples")
        m = {k:sums[k]/max(1.,weights[k]) for k in sums}; m.update(confusion)
        m["coordinate_nll"] = self.config["lambda_anchor"]*m["anchor_nll"] + self.config["lambda_embedding"]*m["embedding_nll"]
        m["pen_ce"] = self.config["lambda_eos"]*m["eos_ce"]
        m["auxiliary_loss"] = self.config["lambda_stroke"]*m["stroke_reconstruction"]
        m["reconstruction"] = m["reconstruction_loss"] = m["coordinate_nll"]+m["pen_ce"]+m["auxiliary_loss"]
        m["kl"] = m["KL_loss"] = m["global_kl"]
        m["stroke_kl"] = m["stroke_kl_objective"] = 0.
        m["kl_objective"] = m["global_kl_objective"]
        m["loss"] = m["total_loss"] = m["reconstruction"]+beta*m["kl_objective"]
        m["beta_effective"] = beta
        m["eos_precision"] = m["eos_tp"]/max(1.,m["eos_tp"]+m["eos_fp"])
        m["eos_recall"] = m["eos_tp"]/max(1.,m["eos_tp"]+m["eos_fn"])
        return m
