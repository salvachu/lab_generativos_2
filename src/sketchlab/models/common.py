"""Shared variable-length batching; no training sketch is truncated."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from sketchlab.representation import encode_deltas


def validate_sketch(strokes: Sequence[np.ndarray]) -> list[np.ndarray]:
    result = []
    for stroke in strokes:
        a = np.asarray(stroke)
        if a.ndim != 2 or a.shape[1] != 2 or not len(a) or not np.isfinite(a).all():
            raise ValueError("Each stroke must be a nonempty finite (N, 2) array")
        result.append(a)
    return result


def tokens(strokes: Sequence[np.ndarray], scale: float = 256.0) -> np.ndarray:
    """Relative model view; canonical absolute archive lives in representation."""
    return encode_deltas(strokes, scale)


def batch_tokens(sketches: Sequence[Sequence[np.ndarray]], device: torch.device,
                 scale: float = 256.0) -> tuple[Tensor, Tensor, Tensor]:
    if not sketches:
        raise ValueError("A batch must contain at least one sketch")
    rows = [torch.as_tensor(tokens(s, scale), device=device, dtype=torch.float32) for s in sketches]
    lengths = torch.tensor([len(t) for t in rows], device=device, dtype=torch.long)
    padded = pad_sequence(rows, batch_first=True)
    mask = torch.arange(padded.shape[1], device=device)[None] < lengths[:, None]
    return padded, lengths, mask


def encode_packed(gru: nn.GRU, padded: Tensor, lengths: Tensor) -> Tensor:
    packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, hidden = gru(packed)
    directions = 2 if gru.bidirectional else 1
    return hidden[-directions:].transpose(0, 1).reshape(padded.shape[0], -1)


def feedback_token(raw: Tensor, pen: Tensor) -> Tensor:
    """Detached modal-mixture mean plus categorical mode for scheduled sampling."""
    component = raw[..., 0].argmax(-1)
    means = raw[..., 1:3]
    selected = means.gather(-2, component[..., None, None].expand(*component.shape, 1, 2)).squeeze(-2)
    state = torch.nn.functional.one_hot(pen.argmax(-1), pen.shape[-1]).to(selected.dtype)
    return torch.cat([selected, state], -1).detach()


def prefix_counts_checked(sketches: Sequence[Sequence[np.ndarray]], prefix_counts: Sequence[int]) -> list[int]:
    if len(sketches) != len(prefix_counts):
        raise ValueError("prefix_counts must match batch size")
    result = []
    for s, c in zip(sketches, prefix_counts):
        if int(c) != c or c < 0 or c > len(s):
            raise ValueError("Prefix count must be between zero and sketch length")
        result.append(int(c))
    return result
