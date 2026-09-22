"""Exact absolute stroke-5 archive plus relative model coordinates.

Canonical columns: x/scale, y/scale, pen_down, stroke_end, sketch_end.
Model columns: dx/scale, dy/scale, pen_down, stroke_end, sketch_end.
A point owns its outgoing pen state; EOS is a separate zero-coordinate token.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import numpy as np

DEFAULT_SCALE = 256.0
ORIGIN = (0.0, 0.0)
DOWN, STROKE_END, EOS = 2, 3, 4


def _check_scale(scale: float) -> float:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return float(scale)


def _encode(strokes: Sequence[np.ndarray], scale: float, *, relative: bool) -> np.ndarray:
    """Encode all points, retaining order, with a final EOS; never resample.

    Canonical absolute tokens use float64 and a binary scale, preserving even
    near-zero float32 coordinates. Relative deltas can lose their low bits when
    subtracting a large preceding point, even before casting to model float32.
    """
    scale = _check_scale(scale)
    arrays = [np.asarray(stroke, dtype=np.float64) for stroke in strokes]
    for array in arrays:
        if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
            raise ValueError("Every stroke must have non-empty shape (N, 2)")
        if not np.isfinite(array).all():
            raise ValueError("Coordinates must be finite")
    tokens = np.zeros((sum(map(len, arrays)) + 1, 5), dtype=np.float64)
    previous = np.array(ORIGIN, dtype=np.float64)
    cursor = 0
    for array in arrays:
        n = len(array)
        if relative:
            tokens[cursor, :2] = (array[0] - previous) / scale
            if n > 1:
                tokens[cursor + 1:cursor + n, :2] = np.diff(array, axis=0) / scale
        else:
            tokens[cursor:cursor + n, :2] = array / scale
        tokens[cursor:cursor + n - 1, DOWN] = 1.0
        tokens[cursor + n - 1, STROKE_END] = 1.0
        previous = array[-1]
        cursor += n
    tokens[-1, EOS] = 1.0
    return tokens


def _decode(tokens: np.ndarray, scale: float, *, relative: bool) -> list[np.ndarray]:
    """Decode stroke-5, stopping at EOS and ignoring padding after it.

    State probabilities/logits are accepted using argmax. A finite unfinished
    stroke is retained when a generation cap stops a sequence. No clipping,
    smoothing, recentering, or prefix modification is performed.
    """
    scale = _check_scale(scale)
    tokens = np.asarray(tokens, dtype=np.float64)
    if tokens.ndim != 2 or tokens.shape[1] != 5:
        raise ValueError("tokens must have shape (T, 5)")
    if not np.isfinite(tokens).all():
        raise ValueError("Tokens must be finite")
    strokes: list[np.ndarray] = []
    current: list[np.ndarray] = []
    position = np.array(ORIGIN, dtype=np.float64)
    for token in tokens:
        state = int(token[2:].argmax())
        if state == 2:
            break
        position = position + token[:2] * scale if relative else token[:2] * scale
        current.append(position.copy())
        if state == 1:
            strokes.append(np.asarray(current, dtype=np.float64))
            current = []
    if current:
        strokes.append(np.asarray(current, dtype=np.float64))
    return strokes


def encode_strokes(strokes: Sequence[np.ndarray], scale: float = DEFAULT_SCALE) -> np.ndarray:
    """Canonical lossless ABSOLUTE stroke-5, with a separate EOS token."""
    return _encode(strokes, scale, relative=False)


def decode_tokens(tokens: np.ndarray, scale: float = DEFAULT_SCALE) -> list[np.ndarray]:
    """Decode canonical ABSOLUTE stroke-5 without changing geometry."""
    return _decode(tokens, scale, relative=False)


def encode_deltas(strokes: Sequence[np.ndarray], scale: float = DEFAULT_SCALE) -> np.ndarray:
    """Relative model view. Floating-point subtraction is not universally exact."""
    return _encode(strokes, scale, relative=True)


def decode_deltas(tokens: np.ndarray, scale: float = DEFAULT_SCALE) -> list[np.ndarray]:
    """Decode relative model tokens with cumulative motion from fixed origin."""
    return _decode(tokens, scale, relative=True)


def encode_sample(sample: dict, scale: float = DEFAULT_SCALE) -> dict:
    """Preserve the complete raw sample, including independent metadata copies."""
    return {"version": 1, "coordinate_mode": "absolute", "scale": _check_scale(scale),
            "tokens": encode_strokes(sample["strokes"], scale),
            "stroke_dtypes": [str(stroke.dtype) for stroke in sample["strokes"]],
            "metadata": deepcopy({key: value for key, value in sample.items() if key != "strokes"})}


def decode_sample(encoded: dict) -> dict:
    """Reconstruct raw fields and original array dtypes from encode_sample."""
    if encoded.get("version") != 1 or encoded.get("coordinate_mode") != "absolute":
        raise ValueError("Unsupported canonical sample representation")
    strokes = decode_tokens(encoded["tokens"], encoded["scale"])
    if len(strokes) != len(encoded["stroke_dtypes"]):
        raise ValueError("Stored stroke count does not match token boundaries")
    result = deepcopy(encoded["metadata"])
    result["strokes"] = [stroke.astype(dtype) for stroke, dtype in zip(strokes, encoded["stroke_dtypes"])]
    return result


def pad_sequences(sequences: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Pad complete sequences, with boolean valid mask including EOS.

    Padding is all zero and has no pen-state target; callers must use the mask.
    No arbitrary max-length parameter exists: no point can silently be truncated.
    """
    if not sequences:
        raise ValueError("Cannot pad an empty batch")
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    tokens = np.zeros((len(sequences), int(lengths.max()), 5), dtype=np.float32)
    mask = np.arange(int(lengths.max()))[None, :] < lengths[:, None]
    for index, sequence in enumerate(sequences):
        sequence = np.asarray(sequence)
        if sequence.ndim != 2 or sequence.shape[1] != 5:
            raise ValueError("Each sequence must have shape (T, 5)")
        tokens[index, :len(sequence)] = sequence
    return {"tokens": tokens, "mask": mask, "lengths": lengths}


def collate_sketches(samples: Sequence[dict], scale: float = DEFAULT_SCALE,
                     *, relative: bool = False) -> dict[str, np.ndarray]:
    encoder = encode_deltas if relative else encode_strokes
    return pad_sequences([encoder(sample["strokes"], scale) for sample in samples])


def prefix_token_count(strokes: Sequence[np.ndarray], prefix_count: int) -> int:
    """Number of non-EOS tokens in an observed prefix of complete strokes."""
    if not 0 <= prefix_count <= len(strokes):
        raise ValueError("prefix_count is outside the sketch")
    return sum(len(stroke) for stroke in strokes[:prefix_count])


def in_stroke_mask(tokens: np.ndarray) -> np.ndarray:
    """True only for moves that draw ink (not first points, moves, or EOS)."""
    tokens = np.asarray(tokens)
    states = tokens[..., 2:].argmax(axis=-1)
    mask = np.zeros(states.shape, dtype=bool)
    mask[..., 1:] = (states[..., :-1] == 0) & (states[..., 1:] != 2)
    return mask
