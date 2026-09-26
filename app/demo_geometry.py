"""Geometry helpers used by the historical H13/H20 demo checkpoints."""

from __future__ import annotations

import numpy as np

from sketchlab.stroke_view import stroke_view


SIZE = 40


def points(strokes):
    parts = []
    for stroke in strokes:
        path = np.asarray(stroke, dtype=np.float32)
        if len(path) == 0:
            continue
        for a, b in zip(path[:-1], path[1:]):
            k = max(2, int(np.ceil(np.linalg.norm(b - a) / 4)))
            parts.append(a[None, :] + np.linspace(0, 1, k, endpoint=False, dtype=np.float32)[:, None] * (b - a)[None, :])
        parts.append(path[-1:])
    return np.concatenate(parts) if parts else np.empty((0, 2), dtype=np.float32)


def feature(strokes, canvas=256):
    xy = points(strokes)
    grid = np.zeros((SIZE, SIZE), dtype=np.float32)
    if len(xy) == 0:
        return grid.ravel()
    ij = np.clip(np.floor(xy * SIZE / canvas).astype(int), 0, SIZE - 1)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            shifted = np.clip(ij + [dx, dy], 0, SIZE - 1)
            grid[shifted[:, 1], shifted[:, 0]] = 1.
    return grid.ravel()


def register(exemplar_prefix, target_prefix, remaining):
    a, b = points(exemplar_prefix), points(target_prefix)
    if len(a) == 0 or len(b) == 0:
        return [np.asarray(s) for s in remaining]
    delta = np.median(b, axis=0) - np.median(a, axis=0)
    return [np.asarray(s, dtype=np.float32) + delta for s in remaining]


def register_fit(exemplar_prefix, target_prefix, remaining, canvas=512):
    a, b = points(exemplar_prefix), points(target_prefix)
    if len(a) == 0 or len(b) == 0:
        return [np.asarray(s, dtype=np.float32) for s in remaining]
    source = np.median(a, axis=0)
    target = np.median(b, axis=0)
    source_spread = np.percentile(a, 90, axis=0) - np.percentile(a, 10, axis=0)
    target_spread = np.percentile(b, 90, axis=0) - np.percentile(b, 10, axis=0)
    ratio = target_spread[source_spread > 8] / source_spread[source_spread > 8]
    scale = float(np.clip(np.median(ratio), .7, 1.3)) if len(ratio) else 1.
    raw = [np.asarray(s, dtype=np.float32) for s in remaining]
    if not raw:
        return raw
    all_points = np.concatenate(raw)
    deviation = (all_points - source) * scale
    factor = 1.
    for axis in range(2):
        high = float(max(deviation[:, axis].max(), 0.))
        low = float(max(-deviation[:, axis].min(), 0.))
        if high > 0:
            factor = min(factor, max(0., (canvas - 4. - target[axis]) / high))
        if low > 0:
            factor = min(factor, max(0., (target[axis] - 4.) / low))
    scale *= max(0., min(1., factor))
    return [((stroke - source) * scale + target).astype(np.float32) for stroke in raw]


def resample(stroke):
    view = stroke_view(stroke, samples=16)
    return (view['relative'] + view['anchor']) * 256.
