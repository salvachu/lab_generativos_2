"""Model-F-only arc-length view. Canonical arrays are never modified."""
from __future__ import annotations

import numpy as np
import torch

VIEW_VERSION = "f-anchor-relative-arclength-v1"


def stroke_view(stroke, samples=16, scale=256.0):
    a = np.asarray(stroke, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError("stroke must be a nonempty finite (N,2) array")
    if not isinstance(samples, int) or samples < 2 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("samples >= 2 and positive finite scale required")
    anchor = a[0].copy()
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))]
    t = np.linspace(0., 1., samples)
    if arc[-1] == 0:
        relative = np.zeros((samples, 2))
    else:
        # Consecutive coincident vertices have identical geometry; remove only
        # from this interpolation view to give np.interp strictly increasing t.
        keep = np.r_[True, np.diff(arc) > 0]
        relative = np.stack([np.interp(t, arc[keep]/arc[-1], (a-anchor)[keep, j]) for j in range(2)], -1)/scale
    return {"anchor": anchor/scale, "relative": relative, "t": t,
            "arc_length": float(arc[-1]), "original_points": len(a)}


def sketch_view(sketches, samples=16, scale=256.0, device="cpu"):
    if not sketches:
        raise ValueError("batch cannot be empty")
    k = max(1, max(map(len, sketches)))
    anchors = np.zeros((len(sketches), k, 2), dtype=np.float32)
    relative = np.zeros((len(sketches), k, samples, 2), dtype=np.float32)
    mask = np.zeros((len(sketches), k), dtype=bool)
    for b, sketch in enumerate(sketches):
        for j, stroke in enumerate(sketch):
            view = stroke_view(stroke, samples, scale)
            anchors[b, j], relative[b, j], mask[b, j] = view["anchor"], view["relative"], True
    return {"anchors": torch.as_tensor(anchors, device=device),
            "relative": torch.as_tensor(relative, device=device),
            "stroke_mask": torch.as_tensor(mask, device=device),
            "point_mask": torch.as_tensor(np.broadcast_to(mask[..., None], relative.shape[:-1]).copy(), device=device),
            "t": torch.linspace(0., 1., samples, device=device)}
