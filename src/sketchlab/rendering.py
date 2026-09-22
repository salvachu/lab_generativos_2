"""Vector-first rendering with explicit stroke boundaries and prefix colors."""
from __future__ import annotations

from html import escape
from pathlib import Path
from collections.abc import Sequence
import math
import numpy as np

# A train-derived display envelope, not a claim about the original drawing UI.
DEFAULT_CANVAS = (-4.0, 508.0, -4.0, 508.0)
PREFIX_COLOR = "#1976d2"
GENERATED_COLOR = "#e66924"


def _bounds(canvas) -> tuple[float, float, float, float]:
    if np.isscalar(canvas):
        return (0.0, float(canvas), 0.0, float(canvas))
    if len(canvas) == 2:
        return (0.0, float(canvas[0]), 0.0, float(canvas[1]))
    if len(canvas) != 4:
        raise ValueError("canvas must be a size, (width,height), or (xmin,xmax,ymin,ymax)")
    return tuple(map(float, canvas))


def render_svg(strokes: Sequence[np.ndarray], prefix_count: int = 0,
               canvas=DEFAULT_CANVAS, *, title: str = "Vector sketch",
               show_order: bool = False) -> str:
    xmin, xmax, ymin, ymax = _bounds(canvas)
    width, height = xmax - xmin, ymax - ymin
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin:g} {ymin:g} {width:g} {height:g}" width="512" height="512" role="img">',
              f'<title>{escape(title)}</title>',
              f'<rect x="{xmin:g}" y="{ymin:g}" width="{width:g}" height="{height:g}" fill="white"/>']
    for index, stroke in enumerate(strokes):
        array = np.asarray(stroke)
        if len(array) == 0:
            continue
        color = PREFIX_COLOR if index < prefix_count else GENERATED_COLOR
        # 17 significant digits round-trip finite float64 coordinates in exports.
        points = " ".join(f"{float(x):.17g},{float(y):.17g}" for x, y in array)
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
        if len(array) == 1:
            pieces.append(f'<circle cx="{array[0, 0]:.17g}" cy="{array[0, 1]:.17g}" r="1" fill="{color}"/>')
        if show_order:
            pieces.append(f'<text x="{array[0, 0]:.17g}" y="{array[0, 1]:.17g}" font-size="10" fill="#637183">{index + 1}</text>')
    pieces.append('</svg>')
    return "\n".join(pieces)


def render_grid(sketches: Sequence, path: str | Path, titles: Sequence[str] | None = None,
                prefix_counts: Sequence[int] | None = None, canvas=DEFAULT_CANVAS,
                *, ncols: int = 4, show_order: bool = False) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not sketches:
        raise ValueError("Cannot render an empty grid")
    ncols = min(ncols, len(sketches))
    nrows = math.ceil(len(sketches) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows), squeeze=False)
    xmin, xmax, ymin, ymax = _bounds(canvas)
    for index, ax in enumerate(axes.ravel()):
        if index >= len(sketches):
            ax.set_visible(False)
            continue
        sample = sketches[index]
        strokes = sample["strokes"] if isinstance(sample, dict) else sample
        observed = prefix_counts[index] if prefix_counts is not None else 0
        for stroke_index, stroke in enumerate(strokes):
            array = np.asarray(stroke)
            color = PREFIX_COLOR if stroke_index < observed else GENERATED_COLOR
            ax.plot(array[:, 0], array[:, 1], color=color, lw=1.1,
                    marker="." if len(array) == 1 else None)
            if show_order:
                ax.text(*array[0], str(stroke_index + 1), fontsize=5, color="#637183")
        ax.set(xlim=(xmin, xmax), ylim=(ymax, ymin), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None:
            ax.set_title(titles[index], fontsize=8, wrap=True)
    figure.tight_layout(pad=0.7)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor="white")
    plt.close(figure)
    return path


def render(strokes, path: str | Path | None = None, **kwargs):
    """Return an SVG string, optionally saving it on disk."""
    svg = render_svg(strokes, **kwargs)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8")
    return svg
