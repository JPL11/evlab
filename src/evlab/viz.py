"""Rendering helpers (require the [viz] extra)."""

from __future__ import annotations

import numpy as np

from .formats import EventData
from .representations import accumulate_frame, time_surface


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        return matplotlib
    except ImportError as e:
        raise ImportError("visualization requires: pip install evlab[viz]") from e


def render_frame(data: EventData, mode: str, out_path: str, **kwargs) -> None:
    """Render a single image (PNG) of the stream in the given mode."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if mode == "accumulate":
        img = accumulate_frame(data, clip=kwargs.get("clip", 3))
        cmap, vmin, vmax = "RdBu_r", -abs(img).max() or 1, abs(img).max() or 1
    elif mode == "time-surface":
        surf = time_surface(data, tau_us=kwargs.get("tau_us", 30000.0))
        img = surf[1] - surf[0]  # ON minus OFF
        cmap, vmin, vmax = "RdBu_r", -1, 1
    else:
        raise ValueError(f"unknown mode '{mode}' (accumulate, time-surface)")

    fig, ax = plt.subplots(figsize=(6, 6 * data.height / max(data.width, 1)))
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def render_gif(
    data: EventData, out_path: str, window_ms: float = 33.0, clip: int = 3, fps: int = 30
) -> int:
    """Render the stream as an animated GIF of accumulated frames.

    Returns the number of frames written.
    """
    _require_matplotlib()
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("GIF export requires: pip install evlab[viz]") from e

    ev = data.events
    if len(ev) == 0:
        raise ValueError("cannot render an empty stream")

    window_us = int(window_ms * 1000)
    t0, t1 = int(ev["t"][0]), int(ev["t"][-1])
    frames = []
    for start in range(t0, t1 + 1, window_us):
        mask = (ev["t"] >= start) & (ev["t"] < start + window_us)
        chunk = EventData(ev[mask].copy(), data.width, data.height)
        img = accumulate_frame(chunk, clip=clip).astype(np.float32)
        # Map [-clip, clip] -> grayscale with ON=white, OFF=black on gray bg
        norm = ((img + clip) / (2 * clip) * 255).astype(np.uint8)
        frames.append(Image.fromarray(norm, mode="L"))

    if not frames:
        raise ValueError("no frames produced; try a larger --window")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return len(frames)
