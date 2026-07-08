"""Dense representations of event streams (voxel grids, time surfaces, frames)."""

from __future__ import annotations

import numpy as np

from .formats import EventData


def voxel_grid(data: EventData, bins: int = 10) -> np.ndarray:
    """Accumulate events into a (bins, height, width) voxel grid.

    Polarity is signed (+1/-1) and events are bilinearly distributed across
    the two temporal bins they straddle, matching the common formulation
    from Zhu et al. (EV-FlowNet) used by most learning pipelines.
    """
    ev = data.events
    grid = np.zeros((bins, data.height, data.width), dtype=np.float32)
    if len(ev) == 0:
        return grid

    t = ev["t"].astype(np.float64)
    t0, t1 = t[0], t[-1]
    denom = max(t1 - t0, 1.0)
    # Scaled timestamp in [0, bins - 1]
    tau = (t - t0) / denom * (bins - 1)
    left = np.floor(tau).astype(np.int64)
    right = np.minimum(left + 1, bins - 1)
    w_right = tau - left
    w_left = 1.0 - w_right

    pol = ev["p"].astype(np.float32) * 2.0 - 1.0
    ys = ev["y"].astype(np.intp)
    xs = ev["x"].astype(np.intp)

    np.add.at(grid, (left, ys, xs), (pol * w_left).astype(np.float32))
    np.add.at(grid, (right, ys, xs), (pol * w_right).astype(np.float32))
    return grid


def time_surface(data: EventData, tau_us: float = 30000.0, at_t: int | None = None) -> np.ndarray:
    """Exponentially-decayed time surface, one channel per polarity.

    Returns a (2, height, width) float32 array where each pixel holds
    ``exp(-(at_t - t_last) / tau_us)`` for the most recent event of that
    polarity, or 0 if the pixel never fired.
    """
    ev = data.events
    surface = np.zeros((2, data.height, data.width), dtype=np.float32)
    if len(ev) == 0:
        return surface

    if at_t is None:
        at_t = int(ev["t"][-1])

    last = np.full((2, data.height, data.width), -np.inf, dtype=np.float64)
    idx = (ev["p"].astype(np.intp), ev["y"].astype(np.intp), ev["x"].astype(np.intp))
    # Later events overwrite earlier ones at the same pixel (events are t-sorted).
    last[idx] = ev["t"].astype(np.float64)

    fired = np.isfinite(last)
    surface[fired] = np.exp(-(at_t - last[fired]) / tau_us).astype(np.float32)
    return surface


def accumulate_frame(data: EventData, clip: int = 3) -> np.ndarray:
    """Signed event-count frame in [-clip, clip], shape (height, width)."""
    ev = data.events
    frame = np.zeros((data.height, data.width), dtype=np.int32)
    if len(ev) == 0:
        return frame
    pol = ev["p"].astype(np.int32) * 2 - 1
    np.add.at(frame, (ev["y"].astype(np.intp), ev["x"].astype(np.intp)), pol)
    return np.clip(frame, -clip, clip)
