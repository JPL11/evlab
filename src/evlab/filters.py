"""Event stream denoising filters.

Each filter has a mask function (returns the boolean keep-mask, used by
`evlab denoise-bench` to score against ground truth) and a public wrapper
that applies it. Returned EventData shares no mutable state with the input;
per-event metadata (like the ``signal`` labels) is subset alongside events.
"""

from __future__ import annotations

import numpy as np

from .formats import EventData

# Sentinel for "never fired": far in the past, but small enough that
# `t - sentinel` cannot overflow int64 for microsecond timestamps.
_NEVER = np.int64(np.iinfo(np.int64).min // 4)


def _apply_mask(data: EventData, keep: np.ndarray) -> EventData:
    meta = {}
    for key, value in data.meta.items():
        arr = np.asarray(value) if not np.isscalar(value) else None
        if arr is not None and arr.shape[:1] == (len(data.events),):
            meta[key] = arr[keep].copy()
        else:
            meta[key] = value
    return EventData(data.events[keep].copy(), data.width, data.height, meta)


def background_activity_mask(
    data: EventData, time_window_us: int = 5000, neighborhood: int = 1
) -> np.ndarray:
    """Keep-mask for the nearest-neighbor / background-activity filter.

    An event survives if at least one other event occurred within
    ``time_window_us`` in its ``(2*neighborhood+1)^2 - 1`` spatial
    neighborhood. This is the classic background-activity filter used by
    DVS pipelines.
    """
    ev = data.events
    keep = np.zeros(len(ev), dtype=bool)
    if len(ev) == 0:
        return keep

    # last_seen[y, x] = timestamp of the most recent event at that pixel
    last_seen = np.full(
        (data.height + 2 * neighborhood, data.width + 2 * neighborhood), _NEVER, dtype=np.int64
    )

    xs = ev["x"].astype(np.int64) + neighborhood
    ys = ev["y"].astype(np.int64) + neighborhood
    ts = ev["t"]

    n = neighborhood
    for i in range(len(ev)):
        x, y, t = xs[i], ys[i], ts[i]
        window = last_seen[y - n : y + n + 1, x - n : x + n + 1]
        # Exclude the center pixel: a pixel refiring alone is still noise.
        center = window[n, n]
        window[n, n] = _NEVER
        keep[i] = bool((t - window <= time_window_us).any())
        window[n, n] = center
        last_seen[y, x] = t
    return keep


def background_activity_filter(
    data: EventData, time_window_us: int = 5000, neighborhood: int = 1
) -> EventData:
    """Apply the background-activity filter (see `background_activity_mask`)."""
    return _apply_mask(data, background_activity_mask(data, time_window_us, neighborhood))


def refractory_mask(data: EventData, refractory_us: int = 1000) -> np.ndarray:
    """Keep-mask dropping events within ``refractory_us`` of the previous
    event at the same pixel (hot-pixel / oscillation suppression)."""
    ev = data.events
    keep = np.zeros(len(ev), dtype=bool)
    if len(ev) == 0:
        return keep

    last_seen = np.full((data.height, data.width), _NEVER, dtype=np.int64)
    xs = ev["x"].astype(np.intp)
    ys = ev["y"].astype(np.intp)
    ts = ev["t"]

    for i in range(len(ev)):
        x, y, t = xs[i], ys[i], ts[i]
        keep[i] = (t - last_seen[y, x]) > refractory_us
        if keep[i]:
            last_seen[y, x] = t
    return keep


def refractory_filter(data: EventData, refractory_us: int = 1000) -> EventData:
    """Apply the refractory filter (see `refractory_mask`)."""
    return _apply_mask(data, refractory_mask(data, refractory_us))


FILTERS = {
    "baf": background_activity_filter,
    "refractory": refractory_filter,
}

MASKS = {
    "baf": background_activity_mask,
    "refractory": refractory_mask,
}
