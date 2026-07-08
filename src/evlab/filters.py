"""Event stream denoising filters.

All filters take and return :class:`~evlab.formats.EventData`; the returned
object shares no mutable state with the input.
"""

from __future__ import annotations

import numpy as np

from .formats import EventData


def background_activity_filter(
    data: EventData, time_window_us: int = 5000, neighborhood: int = 1
) -> EventData:
    """Remove isolated noise events (nearest-neighbor / BAF filter).

    An event survives if at least one other event occurred within
    ``time_window_us`` in its ``(2*neighborhood+1)^2 - 1`` spatial
    neighborhood. This is the classic background-activity filter used by
    DVS pipelines.
    """
    ev = data.events
    if len(ev) == 0:
        return EventData(ev.copy(), data.width, data.height, dict(data.meta))

    # Sentinel for "never fired": far in the past, but small enough that
    # `t - sentinel` cannot overflow int64 for microsecond timestamps.
    never = np.int64(np.iinfo(np.int64).min // 4)

    # last_seen[y, x] = timestamp of the most recent event at that pixel
    last_seen = np.full(
        (data.height + 2 * neighborhood, data.width + 2 * neighborhood), never, dtype=np.int64
    )
    keep = np.zeros(len(ev), dtype=bool)

    xs = ev["x"].astype(np.int64) + neighborhood
    ys = ev["y"].astype(np.int64) + neighborhood
    ts = ev["t"]

    n = neighborhood
    for i in range(len(ev)):
        x, y, t = xs[i], ys[i], ts[i]
        window = last_seen[y - n : y + n + 1, x - n : x + n + 1]
        # Exclude the center pixel: a pixel refiring alone is still noise.
        center = window[n, n]
        window[n, n] = never
        keep[i] = bool((t - window <= time_window_us).any())
        window[n, n] = center
        last_seen[y, x] = t

    return EventData(ev[keep].copy(), data.width, data.height, dict(data.meta))


def refractory_filter(data: EventData, refractory_us: int = 1000) -> EventData:
    """Drop events that fire within ``refractory_us`` of the previous event
    at the same pixel (hot-pixel / oscillation suppression)."""
    ev = data.events
    if len(ev) == 0:
        return EventData(ev.copy(), data.width, data.height, dict(data.meta))

    never = np.int64(np.iinfo(np.int64).min // 4)
    last_seen = np.full((data.height, data.width), never, dtype=np.int64)
    keep = np.zeros(len(ev), dtype=bool)

    xs = ev["x"].astype(np.intp)
    ys = ev["y"].astype(np.intp)
    ts = ev["t"]

    for i in range(len(ev)):
        x, y, t = xs[i], ys[i], ts[i]
        keep[i] = (t - last_seen[y, x]) > refractory_us
        if keep[i]:
            last_seen[y, x] = t

    return EventData(ev[keep].copy(), data.width, data.height, dict(data.meta))


FILTERS = {
    "baf": background_activity_filter,
    "refractory": refractory_filter,
}
