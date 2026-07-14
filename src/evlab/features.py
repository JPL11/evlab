"""Causal per-window statistics for event-stream monitoring.

The 12-dimensional feature map used by runtime corruption monitors
(EvCorrupt-Bench style): each contiguous window of the stream is summarized
by statistics that respond to the documented sensor failure modes — rate
excursions (bursts, congestion), polarity skew (encoder faults), spatial
concentration (hot pixels), spatial coverage (dead regions), and temporal
regularity (flicker, leak events). All features are causal: a window's
features depend only on that window and the one before it.
"""

from __future__ import annotations

import numpy as np

from .formats import EventData

FEATURE_NAMES = [
    "event_rate",  # events per second in the window
    "polarity_fraction",  # share of ON events
    "active_pixel_fraction",  # pixels with >=1 event / all pixels
    "hot_top1_share",  # share of events from the single busiest pixel
    "hot_top_0p1pct_share",  # ... from the busiest 0.1% of pixels
    "hot_top_1pct_share",  # ... from the busiest 1% of pixels
    "spatial_entropy",  # entropy of the per-pixel count map / log(#pixels)
    "fano_1ms",  # variance/mean of event counts in 1 ms bins
    "fano_5ms",  # variance/mean of event counts in 5 ms bins
    "iet_cv",  # coefficient of variation of inter-event times
    "persist_pixel_frac",  # active pixels also active in previous window
    "persist_event_share",  # events landing on previous window's pixels
]


def _fano(t_rel: np.ndarray, window_us: int, bin_us: int) -> float:
    n_bins = max(window_us // bin_us, 1)
    counts = np.bincount(np.minimum(t_rel // bin_us, n_bins - 1), minlength=n_bins)
    mean = counts.mean()
    return float(counts.var() / mean) if mean > 0 else 0.0


def window_features(data: EventData, window_us: int = 50_000):
    """Compute the 12 per-window features.

    Returns ``(features, starts)`` where ``features`` has shape
    ``(n_windows, 12)`` (rows ordered as ``FEATURE_NAMES``) and ``starts``
    holds each window's start timestamp in microseconds.
    """
    ev = data.events
    n_px = data.width * data.height
    if len(ev) == 0 or n_px == 0:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, np.int64)

    t = ev["t"].astype(np.int64)
    t0 = int(t[0])
    win = ((t - t0) // window_us).astype(np.int64)
    n_win = int(win[-1]) + 1
    pix = ev["x"].astype(np.int64) * data.height + ev["y"].astype(np.int64)

    feats = np.zeros((n_win, len(FEATURE_NAMES)))
    starts = t0 + np.arange(n_win, dtype=np.int64) * window_us
    bounds = np.searchsorted(win, np.arange(n_win + 1))
    prev_active: np.ndarray | None = None

    for k in range(n_win):
        lo, hi = bounds[k], bounds[k + 1]
        n = hi - lo
        if n == 0:
            # Empty window: all-zero features; persistence resets.
            prev_active = np.zeros(0, np.int64)
            continue
        tk = t[lo:hi] - (t0 + k * window_us)
        pk = pix[lo:hi]
        counts = np.bincount(pk, minlength=0)
        active = np.flatnonzero(counts)
        per_px = counts[active].astype(float)
        sorted_px = np.sort(per_px)[::-1]

        top = lambda frac: sorted_px[: max(int(np.ceil(n_px * frac)), 1)].sum() / n
        probs = per_px / n
        entropy = float(-(probs * np.log(probs)).sum() / np.log(n_px))

        iet = np.diff(tk.astype(float))
        iet = iet[iet >= 0]
        iet_cv = float(iet.std() / iet.mean()) if len(iet) and iet.mean() > 0 else 0.0

        if prev_active is not None and len(prev_active):
            inter = np.intersect1d(active, prev_active, assume_unique=True)
            persist_px = len(inter) / len(active)
            persist_ev = float(np.isin(pk, prev_active).mean())
        else:
            persist_px = 0.0
            persist_ev = 0.0

        feats[k] = [
            n / (window_us / 1e6),
            float(ev["p"][lo:hi].mean()),
            len(active) / n_px,
            float(sorted_px[0] / n),
            float(top(0.001)),
            float(top(0.01)),
            entropy,
            _fano(tk, window_us, 1_000),
            _fano(tk, window_us, 5_000),
            iet_cv,
            persist_px,
            persist_ev,
        ]
        prev_active = active

    return feats, starts
