"""Synthetic event streams with ground-truth signal/noise labels.

Used by ``evlab synth`` and ``evlab denoise-bench`` to score denoising
filters with known ground truth: real labeled denoising datasets are rare,
and a controllable generator makes filter comparisons reproducible.
"""

from __future__ import annotations

import numpy as np

from .formats import EventData, from_arrays


def moving_bar(
    width: int = 240,
    height: int = 240,
    duration_us: int = 1_000_000,
    signal_rate_hz: float = 20_000.0,
    noise_rate_hz: float = 5_000.0,
    seed: int = 0,
) -> EventData:
    """A vertical bar sweeping left to right over uniform background noise.

    Returns EventData whose ``meta['signal']`` is a boolean array marking
    ground-truth signal events (True) vs. noise events (False), aligned
    with the t-sorted event array.
    """
    rng = np.random.default_rng(seed)

    n_sig = int(signal_rate_hz * duration_us / 1e6)
    t_sig = np.sort(rng.integers(0, duration_us, n_sig))
    # Bar leading edge moves across the full width over the recording.
    edge = (t_sig / duration_us * (width - 4)).astype(np.int64)
    x_sig = edge + rng.integers(0, 3, n_sig)
    y_sig = rng.integers(0, height, n_sig)
    # ON events at the leading edge, OFF at the trailing edge.
    p_sig = (rng.random(n_sig) < 0.5).astype(np.int8)

    n_noise = int(noise_rate_hz * duration_us / 1e6)
    t_noise = rng.integers(0, duration_us, n_noise)
    x_noise = rng.integers(0, width, n_noise)
    y_noise = rng.integers(0, height, n_noise)
    p_noise = rng.integers(0, 2, n_noise).astype(np.int8)

    x = np.concatenate([x_sig, x_noise])
    y = np.concatenate([y_sig, y_noise])
    t = np.concatenate([t_sig, t_noise])
    p = np.concatenate([p_sig, p_noise])
    is_signal = np.concatenate([np.ones(n_sig, bool), np.zeros(n_noise, bool)])

    # Sort everything by time with the same permutation from_arrays will
    # apply (stable argsort), so the label mask stays aligned.
    order = np.argsort(t, kind="stable")
    data = from_arrays(x[order], y[order], t[order], p[order], width, height)
    data.meta["signal"] = is_signal[order]
    return data


GENERATORS = {
    "moving-bar": moving_bar,
}
