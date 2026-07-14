"""Tests for per-window monitoring features."""

import numpy as np

from evlab.corrupt import SEVERITIES, apply_schedule
from evlab.features import FEATURE_NAMES, window_features
from evlab.synth import moving_bar


def episode(name, start_us, end_us, severity="high"):
    return {
        "start_us": start_us,
        "end_us": end_us,
        "type": name,
        "severity": severity,
        "params": dict(SEVERITIES[name][severity]),
    }


def test_shapes_and_names():
    data = moving_bar(duration_us=500_000, seed=0)
    feats, starts = window_features(data, window_us=50_000)
    assert feats.shape == (10, len(FEATURE_NAMES))
    assert len(starts) == 10
    assert np.isfinite(feats).all()


def test_event_rate_and_polarity_exact():
    data = moving_bar(duration_us=200_000, signal_rate_hz=10_000, noise_rate_hz=0, seed=1)
    feats, _ = window_features(data, window_us=100_000)
    # Total events recovered from the rate column.
    total = (feats[:, FEATURE_NAMES.index("event_rate")] * 0.1).sum()
    assert abs(total - len(data.events)) < 1e-6
    pol = feats[:, FEATURE_NAMES.index("polarity_fraction")]
    assert ((0 <= pol) & (pol <= 1)).all()


def test_hot_pixels_raise_concentration():
    clean = moving_bar(duration_us=1_000_000, signal_rate_hz=20_000, noise_rate_hz=0, seed=2)
    hot = apply_schedule(clean, [episode("hot-pixels", 0, 1_000_000)], seed=0)
    fc, _ = window_features(clean, 100_000)
    fh, _ = window_features(hot, 100_000)
    i = FEATURE_NAMES.index("hot_top_1pct_share")
    assert fh[:, i].mean() > fc[:, i].mean() * 1.5


def test_dead_region_lowers_active_fraction():
    clean = moving_bar(duration_us=1_000_000, signal_rate_hz=20_000, noise_rate_hz=5_000, seed=3)
    dead = apply_schedule(clean, [episode("dead-region", 0, 1_000_000, "high")], seed=0)
    i = FEATURE_NAMES.index("active_pixel_fraction")
    fc, _ = window_features(clean, 100_000)
    fd, _ = window_features(dead, 100_000)
    assert fd[:, i].mean() < fc[:, i].mean()


def test_persistence_high_for_static_hot_pixels():
    clean = moving_bar(duration_us=1_000_000, signal_rate_hz=5_000, noise_rate_hz=0, seed=4)
    hot = apply_schedule(clean, [episode("hot-pixels", 0, 1_000_000)], seed=0)
    i = FEATURE_NAMES.index("persist_event_share")
    fh, _ = window_features(hot, 100_000)
    # Hot pixels fire in every window, so events increasingly land on
    # previously-active pixels (skip window 0, which has no predecessor).
    assert fh[1:, i].mean() > 0.5


def test_causality_prefix_invariance():
    # Features of the first k windows must not change when later events are
    # appended: truncate the stream and compare prefixes.
    data = moving_bar(duration_us=1_000_000, signal_rate_hz=10_000, seed=5)
    full, _ = window_features(data, 100_000)
    half_events = data.events[data.events["t"] < data.events["t"][0] + 500_000]
    from evlab.formats import EventData

    half = EventData(half_events.copy(), data.width, data.height)
    part, _ = window_features(half, 100_000)
    assert np.allclose(full[: len(part) - 1], part[: len(part) - 1])
