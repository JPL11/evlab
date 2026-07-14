"""Tests for the corruption injection suite."""

import json

import numpy as np
import pytest

from evlab import formats
from evlab.corrupt import (
    SEVERITIES,
    TYPES,
    apply_schedule,
    detect_clock_quantum,
    load_recipe,
    make_schedule,
    save_recipe,
)
from evlab.synth import moving_bar


@pytest.fixture()
def stream():
    # 2s recording so multi-second episodes fit.
    return moving_bar(duration_us=2_000_000, signal_rate_hz=50_000, noise_rate_hz=0, seed=1)


def episode(name, start_us, end_us, severity="high", **overrides):
    params = dict(SEVERITIES[name][severity])
    params.update(overrides)
    return {
        "start_us": start_us,
        "end_us": end_us,
        "type": name,
        "severity": severity,
        "params": params,
    }


def test_schedule_coverage_and_determinism():
    a = make_schedule(10_000_000, sorted(TYPES), coverage=0.45, seed=7)
    b = make_schedule(10_000_000, sorted(TYPES), coverage=0.45, seed=7)
    assert a == b
    covered = sum(e["end_us"] - e["start_us"] for e in a)
    assert 0.35 <= covered / 10_000_000 <= 0.60
    # Episodes are disjoint and ordered.
    for prev, nxt in zip(a, a[1:]):
        assert prev["end_us"] <= nxt["start_us"]


def test_apply_is_deterministic(stream):
    eps = make_schedule(stream.duration_us, ["burst", "hot-pixels"], seed=3)
    r1 = apply_schedule(stream, eps, seed=3)
    r2 = apply_schedule(stream, eps, seed=3)
    assert np.array_equal(r1.events, r2.events)
    assert np.array_equal(r1.meta["corruption"], r2.meta["corruption"])


def test_labels_align_and_only_inside_episodes(stream):
    eps = [episode("burst", 500_000, 1_000_000)]
    out = apply_schedule(stream, eps, seed=0)
    lab = out.meta["corruption"]
    assert len(lab) == len(out.events)
    t = out.events["t"].astype(np.int64) - int(stream.events["t"][0])
    corrupted = lab == TYPES["burst"]
    assert corrupted.any()
    assert (t[corrupted] >= 500_000).all() and (t[corrupted] < 1_000_000).all()
    # Outside the episode nothing is labeled.
    outside = (t < 500_000) | (t >= 1_000_000)
    assert (lab[outside] == 0).all()


def test_burst_rate_scales(stream):
    lo = apply_schedule(stream, [episode("burst", 0, 2_000_000, "low")], seed=0)
    hi = apply_schedule(stream, [episode("burst", 0, 2_000_000, "high")], seed=0)
    n_lo = int((lo.meta["corruption"] != 0).sum())
    n_hi = int((hi.meta["corruption"] != 0).sum())
    assert 2.0 < n_hi / n_lo < 4.0  # 3x vs 1x base rate


def test_dead_region_drops_events(stream):
    out = apply_schedule(stream, [episode("dead-region", 0, 2_000_000)], seed=0)
    prm = out.meta["schedule"][0]["params"]
    ev = out.events
    inside = (
        (ev["x"] >= prm["x"])
        & (ev["x"] < prm["x"] + prm["w"])
        & (ev["y"] >= prm["y"])
        & (ev["y"] < prm["y"] + prm["h"])
    )
    assert not inside.any()
    assert len(out.events) < len(stream.events)


def test_congestion_preserves_order_and_count(stream):
    out = apply_schedule(stream, [episode("congestion", 0, 2_000_000)], seed=0)
    assert len(out.events) == len(stream.events)
    t = out.events["t"].astype(np.int64)
    assert (np.diff(t) >= 0).all()
    # Released timestamps land on stall boundaries.
    stall = SEVERITIES["congestion"]["high"]["stall_us"]
    rel = t - int(stream.events["t"][0])
    lab = out.meta["corruption"]
    on_boundary = (rel[lab == TYPES["congestion"]] % stall == 0) | (
        rel[lab == TYPES["congestion"]] == 2_000_000 - 1
    )
    assert on_boundary.all()


def test_polarity_forces_single_polarity(stream):
    out = apply_schedule(stream, [episode("polarity", 0, 2_000_000, "high")], seed=0)
    lab = out.meta["corruption"]
    assert (out.events["p"][lab == TYPES["polarity"]] == 1).all()
    # High severity affects every event in the episode.
    assert (lab == TYPES["polarity"]).sum() == len(stream.events)


def test_hot_pixels_confined_to_k_pixels(stream):
    out = apply_schedule(stream, [episode("hot-pixels", 0, 2_000_000, "low")], seed=0)
    lab = out.meta["corruption"]
    hot = out.events[lab == TYPES["hot-pixels"]]
    coords = {(int(x), int(y)) for x, y in zip(hot["x"], hot["y"])}
    assert len(coords) <= SEVERITIES["hot-pixels"]["low"]["pixels"]
    # Leak events are ON-dominant.
    assert hot["p"].mean() > 0.8


def test_timestamps_snapped_to_clock_quantum():
    # Millisecond-quantized recording: injected events must land on the grid.
    base = moving_bar(duration_us=2_000_000, signal_rate_hz=5_000, noise_rate_hz=0, seed=2)
    ev = base.events.copy()
    ev["t"] = (ev["t"] // 1000) * 1000
    ms = formats.EventData(events=ev, width=base.width, height=base.height)
    assert detect_clock_quantum(ms.events["t"]) == 1000
    out = apply_schedule(ms, [episode("burst", 0, 2_000_000)], seed=0)
    assert (out.events["t"] % 1000 == 0).all()


def test_recipe_roundtrip(tmp_path, stream):
    eps = make_schedule(stream.duration_us, ["flicker", "dead-region"], seed=11)
    p = tmp_path / "recipe.json"
    save_recipe(eps, 11, str(p))
    eps2, seed2 = load_recipe(str(p))
    assert seed2 == 11 and eps2 == eps
    r1 = apply_schedule(stream, eps, seed=11)
    r2 = apply_schedule(stream, eps2, seed=seed2)
    assert np.array_equal(r1.events, r2.events)
    json.loads(p.read_text())  # valid JSON


def test_labels_survive_npz_roundtrip(tmp_path, stream):
    out = apply_schedule(stream, [episode("burst", 0, 1_000_000)], seed=0)
    p = str(tmp_path / "c.npz")
    formats.save(out, p)
    back = formats.load(p)
    assert np.array_equal(back.meta["corruption"], out.meta["corruption"])
