"""Physically modeled corruption injection for event streams.

Implements the EvCorrupt-Bench corruption model: six failure modes of real
event sensors, injected as timed episodes with exact ground truth. Each
output event carries a label (0 = clean, else the corruption type id), and
the episode schedule can be exported/replayed as a JSON recipe so a
benchmark is exactly reproducible from the source recording.

Injected and modified timestamps are snapped to the recording's detected
clock quantum. Without this, injected events carry a finer timestamp grid
than the sensor's, and a detector can separate corrupted windows from the
grid alone rather than from the corruption.
"""

from __future__ import annotations

import json

import numpy as np

from .formats import EVENT_DTYPE, EventData

# Corruption type ids, stable across releases (0 means clean).
TYPES = {
    "hot-pixels": 1,
    "flicker": 2,
    "burst": 3,
    "dead-region": 4,
    "congestion": 5,
    "polarity": 6,
}
TYPE_NAMES = {v: k for k, v in TYPES.items()}

# Severity presets: documented failure-mode parameters, two levels each.
SEVERITIES = {
    "hot-pixels": {"low": {"pixels": 30, "rate_hz": 500.0}, "high": {"pixels": 150, "rate_hz": 2000.0}},
    "flicker": {"low": {"freq_hz": 100.0, "area": 0.3, "rate_scale": 1.0}, "high": {"freq_hz": 120.0, "area": 1.0, "rate_scale": 1.0}},
    "burst": {"low": {"rate_scale": 1.0}, "high": {"rate_scale": 3.0}},
    "dead-region": {"low": {"area": 0.15}, "high": {"area": 0.40}},
    "congestion": {"low": {"stall_us": 2000}, "high": {"stall_us": 10000}},
    "polarity": {"low": {"fraction": 0.7}, "high": {"fraction": 1.0}},
}


def detect_clock_quantum(t: np.ndarray) -> int:
    """The recording's timestamp grid in microseconds (>= 1).

    Computed as the GCD of a sample of positive inter-event deltas: 1 for
    native microsecond sensors, 1000 for millisecond-quantized conversions.
    """
    if len(t) < 3:
        return 1
    d = np.diff(t[: min(len(t), 100_000)])
    d = d[d > 0]
    if len(d) == 0:
        return 1
    return int(np.gcd.reduce(d.astype(np.int64)))


def make_schedule(
    duration_us: int,
    types: list[str],
    severity: str = "high",
    coverage: float = 0.45,
    episode_range_s: tuple[float, float] = (1.5, 3.0),
    seed: int = 0,
) -> list[dict]:
    """Draw non-overlapping corruption episodes covering ~``coverage`` of the
    stream, cycling through ``types``. Returns a list of episode dicts
    (start_us, end_us, type, severity, params) with exact onsets."""
    rng = np.random.default_rng(seed)
    lo, hi = (int(episode_range_s[0] * 1e6), int(episode_range_s[1] * 1e6))
    # Short recordings still get one episode, scaled to fit.
    if duration_us < lo:
        lo = max(duration_us // 3, 1)
        hi = max(duration_us // 2, 2)
    episodes = []
    corrupted = 0
    cursor = 0
    i = 0
    while corrupted < coverage * duration_us:
        length = int(rng.integers(lo, hi + 1))
        # Draw a gap so episodes spread over the stream instead of packing
        # at the front; keep it small enough to reach the target coverage.
        max_gap = max(int((duration_us * (1 - coverage)) / max(len(types), 2)), 1)
        gap = int(rng.integers(0, max_gap))
        start = cursor + gap
        end = start + length
        if end > duration_us:
            break
        name = types[i % len(types)]
        episodes.append(
            {
                "start_us": start,
                "end_us": end,
                "type": name,
                "severity": severity,
                "params": dict(SEVERITIES[name][severity]),
            }
        )
        corrupted += length
        cursor = end
        i += 1
    return episodes


def _snap(t: np.ndarray, quantum: int) -> np.ndarray:
    if quantum <= 1:
        return np.round(t).astype(np.int64)
    return (np.round(t / quantum) * quantum).astype(np.int64)


def _base_rate(data: EventData) -> float:
    dur = data.duration_us
    return len(data.events) / (dur / 1e6) if dur else 0.0


def apply_schedule(data: EventData, episodes: list[dict], seed: int = 0) -> EventData:
    """Apply corruption episodes to a stream.

    Returns new EventData whose ``meta['corruption']`` labels every output
    event: 0 for untouched events, else the type id that injected or
    modified it. Events dropped by dead regions are recorded only in the
    schedule (``meta['schedule']``), since they do not appear in the output.
    """
    rng = np.random.default_rng(seed)
    ev = data.events
    t0 = int(ev["t"][0]) if len(ev) else 0
    quantum = detect_clock_quantum(ev["t"])
    base_rate = _base_rate(data)
    W, H = data.width, data.height

    keep = np.ones(len(ev), bool)
    labels = np.zeros(len(ev), np.int8)
    t_out = ev["t"].astype(np.int64).copy()
    p_out = ev["p"].copy()
    injected = []  # (x, y, t, p, type_id) arrays per episode

    for epi in episodes:
        a, b = t0 + epi["start_us"], t0 + epi["end_us"]
        name, prm = epi["type"], epi["params"]
        tid = TYPES[name]
        in_epi = (t_out >= a) & (t_out < b)

        if name == "hot-pixels":
            k, rate = int(prm["pixels"]), float(prm["rate_hz"])
            px = rng.integers(0, W, k)
            py = rng.integers(0, H, k)
            period = 1e6 / rate
            for j in range(k):
                # Quasi-periodic: per-interval jitter around the pixel period.
                n = int((b - a) / period) + 1
                gaps = period * (1 + rng.uniform(-0.2, 0.2, n))
                ts = a + np.cumsum(gaps)
                ts = ts[ts < b]
                pol = (rng.random(len(ts)) < 0.9).astype(np.int8)  # ON-dominant leak
                injected.append((np.full(len(ts), px[j]), np.full(len(ts), py[j]), ts, pol, tid))

        elif name == "flicker":
            f, area, scale = float(prm["freq_hz"]), float(prm["area"]), float(prm["rate_scale"])
            side = np.sqrt(area)
            rw, rh = max(int(W * side), 1), max(int(H * side), 1)
            rx, ry = rng.integers(0, W - rw + 1), rng.integers(0, H - rh + 1)
            # Rate follows |d/dt log L| for L = L0(1 + m sin 2pi f t):
            # proportional to |cos 2pi f t|. Thin a homogeneous Poisson
            # stream by that profile; ON while intensity rises (cos > 0).
            mean_rate = scale * base_rate * area
            n = rng.poisson(mean_rate * (b - a) / 1e6 * (np.pi / 2))
            ts = rng.uniform(a, b, n)
            phase = np.cos(2 * np.pi * f * (ts - a) / 1e6)
            sel = rng.random(n) < np.abs(phase)
            ts = ts[sel]
            pol = (phase[sel] > 0).astype(np.int8)
            xs = rng.integers(rx, rx + rw, len(ts))
            ys = rng.integers(ry, ry + rh, len(ts))
            injected.append((xs, ys, ts, pol, tid))

        elif name == "burst":
            scale = float(prm["rate_scale"])
            n = rng.poisson(scale * base_rate * (b - a) / 1e6)
            ts = rng.uniform(a, b, n)
            injected.append(
                (rng.integers(0, W, n), rng.integers(0, H, n), ts, rng.integers(0, 2, n).astype(np.int8), tid)
            )

        elif name == "dead-region":
            area = float(prm["area"])
            side = np.sqrt(area)
            rw, rh = max(int(W * side), 1), max(int(H * side), 1)
            rx, ry = rng.integers(0, W - rw + 1), rng.integers(0, H - rh + 1)
            inside = in_epi & (ev["x"] >= rx) & (ev["x"] < rx + rw) & (ev["y"] >= ry) & (ev["y"] < ry + rh)
            keep &= ~inside
            epi["params"] = {**prm, "x": int(rx), "y": int(ry), "w": int(rw), "h": int(rh)}

        elif name == "congestion":
            stall = int(prm["stall_us"])
            # Stall-and-release: events accumulate for a stall period and are
            # released together at its end. Order-preserving, one-sided.
            tt = t_out[in_epi]
            t_out[in_epi] = np.minimum(((tt - a) // stall + 1) * stall + a, b - 1)
            labels[in_epi] = tid

        elif name == "polarity":
            frac = float(prm["fraction"])
            hit = in_epi & (rng.random(len(ev)) < frac)
            p_out[hit] = 1
            labels[hit] = tid

        else:
            raise ValueError(f"unknown corruption type: {name}")

    # Assemble surviving originals + injected events, snap injected/jittered
    # timestamps onto the recording's clock grid, and re-sort.
    xs = [ev["x"][keep]]
    ys = [ev["y"][keep]]
    ts = [_snap(t_out[keep], quantum)]
    ps = [p_out[keep]]
    ls = [labels[keep]]
    for x_i, y_i, t_i, p_i, tid in injected:
        xs.append(np.asarray(x_i, np.int64))
        ys.append(np.asarray(y_i, np.int64))
        ts.append(_snap(np.asarray(t_i), quantum))
        ps.append(np.asarray(p_i, np.int8))
        ls.append(np.full(len(t_i), tid, np.int8))

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    p = np.concatenate(ps)
    lab = np.concatenate(ls)
    order = np.argsort(t, kind="stable")

    out = np.empty(len(t), EVENT_DTYPE)
    out["x"], out["y"], out["t"], out["p"] = x[order], y[order], t[order], p[order]
    result = EventData(events=out, width=W, height=H)
    result.meta["corruption"] = lab[order]
    result.meta["schedule"] = episodes
    return result


def save_recipe(episodes: list[dict], seed: int, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"version": 1, "seed": seed, "episodes": episodes}, f, indent=2)


def load_recipe(path: str) -> tuple[list[dict], int]:
    with open(path) as f:
        r = json.load(f)
    return r["episodes"], int(r.get("seed", 0))
