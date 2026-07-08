"""Metrics for comparing and characterizing event streams."""

from __future__ import annotations

import numpy as np

from .formats import EventData


def summary(data: EventData) -> dict:
    """Basic stream statistics used by `evlab info`."""
    ev = data.events
    n = len(ev)
    out = {
        "num_events": n,
        "width": data.width,
        "height": data.height,
        "duration_s": data.duration_us / 1e6,
        "event_rate_hz": data.event_rate,
    }
    if n:
        out["t_start_us"] = int(ev["t"][0])
        out["t_end_us"] = int(ev["t"][-1])
        out["polarity_balance"] = float(ev["p"].mean())
        active = len(np.unique(ev["y"].astype(np.int64) * data.width + ev["x"]))
        out["active_pixels"] = active
        out["active_pixel_fraction"] = active / max(data.width * data.height, 1)
    return out


def retention(original: EventData, filtered: EventData) -> float:
    """Fraction of events surviving a filter."""
    if len(original.events) == 0:
        return 1.0
    return len(filtered.events) / len(original.events)


def event_structural_ratio(data: EventData, patch: int = 8) -> float:
    """Ratio of event mass in the densest patches vs. total (0..1).

    A crude but dependency-free proxy for signal structure: real scenes
    concentrate events on edges/objects; uniform sensor noise spreads them
    evenly. Higher = more structured.
    """
    ev = data.events
    if len(ev) == 0:
        return 0.0
    counts = np.zeros((data.height, data.width), dtype=np.int64)
    np.add.at(counts, (ev["y"].astype(np.intp), ev["x"].astype(np.intp)), 1)

    ph = max(data.height // patch, 1)
    pw = max(data.width // patch, 1)
    pooled = counts[: ph * patch, : pw * patch].reshape(ph, patch, pw, patch).sum(axis=(1, 3))
    flat = np.sort(pooled.ravel())[::-1]
    top = flat[: max(len(flat) // 10, 1)].sum()
    return float(top / max(flat.sum(), 1))


def compare(reference: EventData, other: EventData) -> dict:
    """Metrics comparing two streams (e.g. clean vs denoised)."""
    return {
        "retention": retention(reference, other),
        "event_rate_ref_hz": reference.event_rate,
        "event_rate_other_hz": other.event_rate,
        "structure_ref": event_structural_ratio(reference),
        "structure_other": event_structural_ratio(other),
    }
