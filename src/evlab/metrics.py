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


def denoise_score(data: EventData, keep_mask) -> dict:
    """Score a denoising keep-mask against ground-truth labels.

    Requires ``data.meta['signal']`` (boolean per-event array, True =
    signal), as produced by `evlab.synth`. Positive class = signal kept.
    """
    import numpy as np

    if "signal" not in data.meta:
        raise ValueError("no ground-truth labels: data.meta['signal'] missing (see `evlab synth`)")
    signal = np.asarray(data.meta["signal"], dtype=bool)
    keep = np.asarray(keep_mask, dtype=bool)
    if signal.shape != keep.shape:
        raise ValueError(f"label/mask shape mismatch: {signal.shape} vs {keep.shape}")

    tp = int((keep & signal).sum())
    fp = int((keep & ~signal).sum())
    fn = int((~keep & signal).sum())
    n_noise = int((~signal).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "signal_events": int(signal.sum()),
        "noise_events": n_noise,
        "kept_events": int(keep.sum()),
        "noise_removed": (n_noise - fp) / max(n_noise, 1),
    }


def compare(reference: EventData, other: EventData) -> dict:
    """Metrics comparing two streams (e.g. clean vs denoised)."""
    return {
        "retention": retention(reference, other),
        "event_rate_ref_hz": reference.event_rate,
        "event_rate_other_hz": other.event_rate,
        "structure_ref": event_structural_ratio(reference),
        "structure_other": event_structural_ratio(other),
    }
