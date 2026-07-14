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


# ---------------------------------------------------------------------------
# corruption benchmark scoring (labels + schedules from `evlab corrupt`)
# ---------------------------------------------------------------------------


def corruption_score(data: EventData, keep_mask) -> dict:
    """Score a keep-mask against per-event corruption labels.

    Treats the filter as a corruption-removal classifier: ``recall`` is the
    fraction of corrupted events removed (also broken out per type),
    ``precision`` the fraction of removed events that were corrupted, and
    ``clean_retention`` the fraction of clean events kept.
    """
    from .corrupt import TYPE_NAMES

    keep = np.asarray(keep_mask, bool)
    labels = np.asarray(data.meta["corruption"])
    corrupted = labels != 0
    removed = ~keep

    tp = int((removed & corrupted).sum())
    n_removed = int(removed.sum())
    n_corrupted = int(corrupted.sum())
    n_clean = int((~corrupted).sum())
    precision = tp / n_removed if n_removed else 0.0
    recall = tp / n_corrupted if n_corrupted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_type = {}
    for tid in np.unique(labels[corrupted]):
        of_type = labels == tid
        per_type[TYPE_NAMES[int(tid)]] = {
            "events": int(of_type.sum()),
            "removed": float((removed & of_type).sum() / of_type.sum()),
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clean_retention": float(keep[~corrupted].sum() / n_clean) if n_clean else 0.0,
        "per_type": per_type,
    }


def window_labels(data: EventData, schedule: list[dict], window_us: int = 50_000):
    """Per-window ground truth from an episode schedule.

    Windows mostly (>50%) inside an episode get that episode's type id,
    windows overlapping no episode get 0, and partial-boundary windows get
    -1 (excluded), following the EvCorrupt-Bench protocol.
    """
    from .corrupt import TYPES

    t = data.events["t"].astype(np.int64)
    t0 = int(t[0]) if len(t) else 0
    n_win = int((int(t[-1]) - t0) // window_us) + 1 if len(t) else 0
    labels = np.zeros(n_win, np.int8)
    for k in range(n_win):
        a, b = t0 + k * window_us, t0 + (k + 1) * window_us
        for epi in schedule:
            ea, eb = t0 + epi["start_us"], t0 + epi["end_us"]
            overlap = max(0, min(b, eb) - max(a, ea))
            if overlap == 0:
                continue
            if overlap > window_us // 2:
                labels[k] = TYPES[epi["type"]]
            else:
                labels[k] = -1
            break
    return labels


def auroc(scores, binary_labels) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) statistic."""
    s = np.asarray(scores, float)
    y = np.asarray(binary_labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # Midranks for ties.
    for v in np.unique(s):
        tied = s == v
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def time_to_detection(
    scores, labels, schedule, window_us: int = 50_000, fpr: float = 0.05
) -> dict:
    """Median time-to-detection at a fixed false-positive rate.

    The threshold is the (1-fpr) quantile of clean-window scores. For each
    episode, detection time is the delay from onset to the first flagged
    window; undetected episodes count in ``detected`` but not the median.
    """
    s = np.asarray(scores, float)
    lab = np.asarray(labels)
    clean = lab == 0
    if not clean.any():
        return {"threshold": float("nan"), "detected": 0, "episodes": len(schedule), "median_ttd_ms": float("nan")}
    threshold = float(np.quantile(s[clean], 1 - fpr))
    flagged = s > threshold
    ttds = []
    detected = 0
    for j, epi in enumerate(schedule):
        k0 = epi["start_us"] // window_us
        k1 = (epi["end_us"] - 1) // window_us
        ks = np.arange(k0, min(k1 + 1, len(s)))
        hit = ks[flagged[ks] & (lab[ks] > 0)] if len(ks) else []
        if len(hit):
            detected += 1
            ttds.append((int(hit[0]) + 1) * window_us - epi["start_us"])
    return {
        "threshold": threshold,
        "detected": detected,
        "episodes": len(schedule),
        "median_ttd_ms": float(np.median(ttds) / 1000) if ttds else float("nan"),
    }
