"""Canonical event representation and file format loaders/savers.

The canonical in-memory representation is a numpy structured array with
fields ``x`` (uint16), ``y`` (uint16), ``t`` (int64, microseconds), and
``p`` (int8, polarity 0/1), sorted by ``t``, plus a metadata dict carrying
at least ``width`` and ``height``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

EVENT_DTYPE = np.dtype([("x", "u2"), ("y", "u2"), ("t", "i8"), ("p", "i1")])


@dataclass
class EventData:
    """Events plus sensor metadata."""

    events: np.ndarray
    width: int
    height: int
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.events.dtype != EVENT_DTYPE:
            raise ValueError(f"events must have dtype {EVENT_DTYPE}, got {self.events.dtype}")

    @property
    def duration_us(self) -> int:
        if len(self.events) == 0:
            return 0
        return int(self.events["t"][-1] - self.events["t"][0])

    @property
    def event_rate(self) -> float:
        """Mean events per second over the recording."""
        dur = self.duration_us
        if dur == 0:
            return 0.0
        return len(self.events) / (dur / 1e6)


def from_arrays(x, y, t, p, width: int | None = None, height: int | None = None) -> EventData:
    """Build EventData from separate coordinate arrays (any numeric dtype)."""
    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t)
    p = np.asarray(p)
    if not (len(x) == len(y) == len(t) == len(p)):
        raise ValueError("x, y, t, p must have equal lengths")

    events = np.empty(len(x), dtype=EVENT_DTYPE)
    events["x"] = x
    events["y"] = y
    events["t"] = t
    # Normalize polarity: accept {0,1}, {-1,1}, or booleans.
    p = p.astype(np.int8)
    p[p < 0] = 0
    events["p"] = p

    order = np.argsort(events["t"], kind="stable")
    events = events[order]

    if width is None:
        width = int(events["x"].max()) + 1 if len(events) else 0
    if height is None:
        height = int(events["y"].max()) + 1 if len(events) else 0
    return EventData(events, width=width, height=height)


# ---------------------------------------------------------------------------
# npz
# ---------------------------------------------------------------------------

_NPZ_KEY_CANDIDATES = {
    "x": ["x", "xs"],
    "y": ["y", "ys"],
    "t": ["t", "ts", "timestamp", "timestamps", "time"],
    "p": ["p", "ps", "pol", "polarity", "polarities"],
}


def _resolve_npz_keys(available: set[str]) -> dict[str, str]:
    resolved = {}
    for canon, candidates in _NPZ_KEY_CANDIDATES.items():
        for cand in candidates:
            if cand in available:
                resolved[canon] = cand
                break
        else:
            raise KeyError(
                f"could not find a key for '{canon}' in npz file "
                f"(available: {sorted(available)}; accepted: {candidates})"
            )
    return resolved


def load_npz(path: str) -> EventData:
    with np.load(path) as data:
        available = set(data.files)
        if "events" in available and data["events"].dtype.names:
            ev = data["events"]
            width = int(data["width"]) if "width" in available else None
            height = int(data["height"]) if "height" in available else None
            out = from_arrays(ev["x"], ev["y"], ev["t"], ev["p"], width, height)
        else:
            keys = _resolve_npz_keys(available)
            width = int(data["width"]) if "width" in available else None
            height = int(data["height"]) if "height" in available else None
            out = from_arrays(
                data[keys["x"]], data[keys["y"]], data[keys["t"]], data[keys["p"]], width, height
            )
        # Ground-truth signal/noise labels (written by `evlab synth`). The
        # events were saved t-sorted and from_arrays' stable sort is the
        # identity on sorted input, so the mask stays aligned.
        if "signal" in available:
            out.meta["signal"] = data["signal"].astype(bool)
        return out


def save_npz(data: EventData, path: str) -> None:
    extra = {}
    if "signal" in data.meta:
        extra["signal"] = np.asarray(data.meta["signal"], dtype=bool)
    np.savez_compressed(
        path, events=data.events, width=np.int64(data.width), height=np.int64(data.height), **extra
    )


# ---------------------------------------------------------------------------
# csv / txt  (columns: t x y p, header optional, delimiter auto)
# ---------------------------------------------------------------------------


def load_csv(path: str, order: str = "txyp") -> EventData:
    if sorted(order) != sorted("txyp"):
        raise ValueError(f"order must be a permutation of 'txyp', got '{order}'")
    arr = np.loadtxt(path, delimiter=None if path.endswith(".txt") else ",", ndmin=2)
    if arr.shape[1] < 4:
        raise ValueError(f"expected at least 4 columns (t x y p), got {arr.shape[1]}")
    cols = {ch: arr[:, i] for i, ch in enumerate(order)}
    return from_arrays(cols["x"], cols["y"], cols["t"], cols["p"])


def save_csv(data: EventData, path: str) -> None:
    ev = data.events
    out = np.column_stack([ev["t"], ev["x"], ev["y"], ev["p"]])
    np.savetxt(path, out, fmt="%d", delimiter="," if path.endswith(".csv") else " ")


# ---------------------------------------------------------------------------
# aedat4 (optional dependency)
# ---------------------------------------------------------------------------


def load_aedat4(path: str) -> EventData:
    try:
        import aedat  # type: ignore
    except ImportError as e:
        raise ImportError(
            "reading .aedat4 requires the 'aedat' package: pip install evlab[aedat]"
        ) from e

    xs, ys, ts, ps = [], [], [], []
    width = height = None
    decoder = aedat.Decoder(path)
    for packet in decoder:
        if "events" in packet:
            ev = packet["events"]
            xs.append(ev["x"])
            ys.append(ev["y"])
            ts.append(ev["t"])
            ps.append(ev["on"])
    if not xs:
        raise ValueError(f"no event packets found in {path}")
    return from_arrays(
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ts),
        np.concatenate(ps),
        width,
        height,
    )


# ---------------------------------------------------------------------------
# Prophesee legacy .dat (2D CD events)
# ---------------------------------------------------------------------------


def load_dat(path: str) -> EventData:
    """Read a Prophesee legacy ``.dat`` file (2D CD events).

    Layout: ASCII header lines starting with ``%``, then one byte event
    type + one byte event size, then 8-byte records of little-endian
    ``(uint32 t_us, uint32 addr)`` with ``x = addr & 0x3FFF``,
    ``y = (addr >> 14) & 0x3FFF``, ``p = (addr >> 28) & 1``. The uint32
    timestamp wraps every ~71 minutes; wraps are unwrapped monotonically.
    """
    width = height = None
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b"%"):
                f.seek(pos)
                break
            text = line[1:].strip().decode("ascii", errors="replace")
            if text.lower().startswith("width"):
                width = int(text.split()[-1])
            elif text.lower().startswith("height"):
                height = int(text.split()[-1])
        header = f.read(2)
        if len(header) < 2:
            raise ValueError(f"truncated .dat file: {path}")
        ev_size = header[1]
        if ev_size != 8:
            raise ValueError(f"unsupported .dat event size {ev_size} (expected 8): {path}")
        raw = np.frombuffer(f.read(), dtype=np.dtype([("t", "<u4"), ("addr", "<u4")]))

    t = raw["t"].astype(np.int64)
    # Unwrap uint32 timestamp overflows.
    wraps = np.cumsum(np.diff(t, prepend=t[:1]) < 0)
    t += wraps * (np.int64(1) << 32)

    addr = raw["addr"]
    x = addr & 0x3FFF
    y = (addr >> 14) & 0x3FFF
    p = (addr >> 28) & 0x1
    return from_arrays(x, y, t, p, width, height)


# ---------------------------------------------------------------------------
# ROS bags (optional dependency; dvs_msgs/prophesee-style EventArray topics)
# ---------------------------------------------------------------------------


def load_rosbag(path: str, topic: str | None = None) -> EventData:
    """Extract events from a ROS1/ROS2 bag containing EventArray messages.

    Requires the pure-python ``rosbags`` package (``pip install evlab[ros]``).
    Reads every connection whose message type ends in ``EventArray``
    (dvs_msgs, prophesee_event_msgs, ...), or only ``topic`` if given.
    """
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as e:
        raise ImportError(
            "reading ROS bags requires the 'rosbags' package: pip install evlab[ros]"
        ) from e
    from pathlib import Path

    xs, ys, ts, ps = [], [], [], []
    width = height = None
    with AnyReader([Path(path)]) as reader:
        conns = [
            c
            for c in reader.connections
            if c.msgtype.endswith("EventArray") and (topic is None or c.topic == topic)
        ]
        if not conns:
            available = sorted({f"{c.topic} ({c.msgtype})" for c in reader.connections})
            raise ValueError(f"no EventArray topic found in {path}; connections: {available}")
        for conn, _timestamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if getattr(msg, "width", 0):
                width, height = int(msg.width), int(msg.height)
            for ev in msg.events:
                xs.append(ev.x)
                ys.append(ev.y)
                ts.append(int(ev.ts.sec) * 1_000_000 + int(ev.ts.nanosec) // 1000)
                ps.append(bool(ev.polarity))
    if not xs:
        raise ValueError(f"EventArray topic in {path} contained no events")
    return from_arrays(np.array(xs), np.array(ys), np.array(ts), np.array(ps), width, height)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

_LOADERS = {
    ".npz": load_npz,
    ".csv": load_csv,
    ".txt": load_csv,
    ".aedat4": load_aedat4,
    ".dat": load_dat,
    ".bag": load_rosbag,
}

_SAVERS = {
    ".npz": save_npz,
    ".csv": save_csv,
    ".txt": save_csv,
}


def load(path: str) -> EventData:
    """Load events from any supported file format (dispatched on extension)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _LOADERS:
        raise ValueError(f"unsupported input format '{ext}' (supported: {sorted(_LOADERS)})")
    return _LOADERS[ext](path)


def save(data: EventData, path: str) -> None:
    """Save events to any supported output format (dispatched on extension)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _SAVERS:
        raise ValueError(f"unsupported output format '{ext}' (supported: {sorted(_SAVERS)})")
    _SAVERS[ext](data, path)
