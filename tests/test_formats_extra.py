"""Tests for the .dat / ROS bag loaders, synth labels, and denoise scoring."""

import struct

import numpy as np
import pytest

from evlab import formats
from evlab.filters import MASKS, background_activity_filter
from evlab.metrics import denoise_score
from evlab.synth import moving_bar

# ---------------------------------------------------------------------------
# Prophesee .dat
# ---------------------------------------------------------------------------


def _write_dat(path, records, width=640, height=480):
    """records: list of (t, x, y, p)."""
    with open(path, "wb") as f:
        f.write(b"% Date 2026-07-07\n")
        f.write(f"% Width {width}\n".encode())
        f.write(f"% Height {height}\n".encode())
        f.write(bytes([0x0C, 8]))  # event type, event size
        for t, x, y, p in records:
            addr = (x & 0x3FFF) | ((y & 0x3FFF) << 14) | ((p & 0x1) << 28)
            f.write(struct.pack("<II", t, addr))


def test_load_dat(tmp_path):
    path = str(tmp_path / "events.dat")
    _write_dat(path, [(100, 5, 7, 1), (200, 300, 400, 0), (300, 16383, 1, 1)])
    data = formats.load(path)
    assert data.width == 640 and data.height == 480
    assert list(data.events["t"]) == [100, 200, 300]
    assert list(data.events["x"]) == [5, 300, 16383]
    assert list(data.events["y"]) == [7, 400, 1]
    assert list(data.events["p"]) == [1, 0, 1]


def test_load_dat_timestamp_wrap(tmp_path):
    path = str(tmp_path / "wrap.dat")
    near_max = 2**32 - 10
    _write_dat(path, [(near_max, 1, 1, 1), (5, 2, 2, 0)])  # wraps between records
    data = formats.load(path)
    ts = data.events["t"]
    assert ts[0] == near_max
    assert ts[1] == 5 + 2**32
    assert ts[1] > ts[0]


def test_load_dat_rejects_bad_event_size(tmp_path):
    path = tmp_path / "bad.dat"
    path.write_bytes(b"% Height 10\n" + bytes([0x0C, 4]) + b"\x00" * 8)
    with pytest.raises(ValueError, match="event size"):
        formats.load(str(path))


# ---------------------------------------------------------------------------
# ROS bag (round-trip through the rosbags writer)
# ---------------------------------------------------------------------------

EVENT_MSGDEF = """\
uint16 x
uint16 y
time ts
bool polarity
"""

EVENTARRAY_MSGDEF = """\
std_msgs/Header header
uint32 height
uint32 width
dvs_msgs/Event[] events
"""


def _write_bag(path):
    pytest.importorskip("rosbags")
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    types = {}
    types.update(get_types_from_msg(EVENT_MSGDEF, "dvs_msgs/msg/Event"))
    types.update(get_types_from_msg(EVENTARRAY_MSGDEF, "dvs_msgs/msg/EventArray"))
    typestore.register(types)

    Event = typestore.types["dvs_msgs/msg/Event"]
    EventArray = typestore.types["dvs_msgs/msg/EventArray"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    events = [
        Event(x=1, y=2, ts=Time(sec=0, nanosec=1000), polarity=True),
        Event(x=3, y=4, ts=Time(sec=0, nanosec=2000), polarity=False),
    ]
    msg = EventArray(
        header=Header(seq=0, stamp=Time(sec=0, nanosec=0), frame_id="cam"),
        height=480,
        width=640,
        events=events,
    )
    with Writer(path) as writer:
        conn = writer.add_connection("/dvs/events", "dvs_msgs/msg/EventArray", typestore=typestore)
        writer.write(conn, 42, typestore.serialize_ros1(msg, "dvs_msgs/msg/EventArray"))


def test_load_rosbag(tmp_path):
    bag = tmp_path / "events.bag"
    _write_bag(bag)
    data = formats.load(str(bag))
    assert data.width == 640 and data.height == 480
    assert list(data.events["x"]) == [1, 3]
    assert list(data.events["t"]) == [1, 2]  # ns -> us
    assert list(data.events["p"]) == [1, 0]


def test_load_rosbag_no_topic(tmp_path):
    pytest.importorskip("rosbags")
    from rosbags.rosbag1 import Writer

    bag = tmp_path / "empty.bag"
    with Writer(bag):
        pass
    with pytest.raises(ValueError, match="no EventArray topic"):
        formats.load(str(bag))


# ---------------------------------------------------------------------------
# synth + labeled denoise benchmark
# ---------------------------------------------------------------------------


def test_synth_labels_roundtrip(tmp_path):
    data = moving_bar(
        width=64, height=64, duration_us=100_000, signal_rate_hz=5000, noise_rate_hz=1000, seed=1
    )
    assert "signal" in data.meta
    assert len(data.meta["signal"]) == len(data.events)

    path = str(tmp_path / "labeled.npz")
    formats.save(data, path)
    loaded = formats.load(path)
    assert np.array_equal(loaded.meta["signal"], data.meta["signal"])
    assert np.array_equal(loaded.events, data.events)


def test_filter_subsets_labels():
    data = moving_bar(
        width=64, height=64, duration_us=100_000, signal_rate_hz=5000, noise_rate_hz=1000, seed=1
    )
    filtered = background_activity_filter(data, time_window_us=3000)
    assert len(filtered.meta["signal"]) == len(filtered.events)


def test_denoise_score_baf_beats_chance():
    data = moving_bar(
        width=128, height=128, duration_us=500_000, signal_rate_hz=20000, noise_rate_hz=5000, seed=0
    )
    mask = MASKS["baf"](data, time_window_us=3000)
    score = denoise_score(data, mask)
    base_rate = score["signal_events"] / (score["signal_events"] + score["noise_events"])
    assert score["precision"] > base_rate  # better than keeping everything
    assert score["recall"] > 0.5
    assert 0 < score["noise_removed"] <= 1


def test_denoise_score_requires_labels():
    from evlab import from_arrays

    data = from_arrays([1], [1], [1], [1], 4, 4)
    with pytest.raises(ValueError, match="signal"):
        denoise_score(data, np.array([True]))
