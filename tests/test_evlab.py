import numpy as np
import pytest

from evlab import formats, from_arrays
from evlab.filters import background_activity_filter, refractory_filter
from evlab.metrics import compare, retention, summary
from evlab.representations import accumulate_frame, time_surface, voxel_grid


@pytest.fixture
def sample():
    # A tight cluster (signal) plus two isolated events (noise).
    x = [10, 11, 10, 11, 50, 3]
    y = [20, 20, 21, 21, 60, 70]
    t = [1000, 1100, 1200, 1300, 5000, 9000]
    p = [1, 0, 1, 1, 1, 0]
    return from_arrays(x, y, t, p, width=64, height=80)


def test_from_arrays_sorts_and_normalizes():
    data = from_arrays([1, 2], [3, 4], [200, 100], [-1, 1])
    assert list(data.events["t"]) == [100, 200]
    assert list(data.events["p"]) == [1, 0]
    assert data.width == 3 and data.height == 5


def test_npz_roundtrip(tmp_path, sample):
    path = str(tmp_path / "events.npz")
    formats.save(sample, path)
    loaded = formats.load(path)
    assert np.array_equal(loaded.events, sample.events)
    assert loaded.width == sample.width and loaded.height == sample.height


def test_npz_alias_keys(tmp_path):
    path = str(tmp_path / "alias.npz")
    np.savez(path, xs=[1], ys=[2], timestamps=[3], polarities=[1])
    loaded = formats.load(path)
    assert loaded.events["x"][0] == 1
    assert loaded.events["t"][0] == 3


def test_csv_roundtrip(tmp_path, sample):
    path = str(tmp_path / "events.csv")
    formats.save(sample, path)
    loaded = formats.load(path)
    assert np.array_equal(loaded.events["t"], sample.events["t"])
    assert np.array_equal(loaded.events["p"], sample.events["p"])


def test_unsupported_format(tmp_path, sample):
    with pytest.raises(ValueError, match="unsupported"):
        formats.save(sample, str(tmp_path / "events.xyz"))


def test_baf_keeps_cluster_drops_isolated(sample):
    out = background_activity_filter(sample, time_window_us=5000)
    # The 4 clustered events survive; the isolated first-of-cluster event is
    # dropped (nothing preceded it), as are the two spatially isolated ones.
    kept = set(zip(out.events["x"].tolist(), out.events["y"].tolist()))
    assert (50, 60) not in kept
    assert (3, 70) not in kept
    assert len(out.events) == 3  # cluster minus its first event


def test_refractory_filter():
    data = from_arrays([5, 5, 5], [5, 5, 5], [0, 100, 5000], [1, 1, 1], 10, 10)
    out = refractory_filter(data, refractory_us=1000)
    assert list(out.events["t"]) == [0, 5000]


def test_voxel_grid_shape_and_mass(sample):
    grid = voxel_grid(sample, bins=4)
    assert grid.shape == (4, 80, 64)
    # Signed mass: 4 ON (+1) and 2 OFF (-1) -> +2
    assert grid.sum() == pytest.approx(2.0, abs=1e-4)


def test_time_surface_decay(sample):
    surf = time_surface(sample, tau_us=1000.0)
    assert surf.shape == (2, 80, 64)
    # Most recent event (OFF at (3,70), t=9000) has decay exp(0)=1.
    assert surf[0, 70, 3] == pytest.approx(1.0)
    # Older events decay strictly below 1.
    assert 0 < surf[1, 60, 50] < 1


def test_accumulate_frame(sample):
    frame = accumulate_frame(sample, clip=3)
    assert frame[20, 10] == 1
    assert frame[20, 11] == -1


def test_metrics(sample):
    stats = summary(sample)
    assert stats["num_events"] == 6
    assert stats["active_pixels"] == 6
    filtered = background_activity_filter(sample, time_window_us=5000)
    assert retention(sample, filtered) == 0.5
    result = compare(sample, filtered)
    assert result["retention"] == 0.5


def test_empty_stream():
    data = from_arrays([], [], [], [], width=8, height=8)
    assert data.event_rate == 0.0
    assert voxel_grid(data, bins=2).shape == (2, 8, 8)
    out = background_activity_filter(data)
    assert len(out.events) == 0
