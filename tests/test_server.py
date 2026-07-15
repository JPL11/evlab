"""Tests for the web UI server (requires the ui extra)."""

import numpy as np
import pytest

from evlab import formats
from evlab.corrupt import make_schedule
from evlab.synth import moving_bar

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from evlab.server import create_app  # noqa: E402


@pytest.fixture()
def clip(tmp_path):
    data = moving_bar(duration_us=2_000_000, signal_rate_hz=20_000, noise_rate_hz=5_000, seed=3)
    path = tmp_path / "clip.npz"
    formats.save(data, str(path))
    return path


@pytest.fixture()
def client():
    return TestClient(create_app())


def post(client, clip, **form):
    with open(clip, "rb") as f:
        return client.post("/api/process", files={"file": ("clip.npz", f)}, data=form)


def test_index_serves_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "evlab" in resp.text


def test_denoise_roundtrip(client, clip, tmp_path):
    resp = post(client, clip, op="denoise", filter_name="baf", window="5000")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stats"]["events_out"] < body["stats"]["events_in"]
    assert 0 < body["stats"]["retention"] <= 1
    assert body["filename"] == "clip-denoise.npz"

    dl = client.get(f"/api/result/{body['token']}")
    assert dl.status_code == 200
    out = tmp_path / "result.npz"
    out.write_bytes(dl.content)
    data = formats.load(str(out))
    assert len(data.events) == body["stats"]["events_out"]


def test_corrupt_labels_and_episodes(client, clip):
    resp = post(client, clip, op="corrupt", severity="low", seed="1")
    assert resp.status_code == 200
    stats = resp.json()["stats"]
    assert stats["episodes"] >= 1
    assert stats["corrupted_events"] > 0


def test_corrupt_short_clip_still_gets_episodes():
    # Regression: recordings shorter than the episode maximum used to
    # produce empty schedules for most seeds.
    for seed in range(5):
        assert make_schedule(2_000_000, ["burst"], seed=seed), f"empty schedule for seed {seed}"


def test_unknown_type_is_400(client, clip):
    resp = post(client, clip, op="corrupt", types="nonsense")
    assert resp.status_code == 400
    assert "unknown corruption types" in resp.json()["detail"]


def test_unknown_op_is_400(client, clip):
    resp = post(client, clip, op="explode")
    assert resp.status_code == 400


def test_unknown_token_is_404(client):
    assert client.get("/api/result/deadbeef").status_code == 404


def test_previews_present_when_matplotlib_available(client, clip):
    pytest.importorskip("matplotlib")
    resp = post(client, clip, op="denoise")
    body = resp.json()
    assert body["preview_before"] and body["preview_after"]


def test_corrupt_result_preserves_labels(client, clip, tmp_path):
    resp = post(client, clip, op="corrupt", seed="2")
    body = resp.json()
    dl = client.get(f"/api/result/{body['token']}")
    out = tmp_path / "corrupted.npz"
    out.write_bytes(dl.content)
    data = formats.load(str(out))
    labels = np.asarray(data.meta["corruption"])
    assert int((labels != 0).sum()) == body["stats"]["corrupted_events"]
