"""Tests for mtmc.server (CONTRACT.md §mtmc.server).

No network, no real simulator: fastapi TestClient plus a contract-shaped
fake sim. Never writes into the real web/ (owned by agent C) — the index
placeholder lives only in a pytest tmpdir.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mtmc import events
from mtmc.server import _parse_args, create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "floorplan" / "plan.json"


class FakeSimulation:
    """Minimal stand-in matching the contract Simulation surface:
    ``clock_s`` property + ``step(dt_s) -> list[Observation]``."""

    def __init__(self) -> None:
        self._clock_s = 0.0

    @property
    def clock_s(self) -> float:
        return self._clock_s

    def step(self, dt_s: float) -> list[events.Observation]:
        self._clock_s += dt_s
        return [
            events.Observation(
                ts_s=self._clock_s,
                camera="cam1",
                track_id=1,
                floor_xy=(1.0, 2.0),
                conf=0.9,
                global_id=7,
            )
        ]


@pytest.fixture()
def tmp_root(tmp_path: Path) -> Path:
    """Repo-shaped tree in a tmpdir: real plan.json copy + placeholder index."""
    (tmp_path / "floorplan").mkdir()
    shutil.copyfile(PLAN_PATH, tmp_path / "floorplan" / "plan.json")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text(
        "<!doctype html><title>mtmc placeholder</title>", encoding="utf-8"
    )
    return tmp_path


def test_api_plan_returns_plan_file() -> None:
    # Default root discovery must find the real repo from the package file.
    # Server is plan-agnostic: full-body equality against the file is the
    # whole contract — no assumptions about plan contents here.
    client = TestClient(create_app())
    resp = client.get("/api/plan")
    assert resp.status_code == 200
    assert resp.json() == json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_index_serves_html_when_present(tmp_root: Path) -> None:
    client = TestClient(create_app(root=tmp_root))
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "mtmc placeholder" in resp.text


def test_index_404_when_web_missing(tmp_path: Path) -> None:
    (tmp_path / "floorplan").mkdir()
    shutil.copyfile(PLAN_PATH, tmp_path / "floorplan" / "plan.json")
    client = TestClient(create_app(root=tmp_path))
    assert client.get("/").status_code == 404


def test_static_mount_serves_web_files(tmp_root: Path) -> None:
    client = TestClient(create_app(root=tmp_root))
    resp = client.get("/static/index.html")
    assert resp.status_code == 200
    assert "mtmc placeholder" in resp.text


def test_ws_delivers_valid_tick_frame(tmp_root: Path) -> None:
    app = create_app(root=tmp_root, sim_factory=FakeSimulation)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            frame = json.loads(ws.receive_text())

    # Shape must match events.tick_message exactly.
    reference = json.loads(events.tick_message(0.1, FakeSimulation().step(0.1)))
    assert set(frame) == set(reference) == {"type", "v", "ts_s", "observations"}
    assert frame["type"] == "tick"
    assert frame["v"] == events.SCHEMA_VERSION
    assert isinstance(frame["ts_s"], float)
    assert frame["observations"], "fake sim emits every step; frame must be non-empty"

    obs = frame["observations"][0]
    assert set(obs) == {"ts_s", "camera", "track_id", "floor_xy", "conf", "global_id"}
    assert obs["camera"] == "cam1"
    assert obs["track_id"] == 1
    assert obs["floor_xy"] == [1.0, 2.0]
    assert 0.0 <= obs["conf"] <= 1.0
    assert obs["global_id"] == 7


def test_sim_is_shared_and_survives_disconnect(tmp_root: Path) -> None:
    app = create_app(root=tmp_root, sim_factory=FakeSimulation)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            first_ts = json.loads(ws.receive_text())["ts_s"]
        # Disconnect must not reset or kill the sim: a new client sees the
        # SAME world, further along its clock.
        with client.websocket_connect("/ws") as ws:
            second_ts = json.loads(ws.receive_text())["ts_s"]
    assert second_ts > first_ts


# ---------------------------------------------------------------------------
# M1: --source camera mode (CONTRACT.md §M1 additions)
# ---------------------------------------------------------------------------


class StubCameraWorker:
    """CameraWorkerLike stand-in: an endless supply of canned batches.

    No thread, no cv2, no detector — start/stop only record the lifecycle
    calls the server is contractually required to make.
    """

    camera_id = "live0"

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.latest_jpeg = b"\xff\xd8-stub-jpeg-\xff\xd9"
        self._n = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_batch(self, timeout: float = 0.5) -> list[events.Observation] | None:
        time.sleep(0.005)  # pace the drain loop like a real blocking queue
        self._n += 1
        ts = 0.25 * self._n  # 0.25 is binary-exact: survives round-tripping
        return [
            events.Observation(
                ts_s=ts, camera=self.camera_id, track_id=2 * self._n,
                floor_xy=None, conf=0.9, global_id=None,
            ),
            events.Observation(
                ts_s=ts, camera=self.camera_id, track_id=2 * self._n + 1,
                floor_xy=None, conf=0.5, global_id=None,
            ),
        ]


def camera_app(tmp_root: Path, worker: StubCameraWorker):
    return create_app(root=tmp_root, source="0", camera_worker_factory=lambda: worker)


def test_api_source_sim_mode(tmp_root: Path) -> None:
    client = TestClient(create_app(root=tmp_root, sim_factory=FakeSimulation))
    resp = client.get("/api/source")
    assert resp.status_code == 200
    assert resp.json() == {"mode": "sim", "camera_id": None}


def test_api_source_camera_mode(tmp_root: Path) -> None:
    client = TestClient(camera_app(tmp_root, StubCameraWorker()))
    resp = client.get("/api/source")
    assert resp.status_code == 200
    assert resp.json() == {"mode": "camera", "camera_id": "live0"}


def test_video_mjpg_404_in_sim_mode(tmp_root: Path) -> None:
    client = TestClient(create_app(root=tmp_root, sim_factory=FakeSimulation))
    assert client.get("/video.mjpg").status_code == 404


def test_video_mjpg_frames_carry_latest_jpeg_in_camera_mode(tmp_root: Path) -> None:
    # The body is an INFINITE multipart stream, which TestClient would buffer
    # forever — so probe the endpoint's StreamingResponse generator directly.
    worker = StubCameraWorker()
    app = camera_app(tmp_root, worker)
    route = next(r for r in app.routes if getattr(r, "path", None) == "/video.mjpg")

    async def probe() -> tuple[object, bytes]:
        resp = await route.endpoint()
        gen = resp.body_iterator
        try:
            chunk = await gen.__anext__()
        finally:
            await gen.aclose()
        return resp, chunk

    resp, chunk = asyncio.run(probe())
    assert resp.media_type.startswith("multipart/x-mixed-replace; boundary=")
    assert chunk.startswith(b"--")
    assert b"Content-Type: image/jpeg" in chunk
    assert worker.latest_jpeg in chunk


def test_camera_mode_ws_delivers_tick_frame_and_worker_lifecycle(tmp_root: Path) -> None:
    worker = StubCameraWorker()
    with TestClient(camera_app(tmp_root, worker)) as client:
        assert worker.started, "lifespan startup starts the worker"
        assert not worker.stopped
        with client.websocket_connect("/ws") as ws:
            frame = json.loads(ws.receive_text())
    assert worker.stopped, "lifespan shutdown stops the worker"

    # Same wire shape as sim mode: consumers never branch on source.
    assert set(frame) == {"type", "v", "ts_s", "observations"}
    assert frame["type"] == "tick"
    assert frame["v"] == events.SCHEMA_VERSION
    assert frame["observations"]
    assert frame["ts_s"] == frame["observations"][0]["ts_s"]
    for obs in frame["observations"]:
        assert set(obs) == {"ts_s", "camera", "track_id", "floor_xy", "conf", "global_id"}
        assert obs["camera"] == "live0"
        assert obs["floor_xy"] is None
        assert obs["global_id"] is None
        assert 0.0 <= obs["conf"] <= 1.0


# ---------------------------------------------------------------------------
# M2: tracker plumbing (CONTRACT.md §M2 additions)
# ---------------------------------------------------------------------------


@pytest.fixture()
def factory_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake out the DEFAULT worker factory's three lazy imports so tests can
    run it with no model weights, no device, and no mtmc.tracker (agent A's
    module, built in parallel). Everything constructed is recorded."""
    created: dict[str, Any] = {}

    class FakeByteTracker:
        def __init__(self, **kwargs: float) -> None:
            created["tracker"] = self
            created["tracker_kwargs"] = kwargs

    # The real module may not exist yet — inject a stand-in where the lazy
    # `from mtmc.tracker import ByteTracker` will find it.
    tracker_mod = types.ModuleType("mtmc.tracker")
    tracker_mod.ByteTracker = FakeByteTracker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mtmc.tracker", tracker_mod)

    import mtmc.camera
    import mtmc.detector

    class FakeDetector:
        def __init__(self, **kwargs: float) -> None:
            created["detector_kwargs"] = kwargs

    monkeypatch.setattr(mtmc.detector, "PersonDetector", FakeDetector)

    class FakeWorker(StubCameraWorker):
        def __init__(
            self,
            source: str,
            detector: Any,
            cam_fps: float = 5.0,
            *,
            tracker: Any = None,
        ) -> None:
            super().__init__()
            created["worker"] = self
            created["source"] = source
            created["cam_fps"] = cam_fps
            created["worker_tracker"] = tracker

    monkeypatch.setattr(mtmc.camera, "CameraWorker", FakeWorker)
    return created


def test_camera_mode_default_factory_builds_bytetracker(
    tmp_root: Path, factory_probe: dict[str, Any]
) -> None:
    # No injected factory: create_app must wire the default one (tracker ON).
    app = create_app(root=tmp_root, source="0")
    worker = app.state.hub.ensure_worker()

    assert worker is factory_probe["worker"]
    assert factory_probe["source"] == "0"
    assert factory_probe["worker_tracker"] is factory_probe["tracker"], (
        "the constructed ByteTracker is handed to the worker"
    )
    # Contract M2 defaults, by keyword.
    assert factory_probe["tracker_kwargs"] == {
        "track_thresh": 0.5,
        "match_thresh": 0.8,
        "track_buffer_s": 2.0,
        "min_hits": 3,
    }
    # Tracked mode lowers the detector floor to feed ByteTrack's round 2.
    assert factory_probe["detector_kwargs"] == {"conf": pytest.approx(0.1)}


def test_tracker_flags_thread_through_create_app(
    tmp_root: Path, factory_probe: dict[str, Any]
) -> None:
    app = create_app(
        root=tmp_root,
        source="0",
        track_thresh=0.6,
        match_thresh=0.9,
        track_buffer_s=1.5,
        min_hits=5,
    )
    app.state.hub.ensure_worker()
    assert factory_probe["tracker_kwargs"] == {
        "track_thresh": 0.6,
        "match_thresh": 0.9,
        "track_buffer_s": 1.5,
        "min_hits": 5,
    }


def test_no_track_restores_m1_worker(tmp_root: Path, factory_probe: dict[str, Any]) -> None:
    app = create_app(root=tmp_root, source="0", track=False)
    app.state.hub.ensure_worker()

    assert factory_probe["worker_tracker"] is None
    assert "tracker" not in factory_probe, "--no-track never constructs a ByteTracker"
    assert factory_probe["detector_kwargs"] == {}, "detector keeps its own conf default"


def test_cli_tracking_flags_and_defaults() -> None:
    args = _parse_args([])
    assert args.source == "sim"
    assert (args.track_thresh, args.match_thresh, args.track_buffer, args.min_hits) == (
        0.5,
        0.8,
        2.0,
        3,
    )
    assert args.no_track is False

    args = _parse_args(
        ["--source", "0", "--no-track", "--track-thresh", "0.7", "--match-thresh", "0.85",
         "--track-buffer", "1.5", "--min-hits", "2"]
    )
    assert args.no_track is True
    assert (args.track_thresh, args.match_thresh, args.track_buffer, args.min_hits) == (
        0.7,
        0.85,
        1.5,
        2,
    )
