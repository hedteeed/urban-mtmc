"""Tests for mtmc.server (CONTRACT.md §mtmc.server).

No network, no real simulator: fastapi TestClient plus a contract-shaped
fake sim. Never writes into the real web/ (owned by agent C) — the index
placeholder lives only in a pytest tmpdir.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mtmc import events
from mtmc.server import create_app

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


def test_api_plan_returns_plan_with_six_cameras() -> None:
    # Default root discovery must find the real repo from the package file.
    client = TestClient(create_app())
    resp = client.get("/api/plan")
    assert resp.status_code == 200
    body = resp.json()
    assert body == json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert len(body["cameras"]) == 6


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
