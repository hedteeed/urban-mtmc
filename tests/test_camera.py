"""Tests for mtmc.camera.CameraWorker (CONTRACT.md §M1 additions).

No physical camera, no model file, no onnxruntime: both collaborators — the
cv2 capture and the detector — are injected fakes. Timing assertions are kept
loose enough for a loaded machine (they prove throttling, not precision).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from mtmc.camera import CameraWorker, CaptureLike
from mtmc.events import Observation

FRAME_SHAPE = (48, 64, 3)


class FakeCapture:
    """cv2.VideoCapture stand-in: instant synthetic (all-black) frames."""

    def __init__(self) -> None:
        self.frames_read = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 — cv2's casing
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.frames_read += 1
        return True, np.zeros(FRAME_SHAPE, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class DeadCapture(FakeCapture):
    """Opens fine, then immediately loses the stream."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        return False, None


class FakeDetector:
    """Always sees the same two people; counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        self.calls += 1
        return [(4.0, 4.0, 30.0, 44.0, 0.88), (20.0, 6.0, 60.0, 46.0, 0.61)]


def make_worker(
    cam_fps: float = 50.0,
    queue_maxsize: int = 512,
    capture_factory: Callable[[str], CaptureLike] | None = None,
    **kw: float,
) -> tuple[CameraWorker, FakeDetector]:
    detector = FakeDetector()
    factory = capture_factory or (lambda source: FakeCapture())
    worker = CameraWorker(
        "0",
        detector,
        cam_fps=cam_fps,
        capture_factory=factory,
        queue_maxsize=queue_maxsize,
        **kw,
    )
    return worker, detector


def drain(worker: CameraWorker, min_batches: int, deadline_s: float = 5.0) -> list[list[Observation]]:
    """Collect batches off the queue until min_batches or the deadline."""
    batches: list[list[Observation]] = []
    t_end = time.monotonic() + deadline_s
    while len(batches) < min_batches and time.monotonic() < t_end:
        batch = worker.get_batch(timeout=0.1)
        if batch is not None:
            batches.append(batch)
    return batches


def test_emits_observation_batches_at_about_cam_fps() -> None:
    worker, _ = make_worker(cam_fps=25.0)
    worker.start()
    try:
        t0 = time.monotonic()
        batches = drain(worker, min_batches=6)
        elapsed = time.monotonic() - t0
    finally:
        worker.stop()

    assert len(batches) >= 6
    # 6 batches at 25 fps require >= 5 inter-batch periods (~0.2 s). The 0.5×
    # lower bound proves throttling exists (an unthrottled fake source would
    # finish in ~milliseconds); the upper bound only rules out a stall.
    assert elapsed >= 5 * (1.0 / 25.0) * 0.5
    assert elapsed < 5.0

    for batch in batches:
        assert batch, "only non-empty batches are queued"
        assert len({obs.ts_s for obs in batch}) == 1, "one instant per batch"
        for obs in batch:
            assert isinstance(obs, Observation)
            assert obs.camera == "live0"
            assert obs.floor_xy is None  # no calibration until M3
            assert obs.global_id is None  # no ground truth from a real camera
            assert 0.0 <= obs.conf <= 1.0
            assert obs.ts_s >= 0.0
    ts = [batch[0].ts_s for batch in batches]
    assert ts == sorted(ts), "invariant 1: non-decreasing timestamps"


def test_reads_continuously_but_detects_at_sampled_rate() -> None:
    captures: list[FakeCapture] = []

    def factory(source: str) -> FakeCapture:
        cap = FakeCapture()
        captures.append(cap)
        return cap

    worker, detector = make_worker(cam_fps=10.0, capture_factory=factory)
    worker.start()
    try:
        drain(worker, min_batches=3)
    finally:
        worker.stop()

    # Freshest-frame loop: many reads per detection, not one detect per read.
    assert captures and captures[0].frames_read > detector.calls


def test_track_ids_fresh_monotonic_never_reused() -> None:
    worker, _ = make_worker(cam_fps=100.0)
    worker.start()
    try:
        batches = drain(worker, min_batches=8)
    finally:
        worker.stop()

    ids = [obs.track_id for batch in batches for obs in batch]
    assert len(ids) >= 16
    assert len(ids) == len(set(ids)), "invariant 2: ids never reused"
    assert all(b > a for a, b in zip(ids, ids[1:])), "fresh monotonic ids"


def test_queue_overflow_drops_oldest_batches() -> None:
    worker, detector = make_worker(cam_fps=200.0, queue_maxsize=3)
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while detector.calls < 10 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        worker.stop()  # joins the thread: calls and queue are final below

    n_batches = detector.calls  # every detect call queued a 2-observation batch
    assert n_batches >= 10

    remaining: list[list[Observation]] = []
    while (batch := worker.get_batch(timeout=0.01)) is not None:
        remaining.append(batch)

    assert len(remaining) == 3, "queue holds exactly maxsize batches"
    ids = [obs.track_id for batch in remaining for obs in batch]
    total_ids = 2 * n_batches
    # Drop-OLDEST: the survivors are exactly the newest ids ever issued.
    assert ids == list(range(total_ids - 6, total_ids))


def test_latest_jpeg_is_nonempty_annotated_jpeg() -> None:
    worker, _ = make_worker(cam_fps=100.0)
    assert worker.latest_jpeg is None, "no preview before the first frame"
    worker.start()
    try:
        assert drain(worker, min_batches=1)
        jpeg = worker.latest_jpeg
    finally:
        worker.stop()

    assert isinstance(jpeg, bytes) and len(jpeg) > 100
    assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"  # JPEG SOI/EOI
    img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == FRAME_SHAPE
    assert int(img.sum()) > 0, "boxes + conf labels drawn on an all-black frame"


class FlakyOpens:
    """1st open raises, 2nd yields a dying stream, 3rd (and later) works."""

    def __init__(self) -> None:
        self.opens = 0
        self.captures: list[FakeCapture] = []

    def __call__(self, source: str) -> FakeCapture:
        self.opens += 1
        if self.opens == 1:
            raise RuntimeError("device busy")
        cap = DeadCapture() if self.opens == 2 else FakeCapture()
        self.captures.append(cap)
        return cap


def test_reconnects_after_open_failure_and_stream_loss() -> None:
    factory = FlakyOpens()
    worker, _ = make_worker(cam_fps=100.0, capture_factory=factory, reconnect_delay_s=0.02)
    worker.start()
    try:
        batches = drain(worker, min_batches=2)
    finally:
        worker.stop()

    assert len(batches) >= 2, "recovered and kept streaming — never crashed"
    assert factory.opens >= 3
    assert factory.captures[0].released, "the dead stream's handle was released"


def test_stop_joins_thread_cleanly_and_is_idempotent() -> None:
    worker, _ = make_worker()
    worker.start()
    worker.start()  # second start is a no-op, not a second thread
    threads = [t for t in threading.enumerate() if t.name == "mtmc-camera-live0"]
    assert len(threads) == 1
    worker.stop()
    assert not any(t.is_alive() for t in threads)
    worker.stop()  # safe to call twice


def test_module_imports_without_touching_onnxruntime() -> None:
    """The detector is injected, never imported: importing mtmc.camera must
    not pull in onnxruntime (contract: camera mode only pays for it)."""
    proc = subprocess.run(
        [sys.executable, "-c", "import sys, mtmc.camera; assert 'onnxruntime' not in sys.modules"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# M2: injected tracker (CONTRACT.md §M2 additions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeTrack:
    """Contract Track shape: track_id + box + conf."""

    track_id: int
    box: tuple[float, float, float, float]
    conf: float


class FakeTracker:
    """ByteTracker stand-in: records every update() call and returns a canned
    confirmed-track list ([] = nothing confirmed yet)."""

    def __init__(self, tracks: list[FakeTrack] | None = None) -> None:
        self.calls: list[tuple[list[tuple[float, float, float, float, float]], float]] = []
        self.tracks = tracks or []

    def update(
        self, detections: list[tuple[float, float, float, float, float]], ts_s: float
    ) -> list[FakeTrack]:
        self.calls.append((list(detections), ts_s))
        return list(self.tracks)


CONFIRMED = [
    FakeTrack(track_id=7, box=(5.0, 5.0, 29.0, 43.0), conf=0.83),
    FakeTrack(track_id=9, box=(21.0, 7.0, 59.0, 45.0), conf=0.44),
]


def test_tracker_receives_every_detection_with_frame_ts() -> None:
    tracker = FakeTracker(CONFIRMED)
    worker, _ = make_worker(cam_fps=100.0, tracker=tracker)
    worker.start()
    try:
        batches = drain(worker, min_batches=3)
    finally:
        worker.stop()  # joins the thread: tracker.calls is final below

    # ALL detections are forwarded, unfiltered — including the low-conf one
    # the tracker's round 2 needs — with the frame's own timestamp.
    expected_dets = FakeDetector().detect(np.zeros(FRAME_SHAPE, dtype=np.uint8))
    assert len(tracker.calls) >= 3
    for dets, ts in tracker.calls:
        assert dets == expected_dets
        assert ts >= 0.0
    ts_calls = [ts for _dets, ts in tracker.calls]
    assert ts_calls == sorted(ts_calls), "invariant 1 upstream of the tracker"

    # Observations mirror the CONFIRMED tracks the tracker returned: its ids,
    # its confs, one Observation each, stamped with the update's ts.
    ts_seen = set(ts_calls)
    for batch in batches:
        assert [obs.track_id for obs in batch] == [7, 9]
        assert [obs.conf for obs in batch] == pytest.approx([0.83, 0.44])
        assert len({obs.ts_s for obs in batch}) == 1, "one instant per batch"
        assert batch[0].ts_s in ts_seen, "batch ts is the tracker-update ts"
        for obs in batch:
            assert obs.camera == "live0"
            assert obs.floor_xy is None  # no calibration until M3
            assert obs.global_id is None  # real pipelines carry no ground truth


def test_no_observations_while_tracker_confirms_nothing() -> None:
    tracker = FakeTracker(tracks=[])
    worker, detector = make_worker(cam_fps=200.0, tracker=tracker)
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while len(tracker.calls) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        worker.stop()

    assert len(tracker.calls) >= 5, "detector and tracker kept running"
    assert detector.calls >= 5
    assert worker.get_batch(timeout=0.05) is None, "tentative tracks never emit"


def test_tracked_jpeg_labels_carry_track_id(monkeypatch: pytest.MonkeyPatch) -> None:
    texts: list[str] = []
    real_put_text = cv2.putText

    def spying_put_text(img: np.ndarray, text: str, *args: object, **kwargs: object) -> object:
        texts.append(text)
        return real_put_text(img, text, *args, **kwargs)

    monkeypatch.setattr("mtmc.camera.cv2.putText", spying_put_text)

    worker, _ = make_worker(cam_fps=100.0, tracker=FakeTracker(CONFIRMED))
    worker.start()
    try:
        assert drain(worker, min_batches=1)
        jpeg = worker.latest_jpeg
    finally:
        worker.stop()  # joined before monkeypatch undo: no cross-thread race

    assert isinstance(jpeg, bytes) and jpeg[:2] == b"\xff\xd8"
    # Drawn text = contract label with the middot swapped for a hyphen
    # (Hershey fonts render non-ASCII as "?"); label semantics identical.
    assert "ID 7 - 0.83" in texts and "ID 9 - 0.44" in texts
    assert all(t.startswith("ID ") for t in texts), "no M1 'person' labels in tracked mode"


class ExplodingOnceTracker(FakeTracker):
    """First update raises; the capture loop must skip the frame, not die."""

    def update(
        self, detections: list[tuple[float, float, float, float, float]], ts_s: float
    ) -> list[FakeTrack]:
        first = not self.calls
        result = super().update(detections, ts_s)
        if first:
            raise RuntimeError("kalman went singular")
        return result


def test_tracker_failure_skips_frame_but_worker_survives() -> None:
    worker, _ = make_worker(cam_fps=100.0, tracker=ExplodingOnceTracker(CONFIRMED))
    worker.start()
    try:
        batches = drain(worker, min_batches=2)
    finally:
        worker.stop()

    assert len(batches) >= 2, "kept streaming after the tracker blew up once"


def test_module_imports_without_scipy_or_mtmc_tracker() -> None:
    """The tracker is injected, never imported: importing mtmc.camera must not
    pull in mtmc.tracker or scipy (contract M2: lazy, tracked-mode-only cost)."""
    code = (
        "import sys, mtmc.camera; "
        "assert 'mtmc.tracker' not in sys.modules; "
        "assert 'scipy' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
