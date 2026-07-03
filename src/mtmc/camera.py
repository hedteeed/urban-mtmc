"""mtmc.camera — a real video source (webcam / RTSP) as an Observation stream.

CONTRACT.md §M1: ``cv2.VideoCapture`` over the source, frames read
continuously (freshest wins), detection sampled at ~``cam_fps``. With no
tracker, each detection becomes ONE ``Observation``: ``floor_xy=None`` (no
calibration until M3), ``global_id=None`` (real pipelines carry no ground
truth), and ``track_id`` = a fresh monotonic id — every detection is its own
one-frame tracklet, so invariant 2 holds and the fragmentation is honest.
``ts_s`` is seconds since worker start on a monotonic clock (invariant 1).

CONTRACT.md §M2: an injected tracker replaces the per-detection ids. The
tracker sees EVERY detection plus the frame's ``ts_s``; Observations are
emitted only for the CONFIRMED tracks it returns (its ``track_id``/``conf``),
and the preview labels become ``"ID {track_id} · {conf:.2f}"``.
``tracker=None`` preserves M1 behaviour exactly.

Constraints honoured here:

* The detector AND the tracker are INJECTED, never imported — this module
  imports without onnxruntime or scipy present, and tests inject fakes
  (no device, no model file, no mtmc.tracker).
* Batches cross the thread boundary through a SMALL queue with drop-oldest
  overflow: a stalled consumer resumes on fresh data, never a backlog.
* Camera open failure or stream loss logs, sleeps, and reconnects in a
  loop — the process never dies with the camera.
* ``latest_jpeg`` (annotated boxes + conf labels) is kept under a lock for
  the ``/video.mjpg`` preview.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Protocol

import cv2
import numpy as np

from mtmc.events import Observation

logger = logging.getLogger("mtmc.camera")

DEFAULT_QUEUE_MAXSIZE = 8  # small on purpose: fresher beats fuller
DEFAULT_RECONNECT_DELAY_S = 2.0
DEFAULT_JPEG_QUALITY = 80
_FAST_READ_S = 0.001  # a read this quick means a non-blocking source...
_SPIN_WAIT_S = 0.002  # ...so pause a sliver instead of busy-spinning

_BOX_BGR = (255, 229, 0)  # dashboard accent #00e5ff, as BGR


class DetectorLike(Protocol):
    """The ``mtmc.detector.PersonDetector`` surface (injected, never imported)."""

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]: ...


class TrackLike(Protocol):
    """The ``mtmc.tracker.Track`` surface the worker reads (a frozen dataclass
    on agent A's side; read-only properties keep both satisfiable)."""

    @property
    def track_id(self) -> int: ...

    @property
    def box(self) -> tuple[float, float, float, float]: ...

    @property
    def conf(self) -> float: ...


class TrackerLike(Protocol):
    """The ``mtmc.tracker.ByteTracker`` surface (injected, never imported)."""

    def update(
        self, detections: list[tuple[float, float, float, float, float]], ts_s: float
    ) -> list[TrackLike]: ...


class CaptureLike(Protocol):
    """The slice of ``cv2.VideoCapture`` the worker uses (fakeable in tests)."""

    def isOpened(self) -> bool: ...  # noqa: N802 — cv2's casing

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


def _open_capture(source: str) -> CaptureLike:
    """``"0"``/``"1"`` -> local webcam index; anything else (rtsp://…) is a URL."""
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    # Network sources: a half-dead RTSP peer can block read() for tens of
    # seconds inside FFmpeg, stalling stop() and reconnect. Bound both open
    # and read so a stuck stream returns False and the reconnect loop runs.
    return cv2.VideoCapture(
        source,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
    )


class CameraWorker:
    """Capture thread; public surface: ``start``/``stop``/``get_batch``/``latest_jpeg``."""

    def __init__(
        self,
        source: str,
        detector: DetectorLike,
        cam_fps: float = 5.0,
        camera_id: str = "live0",
        *,
        tracker: TrackerLike | None = None,  # None = M1 per-detection ids (contract M2)
        capture_factory: Callable[[str], CaptureLike] = _open_capture,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ) -> None:
        if cam_fps <= 0:
            raise ValueError("cam_fps must be > 0")
        self.camera_id = camera_id
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._period_s = 1.0 / cam_fps
        self._capture_factory = capture_factory
        self._reconnect_delay_s = reconnect_delay_s
        self._jpeg_quality = jpeg_quality
        self._queue: queue.Queue[list[Observation]] = queue.Queue(maxsize=queue_maxsize)
        self._track_ids = itertools.count()  # never reset, even across reconnects (invariant 2)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None

    # -- public ------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return  # idempotent
        self._stop.clear()
        self._t0 = time.monotonic()  # ts_s = monotonic seconds since worker start
        self._thread = threading.Thread(
            target=self._run, name=f"mtmc-camera-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self, join_timeout_s: float = 5.0) -> None:
        """Signal the capture thread and join it; safe to call twice."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_s)
        self._thread = None

    def get_batch(self, timeout: float = 0.5) -> list[Observation] | None:
        """Blocking read for the asyncio hub's executor bridge; None when idle."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    # -- capture thread ------------------------------------------------------

    def _run(self) -> None:
        cap: CaptureLike | None = None
        next_detect_s = time.monotonic()
        try:
            while not self._stop.is_set():
                if cap is None:
                    cap = self._try_open()
                    if cap is None:  # open failed: back off, retry, never crash
                        self._stop.wait(self._reconnect_delay_s)
                        continue
                    next_detect_s = time.monotonic()
                read_t0 = time.monotonic()
                try:
                    ok, frame = cap.read()
                except Exception:
                    logger.exception("camera %s: read raised", self.camera_id)
                    ok, frame = False, None
                if not ok or frame is None:  # stream loss: reopen after a delay
                    logger.warning(
                        "camera %s: stream lost, reconnecting in %.1fs",
                        self.camera_id,
                        self._reconnect_delay_s,
                    )
                    self._release_quietly(cap)
                    cap = None
                    self._stop.wait(self._reconnect_delay_s)
                    continue
                now = time.monotonic()
                if now < next_detect_s:
                    # Keep draining the source so `frame` stays the freshest;
                    # only a non-blocking source needs an anti-spin pause.
                    if now - read_t0 < _FAST_READ_S:
                        self._stop.wait(min(_SPIN_WAIT_S, next_detect_s - now))
                    continue
                self._process(frame)
                next_detect_s += self._period_s
                if next_detect_s <= time.monotonic():  # detector slower than the
                    next_detect_s = time.monotonic() + self._period_s  # period: resync, no burst
        finally:
            if cap is not None:
                self._release_quietly(cap)

    def _try_open(self) -> CaptureLike | None:
        try:
            cap = self._capture_factory(self._source)
            if cap.isOpened():
                logger.info("camera %s: source %r opened", self.camera_id, self._source)
                return cap
            logger.warning("camera %s: cannot open source %r", self.camera_id, self._source)
            self._release_quietly(cap)
        except Exception:
            logger.exception("camera %s: opening %r raised", self.camera_id, self._source)
        return None

    @staticmethod
    def _release_quietly(cap: CaptureLike) -> None:
        try:
            cap.release()
        except Exception:  # noqa: BLE001 — a dead handle may refuse release
            pass

    def _process(self, frame: np.ndarray) -> None:
        ts_s = time.monotonic() - self._t0
        try:
            detections = self._detector.detect(frame)
        except Exception:
            logger.exception("camera %s: detector failed; frame skipped", self.camera_id)
            return
        if self._tracker is None:
            labelled = [
                ((x1, y1, x2, y2), f"person {conf:.2f}") for x1, y1, x2, y2, conf in detections
            ]
            batch = [
                Observation(
                    ts_s=ts_s,
                    camera=self.camera_id,
                    track_id=next(self._track_ids),  # one-frame tracklet (contract M1)
                    floor_xy=None,  # no calibration until M3
                    conf=min(1.0, max(0.0, float(conf))),  # invariant 5
                    global_id=None,  # real pipelines carry no ground truth
                )
                for _x1, _y1, _x2, _y2, conf in detections
            ]
        else:
            # The tracker sees EVERY detection — ByteTrack's round-2 rescue
            # feeds on the low-conf ones a naive threshold would discard —
            # and every sampled frame, even an empty one: coasting/expiry
            # advance on ts_s, not on detections (contract M2).
            try:
                tracks = self._tracker.update(detections, ts_s)
            except Exception:
                logger.exception("camera %s: tracker failed; frame skipped", self.camera_id)
                return
            labelled = [(t.box, f"ID {t.track_id} · {t.conf:.2f}") for t in tracks]
            batch = [
                Observation(
                    ts_s=ts_s,
                    camera=self.camera_id,
                    track_id=t.track_id,  # stable, never reused — tracker's invariant 2
                    floor_xy=None,  # no calibration until M3
                    conf=min(1.0, max(0.0, float(t.conf))),  # invariant 5
                    global_id=None,  # real pipelines carry no ground truth
                )
                for t in tracks  # CONFIRMED only: tentative tracks never emit
            ]
        # JPEG first, then queue: a consumer woken by the queue may read
        # latest_jpeg immediately and must never see a stale/absent preview.
        self._store_jpeg(self._annotate(frame, labelled))
        if batch:  # empty frames produce no events, matching the sim's cadence
            self._push(batch)

    def _push(self, batch: list[Observation]) -> None:
        """Drop-OLDEST on overflow: backpressure by staleness, not memory."""
        while True:
            try:
                self._queue.put_nowait(batch)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:  # consumer drained it concurrently — fine
                    pass

    def _annotate(
        self,
        frame: np.ndarray,
        labelled_boxes: list[tuple[tuple[float, float, float, float], str]],
    ) -> bytes | None:
        # Labels draw as a filled chip + large dark text (readable on the
        # dashboard's scaled-down tile). Hershey fonts render non-ASCII (the
        # contract label's middot) as "?", so the DRAWN text swaps it for a
        # hyphen; the contract string itself is unchanged elsewhere.
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        for (x1, y1, x2, y2), label in labelled_boxes:
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(frame, p1, p2, _BOX_BGR, 3)
            text = label.replace(" · ", " - ")
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
            # Chip above the box; if the box touches the frame top, inside it.
            cy = p1[1] - 8 if p1[1] - th - baseline - 8 >= 0 else p1[1] + th + baseline + 8
            cv2.rectangle(
                frame,
                (p1[0], cy - th - baseline),
                (p1[0] + tw + 10, cy + baseline),
                _BOX_BGR,
                -1,
            )
            cv2.putText(frame, text, (p1[0] + 5, cy), font, scale, (10, 15, 20), thick, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        return buf.tobytes() if ok else None

    def _store_jpeg(self, jpeg: bytes | None) -> None:
        if jpeg is None:
            return
        with self._jpeg_lock:
            self._latest_jpeg = jpeg
