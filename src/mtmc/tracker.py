"""mtmc.tracker — ByteTrack over constant-velocity Kalman boxes (CONTRACT.md §M2).

Self-contained: numpy + scipy's ``linear_sum_assignment`` only — no cv2, no
onnxruntime — so the module imports alone, tests included.

Conventions this module pins (tests enforce every one):

* Association cost is ``1 - IoU`` between the Kalman-PREDICTED track box and
  the detection box. A Hungarian pair survives only when
  ``cost <= match_thresh``, i.e. ``IoU >= 1 - match_thresh`` (defaults: pairs
  below IoU 0.2 are rejected). Both rounds use the same gate.
* Round 1: ALL live tracks — confirmed, coasting and tentative in one pool,
  Hungarian arbitrates globally — vs detections with conf >= track_thresh.
  Round 2 (the ByteTrack rescue): tracks left unmatched by round 1 vs
  detections with conf in [0.1, track_thresh); detections under 0.1 are
  dropped (the contract floor). Only round-1 leftovers with
  conf >= track_thresh may START a track — low-conf detections can rescue an
  existing track but never create one.
* ``update()`` returns confirmed tracks matched AT THIS CALL only. A coasting
  confirmed track keeps its id and stays matchable while
  ``ts_s - last_match_ts <= track_buffer_s``, but is silent until re-matched
  (consumers already tolerate gaps: the dashboard drops a dot after 3 s
  unseen). Past the buffer the track is retired BEFORE association, so a
  stale prediction can never swallow a fresh detection.
* Tentative tracks carry NO public id. One ``itertools.count`` hands out ids
  at CONFIRMATION — ``min_hits`` consecutive matches, the spawning detection
  counting as hit 1, one miss while tentative kills the track. A ghost that
  dies tentative therefore never burns a number: emitted ids are strictly
  increasing in confirmation order and never reused (events invariant 2).
  ``min_hits <= 1`` degenerates to confirm-on-sight.
* Time is the caller's ``ts_s`` on one shared clock. dt between calls may
  jitter; a non-positive delta degrades to dt = 0 (predict is then the
  identity — the very first update of a track therefore moves nothing:
  velocity starts at 0). No wall clock, no RNG anywhere in this module.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

_MIN_CONF = 0.1  # contract floor: detections below this never participate

# Noise scales follow ByteTrack practice: uncertainty is proportional to box
# height, because near/large targets move more pixels per second than
# far/small ones. Velocities are per SECOND here (dt comes from ts_s), so the
# per-frame ByteTrack weights are restated per second at the nominal 5 fps.
_STD_POS = 1.0 / 20.0  # position/size std, fraction of box height
_STD_VEL = 1.0 / 32.0  # velocity process std, fraction of height per second
_INIT_VEL_STD = 1.0  # loose velocity prior (fraction of height per second):
#                      the prior MEAN is 0 so the first predict moves nothing,
#                      but two updates suffice to lock onto the real motion —
#                      that convergence is what carries ids through crossings.

_EPS = 1e-9


@dataclass(frozen=True)
class Track:
    """One confirmed track as of the ``update()`` call that returned it."""

    track_id: int  # monotonic, never reused (invariant 2)
    box: tuple[float, float, float, float]  # Kalman-filtered, xyxy source pixels
    conf: float  # last matched detection's conf


class _KalmanBox:
    """Constant-velocity filter over state (cx, cy, w, h, vcx, vcy, vw, vh)."""

    __slots__ = ("x", "P")

    def __init__(self, z: np.ndarray) -> None:
        # Velocity prior: mean 0 (first predict must not move the box) with a
        # covariance loose enough that real motion is learned in ~2 updates.
        self.x = np.concatenate([z, np.zeros(4)])
        h = max(float(z[3]), 1.0)
        self.P = np.diag([(2.0 * _STD_POS * h) ** 2] * 4 + [(_INIT_VEL_STD * h) ** 2] * 4)

    def predict(self, dt: float) -> None:
        """Advance by dt seconds. dt may jitter call-to-call; dt <= 0 is a no-op."""
        if dt <= 0.0:
            return
        f = np.eye(8)
        f[0, 4] = f[1, 5] = f[2, 6] = f[3, 7] = dt
        h = max(float(self.x[3]), 1.0)
        # Process variance grows linearly with elapsed time, so jittery or
        # missing samples accumulate exactly as much uncertainty as the gap.
        q = np.array([(_STD_POS * h) ** 2] * 4 + [(_STD_VEL * h) ** 2] * 4) * dt
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + np.diag(q)

    def correct(self, z: np.ndarray) -> None:
        """Fold in a measured (cx, cy, w, h). Measurement noise scales with height."""
        h = max(float(z[3]), 1.0)
        s = self.P[:4, :4] + np.diag([(_STD_POS * h) ** 2] * 4)
        gain = np.linalg.solve(s, self.P[:4, :]).T  # K = P Hᵀ S⁻¹, H = [I₄ | 0]
        self.x = self.x + gain @ (z - self.x[:4])
        self.P = self.P - gain @ self.P[:4, :]
        self.P = (self.P + self.P.T) * 0.5  # symmetry must survive float error

    @property
    def box(self) -> np.ndarray:
        """Current state as xyxy. Long coasts must never collapse or invert a box."""
        cx, cy, w, h = self.x[:4]
        w, h = max(float(w), 1.0), max(float(h), 1.0)
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class _TrackState:
    """Internal per-track record. ``track_id is None`` == tentative."""

    __slots__ = ("kf", "hits", "conf", "last_match_ts", "track_id")

    def __init__(self, z: np.ndarray, conf: float, ts_s: float) -> None:
        self.kf = _KalmanBox(z)
        self.hits = 1  # the spawning detection counts as the first hit
        self.conf = conf
        self.last_match_ts = ts_s
        self.track_id: int | None = None


def _to_cxcywh(d: tuple[float, float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = d[:4]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (Na, 4) x (Nb, 4) xyxy boxes -> (Na, Nb)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, _EPS)


class ByteTracker:
    """Two-round IoU association over Kalman predictions (CONTRACT.md §M2)."""

    def __init__(
        self,
        *,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer_s: float = 2.0,
        min_hits: int = 3,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer_s = track_buffer_s
        self.min_hits = min_hits
        self._tracks: list[_TrackState] = []  # creation order — output order too
        self._ids = itertools.count(1)  # one counter for the whole run: ids never reused
        self._last_ts: float | None = None

    def _associate(
        self,
        tracks: list[_TrackState],
        dets: list[tuple[float, float, float, float, float]],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian on cost = 1 - IoU; a pair is kept only if cost <= match_thresh.

        Returns (matches as (track_idx, det_idx), unmatched track idxs,
        unmatched det idxs) — all index into the arguments, all in
        deterministic ascending order.
        """
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        preds = np.stack([t.kf.box for t in tracks])
        boxes = np.array([d[:4] for d in dets], dtype=float)
        cost = 1.0 - _iou_matrix(preds, boxes)
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= self.match_thresh]
        got_t = {r for r, _ in matches}
        got_d = {c for _, c in matches}
        return (
            matches,
            [i for i in range(len(tracks)) if i not in got_t],
            [j for j in range(len(dets)) if j not in got_d],
        )

    def update(
        self,
        detections: list[tuple[float, float, float, float, float]],
        ts_s: float,
    ) -> list[Track]:
        """One tick: (x1, y1, x2, y2, conf) detections at ts_s -> confirmed tracks.

        Deterministic: same call sequence, same results. ts_s must be
        non-decreasing (events invariant 1); a backwards step is clamped to
        dt = 0 rather than corrupting the filters.
        """
        dt = 0.0 if self._last_ts is None else max(ts_s - self._last_ts, 0.0)
        self._last_ts = ts_s

        # Retire tracks already coasted past the buffer at this instant —
        # matchable "up to track_buffer_s" means at gap <= buffer, never beyond.
        self._tracks = [t for t in self._tracks if ts_s - t.last_match_ts <= self.track_buffer_s]

        for t in self._tracks:
            t.kf.predict(dt)

        dets = [d for d in detections if d[4] >= _MIN_CONF]
        high = [d for d in dets if d[4] >= self.track_thresh]
        low = [d for d in dets if d[4] < self.track_thresh]

        # Round 1: every live track vs high-conf detections.
        matches1, unmatched1, u_high = self._associate(self._tracks, high)
        # Round 2 (rescue): round-1 leftovers vs low-conf detections. Leftover
        # low-conf detections are discarded — they never start tracks.
        pool2 = [self._tracks[i] for i in unmatched1]
        matches2, _, _ = self._associate(pool2, low)

        matched: list[tuple[_TrackState, tuple[float, float, float, float, float]]] = [
            (self._tracks[i], high[j]) for i, j in matches1
        ] + [(pool2[i], low[j]) for i, j in matches2]

        matched_ids = {id(t) for t, _ in matched}
        for t, d in matched:
            t.kf.correct(_to_cxcywh(d))
            t.hits += 1
            t.conf = d[4]
            t.last_match_ts = ts_s
            if t.track_id is None and t.hits >= self.min_hits:
                t.track_id = next(self._ids)  # id exists only from confirmation on

        # One miss kills a tentative track ("consecutive" is literal);
        # confirmed tracks coast instead and answer to the buffer above.
        self._tracks = [t for t in self._tracks if t.track_id is not None or id(t) in matched_ids]

        # Only unmatched HIGH-conf detections may found a track.
        for j in u_high:
            t = _TrackState(_to_cxcywh(high[j]), high[j][4], ts_s)
            if self.min_hits <= 1:  # degenerate config: confirm on sight
                t.track_id = next(self._ids)
            self._tracks.append(t)

        out: list[Track] = []
        for t in self._tracks:  # creation order: deterministic output order
            if t.track_id is not None and t.last_match_ts == ts_s:
                b = t.kf.box
                out.append(
                    Track(
                        track_id=t.track_id,
                        box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        conf=float(t.conf),
                    )
                )
        return out
