"""Event schema v1 — the frozen contract between simulator/pipeline and dashboard.

This is the M0 ancestor of the live event stream a real camera pipeline will
emit (detector -> tracker -> Re-ID -> homography -> this). The dashboard is a
pure consumer: it performs no geometry and never branches on whether events
came from the simulator or from real cameras.

Invariants consumers may rely on (change requires a version bump):

  1. ``ts_s`` is seconds on ONE shared clock, monotonically non-decreasing
     across ticks. Streams are joined by time, never by arrival order.
  2. ``(camera, track_id)`` identifies one per-camera tracklet. Track ids are
     stable while a person stays in view and are NEVER reused within a run.
  3. ``floor_xy`` is metres in the frame of the floor the OBSERVING CAMERA
     is mounted on (origin top-left, x right, y down); the floor itself is
     derived from ``camera`` via the plan — cameras never see other floors.
     Absent calibration/projection it is ``None`` — but the observation is
     still emitted (boxes survive without geometry).
  4. ``global_id`` is GROUND TRUTH from the simulator, present only for
     debugging/eval overlays. Real pipelines omit it until the cross-camera
     engine (M6) assigns one. Dashboards must render correctly without it.
  5. ``conf`` is in [0, 1].
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Observation:
    """One tracklet update from one camera at one instant."""

    ts_s: float                      # shared-clock timestamp, seconds
    camera: str                      # camera id, e.g. "cam3"
    track_id: int                    # per-camera tracklet id (invariant 2)
    floor_xy: tuple[float, float] | None   # metres on floor plan (invariant 3)
    conf: float                      # detector/tracker confidence, 0..1
    global_id: int | None = None     # ground truth only (invariant 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["floor_xy"] = list(self.floor_xy) if self.floor_xy is not None else None
        return d


def tick_message(ts_s: float, observations: list[Observation]) -> str:
    """One WebSocket frame: everything all cameras saw this tick.

    Shape: {"type": "tick", "v": 1, "ts_s": float, "observations": [...]}
    """
    return json.dumps(
        {
            "type": "tick",
            "v": SCHEMA_VERSION,
            "ts_s": round(ts_s, 3),
            "observations": [o.to_dict() for o in observations],
        }
    )
