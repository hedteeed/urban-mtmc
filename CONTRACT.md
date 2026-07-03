# M0 Contract — frozen interfaces

Three components are built independently against this document. Change it
first, bump versions, then change code (the mantis manifest discipline).

## Coordinate system

Metres. Per-floor frames: origin top-left of each floor, x right, y down.
`yaw_deg`: 0 = +x (east), positive clockwise (so +90 faces +y / south).
Floor plan definition: [`floorplan/plan.json`](floorplan/plan.json), schema v2:

- `floors[]` — each with `id`, `name`, `size_m`, `walkable[]` rects,
  `rooms[]` (visual), `stairs[]` footprint rects (with `to` floor id).
- `waypoints` — `{name: {floor, xy}}`. `edges` — plain `[a, b]` pairs or
  `{from, to, kind: "stairs"}` for cross-floor connections.
- `cameras[]` — each carries a `floor` id. **A camera only ever observes
  people on its own floor.** The event schema does not carry a floor field:
  consumers derive an observation's floor from its `camera` via the plan.

## Event stream

Defined in [`src/mtmc/events.py`](src/mtmc/events.py) — `Observation`,
`tick_message()`, `SCHEMA_VERSION = 1`, five written invariants.
WebSocket frames are `tick_message` JSON: one frame per simulator tick
containing all cameras' observations for that instant.

## Module interfaces

### `mtmc.simulator` (owner: agent A)

```python
class Simulation:
    def __init__(self, plan: dict, seed: int = 42,
                 n_people_target: int = 12, obs_hz: float = 5.0): ...
    @property
    def clock_s(self) -> float: ...          # simulation clock, starts at 0.0
    def step(self, dt_s: float) -> list[Observation]:
        """Advance the world by dt_s. Returns observations produced during
        this step (possibly empty — cameras sample at obs_hz, not every step)."""
```

Rules:
- Deterministic for a given seed. NO wall clock, NO global random — a seeded
  `random.Random` instance only (mantis lesson: reproducibility is a feature).
- People walk the waypoint graph between entrances at 1.0–1.6 m/s with jitter,
  sometimes dwelling at `dwell_points`. Spawn/despawn at `entrances`, holding
  the population near `n_people_target`.
- Multi-floor: every walker is on exactly one floor at any instant (the floor
  of the edge endpoint they are nearest along a stairs edge; floor flips at
  the edge midpoint). On `kind: "stairs"` edges speed drops to ~0.6 m/s and
  heading jitter is suppressed. Walkable-containment checks use the walker's
  current floor's rects. A camera observes a walker only when
  `walker.floor == camera.floor` (plus range/FOV as before).
- A camera observes a person when within `range_m` AND within `fov_deg/2` of
  its yaw. Per observation: gaussian position noise (σ≈0.15 m), dropout
  (~5% of samples silently missed), `conf = clamp(0.9 − 0.3·dist/range ± noise)`.
- Per-camera `track_id`: assigned when a person enters a camera's view; if
  unseen by that camera for >2 s, the next sighting gets a NEW track id
  (simulates a real tracker's `track_buffer`). Ids never reused (invariant 2).
- `global_id` = the person's true id, attached to every observation
  (dashboard treats it as debug-only).

### `mtmc.server` (owner: agent B)

```python
def main() -> None: ...   # console script `mtmc-server`; uvicorn on :8100
```

- `GET /` → serves `web/index.html`; `/static/*` → files in `web/`.
- `GET /api/plan` → contents of `floorplan/plan.json`.
- `WS /ws` → on connect, stream `tick_message` frames in real time:
  run `Simulation` at 10 Hz steps, pace with `asyncio.sleep`, broadcast each
  non-empty tick to all connected clients. One shared simulation per server
  (all clients see the same world). Client disconnect must not kill the sim.
- CLI flags: `--port` (8100), `--seed` (42), `--people` (12).

### M1 additions — real camera source

One server, two source modes; the event stream and dashboard contract do not
change shape (mantis lesson: consumers never branch on source).

- New CLI flag `--source` (default `"sim"`): `sim` | webcam index (`"0"`) |
  RTSP URL (`rtsp://…`). In camera mode the hub broadcasts observations from
  `mtmc.camera.CameraWorker` instead of the simulator.
- `mtmc.detector.PersonDetector` (owner: agent A): ONNX Runtime, COCO-
  pretrained YOLOX-S at 640, mantis inference conventions exactly — letterbox
  pad 114 top-left BGR 0-255 no normalization, score = obj × class, person
  class only, class-aware NMS (IoU 0.7), un-letterbox to source pixels.
  `detect(frame_bgr) -> list[(x1, y1, x2, y2, conf)]`. Model file lives at
  `models/yolox_s.onnx` (gitignored); `python -m mtmc.get_model` downloads it
  from a pinned URL and verifies a pinned sha256. Detector raises a clear
  error naming that command if the file is missing. All detector tests that
  need weights skip when the file is absent (mantis oracle-test pattern).
- `mtmc.camera.CameraWorker` (owner: agent B): cv2.VideoCapture over the
  source; samples ~5 fps (`--cam-fps`); per frame: detect → emit one
  `Observation` per detection with `camera="live0"`, `floor_xy=None`,
  `global_id=None`, `conf` from the detector, and `track_id` = a fresh
  monotonic id per detection (detector-only milestone: every detection is its
  own one-frame tracklet — invariant 2 holds, fragmentation is expected and
  honest; M2 replaces this with ByteTrack). Timestamps: seconds since worker
  start on a monotonic clock. Keeps `latest_jpeg` (annotated boxes + conf
  labels, JPEG-encoded) for the preview stream. Runs in a thread; asyncio
  hub reads its queue. Camera loss → log + reconnect loop, server stays up.
- New endpoints: `GET /api/source` → `{"mode": "sim"|"camera", "camera_id":
  "live0"|null}`; `GET /video.mjpg` → multipart/x-mixed-replace MJPEG of
  `latest_jpeg` (camera mode only; 404 in sim mode).

### M2 additions — real tracking (ByteTrack)

- `mtmc.tracker` (owner: agent A). Self-contained ByteTrack: numpy + scipy
  (`linear_sum_assignment`) only; no cv2, no onnxruntime, importable alone.

```python
class ByteTracker:
    def __init__(self, *, track_thresh: float = 0.5, match_thresh: float = 0.8,
                 track_buffer_s: float = 2.0, min_hits: int = 3): ...
    def update(self, detections: list[tuple[float, float, float, float, float]],
               ts_s: float) -> list[Track]:
        """detections = (x1, y1, x2, y2, conf) in source pixels, ANY conf ≥
        ~0.1. Returns CONFIRMED tracks only, each with a stable track_id."""

@dataclass(frozen=True)
class Track:
    track_id: int          # monotonic, never reused (invariant 2)
    box: tuple[float, float, float, float]
    conf: float            # last matched detection's conf
```

  Mechanics per update: Kalman predict (state cx, cy, w, h + velocities,
  constant-velocity, dt from ts_s deltas — sampling may jitter); round 1 =
  Hungarian on IoU cost between predictions and detections with conf ≥
  track_thresh, gated at match_thresh; round 2 = remaining unmatched tracks
  vs detections with conf in [0.1, track_thresh) (the ByteTrack rescue);
  unmatched detections above track_thresh start TENTATIVE tracks, confirmed
  after min_hits consecutive matches (never emitted while tentative);
  unmatched tracks coast (prediction only) up to track_buffer_s seconds,
  reclaimable meanwhile, then die. Deterministic: no wall clock, no RNG.

- `mtmc.camera` (owner: agent B): CameraWorker grows a `tracker` argument
  (None = M1 per-detection ids, preserved for tests). With a tracker:
  detector runs at conf ≥ 0.1, tracker consumes ALL detections, Observations
  are emitted per CONFIRMED track (track_id from the tracker, conf = track
  conf). Annotated JPEG labels become "ID {track_id} · {conf:.2f}". Server
  default in camera mode: tracker ON with contract defaults; flags
  `--track-thresh`, `--match-thresh`, `--track-buffer` (seconds),
  `--min-hits`. `--no-track` restores M1 behavior.
- Dashboard: no changes required by contract (same wire shape); sidebar
  counts simply become person-scaled.
- Dashboard (owner: agent C): on load fetch `/api/source`; in camera mode
  show a CAMERA panel (the MJPEG `<img>`) beside the floor panels, badge
  switches to "M1 · LIVE CAMERA", and observations with `floor_xy=null`
  count in sidebar totals (already contract) — no dots expected until M3
  calibration. Camera id `live0` gets its own sidebar row under a "LIVE"
  heading.

### `web/` dashboard (owner: agent C)

Static, no build step, no external network (works offline).
`index.html` + `app.js` + `style.css`, plain ES6, canvas 2D rendering.

- Fetch `/api/plan`, render EVERY floor as its own panel (side by side when
  wide, stacked when narrow), each labeled with the floor `name`, each with
  its own to-scale transform: walkable areas, rooms, stairs footprints
  (distinct hatch/label), waypoint graph (subtle), cameras as wedge-shaped
  view cones (translucent). Observations render on the panel of their
  camera's floor (derive via plan).
- Ground-truth mode: a person's path renders per floor; when the true path
  crosses floors, draw a short dashed connector from the stair footprint on
  one panel to the stair footprint on the other (visual handoff cue).
- Connect to `ws://<host>/ws`, consume tick frames.
- Dots keyed by `(camera, track_id)` — a MARKER POOL: move existing dots,
  never recreate; drop a dot if its tracklet is unseen for 3 s. Dot color =
  camera color (7-color palette, one per camera). Trail: last ~4 s, fading.
- IMPORTANT M0 truth: one person in two cameras' view = TWO dots (fusion does
  not exist yet). A toggle "ground truth" recolors dots by `global_id` and
  draws the true continuous path — showing exactly the gap M5/M6 will close.
- Sidebar: connection status chip, sim clock, event rate (obs/s), per-camera
  active-tracklet counts, total distinct tracklets seen.
- Header: project name + "M0 · SIMULATOR" badge.
- Dark ops-console look: bg #060a10, panel #0d141d, accent #00e5ff, per-camera
  palette; JetBrains-Mono-ish monospace via system font stack.

## Definition of done (M0)

`uv run mtmc-server` → open http://localhost:8100 → simulated people walk the
mall; dots glide inside camera cones; overlap zone shows double dots; ground
-truth toggle reveals continuous paths; `uv run pytest` green.
