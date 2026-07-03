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
