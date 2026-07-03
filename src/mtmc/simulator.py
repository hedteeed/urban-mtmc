"""Synthetic crowd + camera simulator — M0 stand-in for a real detector/tracker.

Emits the frozen `mtmc.events.Observation` stream from seeded walkers on the
floor-plan waypoint graph. Constraints (CONTRACT.md §mtmc.simulator):
deterministic per seed — one ``random.Random``, no wall clock; per-camera
track ids obey the >2 s track-buffer rule and are never reused within a run;
observations carry gaussian position noise, dropout, and a distance-based
confidence. Cameras sample at ``obs_hz`` with per-camera phase stagger, so a
single ``step()`` may emit observations at several nearby timestamps.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from itertools import pairwise

from .events import Observation

# Walker model
SPEED_RANGE_MPS = (1.0, 1.6)     # per-person constant speed, drawn at spawn
DWELL_PROB = 0.35                # chance to pause when crossing a dwell point
DWELL_RANGE_S = (2.0, 8.0)
WANDER_PROB = 0.3                # chance a route detours via random waypoints
LATERAL_MAX_M = 0.9              # bound on heading-jitter drift off the edge line
SPAWNS_PER_DEFICIT_PER_S = 0.5   # spawn pressure per missing person

# Camera model
POS_NOISE_SIGMA_M = 0.15
CONF_NOISE_SIGMA = 0.04
DROPOUT_P = 0.05
TRACK_BUFFER_S = 2.0             # unseen longer than this => next sighting = new id


@dataclass
class _Person:
    """One walker: a route of waypoint names plus progress along it."""

    pid: int
    route: list[str]
    speed_mps: float
    leg: int = 0            # index into route of the current segment start
    along_m: float = 0.0    # distance travelled along the current segment
    lateral_m: float = 0.0  # heading-jitter offset, perpendicular to the segment
    dwell_left_s: float = 0.0
    done: bool = False


@dataclass
class _Camera:
    """Static camera plus per-run tracker state (ids monotonic, never reused)."""

    cid: str
    x: float
    y: float
    yaw_deg: float
    half_fov_deg: float
    range_m: float
    phase_s: float   # sampling stagger so cameras do not tick in lockstep
    next_k: int = 0  # next sample fires at phase_s + next_k * period
    next_tid: int = 1
    # pid -> (track_id, last_seen_ts); dropout does NOT refresh last_seen.
    tracks: dict[int, tuple[int, float]] = field(default_factory=dict)


class Simulation:
    """Deterministic waypoint-graph crowd observed by noisy virtual cameras.

    All randomness flows through one seeded ``random.Random``; two instances
    with the same seed, plan, and ``step`` call pattern produce identical
    observation streams (a feature, per the contract — never wall clock).
    """

    def __init__(
        self,
        plan: dict,
        seed: int = 42,
        n_people_target: int = 12,
        obs_hz: float = 5.0,
    ) -> None:
        if obs_hz <= 0:
            raise ValueError("obs_hz must be positive")
        self._rng = random.Random(seed)
        self._wp: dict[str, tuple[float, float]] = {
            name: (float(x), float(y)) for name, (x, y) in plan["waypoints"].items()
        }
        self._adj: dict[str, list[tuple[str, float]]] = {n: [] for n in self._wp}
        for a, b in plan["edges"]:
            d = math.dist(self._wp[a], self._wp[b])
            self._adj[a].append((b, d))
            self._adj[b].append((a, d))
        self._entrances: list[str] = list(plan["entrances"])
        self._dwell_points = frozenset(plan.get("dwell_points", ()))
        self._rects = [
            (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            for r in plan["walkable"]
        ]
        self._target = n_people_target
        self._period_s = 1.0 / obs_hz
        self._cameras = [
            _Camera(
                cid=c["id"],
                x=float(c["pos"][0]),
                y=float(c["pos"][1]),
                yaw_deg=float(c["yaw_deg"]),
                half_fov_deg=float(c["fov_deg"]) / 2.0,
                range_m=float(c["range_m"]),
                phase_s=self._rng.uniform(0.0, self._period_s),
            )
            for c in plan["cameras"]
        ]
        self._path_cache: dict[tuple[str, str], list[str] | None] = {}
        self._people: dict[int, _Person] = {}
        self._next_pid = 1
        self._clock = 0.0

    @property
    def clock_s(self) -> float:
        return self._clock

    @property
    def population(self) -> int:
        """Current walker count. Debug/test aid — not part of the frozen contract."""
        return len(self._people)

    def step(self, dt_s: float) -> list[Observation]:
        """Advance the world by dt_s. Returns observations produced during
        this step (possibly empty — cameras sample at obs_hz, not every step)."""
        if dt_s <= 0:
            return []
        t_end = self._clock + dt_s
        # Sample times are phase + k*period (no accumulation drift). Collect all
        # due in (clock, t_end], then advance the world to each in time order so
        # positions are exact at the sampling instant (invariant 1: ts_s ordered).
        events: list[tuple[float, int]] = []
        for ci, cam in enumerate(self._cameras):
            while (t := cam.phase_s + cam.next_k * self._period_s) <= t_end + 1e-9:
                if t > self._clock:
                    events.append((t, ci))
                cam.next_k += 1
        events.sort()
        out: list[Observation] = []
        now = self._clock
        for t, ci in events:
            self._advance(t - now)
            now = t
            out.extend(self._sample_camera(self._cameras[ci], t))
        self._advance(t_end - now)
        self._clock = t_end
        return out

    # -- world dynamics -----------------------------------------------------

    def _advance(self, dt: float) -> None:
        if dt <= 1e-12:
            return
        finished: list[int] = []
        for pid, person in self._people.items():
            self._move(person, dt)
            if person.done:
                finished.append(pid)
        for pid in finished:
            del self._people[pid]
            for cam in self._cameras:
                cam.tracks.pop(pid, None)  # pids never recur; keep state bounded
        deficit = self._target - len(self._people)
        if deficit > 0 and self._rng.random() < min(1.0, SPAWNS_PER_DEFICIT_PER_S * deficit * dt):
            self._spawn()

    def _move(self, p: _Person, dt: float) -> None:
        # Heading jitter as a bounded lateral random walk with pull-back, so
        # true positions stay well inside walkable rects (noise margin 0.5 m).
        p.lateral_m += self._rng.gauss(0.0, 0.35) * math.sqrt(dt) - 0.8 * p.lateral_m * dt
        p.lateral_m = max(-LATERAL_MAX_M, min(LATERAL_MAX_M, p.lateral_m))
        t = dt
        while t > 1e-12 and not p.done:
            if p.dwell_left_s > 0.0:
                used = min(p.dwell_left_s, t)
                p.dwell_left_s -= used
                t -= used
                continue
            a = self._wp[p.route[p.leg]]
            b = self._wp[p.route[p.leg + 1]]
            seg = math.dist(a, b)
            if seg <= 1e-9:  # degenerate edge; skip it
                self._arrive(p)
                continue
            remaining = seg - p.along_m
            if p.speed_mps * t < remaining:
                p.along_m += p.speed_mps * t
                t = 0.0
            else:
                t -= remaining / p.speed_mps
                self._arrive(p)

    def _arrive(self, p: _Person) -> None:
        p.leg += 1
        p.along_m = 0.0
        if p.leg >= len(p.route) - 1:
            p.done = True  # reached destination entrance -> despawn
        elif p.route[p.leg] in self._dwell_points and self._rng.random() < DWELL_PROB:
            p.dwell_left_s = self._rng.uniform(*DWELL_RANGE_S)

    def _spawn(self) -> None:
        if len(self._entrances) < 2:
            return
        rng = self._rng
        src = rng.choice(self._entrances)
        dst = rng.choice([e for e in self._entrances if e != src])
        route = self._route(src, dst)
        if route is None or len(route) < 2:
            return
        self._people[self._next_pid] = _Person(
            pid=self._next_pid,
            route=route,
            speed_mps=rng.uniform(*SPEED_RANGE_MPS),
        )
        self._next_pid += 1

    def _route(self, src: str, dst: str) -> list[str] | None:
        """Shortest path, or (WANDER_PROB) a detour via 1-2 random waypoints."""
        legs = [src]
        if self._rng.random() < WANDER_PROB:
            others = [w for w in self._wp if w != src and w != dst]
            k = min(self._rng.randint(1, 2), len(others))
            legs.extend(self._rng.sample(others, k))
        legs.append(dst)
        full = [src]
        for a, b in pairwise(legs):
            hop = self._shortest(a, b)
            if hop is None:
                return None
            full.extend(hop[1:])
        return full

    def _shortest(self, src: str, dst: str) -> list[str] | None:
        key = (src, dst)
        if key not in self._path_cache:
            best = {src: 0.0}
            prev: dict[str, str] = {}
            pq: list[tuple[float, str]] = [(0.0, src)]  # (dist, name): deterministic ties
            while pq:
                d, node = heapq.heappop(pq)
                if d > best.get(node, math.inf):
                    continue
                if node == dst:
                    break
                for nxt, w in self._adj[node]:
                    nd = d + w
                    if nd < best.get(nxt, math.inf):
                        best[nxt] = nd
                        prev[nxt] = node
                        heapq.heappush(pq, (nd, nxt))
            if dst not in best:
                self._path_cache[key] = None
            else:
                path = [dst]
                while path[-1] != src:
                    path.append(prev[path[-1]])
                self._path_cache[key] = path[::-1]
        return self._path_cache[key]

    # -- geometry -----------------------------------------------------------

    def _position(self, p: _Person) -> tuple[float, float]:
        """True position: along-edge point + tapered lateral offset.

        The taper zeroes the offset near waypoints (no jump when the edge —
        and hence the perpendicular — changes) and the shrink loop guarantees
        the point stays inside the walkable union.
        """
        if p.leg >= len(p.route) - 1:
            return self._wp[p.route[-1]]
        ax, ay = self._wp[p.route[p.leg]]
        bx, by = self._wp[p.route[p.leg + 1]]
        seg = math.dist((ax, ay), (bx, by))
        if seg <= 1e-9:
            return (ax, ay)
        ux, uy = (bx - ax) / seg, (by - ay) / seg
        x, y = ax + ux * p.along_m, ay + uy * p.along_m
        taper = max(0.0, min(1.0, p.along_m / 2.0, (seg - p.along_m) / 2.0))
        lat = p.lateral_m * taper
        for shrink in (1.0, 0.5, 0.25, 0.0):
            cx, cy = x - uy * lat * shrink, y + ux * lat * shrink
            if self._walkable(cx, cy):
                return (cx, cy)
        return (x, y)  # on-edge point is always walkable by plan construction

    def _walkable(self, x: float, y: float) -> bool:
        eps = 1e-9
        return any(
            rx - eps <= x <= rx + rw + eps and ry - eps <= y <= ry + rh + eps
            for rx, ry, rw, rh in self._rects
        )

    # -- observation model ---------------------------------------------------

    def _sample_camera(self, cam: _Camera, ts: float) -> list[Observation]:
        rng = self._rng
        out: list[Observation] = []
        for pid, person in self._people.items():
            px, py = self._position(person)
            dx, dy = px - cam.x, py - cam.y
            dist = math.hypot(dx, dy)
            if dist > cam.range_m:
                continue
            # y-down + clockwise-positive yaw means plain atan2(dy, dx) already
            # measures in the plan's convention.
            bearing = math.degrees(math.atan2(dy, dx))
            off = (bearing - cam.yaw_deg + 180.0) % 360.0 - 180.0
            if abs(off) > cam.half_fov_deg:
                continue
            if rng.random() < DROPOUT_P:
                continue  # silent miss; last_seen not refreshed
            known = cam.tracks.get(pid)
            if known is None or ts - known[1] > TRACK_BUFFER_S:
                tid = cam.next_tid  # fresh id — old ids are never reused
                cam.next_tid += 1
            else:
                tid = known[0]
            cam.tracks[pid] = (tid, ts)
            conf = 0.9 - 0.3 * dist / cam.range_m + rng.gauss(0.0, CONF_NOISE_SIGMA)
            out.append(
                Observation(
                    ts_s=ts,
                    camera=cam.cid,
                    track_id=tid,
                    floor_xy=(
                        px + rng.gauss(0.0, POS_NOISE_SIGMA_M),
                        py + rng.gauss(0.0, POS_NOISE_SIGMA_M),
                    ),
                    conf=max(0.0, min(1.0, conf)),
                    global_id=pid,
                )
            )
        return out
