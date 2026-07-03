"""Contract tests for mtmc.simulator (CONTRACT.md §mtmc.simulator)."""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

from mtmc.events import Observation
from mtmc.simulator import STAIRS_SPEED_MPS, Simulation, _Person

PLAN = json.loads(
    (Path(__file__).resolve().parent.parent / "floorplan" / "plan.json").read_text()
)

# Tiny single-floor v2 plan engineered so every walker's only route (A-B-C)
# crosses camG's wedge (range 7 m), leaves it for >2 s around B, and re-enters
# on the way back — forcing the track-buffer rule deterministically.
GAP_PLAN = {
    "version": 2,
    "name": "gap-lab",
    "floors": [
        {
            "id": "g",
            "name": "GROUND",
            "size_m": [12, 4],
            "walkable": [{"x": 0, "y": 0, "w": 12, "h": 4, "name": "hall"}],
            "rooms": [],
            "stairs": [],
        }
    ],
    "waypoints": {
        "A": {"floor": "g", "xy": [2, 1.5]},
        "B": {"floor": "g", "xy": [11, 2]},
        "C": {"floor": "g", "xy": [2, 2.5]},
    },
    "edges": [["A", "B"], ["B", "C"]],
    "entrances": ["A", "C"],
    "dwell_points": [],
    "cameras": [
        {"id": "camG", "floor": "g", "pos": [0, 2], "yaw_deg": 0, "fov_deg": 100, "range_m": 7}
    ],
}

# Tiny two-floor v2 plan with one stairs edge and one all-seeing camera per
# floor. Geometry is exact so crossing times can be computed analytically:
# GA->GS is 8 m flat, the stairs edge GS->US is 3 m (floor flips at 1.5 m),
# US->UA is 8 m flat. Both cameras cover their entire floor (fov 200, range 100).
CROSS_PLAN = {
    "version": 2,
    "name": "cross-lab",
    "floors": [
        {
            "id": "g",
            "name": "GROUND",
            "size_m": [10, 4],
            "walkable": [{"x": 0, "y": 0, "w": 10, "h": 4, "name": "g-hall"}],
            "rooms": [],
            "stairs": [{"x": 8, "y": 0, "w": 2, "h": 2, "to": "u"}],
        },
        {
            "id": "u",
            "name": "UPPER",
            "size_m": [10, 4],
            "walkable": [{"x": 0, "y": 0, "w": 10, "h": 4, "name": "u-hall"}],
            "rooms": [],
            "stairs": [{"x": 8, "y": 2, "w": 2, "h": 2, "to": "g"}],
        },
    ],
    "waypoints": {
        "GA": {"floor": "g", "xy": [1, 0.5]},
        "GS": {"floor": "g", "xy": [9, 0.5]},
        "US": {"floor": "u", "xy": [9, 3.5]},
        "UA": {"floor": "u", "xy": [1, 3.5]},
    },
    "edges": [["GA", "GS"], {"from": "GS", "to": "US", "kind": "stairs"}, ["US", "UA"]],
    "entrances": ["GA", "UA"],
    "dwell_points": [],
    "cameras": [
        {"id": "camG", "floor": "g", "pos": [5, -1], "yaw_deg": 90, "fov_deg": 200, "range_m": 100},
        {"id": "camU", "floor": "u", "pos": [5, -1], "yaw_deg": 90, "fov_deg": 200, "range_m": 100},
    ],
}


def run(plan: dict = PLAN, seed: int = 42, seconds: float = 30.0, dt: float = 0.1, **kw):
    sim = Simulation(plan, seed=seed, **kw)
    obs: list[Observation] = []
    for _ in range(round(seconds / dt)):
        obs.extend(sim.step(dt))
    return sim, obs


def tid_runs(obs: list[Observation]) -> dict[tuple[str, int], list[tuple[int, float, float]]]:
    """(camera, global_id) -> chronological runs of (track_id, first_ts, last_ts)."""
    runs: dict[tuple[str, int], list[tuple[int, float, float]]] = {}
    for o in obs:
        lst = runs.setdefault((o.camera, o.global_id), [])
        if lst and lst[-1][0] == o.track_id:
            lst[-1] = (o.track_id, lst[-1][1], o.ts_s)
        else:
            lst.append((o.track_id, o.ts_s, o.ts_s))
    return runs


def test_determinism_same_seed() -> None:
    _, a = run(seed=7)
    _, b = run(seed=7)
    assert len(a) > 100
    assert a == b  # byte-identical stream over 30 sim-seconds


def test_different_seeds_differ() -> None:
    _, a = run(seed=1)
    _, b = run(seed=2)
    assert a != b


def test_clock_and_ts_monotonic() -> None:
    sim, obs = run(seconds=10.0)
    assert math.isclose(sim.clock_s, 10.0)
    assert all(x.ts_s <= y.ts_s for x, y in pairwise(obs))
    assert all(0.0 < o.ts_s <= 10.0 for o in obs)


def test_track_ids_never_reused() -> None:
    _, obs = run(seconds=60.0)
    owner: dict[tuple[str, int], int] = {}
    for o in obs:
        key = (o.camera, o.track_id)
        assert owner.setdefault(key, o.global_id) == o.global_id, (
            f"track id {key} reused across people"
        )
    for runs in tid_runs(obs).values():
        ids = [tid for tid, _, _ in runs]
        assert len(set(ids)) == len(ids), "a track id was resurrected after a gap"
        assert ids == sorted(ids)  # fresh ids only ever move forward


def test_new_track_id_after_gap_over_2s() -> None:
    _, obs = run(plan=GAP_PLAN, seconds=60.0, n_people_target=3)
    assert obs, "gap-lab plan produced no observations"
    reentries = 0
    for runs in tid_runs(obs).values():
        for (_, _, prev_last), (_, next_first, _) in pairwise(runs):
            gap = next_first - prev_last
            assert gap > 2.0, f"new id issued after only {gap:.2f}s unseen"
            reentries += 1
    assert reentries >= 3, "expected several forced re-entries in gap-lab"
    # And no tracklet may contain an internal gap over the 2 s track buffer.
    per_tid: dict[tuple[str, int], list[float]] = {}
    for o in obs:
        per_tid.setdefault((o.camera, o.track_id), []).append(o.ts_s)
    for ts in per_tid.values():
        assert all(b - a <= 2.0 + 1e-6 for a, b in pairwise(ts))


def test_observations_only_within_range_and_fov() -> None:
    _, obs = run(seconds=30.0)
    cams = {c["id"]: c for c in PLAN["cameras"]}
    for o in obs:
        c = cams[o.camera]
        dx, dy = o.floor_xy[0] - c["pos"][0], o.floor_xy[1] - c["pos"][1]
        dist = math.hypot(dx, dy)
        assert dist <= c["range_m"] + 0.8  # 0.8 m allowance for position noise
        bearing = math.degrees(math.atan2(dy, dx))
        off = abs((bearing - c["yaw_deg"] + 180.0) % 360.0 - 180.0)
        ang_margin = math.degrees(math.atan2(0.8, max(dist - 0.8, 0.2)))
        assert off <= c["fov_deg"] / 2.0 + ang_margin


def test_conf_within_unit_interval() -> None:
    _, obs = run(seconds=30.0)
    assert all(0.0 <= o.conf <= 1.0 for o in obs)


def test_population_stays_near_target() -> None:
    sim = Simulation(PLAN, seed=42, n_people_target=12)
    counts = []
    for i in range(600):  # 60 sim-seconds at 10 Hz
        sim.step(0.1)
        if i >= 200 and i % 10 == 0:  # sample each second after 20 s warm-up
            counts.append(sim.population)
    assert all(6 <= c <= 18 for c in counts)
    assert sum(counts) / len(counts) >= 9.0


def test_floor_xy_inside_camera_floors_walkable_with_noise_margin() -> None:
    """Per-floor containment: every observation lands inside the walkable
    union OF THE OBSERVING CAMERA'S FLOOR (+noise margin) — v2 frames are
    per-floor, so checking against the wrong floor's rects would be wrong."""
    _, obs = run(seconds=40.0)
    margin = 0.5
    cam_floor = {c["id"]: c["floor"] for c in PLAN["cameras"]}
    rects_by_floor = {
        f["id"]: [
            (r["x"] - margin, r["y"] - margin, r["w"] + 2 * margin, r["h"] + 2 * margin)
            for r in f["walkable"]
        ]
        for f in PLAN["floors"]
    }
    for o in obs:
        rects = rects_by_floor[cam_floor[o.camera]]
        x, y = o.floor_xy
        assert any(rx <= x <= rx + rw and ry <= y <= ry + rh for rx, ry, rw, rh in rects), (
            f"{o.camera}@{o.floor_xy} outside its floor's walkable(+{margin}m)"
        )


def test_camera_ticks_are_staggered() -> None:
    _, obs = run(seconds=30.0)
    period = 1.0 / 5.0
    first_ts = {}
    for o in obs:
        first_ts.setdefault(o.camera, o.ts_s)
    phases = {round(ts % period, 6) for ts in first_ts.values()}
    assert len(first_ts) >= 2 and len(phases) >= 2, "cameras sample in lockstep"


def scripted_walker(plan: dict, route: list[str], speed: float, seed: int = 3):
    """Sim with population control off plus one hand-placed walker (white-box)."""
    sim = Simulation(plan, seed=seed, n_people_target=0)
    person = _Person(pid=1, route=route, speed_mps=speed)
    sim._people[1] = person
    return sim, person


def test_cross_floor_route_traversal_flips_floor_exactly_once() -> None:
    """A walker spawned at DOOR reaches BED1/BED2 via the stairs edge, and its
    floor flips exactly once (at the stairs-edge midpoint) on the way."""
    for dest in ("BED1", "BED2"):
        sim = Simulation(PLAN, seed=11, n_people_target=0)
        route = sim._shortest("DOOR", dest)
        assert route is not None and route[0] == "DOOR" and route[-1] == dest
        hops = set(pairwise(route))
        assert ("STG", "STU") in hops or ("STU", "STG") in hops, (
            "route must climb the only cross-floor edge"
        )
        person = _Person(pid=1, route=route, speed_mps=1.4)
        sim._people[1] = person
        floors = [sim.person_floors()[1]]
        for _ in range(30_000):  # 0.02 s steps; generous cap (600 sim-seconds)
            sim.step(0.02)
            if 1 not in sim._people:
                break
            floors.append(sim.person_floors()[1])
        assert person.done, f"walker never reached {dest}"
        assert floors[0] == "ground" and floors[-1] == "upper"
        flips = sum(1 for f1, f2 in pairwise(floors) if f1 != f2)
        assert flips == 1, f"expected exactly one floor flip, saw {flips}"


def test_camera_never_observes_walker_on_another_floor() -> None:
    """cross-lab geometry is exact: the floor flips at 8 m of hall (1.6 m/s)
    plus half the 3 m stairs edge (0.6 m/s). Every camG observation must be
    strictly before that instant, every camU observation strictly after."""
    sim, _ = scripted_walker(CROSS_PLAN, ["GA", "GS", "US", "UA"], speed=1.6)
    t_flip = 8.0 / 1.6 + 1.5 / STAIRS_SPEED_MPS  # = 7.5 s
    obs: list[Observation] = []
    for _ in range(2000):
        obs.extend(sim.step(0.05))
        if 1 not in sim._people:
            break
    ts = {"camG": [], "camU": []}
    for o in obs:
        ts[o.camera].append(o.ts_s)
    assert len(ts["camG"]) >= 10 and len(ts["camU"]) >= 10  # both floors covered
    assert all(t < t_flip + 1e-6 for t in ts["camG"]), "camG saw an upper-floor walker"
    assert all(t > t_flip - 1e-6 for t in ts["camU"]), "camU saw a ground-floor walker"


def test_stairs_slowdown_and_jitter_suppression() -> None:
    """Crossing the 3 m stairs edge must take >= 3/0.8 s (speed capped well
    below the walker's 1.6 m/s), and the true position stays exactly on the
    edge line (x = 9.0) while on stairs — heading jitter suppressed."""
    sim, person = scripted_walker(CROSS_PLAN, ["GA", "GS", "US", "UA"], speed=1.6)
    stairs_len = 3.0
    t_enter = t_exit = None
    while 1 in sim._people and sim.clock_s < 60.0:
        sim.step(0.02)
        if 1 not in sim._people:
            break
        if person.leg == 1:  # on the GS->US stairs edge
            if t_enter is None:
                t_enter = sim.clock_s
            x, _ = sim._position(person)
            assert abs(x - 9.0) < 1e-9, "lateral jitter applied on a stairs edge"
        elif person.leg > 1 and t_enter is not None and t_exit is None:
            t_exit = sim.clock_s
    assert t_enter is not None and t_exit is not None, "walker never crossed the stairs"
    assert t_exit - t_enter >= stairs_len / 0.8


def test_observation_shape_conforms() -> None:
    _, obs = run(seconds=15.0)
    cam_ids = {c["id"] for c in PLAN["cameras"]}
    people = set()
    for o in obs:
        assert isinstance(o, Observation)
        assert o.camera in cam_ids
        assert isinstance(o.track_id, int) and o.track_id >= 1
        assert isinstance(o.floor_xy, tuple) and len(o.floor_xy) == 2
        assert all(isinstance(v, float) for v in o.floor_xy)
        assert isinstance(o.global_id, int)
        people.add(o.global_id)
    assert len(people) > 5  # a real crowd flowed through
