"""Contract tests for mtmc.simulator (CONTRACT.md §mtmc.simulator)."""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

from mtmc.events import Observation
from mtmc.simulator import Simulation

PLAN = json.loads(
    (Path(__file__).resolve().parent.parent / "floorplan" / "plan.json").read_text()
)

# Tiny plan engineered so every walker's only route (A-B-C) crosses camG's
# wedge (range 7 m), leaves it for >2 s around B, and re-enters on the way
# back — forcing the track-buffer rule deterministically.
GAP_PLAN = {
    "version": 1,
    "name": "gap-lab",
    "size_m": [12, 4],
    "walkable": [{"x": 0, "y": 0, "w": 12, "h": 4, "name": "hall"}],
    "shops": [],
    "waypoints": {"A": [2, 1.5], "B": [11, 2], "C": [2, 2.5]},
    "edges": [["A", "B"], ["B", "C"]],
    "entrances": ["A", "C"],
    "dwell_points": [],
    "cameras": [
        {"id": "camG", "pos": [0, 2], "yaw_deg": 0, "fov_deg": 100, "range_m": 7}
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


def test_floor_xy_inside_walkable_with_noise_margin() -> None:
    _, obs = run(seconds=40.0)
    margin = 0.5
    rects = [
        (r["x"] - margin, r["y"] - margin, r["w"] + 2 * margin, r["h"] + 2 * margin)
        for r in PLAN["walkable"]
    ]
    for o in obs:
        x, y = o.floor_xy
        assert any(rx <= x <= rx + rw and ry <= y <= ry + rh for rx, ry, rw, rh in rects), (
            f"{o.floor_xy} outside walkable(+{margin}m)"
        )


def test_camera_ticks_are_staggered() -> None:
    _, obs = run(seconds=30.0)
    period = 1.0 / 5.0
    first_ts = {}
    for o in obs:
        first_ts.setdefault(o.camera, o.ts_s)
    phases = {round(ts % period, 6) for ts in first_ts.values()}
    assert len(first_ts) >= 2 and len(phases) >= 2, "cameras sample in lockstep"


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
