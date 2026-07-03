"""Tests for mtmc.tracker (CONTRACT.md §M2 ByteTrack).

Every scenario is synthetic with known ground truth at a nominal 5 fps
(dt = 0.2 s). RNG lives in the TESTS only — the tracker itself must be
deterministic (no wall clock, no random state), and test 6 pins that.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtmc.tracker import ByteTracker, Track

DT = 0.2  # 5 fps
W, H = 30.0, 60.0  # nominal pedestrian box, source pixels


def box(cx: float, cy: float, w: float = W, h: float = H) -> tuple[float, float, float, float]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def det(
    cx: float, cy: float, conf: float = 0.9, w: float = W, h: float = H
) -> tuple[float, float, float, float, float]:
    return (*box(cx, cy, w, h), conf)


def cx_of(t: Track) -> float:
    return (t.box[0] + t.box[2]) / 2


def cy_of(t: Track) -> float:
    return (t.box[1] + t.box[3]) / 2


# ------------------------------------------------- 1. single walker, one id


def test_single_walker_noisy_boxes_one_id_forever() -> None:
    rng = np.random.default_rng(0)
    tracker = ByteTracker()
    per_tick: list[list[Track]] = []
    for k in range(75):  # 15 s straight-line walk at 40 px/s
        t = k * DT
        cx = 50.0 + 40.0 * t + rng.normal(0.0, 1.0)
        cy = 120.0 + rng.normal(0.0, 1.0)
        conf = float(np.clip(0.85 + rng.normal(0.0, 0.03), 0.5, 1.0))
        per_tick.append(tracker.update([det(cx, cy, conf)], t))

    ids = {tr.track_id for out in per_tick for tr in out}
    assert len(ids) == 1
    # min_hits=3: silent for two ticks, then emitted EVERY tick.
    assert [len(out) for out in per_tick[:2]] == [0, 0]
    assert all(len(out) == 1 for out in per_tick[2:])
    # Output surface: filtered box stays near truth, conf passes through.
    for k, out in enumerate(per_tick[2:], start=2):
        (tr,) = out
        assert isinstance(tr.box, tuple) and len(tr.box) == 4
        assert cx_of(tr) == pytest.approx(50.0 + 40.0 * k * DT, abs=6.0)
        assert cy_of(tr) == pytest.approx(120.0, abs=6.0)
        assert 0.5 <= tr.conf <= 1.0


def test_single_walker_jittery_dt_one_id() -> None:
    # Sampling intervals wobble (0.12–0.30 s): dt comes from ts_s deltas and
    # must not fragment the track.
    rng = np.random.default_rng(1)
    tracker = ByteTracker()
    t = 0.0
    ids: set[int] = set()
    n_emitted = 0
    for _ in range(60):
        out = tracker.update([det(60.0 + 45.0 * t, 90.0)], t)
        ids.update(tr.track_id for tr in out)
        n_emitted += len(out)
        t += float(rng.uniform(0.12, 0.30))
    assert len(ids) == 1
    assert n_emitted == 58  # every tick from the confirming third one on


# ------------------------------------------------------------- 2. occlusion


def _occlusion_ids(gap_s: float) -> tuple[set[int], set[int]]:
    """Walker at 50 px/s; detections vanish on (2.0, 2.0 + gap_s), then resume
    ON the constant-velocity path. Returns (ids before, ids after) the gap."""
    tracker = ByteTracker()  # track_buffer_s = 2.0
    before: set[int] = set()
    after: set[int] = set()
    for k in range(41):  # 0 .. 8.0 s
        t = round(k * DT, 10)
        visible = t <= 2.0 or t >= 2.0 + gap_s
        dets = [det(40.0 + 50.0 * t, 150.0)] if visible else []
        for tr in tracker.update(dets, t):
            (before if t <= 2.0 else after).add(tr.track_id)
    return before, after


def test_occlusion_within_buffer_keeps_id() -> None:
    # Gone 1.5 s < buffer 2 s: the coasted prediction reclaims the detection.
    before, after = _occlusion_ids(1.5)
    assert len(before) == 1
    assert after == before


def test_occlusion_beyond_buffer_gets_new_id() -> None:
    # Gone 3 s > buffer 2 s: the track died; the resumed walker is a NEW id.
    before, after = _occlusion_ids(3.0)
    assert len(before) == 1 and len(after) == 1
    assert after.isdisjoint(before)
    assert min(after) > max(before)  # monotonic counter: dead id not recycled


# ------------------------------------------------------ 3. low-conf rescue


def test_low_conf_dip_survives_via_round_2() -> None:
    tracker = ByteTracker(track_thresh=0.5)
    low_ticks = {12, 13, 14}  # conf 0.2 in [0.1, 0.5): round-2 material only
    per_tick: list[list[Track]] = []
    for k in range(30):
        t = k * DT
        conf = 0.2 if k in low_ticks else 0.8
        per_tick.append(tracker.update([det(30.0 + 45.0 * t, 200.0, conf)], t))

    ids = {tr.track_id for out in per_tick for tr in out}
    assert len(ids) == 1  # id survived the dip
    assert all(len(per_tick[k]) == 1 for k in range(2, 30))  # no emission gap
    for k in low_ticks:  # rescued ticks emit, carrying the matched det's conf
        assert per_tick[k][0].conf == pytest.approx(0.2)


def test_conf_below_floor_is_invisible() -> None:
    # conf < 0.1 is under the contract floor: it can neither extend nor start
    # a track, so a confirmed track just coasts (silently) through it.
    tracker = ByteTracker()
    out: list[Track] = []
    for k in range(3):
        out = tracker.update([det(100.0, 100.0, 0.9)], k * DT)
    (confirmed,) = out
    assert tracker.update([det(100.0, 100.0, 0.05)], 3 * DT) == []
    (back,) = tracker.update([det(100.0, 100.0, 0.9)], 4 * DT)
    assert back.track_id == confirmed.track_id


# ------------------------------------------------------------- 4. crossing


def _crossing_dets(t: float) -> list[tuple[float, float, float, float, float]]:
    # A walks right at 50 px/s, B walks left at 60 px/s, same y: they cross at
    # t ≈ 2.05 s. Speeds are asymmetric so no tick samples the exact meeting
    # point (which would make the two detections literally identical).
    return [det(100.0 + 50.0 * t, 200.0), det(325.0 - 60.0 * t, 200.0)]


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def test_crossing_is_adversarial_for_greedy_iou() -> None:
    # Strawman: highest-IoU-first greedy against LAST boxes (no motion model).
    # At the swap tick each walker lands nearer the OTHER's previous box, so
    # greedy provably swaps — the scenario is a real trap, not a soft one.
    prev = {"A": box(100.0, 200.0), "B": box(325.0, 200.0)}
    for k in range(1, 21):
        boxes = [b[:4] for b in _crossing_dets(k * DT)]
        ranked = sorted(
            ((_iou(prev[lab], boxes[j]), lab, j) for lab in prev for j in range(2)),
            reverse=True,
        )
        assigned: dict[str, int] = {}
        used: set[int] = set()
        for v, lab, j in ranked:
            if v <= 0.0 or lab in assigned or j in used:
                continue
            assigned[lab] = j
            used.add(j)
        prev = {lab: boxes[j] for lab, j in assigned.items()}
    # Label A ended on B's terminus: greedy followed raw overlap into the swap.
    assert prev["A"][0] == pytest.approx(box(325.0 - 60.0 * 4.0, 200.0)[0], abs=1e-6)


def test_crossing_ids_do_not_swap() -> None:
    # Same trap, real tracker: Hungarian runs on Kalman-PREDICTED boxes, and by
    # the crossing each filter's velocity has converged, so the predictions sit
    # on the far side of the cross and pure IoU cost already separates them.
    tracker = ByteTracker()
    per_tick: list[dict[int, float]] = []
    for k in range(21):  # 0 .. 4.0 s
        out = tracker.update(_crossing_dets(k * DT), k * DT)
        per_tick.append({tr.track_id: cx_of(tr) for tr in out})

    ids = {tid for tick in per_tick for tid in tick}
    assert len(ids) == 2  # no fragmentation through the cross...
    assert all(len(tick) == 2 for tick in per_tick[2:])  # ...and no dropouts
    id_a, id_b = sorted(ids, key=lambda tid: per_tick[2][tid])  # A starts left
    assert per_tick[20][id_a] == pytest.approx(100.0 + 50.0 * 4.0, abs=8.0)
    assert per_tick[20][id_b] == pytest.approx(325.0 - 60.0 * 4.0, abs=8.0)


# ----------------------------------------------------- 5. ghost suppression


def test_single_frame_ghost_never_emits_never_burns_an_id() -> None:
    tracker = ByteTracker()
    per_tick: list[list[Track]] = []
    for k in range(40):
        t = k * DT
        dets = [det(50.0 + 40.0 * t, 100.0)]  # real walker throughout
        if k == 10:
            dets.append(det(500.0, 400.0, 0.95))  # one-frame high-conf ghost
        if k >= 20:
            dets.append(det(600.0, 300.0))  # second real walker, appears later
        per_tick.append(tracker.update(dets, t))

    ids = sorted({tr.track_id for out in per_tick for tr in out})
    assert len(ids) == 2
    # Tentative tracks take ids only at confirmation, so the ghost (killed by
    # its first miss) left the counter untouched: walker 2 got the NEXT id.
    assert ids[1] == ids[0] + 1
    # And nothing was ever emitted anywhere near the ghost.
    for out in per_tick:
        for tr in out:
            assert abs(cx_of(tr) - 500.0) > 50.0 or abs(cy_of(tr) - 400.0) > 50.0


def test_tentative_dies_on_a_single_miss() -> None:
    # min_hits=3 means three CONSECUTIVE matches: hit, hit, miss restarts the
    # count from scratch, so confirmation lands 3 ticks after the miss.
    tracker = ByteTracker(min_hits=3)
    present = [True, True, False, True, True, True]
    sizes = []
    for k, p in enumerate(present):
        dets = [det(100.0 + 10.0 * k, 100.0)] if p else []
        sizes.append(len(tracker.update(dets, k * DT)))
    assert sizes == [0, 0, 0, 0, 0, 1]


# -------------------------------------------------------- 6. determinism


def _busy_sequence() -> list[tuple[float, list[tuple[float, float, float, float, float]]]]:
    """A messy but fixed scenario: noise, an occlusion, a conf dip, clutter."""
    rng = np.random.default_rng(7)
    seq = []
    for k in range(80):
        t = k * DT
        dets: list[tuple[float, float, float, float, float]] = []
        if not 3.0 < t < 4.4:  # walker 1 occluded for 1.4 s mid-run
            conf = 0.2 if 20 <= k < 23 else 0.85  # and dips low-conf later
            dets.append(
                det(20.0 + 30.0 * t + rng.normal(0, 1), 80.0 + rng.normal(0, 1), conf)
            )
        dets.append(det(400.0 - 25.0 * t + rng.normal(0, 1), 260.0 + rng.normal(0, 1), 0.7))
        if k % 17 == 0:  # sporadic one-frame clutter
            dets.append(det(rng.uniform(0, 640), rng.uniform(0, 480), 0.9))
        seq.append((t, dets))
    return seq


def test_identical_input_identical_output_twice() -> None:
    seq = _busy_sequence()

    def run() -> list[list[Track]]:
        tracker = ByteTracker()
        return [tracker.update(list(dets), t) for t, dets in seq]

    # Frozen dataclasses compare by value; same ops on same floats must be
    # bitwise identical — the tracker holds no clock and no RNG.
    assert run() == run()


# --------------------------------------------------- 7. ids never reused


def test_ids_never_reused_across_a_churny_run() -> None:
    def walkers_at(k: int) -> list[tuple[float, float, float, float, float]]:
        t = k * DT
        dets = []
        if k < 30:
            dets.append(det(30.0 + 40.0 * t, 60.0))
        if 15 <= k < 55:
            dets.append(det(500.0 - 30.0 * (t - 3.0), 300.0))
        if 45 <= k < 90:
            dets.append(det(100.0 + 35.0 * (t - 9.0), 400.0))
        if 70 <= k < 90:
            dets.append(det(600.0, 100.0 + 20.0 * (t - 14.0)))
        return dets

    tracker = ByteTracker()
    ticks_by_id: dict[int, list[int]] = {}
    for k in range(90):
        for tr in tracker.update(walkers_at(k), k * DT):
            ticks_by_id.setdefault(tr.track_id, []).append(k)

    assert len(ticks_by_id) == 4  # four walker generations, four ids
    for ks in ticks_by_id.values():
        # One contiguous life per id: an id that stopped never resurfaces.
        assert ks == list(range(ks[0], ks[-1] + 1))
    first_seen = sorted(ticks_by_id, key=lambda tid: ticks_by_id[tid][0])
    assert first_seen == sorted(first_seen)  # handed out in confirmation order


# ------------------------------------------------- gating convention pin


def test_gate_is_cost_leq_match_thresh() -> None:
    # Convention under test: cost = 1 - IoU, pair kept iff cost <= match_thresh
    # i.e. IoU >= 1 - match_thresh. For stationary W=30 boxes shifted by dx,
    # IoU = (30 - dx) / (30 + dx): the 0.2 boundary sits at dx = 20.
    def run(dx: float) -> tuple[int, list[Track]]:
        tracker = ByteTracker(match_thresh=0.8)
        out: list[Track] = []
        for k in range(3):  # confirm a stationary track (velocity stays 0)
            out = tracker.update([det(100.0, 100.0)], k * DT)
        (confirmed,) = out
        return confirmed.track_id, tracker.update([det(100.0 + dx, 100.0)], 3 * DT)

    tid, out = run(15.0)  # IoU 1/3 >= 0.2: inside the gate
    assert [tr.track_id for tr in out] == [tid]

    tid, out = run(24.0)  # IoU 6/54 ≈ 0.11 < 0.2: rejected despite Hungarian
    # The old track coasts silently; the far detection is a fresh TENTATIVE
    # (no id yet) — so this tick emits nothing at all.
    assert out == []
