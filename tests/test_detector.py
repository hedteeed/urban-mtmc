"""Tests for mtmc.detector + mtmc.get_model (CONTRACT.md §M1 detector).

Letterbox geometry and decode/NMS tests are pure numpy — no weights needed.
Anything touching the real model skips when models/yolox_s.onnx is absent
(mantis oracle-test pattern).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mtmc.detector import (
    PAD_VALUE,
    PersonDetector,
    decode_outputs,
    letterbox,
    nms,
    postprocess,
)
from mtmc.get_model import MODEL_PATH, MODEL_SHA256, sha256_of

needs_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/yolox_s.onnx absent — run `python -m mtmc.get_model`",
)

N_COLS = 85  # 4 box + 1 obj + 80 COCO classes
N_ROWS = 80 * 80 + 40 * 40 + 20 * 20  # 8400 rows at imgsz 640


# ---------------------------------------------------------------- letterbox

def test_letterbox_ratio_and_shape() -> None:
    for h, w in [(480, 640), (720, 1280), (640, 640), (100, 30)]:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        canvas, ratio = letterbox(frame, 640)
        assert canvas.shape == (640, 640, 3)
        assert canvas.dtype == np.uint8
        assert ratio == pytest.approx(min(640 / h, 640 / w))


def test_letterbox_top_left_anchor_and_pad_value() -> None:
    # Solid-color wide frame: content lands top-left, pad fills bottom, value 114.
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    canvas, ratio = letterbox(frame, 640)
    assert ratio == pytest.approx(1.0)
    assert (canvas[:360, :, :] == 200).all()  # content anchored at (0, 0)
    assert (canvas[360:, :, :] == PAD_VALUE).all()  # bottom pad, exactly 114

    # Tall frame: pad fills the right side instead.
    frame = np.full((640, 320, 3), 50, dtype=np.uint8)
    canvas, ratio = letterbox(frame, 640)
    assert ratio == pytest.approx(1.0)
    assert (canvas[:, :320, :] == 50).all()
    assert (canvas[:, 320:, :] == PAD_VALUE).all()


def test_letterbox_downscale_geometry() -> None:
    # 720x1280 -> ratio 0.5 -> 360x640 content region, rest pad.
    frame = np.full((720, 1280, 3), 99, dtype=np.uint8)
    canvas, ratio = letterbox(frame, 640)
    assert ratio == pytest.approx(0.5)
    assert (canvas[:360, :640, :] == 99).all()
    assert (canvas[360:, :, :] == PAD_VALUE).all()


# --------------------------------------------------------------- grid decode

def _raw(rows: int = N_ROWS) -> np.ndarray:
    return np.zeros((rows, N_COLS), dtype=np.float32)


def test_decode_grid_offsets_and_exp_wh() -> None:
    raw = _raw()
    # Stride-8 level, first cell (gx=0, gy=0): xy passes through * 8, wh = exp * 8.
    raw[0, :4] = [0.25, 0.5, 0.0, math.log(2.0)]
    # Stride-16 level starts at 80*80; cell (gx=5, gy=3) is row 6400 + 3*40 + 5.
    i16 = 80 * 80 + 3 * 40 + 5
    raw[i16, :4] = [0.5, 0.25, math.log(2.0), math.log(3.0)]
    # Stride-32 level starts at 6400 + 1600; cell (gx=1, gy=2) is row 8000 + 2*20 + 1.
    i32 = 80 * 80 + 40 * 40 + 2 * 20 + 1
    raw[i32, :4] = [0.0, 0.0, 0.0, 0.0]

    out = decode_outputs(raw, 640)
    np.testing.assert_allclose(out[0, :4], [0.25 * 8, 0.5 * 8, 8.0, 16.0], rtol=1e-5)
    np.testing.assert_allclose(
        out[i16, :4], [(5 + 0.5) * 16, (3 + 0.25) * 16, 32.0, 48.0], rtol=1e-5
    )
    np.testing.assert_allclose(out[i32, :4], [1 * 32, 2 * 32, 32.0, 32.0], rtol=1e-5)
    # obj/cls columns pass through untouched (already sigmoided in-graph).
    raw[0, 4] = 0.7
    assert decode_outputs(raw, 640)[0, 4] == pytest.approx(0.7)


def test_decode_row_count_matches_levels() -> None:
    # decode must line up 1:1 with 80^2 + 40^2 + 20^2 rows; wh strictly positive after exp.
    out = decode_outputs(_raw(), 640)
    assert out.shape == (N_ROWS, N_COLS)
    assert (out[:, 2:4] > 0).all()


# ---------------------------------------------------------------------- NMS

def test_nms_suppresses_high_iou_keeps_disjoint() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],  # A, best score
            [0.0, 0.0, 10.0, 10.5],  # IoU with A ~0.95 -> suppressed at 0.7
            [50.0, 50.0, 60.0, 60.0],  # disjoint -> kept
        ]
    )
    scores = np.array([0.9, 0.8, 0.6])
    assert nms(boxes, scores, 0.7) == [0, 2]


def test_nms_keeps_moderate_overlap_below_threshold() -> None:
    # IoU([0,0,10,10], [4,0,14,10]) = 60/140 ~ 0.43 < 0.7 -> both survive.
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 14.0, 10.0]])
    scores = np.array([0.9, 0.8])
    assert nms(boxes, scores, 0.7) == [0, 1]


def test_nms_orders_by_score() -> None:
    boxes = np.array([[0.0, 0.0, 5.0, 5.0], [20.0, 20.0, 25.0, 25.0]])
    scores = np.array([0.3, 0.9])
    assert nms(boxes, scores, 0.7) == [1, 0]


# -------------------------------------------------- postprocess on synthetic

def test_postprocess_person_only_score_product_unletterbox() -> None:
    pred = _raw()
    # Strong person: center (80, 80), 40x80 box, score 0.9 * 0.95 = 0.855.
    pred[0, :4] = [80.0, 80.0, 40.0, 80.0]
    pred[0, 4] = 0.9
    pred[0, 5 + 0] = 0.95
    # Strong NON-person (class 1) elsewhere: must be dropped despite high score.
    pred[1, :4] = [300.0, 300.0, 40.0, 80.0]
    pred[1, 4] = 0.9
    pred[1, 5 + 1] = 0.95
    # Weak person: obj * cls = 0.2 * 0.5 = 0.1 < 0.35 threshold.
    pred[2, :4] = [500.0, 100.0, 40.0, 80.0]
    pred[2, 4] = 0.2
    pred[2, 5 + 0] = 0.5

    dets = postprocess(pred, ratio=0.5, orig_h=960, orig_w=1280, conf_thresh=0.35, nms_thresh=0.7)
    assert len(dets) == 1
    x1, y1, x2, y2, conf = dets[0]
    # Un-letterbox divides by ratio 0.5 (top-left pad: no offset subtraction).
    assert (x1, y1, x2, y2) == pytest.approx((120.0, 80.0, 200.0, 240.0))
    assert conf == pytest.approx(0.855)


def test_postprocess_clamps_to_frame() -> None:
    pred = _raw()
    pred[0, :4] = [630.0, 630.0, 100.0, 100.0]  # spills past the letterbox edge
    pred[0, 4] = 0.9
    pred[0, 5] = 0.9
    dets = postprocess(pred, ratio=1.0, orig_h=640, orig_w=640, conf_thresh=0.35, nms_thresh=0.7)
    assert len(dets) == 1
    x1, y1, x2, y2, _ = dets[0]
    assert 0.0 <= x1 <= x2 <= 639.0
    assert 0.0 <= y1 <= y2 <= 639.0


def test_postprocess_empty_when_nothing_passes() -> None:
    assert postprocess(_raw(), 1.0, 480, 640, 0.35, 0.7) == []


# ------------------------------------------------------------ model-required

def test_missing_model_names_get_model_command() -> None:
    with pytest.raises(FileNotFoundError, match=r"python -m mtmc\.get_model"):
        PersonDetector(model_path="models/definitely_not_here.onnx")


@needs_model
def test_model_hash_matches_pin() -> None:
    assert sha256_of(MODEL_PATH) == MODEL_SHA256


@needs_model
def test_e2e_smoke_synthetic_frame() -> None:
    det = PersonDetector(model_path=str(MODEL_PATH))
    # Pinned 0.1.1rc0 export is a raw-grid export: numpy decode must be active.
    assert det._needs_decode is True
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    dets = det.detect(frame)
    assert isinstance(dets, list)  # noise frame: likely empty, but shape must hold
    for x1, y1, x2, y2, conf in dets:
        assert 0.0 <= x1 <= x2 <= 639.0
        assert 0.0 <= y1 <= y2 <= 479.0
        assert 0.0 <= conf <= 1.0
