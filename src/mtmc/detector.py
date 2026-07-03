"""ONNX Runtime person detector — mantis YOLOX inference conventions, exactly.

Mirrors mantis_detector/inference.py: letterbox to an imgsz square anchored
TOP-LEFT with pad value 114; BGR float 0-255, NO normalization; score =
objectness * class_score; person (COCO class 0) only; greedy NMS at IoU 0.7;
un-letterbox by dividing by the resize ratio; clamp to the source frame.

The pinned 0.1.1rc0 ONNX export emits RAW per-level grids (no in-graph box
decode), so the standard YOLOX numpy decode is applied: strides 8/16/32,
xy = (pred + grid_offset) * stride, wh = exp(pred) * stride. Objectness and
class scores are already sigmoided in the graph. Decode need is probed once at
init rather than assumed, so a decoded-in-graph export also works unchanged.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

STRIDES = (8, 16, 32)  # YOLOX-S FPN levels; 640 -> 80^2 + 40^2 + 20^2 = 8400 rows
PAD_VALUE = 114  # YOLOX letterbox gray
PERSON_CLASS = 0  # COCO class 0


def letterbox(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, float]:
    """Resize ``frame`` (BGR HxWx3) into an imgsz x imgsz canvas, top-left, pad 114."""
    h, w = frame.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    nh, nw = round(h * ratio), round(w * ratio)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:nh, :nw] = resized  # top-left anchor: pad only right/bottom
    return canvas, ratio


def decode_outputs(raw: np.ndarray, imgsz: int) -> np.ndarray:
    """Standard YOLOX grid decode for un-decoded exports.

    ``raw``: (N, 5 + n_classes), rows level-major over stride 8/16/32 grids,
    row-major within each grid. Returns a copy with boxes in letterboxed pixels
    (cx, cy, w, h); obj/cls columns pass through (already sigmoided in-graph).
    """
    grids: list[np.ndarray] = []
    strides: list[np.ndarray] = []
    for stride in STRIDES:
        gs = imgsz // stride
        xv, yv = np.meshgrid(np.arange(gs), np.arange(gs))  # row-major: y outer, x inner
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        strides.append(np.full((gs * gs, 1), stride))
    grid = np.concatenate(grids, axis=0).astype(np.float32)
    stride_col = np.concatenate(strides, axis=0).astype(np.float32)

    out = raw.astype(np.float32, copy=True)
    out[:, :2] = (raw[:, :2] + grid) * stride_col
    out[:, 2:4] = np.exp(raw[:, 2:4]) * stride_col
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy IoU NMS on (N, 4) xyxy boxes. Returns kept indices, score-descending."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = iw * ih
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thresh]
    return keep


def postprocess(
    pred: np.ndarray,
    ratio: float,
    orig_h: int,
    orig_w: int,
    conf_thresh: float,
    nms_thresh: float,
) -> list[tuple[float, float, float, float, float]]:
    """Decoded (N, 85) predictions -> person (x1, y1, x2, y2, conf) in source pixels.

    score = objectness * best class score; argmax class must be person (class 0)
    — the mantis filter. Single-class NMS == class-aware NMS here. Boxes are
    divided by the letterbox ratio then clamped into the source frame.
    """
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]
    cls_idx = cls_scores.argmax(axis=1)
    scores = obj * cls_scores[np.arange(pred.shape[0]), cls_idx]

    keep = (scores >= conf_thresh) & (cls_idx == PERSON_CLASS)
    if not keep.any():
        return []
    boxes = pred[keep, :4]
    scores = scores[keep]

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack(
        [
            (cx - bw / 2) / ratio,  # un-letterbox: top-left pad means divide only
            (cy - bh / 2) / ratio,
            (cx + bw / 2) / ratio,
            (cy + bh / 2) / ratio,
        ],
        axis=1,
    )

    kept = nms(xyxy, scores, nms_thresh)
    out: list[tuple[float, float, float, float, float]] = []
    for i in kept:
        bx1, by1, bx2, by2 = xyxy[i]
        out.append(
            (
                float(np.clip(bx1, 0.0, orig_w - 1)),
                float(np.clip(by1, 0.0, orig_h - 1)),
                float(np.clip(bx2, 0.0, orig_w - 1)),
                float(np.clip(by2, 0.0, orig_h - 1)),
                float(scores[i]),
            )
        )
    return out


class PersonDetector:
    """COCO-pretrained YOLOX-S person detector on ONNX Runtime CPU."""

    def __init__(
        self,
        model_path: str = "models/yolox_s.onnx",
        imgsz: int = 640,
        conf: float = 0.35,
        nms_iou: float = 0.7,
    ) -> None:
        path = Path(model_path)
        if not path.exists() and not path.is_absolute():
            # Server may start from any cwd; fall back to the repo root.
            path = Path(__file__).resolve().parents[2] / model_path
        if not path.exists():
            raise FileNotFoundError(
                f"YOLOX-S model not found at {model_path!r} — "
                f"run `python -m mtmc.get_model` to download it."
            )
        self.imgsz = imgsz
        self.conf = conf
        self.nms_iou = nms_iou
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name
        self._needs_decode = self._probe_needs_decode()

    def _probe_needs_decode(self) -> bool:
        """One blank-frame inference: decoded exports have wh = exp(.)*stride > 0
        for every row; raw regression logits always contain negatives."""
        blank = np.full((1, 3, self.imgsz, self.imgsz), float(PAD_VALUE), dtype=np.float32)
        out = self.session.run(None, {self._input_name: blank})[0][0]
        return bool((out[:, 2:4] < 0).any())

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """One BGR frame -> person detections as (x1, y1, x2, y2, conf), source pixels."""
        orig_h, orig_w = frame_bgr.shape[:2]
        padded, ratio = letterbox(frame_bgr, self.imgsz)
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)  # BGR 0-255, NO normalization
        pred = self.session.run(None, {self._input_name: blob})[0][0]  # (8400, 85)
        if self._needs_decode:
            pred = decode_outputs(pred, self.imgsz)
        return postprocess(pred, ratio, orig_h, orig_w, self.conf, self.nms_iou)
