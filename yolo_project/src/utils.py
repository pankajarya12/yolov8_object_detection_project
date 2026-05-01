"""Common utility helpers used across training and inference scripts."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


# ---------- Geometry helpers (educational: IoU + NMS) ----------

def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-Union for two boxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(boxes: List[Sequence[float]],
                        scores: List[float],
                        iou_threshold: float = 0.5) -> List[int]:
    """Plain-Python NMS (for teaching). Returns kept indices."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if compute_iou(boxes[i], boxes[j]) < iou_threshold]
    return keep


# ---------- Drawing helpers ----------

def _color_for_class(cls_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(cls_id + 1)
    return tuple(int(c) for c in rng.integers(50, 255, size=3))


def draw_detections(frame: np.ndarray,
                    boxes_xyxy: np.ndarray,
                    confidences: np.ndarray,
                    class_ids: np.ndarray,
                    class_names: Sequence[str]) -> np.ndarray:
    """Draw bounding boxes + labels on a BGR frame in-place and return it."""
    for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, confidences, class_ids):
        cls = int(cls)
        color = _color_for_class(cls)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls] if cls < len(class_names) else cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def overlay_hud(frame: np.ndarray, fps: float, count: int) -> np.ndarray:
    text = f"FPS: {fps:5.1f} | Objects: {count}"
    cv2.rectangle(frame, (8, 8), (260, 38), (0, 0, 0), -1)
    cv2.putText(frame, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


# ---------- FPS meter ----------

class FPSMeter:
    def __init__(self, smoothing: int = 30):
        self.times: List[float] = []
        self.smoothing = smoothing
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self.times.append(dt)
        if len(self.times) > self.smoothing:
            self.times.pop(0)
        avg = sum(self.times) / len(self.times)
        return 1.0 / avg if avg > 0 else 0.0


# ---------- CSV inference logger ----------

class DetectionLogger:
    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.fp)
        self.writer.writerow(
            ["frame", "timestamp", "class_id", "class_name",
             "confidence", "x1", "y1", "x2", "y2"]
        )

    def log(self, frame_idx: int, class_ids, class_names, confs, boxes):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for cid, conf, box in zip(class_ids, confs, boxes):
            cid = int(cid)
            name = class_names[cid] if cid < len(class_names) else str(cid)
            self.writer.writerow([frame_idx, ts, cid, name,
                                  f"{float(conf):.4f}", *[f"{float(v):.1f}" for v in box]])

    def close(self):
        self.fp.close()


# ---------- ROI helpers ----------

def filter_boxes_in_roi(boxes_xyxy: np.ndarray,
                        roi_xyxy: Sequence[int]) -> np.ndarray:
    """Boolean mask of boxes whose center lies inside the ROI."""
    rx1, ry1, rx2, ry2 = roi_xyxy
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    return (cx >= rx1) & (cx <= rx2) & (cy >= ry1) & (cy <= ry2)


def count_per_class(class_ids: Iterable[int], class_names: Sequence[str]) -> dict:
    counts: dict = {}
    for cid in class_ids:
        cid = int(cid)
        name = class_names[cid] if cid < len(class_names) else str(cid)
        counts[name] = counts.get(name, 0) + 1
    return counts
