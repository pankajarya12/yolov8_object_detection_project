"""Unit tests for utility functions."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import (compute_iou, count_per_class, filter_boxes_in_roi,
                   non_max_suppression)


def test_iou_identical():
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint():
    assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial():
    iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.14 < iou < 0.15  # 25 / 175


def test_nms_keeps_highest():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
    scores = [0.9, 0.8, 0.95]
    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
    assert sorted(keep) == [0, 2]


def test_count_per_class():
    counts = count_per_class([0, 0, 1, 2, 1], ["car", "bike", "person"])
    assert counts == {"car": 2, "bike": 2, "person": 1}


def test_roi_filter():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]])
    mask = filter_boxes_in_roi(boxes, (40, 40, 80, 80))
    assert mask.tolist() == [False, True, False]
