"""Detector-layer tests: parsing model output into standard Detections.

These use fake model output (no weights needed). They test OUR filtering/parsing logic,
not the accuracy of a trained network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from module_05_ppe.config_loader import load_config
from module_05_ppe.inference.detector import (ModelNotFoundError, PPEDetector, ScriptedDetector,
                                              parse_result)
from module_05_ppe.schemas import BBox, Detection
from module_05_ppe.tests.helpers import FakeModel, FakeResult
from module_05_ppe.training.augmentations import simulate_blur, simulate_low_light

CFG = load_config()


def test_correct_object_is_detected_with_standard_fields():
    res = FakeResult([[120, 80, 300, 420]], [0.91], [1], ids=[17])
    dets = parse_result(res, CFG, frame_id=1250)
    assert len(dets) == 1 and dets[0].class_name == "helmet" and dets[0].track_id == 17
    row = dets[0].to_standard_dict("CAM-001", "2026-10-02T14:30:15.200Z")
    assert list(row) == ["module", "camera_id", "frame_id", "timestamp", "track_id",
                         "class_name", "confidence", "bbox"]
    assert row["module"] == "module_05_ppe" and row["frame_id"] == 1250
    assert row["bbox"] == {"x1": 120, "y1": 80, "x2": 300, "y2": 420}


def test_wrong_object_class_is_ignored():
    res = FakeResult([[10, 10, 90, 90], [200, 200, 300, 300]], [0.95, 0.95], [3, 1])   # 3 = cap
    dets = parse_result(res, CFG)
    assert [d.class_name for d in dets] == ["helmet"]          # the cap never becomes a helmet


def test_low_confidence_is_filtered_per_class():
    # thresholds in config: person 0.40, helmet 0.45
    res = FakeResult([[0, 0, 100, 300], [0, 0, 100, 300], [0, 0, 50, 50], [0, 0, 50, 50]],
                     [0.45, 0.30, 0.44, 0.46], [0, 0, 1, 1])
    dets = parse_result(res, CFG)
    assert [(d.class_name, round(d.confidence, 2)) for d in dets] == [("person", 0.45), ("helmet", 0.46)]


def test_multiple_objects_all_kept():
    res = FakeResult([[0, 0, 100, 300], [150, 0, 250, 300], [300, 0, 400, 300],
                      [20, 0, 70, 40], [170, 0, 220, 40]], [0.9] * 5, [0, 0, 0, 1, 1], ids=[1, 2, 3, np.nan, np.nan])
    dets = parse_result(res, CFG)
    assert len(dets) == 5
    assert [d.track_id for d in dets] == [1, 2, 3, None, None]


def test_small_object_threshold():
    res = FakeResult([[0, 0, 4, 4], [0, 0, 12, 10]], [0.9, 0.9], [1, 1])      # 16 px^2 vs 120 px^2
    dets = parse_result(res, CFG)
    assert len(dets) == 1 and dets[0].bbox.area == 120


def test_partially_occluded_object_with_moderate_confidence_is_kept():
    res = FakeResult([[100, 100, 130, 125]], [0.47], [1])                       # half-hidden helmet
    assert len(parse_result(res, CFG)) == 1


def test_blurred_object_with_collapsed_confidence_is_dropped():
    res = FakeResult([[100, 100, 150, 140]], [0.28], [1])
    assert parse_result(res, CFG) == []


def test_blur_simulation_really_removes_detail():
    rng = np.random.default_rng(0)
    img = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    sharp = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    blurred = cv2.Laplacian(cv2.cvtColor(simulate_blur(img), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert blurred < sharp * 0.2


def test_low_light_simulation_darkens():
    img = np.full((60, 80, 3), 180, np.uint8)
    assert simulate_low_light(img).mean() < img.mean() * 0.5


def test_empty_model_output_gives_no_detections():
    assert parse_result(FakeResult([], [], []), CFG) == []


def test_detector_uses_tracking_and_parses_result():
    model = FakeModel(FakeResult([[0, 0, 100, 300]], [0.9], [0], ids=[5]))
    det = PPEDetector(CFG, model=model).detect(np.zeros((480, 640, 3), np.uint8), frame_id=3)
    assert model.calls == ["track"] and det[0].track_id == 5 and det[0].frame_id == 3


def test_detector_rejects_model_without_required_classes():
    bad = FakeModel(FakeResult([], [], []), names={0: "person", 1: "helmet"})
    try:
        PPEDetector(CFG, model=bad)
    except ValueError as exc:
        assert "safety_vest" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_missing_weights_give_clear_error():
    try:
        PPEDetector(CFG, weights_path="models/checkpoints/does_not_exist.pt")
    except ModelNotFoundError as exc:
        assert "train" in str(exc).lower()
    else:
        raise AssertionError("expected ModelNotFoundError")


def test_scripted_detector_stamps_frame_id():
    script = {7: [Detection("person", 0.9, BBox(0, 0, 10, 10), 1, frame_id=0)]}
    out = ScriptedDetector(script).detect(np.zeros((10, 10, 3), np.uint8), 7)
    assert out[0].frame_id == 7 and ScriptedDetector(script).detect(None, 8) == []
