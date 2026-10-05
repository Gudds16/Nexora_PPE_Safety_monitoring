"""Model inference ONLY: answers "What did the model detect?".

No business rules here (see rules/event_rules.py) and no PPE association
(see inference/association.py).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from module_05_ppe.config_loader import resolve_path
from module_05_ppe.hardware import inspect_hardware, select_device
from module_05_ppe.schemas import SUPPORTED_CLASSES, BBox, Detection

logger = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained weights file does not exist."""


def _to_numpy(value: Any) -> np.ndarray:
    """Accept torch tensors or arrays."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_result(result: Any, config: Dict[str, Any], frame_id: int = 0) -> List[Detection]:
    """Convert one ultralytics `Results` object into Detection objects.

    Applies per-class confidence thresholds and drops unknown classes and tiny boxes.
    Works on any object exposing result.boxes.{xyxy,conf,cls,id} and result.names.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes.xyxy) == 0:
        return []
    names: Dict[int, str] = dict(result.names)
    xyxy, conf, cls = _to_numpy(boxes.xyxy), _to_numpy(boxes.conf), _to_numpy(boxes.cls)
    ids = _to_numpy(boxes.id) if getattr(boxes, "id", None) is not None else None
    model_cfg = config["model"]
    default_conf = float(model_cfg["confidence_threshold"])
    per_class = model_cfg.get("class_confidence", {}) or {}
    min_area = float(model_cfg.get("min_box_area_px", 0))

    detections: List[Detection] = []
    for i in range(len(xyxy)):
        name = names.get(int(cls[i]))
        if name not in SUPPORTED_CLASSES:
            continue
        if float(conf[i]) < float(per_class.get(name, default_conf)):
            continue
        box = BBox(*[float(v) for v in xyxy[i]])
        if box.area < min_area:
            continue
        track_id = int(ids[i]) if ids is not None and not np.isnan(ids[i]) else None
        detections.append(Detection(name, float(conf[i]), box, track_id, frame_id))
    return detections


class PPEDetector:
    """Wraps the trained YOLO model (+ ByteTrack) and returns standard Detections."""

    def __init__(self, config: Dict[str, Any], model: Any = None, weights_path: Optional[str] = None):
        self.config = config
        self.tracking = bool(config["tracking"].get("enabled", True))
        hw = inspect_hardware()
        self.device = select_device(config["model"].get("device", "auto"), hw)
        self.model = model if model is not None else self._load_model(weights_path)
        self._check_classes()
        self.tracker_cfg = str(resolve_path(config["tracking"].get("tracker_config", "bytetrack.yaml")))

    def _load_model(self, weights_path: Optional[str]) -> Any:
        path = resolve_path(weights_path or self.config["model"]["path"])
        if not path.is_file():
            raise ModelNotFoundError(
                f"Model weights not found: {path}\n"
                "Train first (python module_05_ppe/training/train.py) or pass --weights <file>.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc
        logger.info("Loading weights %s on device %s", path, self.device)
        return YOLO(str(path))

    def _check_classes(self) -> None:
        names = getattr(self.model, "names", None)
        if not names:
            return
        missing = set(SUPPORTED_CLASSES) - set(dict(names).values())
        if missing:
            raise ValueError(f"Model is missing required classes {sorted(missing)}; it has {dict(names)}")

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[Detection]:
        """Run detection (and tracking, if enabled) on one BGR frame."""
        m = self.config["model"]
        lowest = min([float(m["confidence_threshold"])] + [float(v) for v in (m.get("class_confidence") or {}).values()])
        kwargs = dict(imgsz=int(m["input_size"]), conf=lowest, iou=float(m["iou_threshold"]),
                      device=self.device, verbose=False)
        if self.tracking:
            results = self.model.track(frame, persist=True, tracker=self.tracker_cfg, **kwargs)
        else:
            results = self.model.predict(frame, **kwargs)
        return parse_result(results[0], self.config, frame_id)


class ScriptedDetector:
    """Test / demo double: returns pre-written detections instead of running a model.

    `script` maps frame_id -> list[Detection], or is a callable(frame_id) -> list[Detection].
    It lets the whole pipeline (association, rules, evidence, JSON) be exercised
    without weights. It is NOT a model and says nothing about model accuracy.
    """

    def __init__(self, script: Union[Dict[int, Sequence[Detection]], Callable[[int], Sequence[Detection]]]):
        self.script = script

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[Detection]:
        raw = self.script(frame_id) if callable(self.script) else self.script.get(frame_id, [])
        return [Detection(d.class_name, d.confidence, d.bbox, d.track_id, frame_id) for d in raw]
