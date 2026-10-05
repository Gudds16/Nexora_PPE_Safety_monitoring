"""Standard NEXORA data structures shared by every part of Module 05.

Field names follow FileStructureandPrecautions.docx and must not be renamed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

MODULE_ID = "module_05_ppe"

# Approved values only (spec sections 12 and 13).
STATUS_VALUES = ("PENDING", "CONFIRMED", "DISMISSED", "UNCERTAIN", "RESOLVED")
SEVERITY_VALUES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Classes this module understands.
CLASS_PERSON = "person"
CLASS_HELMET = "helmet"
CLASS_VEST = "safety_vest"
SUPPORTED_CLASSES = (CLASS_PERSON, CLASS_HELMET, CLASS_VEST)

EVENT_PPE_VIOLATION = "PPE_VIOLATION"
EVENT_TYPES = (EVENT_PPE_VIOLATION,)


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in pixel coordinates (x1,y1 = top-left)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def intersection_area(self, other: "BBox") -> float:
        iw = min(self.x2, other.x2) - max(self.x1, other.x1)
        ih = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, iw) * max(0.0, ih)

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def fraction_inside(self, other: "BBox") -> float:
        """Share of THIS box's area that lies inside `other` (0..1)."""
        return self.intersection_area(other) / self.area if self.area > 0 else 0.0

    def to_dict(self) -> Dict[str, int]:
        return {k: int(round(v)) for k, v in
                (("x1", self.x1), ("y1", self.y1), ("x2", self.x2), ("y2", self.y2))}


@dataclass
class Detection:
    """One raw model observation in one frame (spec: 'Detection')."""

    class_name: str
    confidence: float
    bbox: BBox
    track_id: Optional[int] = None
    frame_id: int = 0

    def to_standard_dict(self, camera_id: str, timestamp: str,
                         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Standard detection output. Extra module-specific fields are added, never renamed."""
        out: Dict[str, Any] = {
            "module": MODULE_ID,
            "camera_id": camera_id,
            "frame_id": self.frame_id,
            "timestamp": timestamp,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_dict(),
        }
        if extra:
            out.update(extra)
        return out


def split_by_class(detections: List[Detection]):
    """Return (persons, helmets, vests) lists."""
    persons = [d for d in detections if d.class_name == CLASS_PERSON]
    helmets = [d for d in detections if d.class_name == CLASS_HELMET]
    vests = [d for d in detections if d.class_name == CLASS_VEST]
    return persons, helmets, vests
