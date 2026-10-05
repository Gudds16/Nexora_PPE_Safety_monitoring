"""Shared builders for the tests (plain functions, no pytest dependency)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from module_05_ppe.config_loader import load_config
from module_05_ppe.inference.association import WorkerObservation
from module_05_ppe.rules.event_rules import PPEEventEngine
from module_05_ppe.schemas import BBox, Detection

FPS = 10.0


def fast_config(**event_overrides: Any) -> Dict[str, Any]:
    """Config with short temporal windows so tests stay small (5 frames / 0 s / recovery 3 / cooldown 10 s)."""
    cfg = copy.deepcopy(load_config())
    cfg["event"].update(minimum_frames=5, minimum_seconds=0.0, recovery_frames=3,
                        cooldown_seconds=10, track_max_age_frames=30)
    cfg["event"].update(event_overrides)
    return cfg


def obs(track_id: int, helmet: str = "YES", vest: str = "YES", conf: float = 0.9) -> WorkerObservation:
    person = Detection("person", conf, BBox(100, 100, 200, 400), track_id)
    return WorkerObservation(track_id=track_id, person=person,
                             states={"helmet": helmet, "safety_vest": vest},
                             matches={"helmet": None, "safety_vest": None})


def run_frames(engine: PPEEventEngine, per_frame: Sequence[Sequence[WorkerObservation]], start_frame: int = 0):
    """Feed one list of observations per frame at FPS; return all events produced."""
    events = []
    for i, frame_obs in enumerate(per_frame):
        frame_id = start_frame + i
        t = frame_id / FPS
        events += engine.update(frame_obs, frame_id, t, f"2026-10-02T14:30:{int(t):02d}.000Z")
    return events


def repeat(frame_obs: Sequence[WorkerObservation], n: int) -> List[Sequence[WorkerObservation]]:
    return [frame_obs] * n


def make_engine(camera_id: str = "CAM-001", **event_overrides: Any) -> PPEEventEngine:
    return PPEEventEngine(fast_config(**event_overrides), camera_id)


# ---- fake ultralytics objects -------------------------------------------------
class FakeTensor:
    """Mimics a torch tensor: .cpu().numpy()."""

    def __init__(self, data):
        self._a = np.asarray(data, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __len__(self):
        return len(self._a)


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy, self.conf, self.cls = FakeTensor(xyxy), FakeTensor(conf), FakeTensor(cls)
        self.id = FakeTensor(ids) if ids is not None else None


class FakeResult:
    names = {0: "person", 1: "helmet", 2: "safety_vest", 3: "cap"}

    def __init__(self, xyxy, conf, cls, ids=None):
        self.boxes = FakeBoxes(xyxy, conf, cls, ids) if len(xyxy) else None


class FakeModel:
    """Stands in for ultralytics YOLO in PPEDetector tests."""

    def __init__(self, result: FakeResult, names: Optional[Dict[int, str]] = None):
        self._result = result
        self.names = names or FakeResult.names
        self.calls: List[str] = []

    def track(self, frame, **kwargs):
        self.calls.append("track")
        return [self._result]

    def predict(self, frame, **kwargs):
        self.calls.append("predict")
        return [self._result]
