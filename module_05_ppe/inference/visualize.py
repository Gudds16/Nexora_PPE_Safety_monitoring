"""Draw worker/PPE overlays on frames (annotated frames and annotated demo video)."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from module_05_ppe.inference.association import NO, UNKNOWN, YES, WorkerObservation
from module_05_ppe.schemas import Detection

GREEN, RED, GREY, BLUE = (0, 180, 0), (0, 0, 220), (150, 150, 150), (230, 160, 0)


def annotate_frame(frame: np.ndarray, observations: Sequence[WorkerObservation],
                   helmets: Sequence[Detection], vests: Sequence[Detection],
                   banner: str = "") -> np.ndarray:
    """Return a copy of `frame` with worker status boxes. Unassociated PPE is drawn thin blue."""
    out = frame.copy()
    for det in list(helmets) + list(vests):
        b = det.bbox
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), BLUE, 1)
        cv2.putText(out, det.class_name, (int(b.x1), max(10, int(b.y1) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1, cv2.LINE_AA)
    for obs in observations:
        states = list(obs.states.values())
        colour = RED if NO in states else (GREY if UNKNOWN in states else GREEN)
        b = obs.person_bbox
        cv2.rectangle(out, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, 2)
        label = f"#{obs.track_id} H:{obs.states.get('helmet', '?')[0]} V:{obs.states.get('safety_vest', '?')[0]}"
        cv2.putText(out, label, (int(b.x1), min(out.shape[0] - 4, int(b.y2) + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        for match in obs.matches.values():
            if match:
                mb = match.detection.bbox
                cv2.rectangle(out, (int(mb.x1), int(mb.y1)), (int(mb.x2), int(mb.y2)), colour, 2)
    if banner:
        cv2.putText(out, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return out
