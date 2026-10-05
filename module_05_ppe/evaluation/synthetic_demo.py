"""Pipeline-logic demo with SCRIPTED detections (no model involved).

    python module_05_ppe/evaluation/synthetic_demo.py

Renders a small synthetic video of cartoon workers, feeds hand-written detections through the
real pipeline (association -> temporal rules -> events -> evidence -> JSON) and writes
evaluation/results/synthetic_demo/{demo.mp4, annotated_demo.mp4, results.json, evidence/...}.

IMPORTANT: this proves the pipeline plumbing works end to end. It is NOT a model demo and says
nothing about detection accuracy. A real demo needs trained weights:
    python module_05_ppe/inference/pipeline.py --source <video> --camera-id CAM-001 --save-annotated
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from module_05_ppe.config_loader import MODULE_DIR
from module_05_ppe.evaluation.scenarios import (helmet_in_hand, helmet_on_ground_between, helmet_worn,
                                                person, vest_worn)
from module_05_ppe.inference.detector import ScriptedDetector
from module_05_ppe.inference.pipeline import process_video
from module_05_ppe.schemas import Detection

FPS, SECONDS, SIZE = 10, 12, (640, 480)
OUT = MODULE_DIR / "evaluation" / "results" / "synthetic_demo"


def script(frame_id: int) -> List[Detection]:
    """Worker 1 compliant. Worker 2 never wears a helmet (a helmet lies on the floor between them).
    Worker 3 holds a helmet in hand (not worn) from frame 30 and puts nothing on."""
    w1 = person(1, 120, top=120, h=270, w=90)
    w2 = person(2, 260, top=120, h=270, w=90)
    dets = [w1, helmet_worn(w1), vest_worn(w1), w2, vest_worn(w2), helmet_on_ground_between(w1, w2)]
    if frame_id >= 30:
        w3 = person(3, 440, top=120, h=270, w=90)
        dets += [w3, helmet_in_hand(w3), vest_worn(w3)]
    return dets


def render_background(frame_id: int, dets: List[Detection]) -> np.ndarray:
    img = np.full((SIZE[1], SIZE[0], 3), (200, 200, 200), np.uint8)
    cv2.rectangle(img, (0, 400), (SIZE[0], SIZE[1]), (150, 150, 150), -1)
    colours = {"person": (90, 90, 90), "helmet": (0, 200, 255), "safety_vest": (0, 140, 255)}
    for d in sorted(dets, key=lambda x: {"person": 0, "safety_vest": 1, "helmet": 2}[x.class_name]):
        b = d.bbox
        pt1, pt2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        if d.class_name == "person":
            cv2.rectangle(img, pt1, pt2, colours["person"], -1)
            cv2.circle(img, (int(b.center_x), int(b.y1) + 14), 14, (170, 190, 220), -1)
        else:
            cv2.rectangle(img, pt1, pt2, colours[d.class_name], -1)
    cv2.putText(img, f"SYNTHETIC SCRIPTED DEMO  frame {frame_id}", (8, 462), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    video = OUT / "demo.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for i in range(FPS * SECONDS):
        writer.write(render_background(i, script(i)))
    writer.release()

    result = process_video(str(video), "CAM-DEMO", output_dir=str(OUT), detector=ScriptedDetector(script),
                           save_annotated=True, start_time="2026-10-02T14:30:00Z")
    shutil.move(str(OUT / "annotated_video.mp4"), str(OUT / "annotated_demo.mp4"))
    result["files"]["annotated_video"] = "annotated_demo.mp4"
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"events: {len(result['events'])}")
    for e in result["events"]:
        print(f"  {e['event_id']} track={e['track_id']} missing={e['missing_ppe']} severity={e['severity']} t={e['video_time_s']}s")
    print(f"output: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
