"""Module 05 entry point.

    Video -> Detection -> Tracking -> Person-PPE association -> Temporal validation
          -> PPE rules -> Event -> Evidence -> Standard JSON

Integration code only needs:

    from module_05_ppe.inference.pipeline import process_video
    result = process_video(source="video.mp4", camera_id="CAM-001")

CLI:
    python module_05_ppe/inference/pipeline.py --source video.mp4 --camera-id CAM-001
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # allow `python module_05_ppe/inference/pipeline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from module_05_ppe.config_loader import load_config, resolve_path
from module_05_ppe.inference.association import AssociationParams, associate
from module_05_ppe.inference.detector import PPEDetector
from module_05_ppe.inference.evidence import EvidenceRecorder
from module_05_ppe.inference.visualize import annotate_frame
from module_05_ppe.rules.event_rules import PPEEventEngine
from module_05_ppe.schemas import MODULE_ID, split_by_class

logger = logging.getLogger("module_05_ppe.pipeline")
DEFAULT_FPS = 25.0


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _make_output_dir(config: Dict[str, Any], camera_id: str, source: str, output_dir: Optional[str]) -> Path:
    if output_dir:
        out = Path(output_dir).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = resolve_path(config["output"].get("directory", "outputs")) / f"{camera_id}_{Path(source).stem}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def process_video(source: str, camera_id: str, config_path: str = "config/config.yaml",
                  output_dir: Optional[str] = None, detector: Any = None, save_annotated: bool = False,
                  max_frames: Optional[int] = None, start_time: Optional[str] = None,
                  weights_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the full Module 05 pipeline on a video file and return standardized results.

    Args:
        source: path to a video file (MP4 etc.). RTSP/webcam are NOT implemented.
        camera_id: e.g. "CAM-001".
        config_path: YAML config (default config/config.yaml).
        output_dir: where evidence/JSON/annotated video go (default outputs/<run>).
        detector: object with .detect(frame, frame_id); default = trained PPEDetector.
        save_annotated: also write annotated_video.mp4.
        max_frames: stop early (useful for quick tests).
        start_time: ISO timestamp of the first frame (default: now, UTC).
        weights_path: override model.path.

    Returns:
        dict with module, camera_id, events, incidents, workers, processing stats and file paths.
    """
    config = load_config(config_path)
    if not Path(source).is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    out_dir = _make_output_dir(config, camera_id, source, output_dir)
    detector = detector or PPEDetector(config, weights_path=weights_path)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1:
        logger.warning("Video FPS unreadable; assuming %.1f", DEFAULT_FPS)
        fps = DEFAULT_FPS
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    t0_wall = datetime.fromisoformat(start_time.replace("Z", "+00:00")) if start_time else datetime.now(timezone.utc)

    params = AssociationParams.from_config(config["association"])
    engine = PPEEventEngine(config, camera_id)
    recorder = EvidenceRecorder(out_dir, fps, config["evidence"])
    writer = None
    if save_annotated:
        writer = cv2.VideoWriter(str(out_dir / "annotated_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
    det_file = None
    if config["output"].get("write_detections_file", True):
        det_file = open(out_dir / "detections.jsonl", "w", encoding="utf-8")

    events: List[Dict[str, Any]] = []
    incidents: List[Dict[str, Any]] = []
    workers: Dict[int, Dict[str, int]] = {}
    detect_ms: List[float] = []
    frame_id, started = 0, time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and frame_id >= max_frames):
                break
            video_t = frame_id / fps
            stamp = _iso(t0_wall + timedelta(seconds=video_t))

            t_det = time.perf_counter()
            detections = detector.detect(frame, frame_id)
            detect_ms.append((time.perf_counter() - t_det) * 1000.0)

            persons, helmets, vests = split_by_class(detections)
            observations = associate(persons, helmets, vests, params)
            new_events = engine.update(observations, frame_id, video_t, stamp)
            recorder.push(frame)

            need_annotation = writer is not None or bool(new_events)
            annotated = annotate_frame(frame, observations, [h for h in helmets if h.track_id is None],
                                       [v for v in vests if v.track_id is None]) if need_annotation else None
            for event in new_events:
                incident_id = event.event_id.replace("EVT", "INC")
                evidence = recorder.capture(incident_id, frame, annotate_frame(
                    frame, observations, [], [], banner=f"{event.event_id} PPE violation: "
                    + ", ".join(event.missing_ppe)))
                events.append(event.to_dict())
                incidents.append(event.to_incident(incident_id, evidence))
            if writer is not None:
                writer.write(annotated)

            if det_file is not None:
                _write_detections(det_file, observations, persons, helmets, vests, camera_id, stamp, frame_id)
            for obs in observations:
                w = workers.setdefault(obs.track_id, {"frames_seen": 0, "frames_helmet_no": 0,
                                                      "frames_vest_no": 0, "frames_compliant": 0})
                w["frames_seen"] += 1
                w["frames_helmet_no"] += obs.states.get("helmet") == "NO"
                w["frames_vest_no"] += obs.states.get("safety_vest") == "NO"
                w["frames_compliant"] += all(v == "YES" for v in obs.states.values())
            frame_id += 1
    finally:
        cap.release()
        recorder.finalize()
        if writer is not None:
            writer.release()
        if det_file is not None:
            det_file.close()

    wall = time.perf_counter() - started
    result: Dict[str, Any] = {
        "module": MODULE_ID, "module_version": config["module"]["version"], "camera_id": camera_id,
        "source": str(source), "start_time": _iso(t0_wall), "frames_processed": frame_id,
        "video_fps": round(fps, 3), "frame_size": [width, height],
        "events": events, "incidents": incidents,
        "workers": {str(k): v for k, v in sorted(workers.items())},
        "processing": {
            "wall_seconds": round(wall, 3),
            "end_to_end_fps": round(frame_id / wall, 2) if wall > 0 else None,
            "mean_detector_ms": round(sum(detect_ms) / len(detect_ms), 2) if detect_ms else None,
        },
        "files": {"output_dir": str(out_dir),
                  "detections": "detections.jsonl" if det_file is not None else None,
                  "annotated_video": "annotated_video.mp4" if writer is not None else None},
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Done: %d frames, %d events -> %s", frame_id, len(events), out_dir)
    return result


def _write_detections(handle, observations, persons, helmets, vests, camera_id, stamp, frame_id) -> None:
    """One JSON line per detection. PPE track_id = id of the worker it is associated with."""
    owner = {}
    for obs in observations:
        for kind, match in obs.matches.items():
            if match:
                owner[id(match.detection)] = obs.track_id
    for det in list(persons) + list(helmets) + list(vests):
        if det.class_name == "person":
            track = det.track_id
            extra = None
        else:
            track = owner.get(id(det))
            extra = {"associated_with_worker": track is not None}
        row = det.to_standard_dict(camera_id, stamp, extra)
        row["track_id"] = track
        handle.write(json.dumps(row) + "\n")


class PPEMonitoringModule:
    """Object-style wrapper: module.process(source=..., camera_id=...) (spec section 8)."""

    def __init__(self, config_path: str = "config/config.yaml", weights_path: Optional[str] = None):
        self.config_path, self.weights_path = config_path, weights_path

    def process(self, source: str, camera_id: str, **kwargs: Any) -> Dict[str, Any]:
        return process_video(source, camera_id, self.config_path, weights_path=self.weights_path, **kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NEXORA Module 05 - PPE monitoring on a video file")
    parser.add_argument("--source", required=True, help="path to a video file (e.g. MP4)")
    parser.add_argument("--camera-id", default="CAM-001")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--weights", default=None, help="override model.path in the config")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-annotated", action="store_true", help="also write annotated_video.mp4")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-time", default=None, help="ISO time of first frame, e.g. 2026-10-02T14:30:00Z")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        result = process_video(args.source, args.camera_id, args.config, args.output_dir,
                               save_annotated=args.save_annotated, max_frames=args.max_frames,
                               start_time=args.start_time, weights_path=args.weights)
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2
    summary = {k: result[k] for k in ("module", "camera_id", "frames_processed", "processing")}
    summary.update(events=len(result["events"]), results_file=str(Path(result["files"]["output_dir"]) / "results.json"))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
