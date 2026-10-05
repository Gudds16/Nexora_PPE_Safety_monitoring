"""Evidence generation: event frame, annotated frame and a short clip around the event.

Layout (spec section 19):  <output_dir>/evidence/INC-000123/{frame.jpg, annotated_frame.jpg, clip.mp4}
Frames are kept in a small JPEG ring buffer so a 5 s pre-event window does not need GBs of RAM.
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EvidenceRecorder:
    """Call push() for every frame, capture() when an event fires, finalize() at the end."""

    def __init__(self, output_dir: Path, fps: float, cfg: Dict[str, Any]):
        self.root = Path(output_dir) / cfg.get("folder", "evidence")
        self.fps = fps if fps and fps > 0 else 25.0
        self.save_frame = bool(cfg.get("save_frame", True))
        self.save_clip = bool(cfg.get("save_clip", True))
        self.post_frames = int(round(float(cfg.get("post_event_seconds", 5)) * self.fps))
        pre_frames = int(round(float(cfg.get("pre_event_seconds", 5)) * self.fps))
        self.max_width = int(cfg.get("clip_max_width", 960))
        self.quality = int(cfg.get("jpeg_quality", 90))
        self._buffer: Deque[bytes] = deque(maxlen=max(1, pre_frames + 1))
        self._pending: List[Dict[str, Any]] = []
        self._output_dir = Path(output_dir)

    def _encode_small(self, frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else b""

    def push(self, frame: np.ndarray) -> None:
        """Add the current frame to the ring buffer and to any clip still being filled."""
        if not self.save_clip:
            return
        data = self._encode_small(frame)
        self._buffer.append(data)
        for job in list(self._pending):
            job["frames"].append(data)
            job["remaining"] -= 1
            if job["remaining"] <= 0:
                self._write_clip(job)
                self._pending.remove(job)

    def capture(self, incident_id: str, frame: np.ndarray, annotated: np.ndarray) -> Dict[str, Optional[str]]:
        """Save frame images now; schedule the clip. Returns relative evidence paths."""
        folder = self.root / incident_id
        folder.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Optional[str]] = {"frame": None, "annotated_frame": None, "clip": None}
        if self.save_frame:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            cv2.imwrite(str(folder / "frame.jpg"), frame, params)
            cv2.imwrite(str(folder / "annotated_frame.jpg"), annotated, params)
            paths["frame"] = self._rel(folder / "frame.jpg")
            paths["annotated_frame"] = self._rel(folder / "annotated_frame.jpg")
        if self.save_clip:
            self._pending.append({"path": folder / "clip.mp4", "frames": list(self._buffer),
                                  "remaining": self.post_frames})
            paths["clip"] = self._rel(folder / "clip.mp4")
        return paths

    def finalize(self) -> None:
        """Write clips that were still waiting for post-event frames (video ended early)."""
        for job in self._pending:
            self._write_clip(job)
        self._pending.clear()

    def _write_clip(self, job: Dict[str, Any]) -> None:
        frames = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in job["frames"] if b]
        frames = [f for f in frames if f is not None]
        if not frames:
            logger.warning("No frames for clip %s", job["path"])
            return
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(job["path"]), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not writer.isOpened():
            logger.error("Could not open video writer for %s", job["path"])
            return
        for f in frames:
            writer.write(f)
        writer.release()
        logger.info("Wrote evidence clip %s (%d frames)", job["path"], len(frames))

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self._output_dir)).replace("\\", "/")
