"""Business rules: does a stream of per-frame worker observations qualify as a NEXORA event?

This file never touches the model. It answers only:
  "Does this detection qualify as a NEXORA event?"

State machine per worker track (spec: no frame-by-frame alert spam):

  IDLE --violation--> CANDIDATE --(enough frames AND seconds, cooldown over)--> ACTIVE
  ACTIVE / CANDIDATE --(recovery_frames compliant frames)--> IDLE

* One event is emitted when an ACTIVE episode starts; it is NOT repeated while the
  violation continues.
* After an event, the same track cannot raise a new one for `cooldown_seconds`.
* Frames where PPE is UNKNOWN (occluded / too small) neither add to nor reset a streak.
* Short compliant flickers (< recovery_frames) do not reset a streak.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from module_05_ppe.inference.association import NO, UNKNOWN, YES, WorkerObservation
from module_05_ppe.schemas import (EVENT_PPE_VIOLATION, MODULE_ID, SEVERITY_VALUES, BBox)

logger = logging.getLogger(__name__)

IDLE, CANDIDATE, ACTIVE = "IDLE", "CANDIDATE", "ACTIVE"
RULE_NAME = "persistent_missing_ppe"


@dataclass
class PPEEvent:
    """A temporally validated PPE violation (spec: 'Event')."""

    event_id: str
    camera_id: str
    track_id: int
    timestamp: str
    confidence: float
    severity: str
    frame_id: int
    first_frame_id: int
    bbox: BBox
    missing_ppe: List[str]
    video_time_s: float
    rule: Dict[str, Any]
    event_type: str = EVENT_PPE_VIOLATION
    status: str = "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        """Standard event output + module-specific extras."""
        return {
            "event_id": self.event_id, "module": MODULE_ID, "event_type": self.event_type,
            "camera_id": self.camera_id, "track_id": self.track_id, "timestamp": self.timestamp,
            "confidence": round(self.confidence, 4), "severity": self.severity, "status": self.status,
            "rule": self.rule, "missing_ppe": list(self.missing_ppe), "frame_id": self.frame_id,
            "first_violation_frame_id": self.first_frame_id, "video_time_s": round(self.video_time_s, 3),
            "bbox": self.bbox.to_dict(),
        }

    def to_incident(self, incident_id: str, evidence: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
        """Standard incident output (platform-level, reviewable record)."""
        missing = " and ".join(m.replace("_", " ") for m in self.missing_ppe)
        return {
            "incident_id": incident_id, "module": MODULE_ID, "title": "PPE Violation",
            "event_type": self.event_type, "camera_id": self.camera_id, "timestamp": self.timestamp,
            "severity": self.severity, "confidence": round(self.confidence, 4), "status": self.status,
            "description": f"Potential PPE non-compliance: {missing} not detected on worker track "
                           f"{self.track_id}. Human review required.",
            "event_id": self.event_id, "track_id": self.track_id,
            "evidence": evidence or {"frame": None, "annotated_frame": None, "clip": None},
            "review": {"decision": None, "reviewer_id": None, "reviewed_at": None},
        }


@dataclass
class _TrackState:
    state: str = IDLE
    violation_frames: int = 0
    compliant_streak: int = 0
    first_frame: int = 0
    first_time: float = 0.0
    last_seen_frame: int = 0
    cooldown_until: float = -1.0
    missing: Set[str] = field(default_factory=set)
    conf_sum: float = 0.0


def severity_for(missing: Sequence[str], severity_cfg: Dict[str, str]) -> str:
    """Severity rule (documented in README):

    helmet AND vest missing -> both_missing (CRITICAL)
    helmet missing          -> helmet_missing (HIGH): head injury is the most severe outcome
    vest missing            -> safety_vest_missing (MEDIUM): visibility risk, lower direct injury risk
    anything else           -> LOW
    """
    missing_set = set(missing)
    if {"helmet", "safety_vest"} <= missing_set:
        level = severity_cfg.get("both_missing", "CRITICAL")
    elif "helmet" in missing_set:
        level = severity_cfg.get("helmet_missing", "HIGH")
    elif "safety_vest" in missing_set:
        level = severity_cfg.get("safety_vest_missing", "MEDIUM")
    else:
        level = "LOW"
    if level not in SEVERITY_VALUES:
        raise ValueError(f"Invalid severity '{level}' in config; allowed: {SEVERITY_VALUES}")
    return level


class PPEEventEngine:
    """Turns per-frame WorkerObservations into validated events. One engine per camera."""

    def __init__(self, config: Dict[str, Any], camera_id: str, id_start: int = 1):
        ev = config["event"]
        self.camera_id = camera_id
        self.min_frames = int(ev["minimum_frames"])
        self.min_seconds = float(ev.get("minimum_seconds", 0.0))
        self.recovery_frames = int(ev.get("recovery_frames", 1))
        self.cooldown = float(ev.get("cooldown_seconds", 0.0))
        self.max_age = int(ev.get("track_max_age_frames", 30))
        req = config["ppe_requirements"]
        self.required: List[str] = [k for k, flag in (("helmet", req.get("helmet_required")),
                                                      ("safety_vest", req.get("safety_vest_required"))) if flag]
        self.severity_cfg = config["severity"]
        self._tracks: Dict[int, _TrackState] = {}
        self._next_id = id_start

    # ------------------------------------------------------------------ helpers
    def _classify(self, obs: WorkerObservation):
        """Return ('violation', missing) | ('compliant', []) | ('neutral', [])."""
        missing = [k for k in self.required if obs.states.get(k) == NO]
        if missing:
            return "violation", missing
        if all(obs.states.get(k) == YES for k in self.required):
            return "compliant", []
        return "neutral", []

    def _new_event_id(self) -> str:
        event_id = f"EVT-{self._next_id:06d}"
        self._next_id += 1
        return event_id

    # ------------------------------------------------------------------ main
    def update(self, observations: Sequence[WorkerObservation], frame_id: int,
               video_time_s: float, timestamp_iso: str) -> List[PPEEvent]:
        """Process one frame. Returns the (usually empty) list of NEW events."""
        events: List[PPEEvent] = []
        for obs in observations:
            if obs.track_id < 0:      # untracked person: cannot validate over time
                continue
            st = self._tracks.setdefault(obs.track_id, _TrackState(first_frame=frame_id,
                                                                    first_time=video_time_s))
            st.last_seen_frame = frame_id
            kind, missing = self._classify(obs)

            if kind == "violation":
                st.compliant_streak = 0
                if st.state == IDLE:
                    st.state, st.first_frame, st.first_time = CANDIDATE, frame_id, video_time_s
                    st.violation_frames, st.conf_sum, st.missing = 0, 0.0, set()
                st.violation_frames += 1
                st.conf_sum += obs.person.confidence
                st.missing.update(missing)
                ready = (st.state == CANDIDATE and st.violation_frames >= self.min_frames
                         and video_time_s - st.first_time >= self.min_seconds)
                if ready and video_time_s >= st.cooldown_until:
                    events.append(self._emit(obs, st, frame_id, video_time_s, timestamp_iso))
            elif kind == "compliant":
                st.compliant_streak += 1
                if st.compliant_streak >= self.recovery_frames and st.state != IDLE:
                    logger.debug("Track %s recovered (episode ended)", obs.track_id)
                    st.state, st.violation_frames, st.missing, st.conf_sum = IDLE, 0, set(), 0.0
            # 'neutral' (UNKNOWN visibility): hold state, change nothing

        self._expire_tracks(frame_id)
        return events

    def _emit(self, obs: WorkerObservation, st: _TrackState, frame_id: int,
              video_time_s: float, timestamp_iso: str) -> PPEEvent:
        missing = sorted(st.missing)
        st.state = ACTIVE
        st.cooldown_until = video_time_s + self.cooldown
        event = PPEEvent(
            event_id=self._new_event_id(), camera_id=self.camera_id, track_id=obs.track_id,
            timestamp=timestamp_iso, confidence=min(1.0, st.conf_sum / max(1, st.violation_frames)),
            severity=severity_for(missing, self.severity_cfg), frame_id=frame_id,
            first_frame_id=st.first_frame, bbox=obs.person_bbox, missing_ppe=missing,
            video_time_s=video_time_s,
            rule={"name": RULE_NAME, "minimum_frames": self.min_frames,
                  "minimum_seconds": self.min_seconds, "cooldown_seconds": self.cooldown},
        )
        logger.info("%s %s track=%s missing=%s severity=%s", event.event_id, event.event_type,
                    event.track_id, missing, event.severity)
        return event

    def _expire_tracks(self, frame_id: int) -> None:
        stale = [t for t, s in self._tracks.items()
                 if frame_id - s.last_seen_frame > (self.max_age if s.state != ACTIVE else self.max_age * 20)]
        # ACTIVE tracks are kept 20x longer so cooldown still protects against re-alerts
        # when the tracker briefly loses and re-finds the same worker with the same id.
        for track_id in stale:
            del self._tracks[track_id]

    def active_tracks(self) -> List[int]:
        return [t for t, s in self._tracks.items() if s.state == ACTIVE]
