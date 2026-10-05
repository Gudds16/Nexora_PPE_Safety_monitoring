"""Event-layer tests: detections -> temporally validated events (no alert spam)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module_05_ppe.rules.event_rules import severity_for
from module_05_ppe.schemas import MODULE_ID, SEVERITY_VALUES, STATUS_VALUES
from module_05_ppe.tests.helpers import fast_config, make_engine, obs, repeat, run_frames
from module_05_ppe.rules.event_rules import PPEEventEngine

VIOLATING = [obs(1, helmet="NO")]
OK = [obs(1)]


def test_detection_becomes_event_after_enough_frames():
    events = run_frames(make_engine(), repeat(VIOLATING, 5))
    assert len(events) == 1
    e = events[0].to_dict()
    for key in ("event_id", "module", "event_type", "camera_id", "track_id", "timestamp",
                "confidence", "severity", "status"):
        assert key in e
    assert e["module"] == MODULE_ID and e["event_type"] == "PPE_VIOLATION"
    assert e["event_id"] == "EVT-000001" and e["status"] == "PENDING" and e["severity"] == "HIGH"
    assert e["missing_ppe"] == ["helmet"] and e["track_id"] == 1


def test_compliant_worker_produces_no_event():
    assert run_frames(make_engine(), repeat(OK, 100)) == []


def test_persistent_violation_gives_exactly_one_event():
    events = run_frames(make_engine(), repeat(VIOLATING, 300))
    assert len(events) == 1                                # frame-by-frame spam avoided


def test_temporary_violation_gives_no_event():
    events = run_frames(make_engine(), repeat(VIOLATING, 4) + repeat(OK, 20))
    assert events == []


def test_short_compliant_flicker_does_not_reset_the_streak():
    frames = repeat(VIOLATING, 3) + repeat(OK, 2) + repeat(VIOLATING, 3)      # flicker < recovery_frames (3)
    assert len(run_frames(make_engine(), frames)) == 1


def test_recovery_resets_the_streak():
    frames = repeat(VIOLATING, 4) + repeat(OK, 3) + repeat(VIOLATING, 4)      # recovered in between
    assert run_frames(make_engine(), frames) == []


def test_minimum_seconds_is_enforced():
    engine = PPEEventEngine(fast_config(minimum_frames=3, minimum_seconds=2.0), "CAM-001")
    events = run_frames(engine, repeat(VIOLATING, 15))                        # 1.4 s of video at 10 fps
    assert events == []
    events = run_frames(PPEEventEngine(fast_config(minimum_frames=3, minimum_seconds=1.0), "CAM-001"),
                        repeat(VIOLATING, 15))
    assert len(events) == 1


def test_cooldown_blocks_repeat_events_then_allows_one():
    frames = repeat(VIOLATING, 10) + repeat(OK, 5) + repeat(VIOLATING, 200)
    events = run_frames(make_engine(), frames)
    assert len(events) == 2
    assert events[1].video_time_s >= events[0].video_time_s + 10               # only after cooldown


def test_multiple_tracked_workers_are_independent():
    frame = [obs(1, helmet="NO"), obs(2), obs(3, vest="NO")]
    events = run_frames(make_engine(), repeat(frame, 6))
    by_track = {e.track_id: e for e in events}
    assert sorted(by_track) == [1, 3]
    assert by_track[1].severity == "HIGH" and by_track[1].missing_ppe == ["helmet"]
    assert by_track[3].severity == "MEDIUM" and by_track[3].missing_ppe == ["safety_vest"]


def test_both_missing_is_critical():
    events = run_frames(make_engine(), repeat([obs(1, helmet="NO", vest="NO")], 6))
    assert events[0].severity == "CRITICAL" and events[0].missing_ppe == ["helmet", "safety_vest"]


def test_unknown_visibility_never_creates_an_event():
    assert run_frames(make_engine(), repeat([obs(1, helmet="UNKNOWN", vest="UNKNOWN")], 100)) == []


def test_unknown_frames_hold_the_streak_without_adding():
    frames = repeat(VIOLATING, 3) + repeat([obs(1, helmet="UNKNOWN")], 20) + repeat(VIOLATING, 2)
    assert len(run_frames(make_engine(), frames)) == 1


def test_untracked_person_is_ignored():
    assert run_frames(make_engine(), repeat([obs(-1, helmet="NO")], 50)) == []


def test_vest_not_required_is_not_a_violation():
    cfg = fast_config()
    cfg["ppe_requirements"]["safety_vest_required"] = False
    events = run_frames(PPEEventEngine(cfg, "CAM-001"), repeat([obs(1, vest="NO")], 20))
    assert events == []


def test_vanished_track_state_expires():
    frames = repeat(VIOLATING, 3) + repeat([], 40) + repeat(VIOLATING, 3)
    assert run_frames(make_engine(), frames) == []


def test_only_approved_status_and_severity_values():
    events = run_frames(make_engine(), repeat([obs(1, helmet="NO")], 6))
    assert events[0].status in STATUS_VALUES and events[0].severity in SEVERITY_VALUES
    for missing in (["helmet"], ["safety_vest"], ["helmet", "safety_vest"], []):
        assert severity_for(missing, fast_config()["severity"]) in SEVERITY_VALUES


def test_incident_has_standard_structure():
    event = run_frames(make_engine(), repeat(VIOLATING, 6))[0]
    inc = event.to_incident("INC-000001", {"frame": "evidence/INC-000001/frame.jpg",
                                           "annotated_frame": None, "clip": None})
    for key in ("incident_id", "module", "title", "event_type", "camera_id", "timestamp", "severity",
                "confidence", "status", "evidence", "review"):
        assert key in inc
    assert inc["status"] == "PENDING" and inc["review"] == {"decision": None, "reviewer_id": None, "reviewed_at": None}
    assert "intentional" not in inc["description"].lower() and "Human review" in inc["description"]


def test_invalid_severity_in_config_is_rejected():
    try:
        severity_for(["helmet"], {"helmet_missing": "SEVERE"})
    except ValueError:
        return
    raise AssertionError("expected ValueError")
