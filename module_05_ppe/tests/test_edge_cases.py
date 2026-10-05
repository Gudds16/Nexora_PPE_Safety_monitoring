"""Edge cases through the real association + rules (+ a full pipeline run on a tiny synthetic video).

Detections are hand-written, so these tests verify the module's LOGIC under difficult situations;
they do NOT measure how well a trained model copes with real low light, blur, etc.
"""
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import yaml

from module_05_ppe.config_loader import ConfigError, load_config
from module_05_ppe.evaluation import scenarios as sc
from module_05_ppe.inference.association import AssociationParams, associate
from module_05_ppe.inference.detector import ModelNotFoundError, ScriptedDetector
from module_05_ppe.inference.pipeline import process_video
from module_05_ppe.rules.event_rules import PPEEventEngine
from module_05_ppe.schemas import BBox, Detection
from module_05_ppe.tests.helpers import FPS, fast_config
from module_05_ppe.training.augmentations import simulate_low_light

CFG = fast_config()
PARAMS = AssociationParams.from_config(CFG["association"])


def simulate(frames, camera_id="CAM-001", engine=None):
    """frames: list of (persons, helmets, vests) per frame -> (events, engine)."""
    engine = engine or PPEEventEngine(CFG, camera_id)
    events = []
    for i, (persons, helmets, vests) in enumerate(frames):
        events += engine.update(associate(persons, helmets, vests, PARAMS), i, i / FPS, f"t{i}")
    return events, engine


def shifted(det, dx, dy):
    b = det.bbox
    return Detection(det.class_name, det.confidence, BBox(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy), det.track_id)


def worker(track_id, cx, helmet=True, vest=True, top=150):
    p = sc.person(track_id, cx, top=top)
    return p, ([sc.helmet_worn(p)] if helmet else []), ([sc.vest_worn(p)] if vest else [])


def scene(workers):
    persons = [w[0] for w in workers]
    return persons, [h for w in workers for h in w[1]], [v for w in workers for v in w[2]]


# ------------------------------------------------------------------ the spec's edge-case list
def test_empty_scene_gives_no_events():
    events, engine = simulate([([], [], [])] * 50)
    assert events == [] and engine.active_tracks() == []


def test_low_light_darkens_frames_and_missing_detections_do_not_invent_events():
    img = np.full((100, 100, 3), 200, np.uint8)
    assert simulate_low_light(img).mean() < 80
    events, _ = simulate([([], [], [])] * 60)                  # detector sees nothing in the dark
    assert events == []


def test_crowded_scene_all_compliant_has_no_events():
    crowd = scene([worker(i + 1, 60 + i * 105) for i in range(12)])
    events, _ = simulate([crowd] * 30)
    assert events == []


def test_crowded_scene_flags_exactly_the_workers_without_helmets():
    bare = {3, 7, 11}
    crowd = scene([worker(i + 1, 60 + i * 105, helmet=(i + 1) not in bare) for i in range(12)])
    events, _ = simulate([crowd] * 30)
    assert sorted(e.track_id for e in events) == sorted(bare)
    assert all(e.missing_ppe == ["helmet"] for e in events)


def test_occluded_worker_never_triggers_an_event():
    persons, helmets, vests, _ = sc.all_scenarios()["occluded_worker_unknown"]
    events, _ = simulate([(persons, helmets, vests)] * 60)
    assert events == []


def test_camera_movement_jitter_keeps_correct_results():
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(40):
        dx, dy = rng.integers(-8, 9, 2)
        ok = scene([worker(1, 200), worker(2, 400, helmet=False)])
        frames.append(tuple([shifted(d, dx, dy) for d in group] for group in ok))
    events, _ = simulate(frames)
    assert [e.track_id for e in events] == [2]                 # worker 2 only, exactly once


def test_blur_causing_brief_missed_helmets_does_not_raise_false_alert():
    with_helmet, without = scene([worker(1, 300)]), scene([worker(1, 300, helmet=False)])
    frames = ([with_helmet] * 8 + [without] * 2) * 10          # helmet vanishes 2 of every 10 frames
    events, _ = simulate(frames)
    assert events == []


def test_sustained_missed_helmet_does_raise_alert_documented_false_positive_case():
    # If the detector loses a REALLY worn helmet for a long time (heavy blur), a violation is raised.
    # This is a known limitation (README: False Positive Cases); human review resolves it.
    events, _ = simulate([scene([worker(1, 300, helmet=False)])] * 20)
    assert len(events) == 1


def test_partial_visibility_head_cut_off_is_not_a_violation():
    cut = sc.person(1, 300, top=0)
    events, _ = simulate([([cut], [], [sc.vest_worn(cut)])] * 60)
    assert events == []


def test_false_positive_object_a_cap_is_not_a_helmet():
    # The detector has no 'cap' class output; a worker with only a cap has NO helmet detection.
    events, _ = simulate([scene([worker(1, 300, helmet=False)])] * 10)
    assert len(events) == 1 and events[0].missing_ppe == ["helmet"]


def test_false_negative_guard_helmet_in_hand_still_alerts():
    persons, helmets, vests, _ = sc.all_scenarios()["helmet_in_hand"]
    events, _ = simulate([(persons, helmets, vests)] * 10)
    assert len(events) == 1                                    # a nearby helmet must not hide a violation


def test_helmet_between_two_workers_still_alerts_both():
    persons, helmets, vests, _ = sc.all_scenarios()["two_workers_helmet_between_on_ground"]
    events, _ = simulate([(persons, helmets, vests)] * 10)
    assert sorted(e.track_id for e in events) == [1, 2]


def test_multiple_cameras_are_isolated():
    bad = scene([worker(1, 300, helmet=False)])
    good = scene([worker(1, 300)])
    ev_a, eng_a = simulate([bad] * 10, "CAM-001")
    ev_b, eng_b = simulate([good] * 10, "CAM-002")
    assert len(ev_a) == 1 and ev_a[0].camera_id == "CAM-001" and ev_b == []
    ev_c, _ = simulate([bad] * 10, "CAM-003")                  # same track id 1, different camera
    assert len(ev_c) == 1 and ev_c[0].camera_id == "CAM-003" and ev_c[0].event_id == "EVT-000001"


# ------------------------------------------------------------------ full pipeline on a tiny synthetic video
def _write_video(path, frames=40, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), 60 + i, np.uint8))
    writer.release()


def _config_file(directory):
    cfg = copy.deepcopy(CFG)
    cfg.pop("_config_file", None)
    cfg["evidence"].update(pre_event_seconds=1, post_event_seconds=1)
    path = Path(directory) / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _script():
    ok = sc.person(1, 80, top=40, h=170, w=60)
    bad = sc.person(2, 200, top=40, h=170, w=60)
    dets = [ok, sc.helmet_worn(ok), sc.vest_worn(ok), bad, sc.vest_worn(bad)]
    return lambda frame_id: dets


def test_full_pipeline_on_synthetic_video_creates_event_evidence_and_json():
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "clip.mp4"
        _write_video(video)
        out = Path(tmp) / "out"
        result = process_video(str(video), "CAM-009", _config_file(tmp), output_dir=str(out),
                               detector=ScriptedDetector(_script()), save_annotated=True,
                               start_time="2026-10-02T14:30:00Z")
        assert result["module"] == "module_05_ppe" and result["camera_id"] == "CAM-009"
        assert result["frames_processed"] == 40 and len(result["events"]) == 1
        event, incident = result["events"][0], result["incidents"][0]
        assert event["track_id"] == 2 and event["missing_ppe"] == ["helmet"] and event["timestamp"].endswith("Z")
        assert incident["status"] == "PENDING" and incident["incident_id"] == "INC-000001"
        for rel in incident["evidence"].values():
            assert rel and (out / rel).is_file(), rel
        assert (out / "results.json").is_file() and (out / "annotated_video.mp4").is_file()
        first = json.loads((out / "detections.jsonl").read_text().splitlines()[0])
        assert set(first) >= {"module", "camera_id", "frame_id", "timestamp", "track_id", "class_name", "confidence", "bbox"}
        assert json.loads((out / "results.json").read_text())["events"][0]["event_id"] == "EVT-000001"


def test_pipeline_two_cameras_write_separate_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "clip.mp4"
        _write_video(video, frames=30)
        cfg = _config_file(tmp)
        a = process_video(str(video), "CAM-001", cfg, output_dir=str(Path(tmp) / "a"), detector=ScriptedDetector(_script()))
        b = process_video(str(video), "CAM-002", cfg, output_dir=str(Path(tmp) / "b"), detector=ScriptedDetector(lambda f: []))
        assert a["camera_id"] == "CAM-001" and len(a["events"]) == 1
        assert b["camera_id"] == "CAM-002" and b["events"] == []


def test_pipeline_missing_video_and_missing_weights_fail_clearly():
    try:
        process_video("no_such_video.mp4", "CAM-001")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "c.mp4"
        _write_video(video, frames=5)
        try:
            process_video(str(video), "CAM-001", output_dir=str(Path(tmp) / "o"), weights_path="models/checkpoints/none.pt")
        except ModelNotFoundError:
            return
        raise AssertionError("expected ModelNotFoundError")


def test_config_with_wrong_module_id_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = copy.deepcopy(CFG)
        cfg.pop("_config_file", None)
        cfg["module"]["id"] = "module_99_wrong"
        path = Path(tmp) / "bad.yaml"
        path.write_text(yaml.safe_dump(cfg))
        try:
            load_config(str(path))
        except ConfigError:
            return
        raise AssertionError("expected ConfigError")
