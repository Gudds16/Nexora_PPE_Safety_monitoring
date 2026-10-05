# NEXORA Module 05 — PPE & Workplace Safety Monitoring

> **Status (honest summary):** the code, rules, tests and documentation are complete and the pipeline has been
> run end-to-end on a *synthetic scripted* video. **No model has been trained yet and no real dataset has been
> used**, so every detection-accuracy number below is **Not measured**. Training must be run on a GPU machine
> (see section 10).

## 1. Module Overview
Detects **person**, **helmet** and **safety_vest** in video, tracks each worker, decides *which worker wears
which PPE*, and raises a single reviewable `PPE_VIOLATION` event when a worker is persistently missing required
PPE. It exists so that safety supervisors review a short list of evidence-backed incidents instead of watching
camera feeds. It is a monitoring / decision-support tool: it reports *potential PPE non-compliance*, never
intent, negligence or danger, and human review is mandatory.

## 2. Module ID
`module_05_ppe`

## 3. Objective
Turn video into standardized, temporally validated PPE events with evidence, where PPE is associated with the
**correct person** (a helmet lying nearby, held in a hand or on a table never makes a worker compliant).

## 4. Supported Detection Classes
`person`, `helmet`, `safety_vest`.
Other PPE (gloves, safety shoes, glasses, mask, ear protection, face shield, safety suit) are **not** implemented:
the spec says to add them only when enough training data and validation exist. `gloves_required` /
`safety_shoes_required` in the config are ignored (a warning is logged if set to true).

## 5. Detection Criteria
All values are in `config/config.yaml` and are **PROVISIONAL** (not tuned — no validation data exists yet).

| Item | Value |
|---|---|
| Confidence thresholds | person 0.40, helmet 0.45, safety_vest 0.45 (`model.class_confidence`), default 0.50 |
| Minimum box area | 64 px² (smaller boxes are dropped as noise) |
| IoU (NMS) | 0.50 |
| Tracking requirement | persons need a tracker id to take part in event validation; untracked persons get a negative temporary id and are ignored by the event rules |
| PPE counted for a worker only if | it lies in that worker's head region (helmet) or torso region (vest) — see section 6 |

## 6. Event Criteria
```
Detection → Tracking → Person-PPE association → Temporal validation → Rule validation → Event
```
**Association** (`inference/association.py`): geometry, not "helmet somewhere in the frame".
* Head region = top 30 % of the person box (central 70 % width, +10 % above). Helmet must have ≥ 50 % of its box inside it, be no taller than 40 % of the person, and be roughly centred on the head.
* Torso region = 15 %–65 % of the person height. Vest must be ≥ 60 % inside the person box and ≥ 50 % inside the torso region.
* Each helmet/vest is assigned to **at most one** worker (best score first) and each worker gets at most one of each.
* Per worker and PPE type the state is `YES`, `NO` or `UNKNOWN`. `UNKNOWN` (head cut off by the frame top, person smaller than 60 px, or head/torso mostly covered by a worker standing nearer the camera) **never** produces a violation.

**Temporal validation** (`rules/event_rules.py`), per worker track:
* ≥ 15 violating frames **and** ≥ 1.0 s of video → one event; it is not repeated while the violation continues.
* ≥ 5 consecutive compliant frames end an episode; shorter flickers do not reset the streak.
* 10 s cooldown per track before another event can fire. Tracks unseen for 30 frames are forgotten.

## 7. Model Architecture
| Item | Value |
|---|---|
| Detector | YOLO11s via `ultralytics` — **one 3-class detector** (person, helmet, safety_vest). Decision: simpler to train/maintain than separate person + PPE models; both public datasets already label persons. |
| Pretrained weights | `yolo11s.pt` (to be downloaded by ultralytics on first training run — not downloaded here) |
| Trained weights | **None yet** (`models/checkpoints/best.pt` does not exist) |
| Input resolution | 640 |
| Confidence / IoU | see section 5 / 0.50 |
| Tracker | ByteTrack through ultralytics (`config/bytetrack_ppe.yaml`) — **not yet exercised** (ultralytics not installed where this was built) |
| Pose model | not used (optional in the spec; geometry association is used instead) |

## 8. Dataset
Full details: [`datasets/README.md`](datasets/README.md). Candidates named by the spec: Pictor-PPE, SH17, COCO
(person pre-training) and custom industrial CCTV (mandatory for production). **Nothing has been downloaded, no
license has been verified, and no images/videos are in this project yet.**

## 9. Custom Dataset
**Not available yet.** Needed: industrial CCTV clips (factory / construction / warehouse, Indian sites, several
helmet/vest colours, low light, occlusion, crowds, several resolutions), each tagged with a camera/session id.
Camera angle, lighting, indoor/outdoor, day/night, resolution, number of videos/frames and annotation method
must be filled in `datasets/README.md` once collected.

## 10. Training
```bash
python module_05_ppe/training/train.py --dry-run     # checks config, hardware and dataset only
python module_05_ppe/training/train.py               # train (GPU strongly recommended)
python module_05_ppe/training/train.py --resume      # continue an interrupted run
```
| Setting | Default (config `training:`) |
|---|---|
| Epochs / image size | 100 / 640 |
| Batch size | `auto` — chosen from detected VRAM (CUDA) or RAM (Apple MPS), halved automatically on out-of-memory |
| Optimizer | `auto` (ultralytics picks) |
| Augmentation | preset `cctv` (`training/augmentations.py`; small hue shift to keep vest colour meaning, wider brightness for low light, mosaic, flip) |
| Pretrained | `yolo11s.pt` |
| Outputs | `models/checkpoints/best.pt` (only replaced by a *better* model), `last.pt`, `best_meta.json`; run folders in `models/training_runs/`; log in `logs/training.log`; status in `MODULE_PROGRESS.md` |

**Training status: not run** (the build environment had no GPU, no internet and no ultralytics).

## 11. Inference
```bash
python module_05_ppe/inference/pipeline.py --source path/to/video.mp4 --camera-id CAM-001 --save-annotated
```
Python:
```python
from module_05_ppe.inference.pipeline import process_video
result = process_video(source="video.mp4", camera_id="CAM-001", config_path="config/config.yaml")
```
Without trained weights the command stops with a clear "Model weights not found" message. To exercise the
plumbing without a model: `python module_05_ppe/evaluation/synthetic_demo.py` (scripted detections, see section 18).

## 12. Input
* Video **files** readable by OpenCV (MP4 etc.). Only a synthetic MP4 has been tested.
* **Not implemented:** RTSP, HLS, webcam, single images (the spec says not to claim what is not tested).

## 13. Output
`process_video()` returns a dict and writes the same to `results.json`:
`module`, `module_version`, `camera_id`, `source`, `start_time`, `frames_processed`, `video_fps`, `frame_size`,
`events[]`, `incidents[]`, `workers{track_id: frame counts}`, `processing{wall_seconds, end_to_end_fps, mean_detector_ms}`, `files`.
Also written per run: `detections.jsonl` (one standard detection per line), `evidence/INC-xxxxxx/…`, optional `annotated_video.mp4`.
Detection `track_id`: for persons the tracker id; for helmet/vest the id of the worker it is associated with (`null` if unassociated, plus `associated_with_worker`).
Event/incident timestamps = `--start-time` (default: processing start, UTC) + video time.

## 14. Event Types
| Event type | Meaning |
|---|---|
| `PPE_VIOLATION` | a worker track was persistently without required PPE (`missing_ppe` lists `helmet` and/or `safety_vest`) |

**Severity rule** (configurable in `severity:`): helmet **and** vest missing → `CRITICAL`; helmet missing → `HIGH` (head-injury protection); vest missing → `MEDIUM` (visibility risk, lower direct injury risk). **Status** values: `PENDING` (all new events), `CONFIRMED`, `DISMISSED`, `UNCERTAIN`, `RESOLVED` (set later by human review). Event confidence = mean *person-detection* confidence over the violating frames (the absence of a helmet has no detector confidence of its own).

## 15. JSON Output
Real output of the synthetic demo (`evaluation/results/synthetic_demo/results.json`), shortened:
```json
{
  "event_id": "EVT-000001", "module": "module_05_ppe", "event_type": "PPE_VIOLATION",
  "camera_id": "CAM-DEMO", "track_id": 2, "timestamp": "2026-10-02T14:30:01.400Z",
  "confidence": 0.9, "severity": "HIGH", "status": "PENDING",
  "rule": {"name": "persistent_missing_ppe", "minimum_frames": 15, "minimum_seconds": 1.0, "cooldown_seconds": 10.0},
  "missing_ppe": ["helmet"], "frame_id": 14, "bbox": {"x1": 215, "y1": 120, "x2": 305, "y2": 390}
}
```
```json
{
  "incident_id": "INC-000001", "module": "module_05_ppe", "title": "PPE Violation",
  "event_type": "PPE_VIOLATION", "camera_id": "CAM-DEMO", "timestamp": "2026-10-02T14:30:01.400Z",
  "severity": "HIGH", "confidence": 0.9, "status": "PENDING",
  "evidence": {"frame": "evidence/INC-000001/frame.jpg", "annotated_frame": "evidence/INC-000001/annotated_frame.jpg",
               "clip": "evidence/INC-000001/clip.mp4"},
  "review": {"decision": null, "reviewer_id": null, "reviewed_at": null}
}
```
Evidence paths are relative to the run's output folder.

## 16. Evidence
Per incident, in `<output_dir>/evidence/INC-xxxxxx/`: `frame.jpg` (raw event frame), `annotated_frame.jpg` (worker boxes, status, banner), `clip.mp4` (5 s before + 5 s after, downscaled to ≤ 960 px wide, from a JPEG ring buffer). Event JSON also carries `timestamp`, `frame_id`, `first_violation_frame_id`, `bbox`, `event_type`, `confidence`.

## 17. Evaluation
Run: `python module_05_ppe/evaluation/evaluate.py --video sample.mp4` → `evaluation/results/evaluation_report.md` + `evaluation_results.json`.

| Metric | Result |
|---|---|
| Association logic on 14 hand-written scenarios | **14 / 14 passed** (measured; checks geometry logic only, not a model) |
| Unit/edge-case tests | **76 / 76 passed** (see section 18 for how they were run) |
| Precision, Recall, F1 | Not measured |
| mAP@50, mAP@50:95 | Not measured |
| FPS, inference latency, end-to-end latency | Not measured on a real model. (The synthetic demo ran 120 frames at ~141 fps end-to-end on a 1-core CPU with a scripted detector costing ~0.03 ms — this reflects association + rules + I/O only, not model speed.) |
| GPU / CPU / VRAM usage | Not measured |
| False positives / false negatives, false alerts per camera-hour | Not measured (needs ground-truth events: `--gt-events`) |
| Performance under difficult conditions | Not measured (evaluate.py can test synthetic low-light / blur / low-res copies once a model exists) |
| Tracking IDF1 / HOTA | Not measured |

No published dataset metric is reused as a NEXORA result.

## 18. Tested Scenarios
`pytest module_05_ppe/tests` (5 files, 76 tests). In the build sandbox `pytest` was not installed, so the same test functions were executed with a minimal stand-in runner: **76 passed, 0 failed**. Run real `pytest` on your machine to confirm.
All tests use fake model output or hand-written boxes: they verify **module logic**, not neural-network accuracy.
* **PPE association** (`test_ppe_association.py`): helmet worn, held in hand, on a table, lying nearby, partially hidden/occluded worker, no helmet, vest worn / held, helmet between two workers, two workers close together, overlapping workers, crowd of six, head cut off, tiny person, one helmet never serving two workers.
* **Detection** (`test_detection.py`): correct object, wrong object (cap), low confidence, multiple objects, small object, occluded object, blurred object, empty output, tracker ids, missing weights.
* **Events** (`test_events.py`): detection→event, no event, persistent (one event only), temporary, flicker tolerance, recovery, minimum seconds, cooldown, multiple tracks, severity rules, UNKNOWN handling, approved status/severity values, incident structure.
* **Edge cases** (`test_edge_cases.py`): empty scene, low light, crowded scene (12 workers), occlusion, camera-movement jitter, blur, partial visibility, false positive / false negative situations, multiple cameras, full pipeline on a tiny synthetic MP4 (event + evidence files + JSON), missing video/weights, bad config.
* **Support code** (`test_training_utils.py`): batch-size selection, error diagnosis, never-overwrite-better-checkpoint, dataset validation and session-leakage detection, label remapping, event matching metrics, progress file.

**Demo:** `python module_05_ppe/evaluation/synthetic_demo.py` renders a cartoon video, runs scripted detections through the real pipeline and produced exactly 2 events (workers 2 and 3) from 120 frames, none for the compliant worker. Files in `evaluation/results/synthetic_demo/`: `demo.mp4`, `annotated_demo.mp4`, `results.json`, `detections.jsonl`, `evidence/`. **This is a pipeline demo, not a model demo.**

## 19. Known Limitations
* No trained model, no real data, no measured accuracy yet; all thresholds are provisional.
* Association is 2-D box geometry: it cannot see depth, so a helmet held at head height in front of a bare head can look worn; pose estimation (optional in the spec) is not used.
* Tracker ID switches create a new track and can produce a second event for the same person (cooldown is per track id).
* "Nearer worker" occlusion uses feet position — unreliable with unusual camera angles.
* Only helmet and vest; only video files; no RTSP/webcam; no zone rules (the spec mentions zones — not implemented).
* ByteTrack integration, training, validation and speed scripts have never been executed against real ultralytics in the build environment.
* Event timestamps are processing-start based unless `--start-time` is supplied.

## 20. False Positive Cases
(Alerts that may be wrong — human review required)
* A worn helmet missed by the detector for longer than ~1.5 s (heavy blur, glare, very small/far worker).
* Unusual headgear/coloured helmets or hi-vis vests the detector was never trained on.
* A second track created for the same worker after an ID switch.
* Seated/crouching/bending workers whose head or torso falls outside the assumed body proportions.

## 21. False Negative Cases
(Violations that may be missed)
* Worker never detected as a person, or a brief appearance shorter than 15 frames / 1 s.
* Workers judged `UNKNOWN` (occluded, cut off, very small) — deliberately no alert.
* A cap, hood or shirt falsely detected as helmet/vest, or a helmet worn on the back of the head outside the assumed head region.
* A helmet held right at head height in front of a bare head.

## 22. Hardware Requirements
**Not measured.** Suggestions only: training — NVIDIA GPU with ≥ 6 GB VRAM recommended (Apple Silicon MPS works but slower; CPU-only is impractical); the code picks a batch size automatically. Inference — real-time speed on any given GPU/CPU is unknown until `evaluate.py --video` is run. RAM/CPU needs: not measured. The tests and demo run on a 1-core, ~4 GB CPU machine.

## 23. Integration Instructions
```python
from module_05_ppe.inference.pipeline import process_video   # run from the project root
result = process_video(source="video.mp4", camera_id="CAM-001")
for incident in result["incidents"]:
    ...   # standard incident dicts: incident_id, module, title, event_type, camera_id, timestamp, severity, confidence, status, evidence, review
```
* The integration layer needs no model details: `module_05_ppe/models/checkpoints/best.pt` + `config/config.yaml` are the only inputs.
* Field names follow `FileStructureandPrecautions.docx`; extra module fields are additive.
* Frontend should show `PENDING` incidents for review and write back `CONFIRMED / DISMISSED / UNCERTAIN / RESOLVED`.
* This module is standalone (no `ml/common/` dependency). If NEXORA later provides shared tracking/video readers, replace `PPEDetector` tracking and the `cv2.VideoCapture` loop in `pipeline.py` — the rules and association code do not depend on them.
* Model files are git-ignored; use Git LFS or external storage for `best.pt`.

## 24. Dependencies
`requirements.txt` (project root) — version **ranges** because the build environment had no internet to confirm pins: `ultralytics>=8.3.0,<9`, `torch>=2.2`, `opencv-python>=4.8`, `numpy>=1.24`, `PyYAML>=6.0`, `psutil>=5.9`, `pytest>=7.4`. Tests/demo were run with Python 3.12.3, numpy 2.4.4, opencv 4.13.0, PyYAML 6.0.3. After installing, run `pip freeze > requirements.lock.txt` to pin exact versions.

## 25. Version
v1.0.0 (code complete; model untrained)
