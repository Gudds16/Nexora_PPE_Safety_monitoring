"""Module 05 evaluation. Produces evaluation/results/evaluation_results.json and evaluation_report.md.

    python module_05_ppe/evaluation/evaluate.py --video sample.mp4
    python module_05_ppe/evaluation/evaluate.py --video sample.mp4 --gt-events gt.json --results run1/results.json

Sections (each is measured, or reported as "Not measured" with the reason - never estimated):
  1. association_scenarios  - logic check on hand-written geometry (always runs, needs no model)
  2. detection_metrics      - precision/recall/F1/mAP50/mAP50-95 on the test split (needs model + dataset)
  3. speed                  - FPS, latency, CPU, GPU memory on --video (needs model + video)
  4. difficult_conditions   - detection metrics on low-light / blurred / low-res copies of the split
  5. event_level            - missed events, false alerts, false alerts per camera-hour
                              (needs --gt-events ground truth JSON + --results pipeline outputs)

Ground-truth file format for --gt-events:
  {"videos": [{"camera_id": "CAM-001", "duration_s": 600, "events": [{"time_s": 42.0}, ...]}]}
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from module_05_ppe.config_loader import MODULE_DIR, load_config, resolve_path
from module_05_ppe.evaluation.scenarios import all_scenarios
from module_05_ppe.hardware import inspect_hardware, select_device
from module_05_ppe.inference.association import AssociationParams, associate
from module_05_ppe.progress import ProgressTracker
from module_05_ppe.training.augmentations import CONDITIONS
from module_05_ppe.training.dataset_tools import labels_dir_for, list_images, resolve_split_dirs, validate_dataset

logger = logging.getLogger("module_05_ppe.evaluate")
RESULTS_DIR = MODULE_DIR / "evaluation" / "results"
NOT_MEASURED = "Not measured"


def not_measured(reason: str) -> Dict[str, str]:
    return {"status": NOT_MEASURED, "reason": reason}


# ----------------------------------------------------------------------------- 1. association scenarios
def evaluate_association_scenarios(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run every synthetic scenario and compare worker states with the expected ones."""
    params = AssociationParams.from_config(config["association"])
    rows = []
    for name, (persons, helmets, vests, expected) in all_scenarios().items():
        got = {o.track_id: {"helmet": o.states["helmet"], "safety_vest": o.states["safety_vest"]}
               for o in associate(persons, helmets, vests, params)}
        rows.append({"scenario": name, "passed": got == expected, "expected": expected, "got": got})
    passed = sum(r["passed"] for r in rows)
    return {"status": "measured", "note": "hand-written geometry, checks logic only - not model accuracy",
            "passed": passed, "total": len(rows), "scenarios": rows}


# ----------------------------------------------------------------------------- 5. event-level metrics
def match_events(gt_times: Sequence[float], pred_times: Sequence[float], tolerance_s: float) -> Dict[str, int]:
    """One-to-one greedy matching of predicted events to ground-truth events within a time tolerance."""
    unmatched_gt = sorted(gt_times)
    tp = 0
    for t in sorted(pred_times):
        best = min(unmatched_gt, key=lambda g: abs(g - t), default=None)
        if best is not None and abs(best - t) <= tolerance_s:
            unmatched_gt.remove(best)
            tp += 1
    return {"true_positive": tp, "false_alerts": len(pred_times) - tp, "missed_events": len(unmatched_gt)}


def evaluate_events(gt_path: Path, result_paths: Sequence[Path], tolerance_s: float) -> Dict[str, Any]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    preds: Dict[str, List[float]] = {}
    for rp in result_paths:
        res = json.loads(rp.read_text(encoding="utf-8"))
        preds.setdefault(res["camera_id"], []).extend(e["video_time_s"] for e in res["events"])
    totals = {"true_positive": 0, "false_alerts": 0, "missed_events": 0}
    hours = 0.0
    for video in gt["videos"]:
        hours += float(video["duration_s"]) / 3600.0
        m = match_events([e["time_s"] for e in video["events"]], preds.get(video["camera_id"], []), tolerance_s)
        for k in totals:
            totals[k] += m[k]
    tp, fa, miss = totals["true_positive"], totals["false_alerts"], totals["missed_events"]
    return {"status": "measured", **totals, "tolerance_s": tolerance_s, "camera_hours": round(hours, 4),
            "event_precision": tp / (tp + fa) if tp + fa else None,
            "event_recall": tp / (tp + miss) if tp + miss else None,
            "false_alerts_per_camera_hour": fa / hours if hours > 0 else None,
            "note": "Assumes one camera/video per camera_id in --results; matching is by time only."}


# ----------------------------------------------------------------------------- 3. speed
def evaluate_speed(config: Dict[str, Any], video: Path, weights: Optional[str], sample_frames: int) -> Dict[str, Any]:
    from module_05_ppe.inference.detector import PPEDetector
    from module_05_ppe.inference.pipeline import process_video
    detector = PPEDetector(config, weights_path=weights)
    cap, latencies = cv2.VideoCapture(str(video)), []
    for i in range(sample_frames):
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        detector.detect(frame, i)
        latencies.append((time.perf_counter() - t) * 1000.0)
    cap.release()
    if len(latencies) < 5:
        return not_measured("video too short for speed measurement")
    warm = sorted(latencies[3:])                       # drop first frames (model warm-up)
    out: Dict[str, Any] = {"status": "measured", "frames_timed": len(warm), "device": detector.device,
                           "mean_inference_ms": sum(warm) / len(warm), "p95_inference_ms": warm[int(0.95 * (len(warm) - 1))],
                           "inference_fps": 1000.0 * len(warm) / sum(warm)}
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        res = process_video(str(video), "CAM-EVAL", output_dir=tmp, detector=detector, max_frames=sample_frames)
    wall = time.perf_counter() - wall0
    out.update(end_to_end_fps=res["processing"]["end_to_end_fps"],
               end_to_end_ms_per_frame=1000.0 * wall / max(1, res["frames_processed"]),
               process_cpu_percent=100.0 * (time.process_time() - cpu0) / wall if wall > 0 else None)
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:
        pass
    out.setdefault("gpu_peak_memory_mb", NOT_MEASURED + " (no CUDA device)")
    return out


# ----------------------------------------------------------------------------- 4. difficult conditions
def build_degraded_dataset(data_yaml: Path, split: str, condition: str, dest: Path, limit: int = 300) -> Path:
    """Copy up to `limit` images of a split with a degradation applied; return the new data.yaml."""
    names, split_dirs = resolve_split_dirs(data_yaml)
    img_out, lab_out = dest / "images", dest / "labels"
    img_out.mkdir(parents=True), lab_out.mkdir(parents=True)
    n = 0
    for d in split_dirs[split]:
        for img_path in list_images(d):
            if n >= limit:
                break
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            cv2.imwrite(str(img_out / img_path.name), CONDITIONS[condition](frame))
            lab = labels_dir_for(d) / (img_path.stem + ".txt")
            if lab.is_file():
                shutil.copy2(lab, lab_out / lab.name)
            n += 1
    yaml_path = dest / "data.yaml"
    yaml_path.write_text(json.dumps({"path": str(dest), "train": "images", "val": "images",
                                     "names": {i: n_ for i, n_ in enumerate(names)}}), encoding="utf-8")
    return yaml_path


def evaluate_difficult(config: Dict[str, Any], weights: Path, data_yaml: Path, split: str) -> Dict[str, Any]:
    from module_05_ppe.training.validate import run_validation
    device = select_device(config["model"].get("device", "auto"), inspect_hardware())
    out: Dict[str, Any] = {"status": "measured",
                           "note": "Synthetic degradations of the same split (not real night/blur footage)."}
    for cond in CONDITIONS:
        with tempfile.TemporaryDirectory() as tmp:
            degraded = build_degraded_dataset(data_yaml, split, cond, Path(tmp) / cond)
            res = run_validation(weights, degraded, "val", int(config["model"]["input_size"]), 8, device)
        out[cond] = {k: res[k] for k in ("precision", "recall", "f1", "mAP50", "mAP50-95")}
    return out


# ----------------------------------------------------------------------------- report
def render_report(results: Dict[str, Any]) -> str:
    lines = ["# NEXORA Module 05 - Evaluation Report", "",
             f"Generated (UTC): {results['generated_utc']}", "",
             "Every number below was measured by evaluate.py. 'Not measured' means it was not.", ""]
    for section, body in results["sections"].items():
        lines += [f"## {section}", ""]
        if body.get("status") == NOT_MEASURED:
            lines += [f"**Not measured** - {body['reason']}", ""]
            continue
        if section == "association_scenarios":
            lines += [f"{body['passed']}/{body['total']} synthetic scenarios passed. {body['note']}", ""]
            lines += [f"- {'PASS' if r['passed'] else 'FAIL'}: {r['scenario']}" for r in body["scenarios"]] + [""]
        else:
            lines += ["```json", json.dumps(body, indent=2), "```", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate NEXORA Module 05")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default=None, help="split for detection metrics (default evaluation.split)")
    ap.add_argument("--video", default=None, help="video for speed measurement")
    ap.add_argument("--gt-events", default=None)
    ap.add_argument("--results", nargs="*", default=[], help="results.json files from process_video")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    ecfg = cfg.get("evaluation", {})
    split = args.split or ecfg.get("split", "test")
    weights = resolve_path(args.weights or cfg["model"]["path"])
    data_yaml = resolve_path(args.data or cfg["training"]["data"])
    sections: Dict[str, Any] = {"association_scenarios": evaluate_association_scenarios(cfg)}

    have_model = weights.is_file()
    report = validate_dataset(data_yaml, cfg["classes"]["names"])
    have_split = report.ok and report.counts.get(split, {}).get("images", 0) > 0
    try:
        import ultralytics  # noqa: F401
        have_ultra = True
    except ImportError:
        have_ultra = False

    if have_model and have_split and have_ultra:
        from module_05_ppe.training.validate import run_validation
        res = run_validation(weights, data_yaml, split, int(cfg["model"]["input_size"]), 8,
                             select_device(cfg["model"].get("device", "auto"), inspect_hardware()))
        sections["detection_metrics"] = {"status": "measured", "split": split,
                                         "images": report.counts[split]["images"], **res}
        sections["difficult_conditions"] = evaluate_difficult(cfg, weights, data_yaml, split)
    else:
        why = ("no trained weights" if not have_model else "dataset split unavailable" if not have_split
               else "ultralytics not installed")
        sections["detection_metrics"] = not_measured(why)
        sections["difficult_conditions"] = not_measured(why)

    if have_model and have_ultra and args.video and Path(args.video).is_file():
        sections["speed"] = evaluate_speed(cfg, Path(args.video), args.weights, int(ecfg.get("speed_sample_frames", 200)))
    else:
        sections["speed"] = not_measured("needs trained weights, ultralytics and --video <file>")

    if args.gt_events and args.results:
        sections["event_level"] = evaluate_events(Path(args.gt_events), [Path(p) for p in args.results],
                                                  float(ecfg.get("event_match_tolerance_s", 3.0)))
    else:
        sections["event_level"] = not_measured("needs --gt-events (ground-truth events) and --results (pipeline outputs)")

    results = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "hardware": inspect_hardware(), "sections": sections}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (RESULTS_DIR / "evaluation_report.md").write_text(render_report(results), encoding="utf-8")
    logger.info("Wrote %s", RESULTS_DIR / "evaluation_report.md")
    measured = [k for k, v in sections.items() if v.get("status") != NOT_MEASURED]
    tracker = ProgressTracker()
    tracker.update(evaluation_status=f"Measured sections: {measured}. Others: Not measured.")
    if "detection_metrics" in measured:
        tracker.complete("evaluation completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
