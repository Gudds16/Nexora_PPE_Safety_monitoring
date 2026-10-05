"""Validate a trained model on the val or test split and save measured metrics.

    python module_05_ppe/training/validate.py                 # val split, models/checkpoints/best.pt
    python module_05_ppe/training/validate.py --split test

Writes evaluation/results/validation_<split>.json with precision, recall, F1, mAP@50 and
mAP@50:95 (overall and per class). Nothing is estimated: if the model or dataset is missing
the script exits with a clear message and writes no metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module_05_ppe.config_loader import MODULE_DIR, load_config, resolve_path
from module_05_ppe.hardware import inspect_hardware, select_device
from module_05_ppe.progress import ProgressTracker
from module_05_ppe.training.dataset_tools import validate_dataset

logger = logging.getLogger("module_05_ppe.validate")
RESULTS_DIR = MODULE_DIR / "evaluation" / "results"


def f1_score(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def run_validation(weights: Path, data_yaml: Path, split: str, imgsz: int, batch: int, device: str,
                   conf: float = 0.001) -> Dict[str, Any]:
    """Run ultralytics validation and return a plain-dict summary."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, device=device,
                        conf=conf, verbose=False, plots=False)
    box = metrics.box
    names = dict(model.names)
    per_class: List[Dict[str, Any]] = []
    for pos, class_id in enumerate(box.ap_class_index):
        p, r = float(box.p[pos]), float(box.r[pos])
        per_class.append({"class": names[int(class_id)], "precision": p, "recall": r, "f1": f1_score(p, r),
                          "mAP50": float(box.ap50[pos]), "mAP50-95": float(box.ap[pos])})
    p, r = float(box.mp), float(box.mr)
    return {"precision": p, "recall": r, "f1": f1_score(p, r), "mAP50": float(box.map50),
            "mAP50-95": float(box.map), "per_class": per_class}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the Module 05 detector")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf is standard for mAP computation")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    weights = resolve_path(args.weights or cfg["model"]["path"])
    data_yaml = resolve_path(args.data or cfg["training"]["data"])
    if not weights.is_file():
        logger.error("Weights not found: %s - train first (python module_05_ppe/training/train.py).", weights)
        return 2
    report = validate_dataset(data_yaml, cfg["classes"]["names"])
    if not report.ok or report.counts.get(args.split, {}).get("images", 0) == 0:
        logger.error("Dataset not usable for split '%s': %s", args.split, report.errors or "no images")
        return 2
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        logger.error("ultralytics is not installed. Run: pip install -r requirements.txt")
        return 3

    hw = inspect_hardware()
    summary = run_validation(weights, data_yaml, args.split, int(cfg["model"]["input_size"]), args.batch,
                             select_device(cfg["model"].get("device", "auto"), hw), args.conf)
    summary.update(split=args.split, images=report.counts[args.split]["images"], weights=str(weights),
                   data=str(data_yaml), conf=args.conf, hardware=hw,
                   measured_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"validation_{args.split}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("P=%.3f R=%.3f F1=%.3f mAP50=%.3f mAP50-95=%.3f -> %s", summary["precision"],
                summary["recall"], summary["f1"], summary["mAP50"], summary["mAP50-95"], out)
    tracker = ProgressTracker()
    tracker.update(validation_status=f"Measured on '{args.split}' split ({summary['images']} images): "
                   f"mAP50={summary['mAP50']:.3f}, mAP50-95={summary['mAP50-95']:.3f}")
    tracker.complete("validation completed", next_action="Run evaluation/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
