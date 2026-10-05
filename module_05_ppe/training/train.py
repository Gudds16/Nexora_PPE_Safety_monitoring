"""Train the Module 05 detector (person, helmet, safety_vest) with ultralytics YOLO.

    python module_05_ppe/training/train.py                 # defaults from config.yaml
    python module_05_ppe/training/train.py --resume        # continue an interrupted run
    python module_05_ppe/training/train.py --dry-run       # check config/dataset/hardware only

What it does automatically
* inspects CPU / RAM / GPU and picks device + starting batch size (override with --batch)
* validates the dataset (class order, label syntax, session leakage) BEFORE training
* on CUDA out-of-memory: halves the batch size and continues from last.pt
* on other failures: diagnoses, then resumes from last.pt (up to training.max_retries times)
* copies last.pt every time and best.pt only when it is better than the one already in
  models/checkpoints/ (a worse run never overwrites a better checkpoint)
* writes logs/training.log and keeps MODULE_PROGRESS.md up to date
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module_05_ppe.config_loader import MODULE_DIR, load_config, resolve_path
from module_05_ppe.hardware import inspect_hardware, select_device, suggest_batch_size
from module_05_ppe.progress import ProgressTracker
from module_05_ppe.training.augmentations import get_augmentation_params
from module_05_ppe.training.dataset_tools import validate_dataset

logger = logging.getLogger("module_05_ppe.train")
CKPT_DIR = MODULE_DIR / "models" / "checkpoints"


# ----------------------------------------------------------------------------- helpers
def setup_logging() -> None:
    log_dir = MODULE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "training.log", encoding="utf-8")])


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def diagnose_error(exc: BaseException) -> str:
    """Turn a training exception into a plain-language hint."""
    text = str(exc).lower()
    if is_oom(exc):
        return "GPU/accelerator ran out of memory -> batch size will be halved and training continued."
    if "no such file" in text or "does not exist" in text or "dataset" in text and "not found" in text:
        return "A dataset or weights path is wrong. Check training.data in config.yaml and datasets/README.md."
    if "urlopen" in text or "connection" in text or "timed out" in text or "resolve" in text:
        return "Network problem while downloading pretrained weights. Download yolo11s.pt manually and set training.model to its path."
    if "cuda" in text and ("not available" in text or "invalid device" in text):
        return "Requested CUDA device is not available. Use --device cpu or --device mps, or install the CUDA build of torch."
    if "mps" in text:
        return "Apple MPS backend error. Retry with --device cpu, or update torch."
    return "Unrecognised error. See logs/training.log for the full traceback."


def read_results_csv(path: Path) -> Dict[str, Any]:
    """Summarise ultralytics results.csv: last epoch and the epoch with the best mAP50-95."""
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        return {}

    def num(row: Dict[str, str], key: str) -> Optional[float]:
        try:
            return float(row[key])
        except (KeyError, ValueError):
            return None

    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: num(r, key) if num(r, key) is not None else -1.0)
    pick = lambda r: {"epoch": int(float(r.get("epoch", 0))),  # noqa: E731
                      "precision": num(r, "metrics/precision(B)"), "recall": num(r, "metrics/recall(B)"),
                      "mAP50": num(r, "metrics/mAP50(B)"), "mAP50-95": num(r, key)}
    return {"epochs_logged": len(rows), "last": pick(rows[-1]), "best": pick(best)}


def read_checkpoint_fitness(path: Path) -> Optional[float]:
    """best_fitness stored inside an ultralytics checkpoint (needs torch)."""
    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        value = ckpt.get("best_fitness")
        return float(value) if value is not None else None
    except Exception as exc:
        logger.warning("Could not read fitness from %s: %s", path, exc)
        return None


def read_checkpoint_epoch(path: Path) -> Optional[int]:
    try:
        import torch
        return int(torch.load(str(path), map_location="cpu", weights_only=False).get("epoch", -1))
    except Exception:
        return None


def promote_checkpoints(weights_dir: Path, ckpt_dir: Path = CKPT_DIR,
                        fitness_reader: Callable[[Path], Optional[float]] = read_checkpoint_fitness,
                        data_note: str = "") -> Dict[str, Any]:
    """Copy last.pt always; copy best.pt only if it beats the checkpoint already stored."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_path = ckpt_dir / "best_meta.json"
    outcome: Dict[str, Any] = {"last_copied": False, "best_copied": False}
    last = weights_dir / "last.pt"
    if last.is_file():
        shutil.copy2(last, ckpt_dir / "last.pt")
        outcome["last_copied"] = True
    best = weights_dir / "best.pt"
    if not best.is_file():
        return outcome
    new_fit = fitness_reader(best)
    old_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    old_fit = old_meta.get("fitness")
    existing = (ckpt_dir / "best.pt").is_file()
    better = (not existing) or old_fit is None or (new_fit is not None and new_fit > old_fit)
    if better:
        shutil.copy2(best, ckpt_dir / "best.pt")
        meta_path.write_text(json.dumps({
            "fitness": new_fit, "source": str(weights_dir),
            "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset": data_note}, indent=2))
        outcome["best_copied"] = True
    else:
        logger.info("Kept existing best.pt (fitness %s >= new %s)", old_fit, new_fit)
    outcome.update(new_fitness=new_fit, previous_fitness=old_fit)
    return outcome


def unique_run_dir(project: Path, name: str) -> str:
    """Never reuse a run folder that already holds weights (protects old checkpoints)."""
    candidate, n = name, 2
    while (project / candidate / "weights" / "last.pt").is_file():
        candidate, n = f"{name}_{n}", n + 1
    return candidate


# ----------------------------------------------------------------------------- main logic
def run_training(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    tracker = ProgressTracker()
    hw = inspect_hardware()
    device = select_device(args.device or cfg["model"].get("device", "auto"), hw)
    imgsz = args.imgsz or int(tcfg["imgsz"])
    model_name = args.model or tcfg["model"]
    epochs = args.epochs or int(tcfg["epochs"])
    batch_arg = args.batch or str(tcfg.get("batch", "auto"))
    batch = suggest_batch_size(hw, imgsz, model_name) if batch_arg == "auto" else int(batch_arg)
    logger.info("Hardware: %s", hw)
    logger.info("Device=%s batch=%d imgsz=%d epochs=%d model=%s", device, batch, imgsz, epochs, model_name)
    tracker.complete("environment verified")
    if hw["accelerator"] == "cpu":
        logger.warning("No GPU detected: CPU training will be extremely slow. Use a GPU machine for real training.")

    data_yaml = resolve_path(args.data or tcfg["data"])
    report = validate_dataset(data_yaml, cfg["classes"]["names"])
    for w in report.warnings:
        logger.warning("dataset: %s", w)
    if not report.ok:
        for e in report.errors[:20]:
            logger.error("dataset: %s", e)
        tracker.update(dataset="Unavailable or invalid - see logs/training.log",
                       next_action="Prepare the dataset (datasets/README.md, training/prepare_dataset.py), then run train.py")
        logger.error("Dataset validation failed; training NOT started.")
        return 2
    tracker.complete("dataset verified")
    tracker.update(dataset=f"Validated: {json.dumps(report.counts)}")
    if args.dry_run:
        logger.info("Dry run OK: configuration, dataset and hardware look usable.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is not installed. Run: pip install -r requirements.txt")
        return 3

    project = resolve_path(tcfg["project_dir"])
    run_name = unique_run_dir(project, tcfg["run_name"]) if not args.resume else tcfg["run_name"]
    aug = get_augmentation_params(tcfg.get("augmentation_preset", "cctv"))
    common = dict(data=str(data_yaml), imgsz=imgsz, device=device, workers=int(tcfg["workers"]),
                  seed=int(tcfg["seed"]), deterministic=True, patience=int(tcfg["patience"]),
                  optimizer=tcfg["optimizer"], project=str(project), exist_ok=True,
                  amp=(hw["accelerator"] == "cuda"), cache=bool(tcfg.get("cache", False)), **aug)
    tracker.complete("training started")
    tracker.update(model=f"{model_name} (fine-tuning in progress)")

    resume_pt = project / run_name / "weights" / "last.pt"
    retries, weights_dirs = 0, []
    mode = "resume" if (args.resume and resume_pt.is_file()) else "fresh"
    if args.resume and mode == "fresh":
        logger.warning("--resume given but %s not found; starting a fresh run.", resume_pt)
    current_name, current_batch, current_epochs, start_weights = run_name, batch, epochs, model_name

    while True:
        try:
            if mode == "resume":
                logger.info("Resuming from %s", resume_pt)
                YOLO(str(resume_pt)).train(resume=True)
            else:
                logger.info("Training run '%s' (batch=%d, epochs=%d, weights=%s)", current_name, current_batch,
                            current_epochs, start_weights)
                YOLO(str(start_weights)).train(name=current_name, batch=current_batch, epochs=current_epochs, **common)
            break
        except KeyboardInterrupt:
            logger.warning("Interrupted. Re-run with --resume to continue from last.pt.")
            tracker.update(next_action="Run: python module_05_ppe/training/train.py --resume")
            break
        except Exception as exc:  # noqa: BLE001 - we diagnose and decide below
            logger.exception("Training failed: %s", exc)
            hint = diagnose_error(exc)
            logger.error("Diagnosis: %s", hint)
            tracker.add_issue(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}: {hint}")
            last = project / current_name / "weights" / "last.pt"
            if is_oom(exc) and current_batch > 2:
                current_batch = max(2, current_batch // 2)
                if last.is_file():
                    done = read_checkpoint_epoch(last)
                    current_epochs = max(1, current_epochs - ((done or -1) + 1))
                    start_weights = str(last)
                    weights_dirs.append(last.parent)
                    current_name = f"{current_name}_b{current_batch}"
                mode = "fresh"
                logger.info("Retrying with batch=%d", current_batch)
                continue
            if retries < int(tcfg.get("max_retries", 2)) and last.is_file():
                retries += 1
                mode, resume_pt = "resume", last
                logger.info("Retry %d/%s: resuming from last.pt", retries, tcfg.get("max_retries", 2))
                continue
            tracker.update(next_action=f"Fix the error above ({hint}) and run train.py --resume")
            return 1

    weights_dirs.append(project / current_name / "weights")
    summary: Dict[str, Any] = {}
    for wdir in weights_dirs:
        promote_checkpoints(wdir, CKPT_DIR, data_note=str(data_yaml))
        summary = read_results_csv(wdir.parent / "results.csv") or summary
    have_best, have_last = (CKPT_DIR / "best.pt").is_file(), (CKPT_DIR / "last.pt").is_file()
    tracker.update(latest_checkpoint=str(CKPT_DIR / "last.pt") if have_last else "None",
                   best_checkpoint=str(CKPT_DIR / "best.pt") if have_best else "None",
                   last_successful_epoch=json.dumps(summary.get("last")) if summary else "None",
                   model=f"{model_name} fine-tuned (metrics are validation-split numbers logged during training): "
                         f"{json.dumps(summary.get('best'))}" if summary else "Not trained")
    if have_best:
        tracker.complete("best model", next_action="Run training/validate.py, then evaluation/evaluate.py")
    return 0 if have_best else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the NEXORA Module 05 PPE detector")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--data", default=None, help="override training.data (YOLO data.yaml)")
    p.add_argument("--model", default=None, help="override training.model (pretrained weights)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", default=None, help="integer or 'auto'")
    p.add_argument("--device", default=None, help="auto | cpu | mps | 0")
    p.add_argument("--resume", action="store_true", help="resume the last interrupted run")
    p.add_argument("--dry-run", action="store_true", help="validate everything but do not train")
    return p


if __name__ == "__main__":
    setup_logging()
    sys.exit(run_training(build_parser().parse_args()))
