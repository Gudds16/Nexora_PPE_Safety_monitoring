"""Build the merged 3-class YOLO dataset (person, helmet, safety_vest).

Inputs
  --source NAME=PATH   a YOLO-format dataset folder containing data.yaml (SH17, a converted
                       Pictor-PPE, ...). Classes are remapped BY NAME; unmapped classes are dropped.
  --custom-root PATH   custom CCTV: PATH/<session_id>/images/*.jpg and PATH/<session_id>/labels/*.txt
                       (labels already in the 3-class order of config.classes.names)
  --custom-split FILE  YAML: {train: [sessionA], val: [sessionB], test: [sessionC]}  (camera/session split)

Public images keep the splits they ship with. Custom sessions are split BY SESSION, never by frame.
Nothing is downloaded here; get the datasets yourself and check their licenses (datasets/README.md).

Example:
  python module_05_ppe/training/prepare_dataset.py --source sh17=datasets/raw/SH17 \
      --custom-root datasets/raw/custom --custom-split datasets/custom_split.yaml
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from module_05_ppe.config_loader import MODULE_DIR, load_config
from module_05_ppe.training.dataset_tools import (SPLITS, labels_dir_for, list_images, resolve_split_dirs,
                                                  validate_dataset)

logger = logging.getLogger("module_05_ppe.prepare_dataset")

# Source class name (lower-case) -> target class. 'hat'/'cap' are deliberately NOT mapped:
# a cap is a hard negative, not a helmet (spec section 12).
DEFAULT_ALIASES: Dict[str, str] = {
    "person": "person", "worker": "person", "people": "person",
    "helmet": "helmet", "hard-hat": "helmet", "hardhat": "helmet", "hard_hat": "helmet",
    "safety_vest": "safety_vest", "safety-vest": "safety_vest", "safety vest": "safety_vest",
    "vest": "safety_vest", "hi-vis": "safety_vest",
}


def build_id_map(source_names: Sequence[str], target_names: Sequence[str],
                 aliases: Dict[str, str]) -> Dict[int, int]:
    """source class id -> target class id, matched by (alias) name."""
    mapping: Dict[int, int] = {}
    for sid, name in enumerate(source_names):
        target = aliases.get(str(name).strip().lower())
        if target in target_names:
            mapping[sid] = list(target_names).index(target)
    return mapping


def remap_label_line(line: str, id_map: Dict[int, int]) -> Optional[str]:
    """Rewrite one YOLO label line with the new class id, or None when the class is dropped."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        new_id = id_map.get(int(float(parts[0])))
    except ValueError:
        return None
    return None if new_id is None else " ".join([str(new_id)] + parts[1:5])


def _copy_pair(image: Path, label_dir: Path, id_map: Optional[Dict[int, int]], out_img: Path,
               out_lab: Path, new_stem: str) -> None:
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, out_img / (new_stem + image.suffix.lower()))
    src = label_dir / (image.stem + ".txt")
    lines: List[str] = []
    if src.is_file():
        for raw in src.read_text(encoding="utf-8").splitlines():
            new = remap_label_line(raw, id_map) if id_map is not None else raw.strip() or None
            if new:
                lines.append(new)
    (out_lab / (new_stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def add_public_source(name: str, root: Path, out: Path, targets: Sequence[str], aliases: Dict[str, str]) -> Dict[str, int]:
    names, split_dirs = resolve_split_dirs(root / "data.yaml")
    id_map = build_id_map(names, targets, aliases)
    logger.info("Source %s: class map %s", name, {names[k]: targets[v] for k, v in id_map.items()})
    if not id_map:
        raise ValueError(f"No classes of '{name}' map to {list(targets)}. Pass --alias to extend the mapping.")
    counts: Dict[str, int] = {}
    for split, dirs in split_dirs.items():
        for d in dirs:
            for img in list_images(d):
                stem = f"{name}-{split}__{img.stem}"
                _copy_pair(img, labels_dir_for(d), id_map, out / split / "images", out / split / "labels", stem)
                counts[split] = counts.get(split, 0) + 1
    return counts


def add_custom_sessions(root: Path, split_file: Path, out: Path) -> Dict[str, int]:
    plan = yaml.safe_load(split_file.read_text(encoding="utf-8"))
    seen: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for split in SPLITS:
        for session in plan.get(split, []) or []:
            if session in seen:
                raise ValueError(f"Session '{session}' listed in both {seen[session]} and {split}")
            seen[session] = split
            img_dir = root / session / "images"
            for img in list_images(img_dir):
                _copy_pair(img, root / session / "labels", None, out / split / "images",
                           out / split / "labels", f"{session}__{img.stem}")
                counts[split] = counts.get(split, 0) + 1
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--alias", action="append", default=[], metavar="SRC_CLASS=TARGET",
                    help="extra class-name mapping, e.g. --alias 'hard hat=helmet'")
    ap.add_argument("--custom-root", default=None)
    ap.add_argument("--custom-split", default=None)
    ap.add_argument("--out", default="datasets/processed")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    targets = cfg["classes"]["names"]
    aliases = dict(DEFAULT_ALIASES)
    aliases.update({k.strip().lower(): v for k, v in (a.split("=", 1) for a in args.alias)})
    out = Path(args.out) if Path(args.out).is_absolute() else MODULE_DIR / args.out
    if out.exists() and any(out.iterdir()):
        logger.error("Output folder %s is not empty; move or delete it first (nothing was changed).", out)
        return 2
    if not args.source and not args.custom_root:
        logger.error("Nothing to do: give at least one --source or --custom-root.")
        return 2

    totals: Dict[str, int] = {}
    try:
        for spec in args.source:
            name, _, path = spec.partition("=")
            for split, n in add_public_source(name, Path(path).expanduser(), out, targets, aliases).items():
                totals[split] = totals.get(split, 0) + n
        if args.custom_root:
            if not args.custom_split:
                logger.error("--custom-root needs --custom-split (split by camera/session, not by frame).")
                return 2
            for split, n in add_custom_sessions(Path(args.custom_root), Path(args.custom_split), out).items():
                totals[split] = totals.get(split, 0) + n
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    logger.info("Images written per split: %s", totals)
    report = validate_dataset(MODULE_DIR / "datasets" / "data.yaml", targets)
    for w in report.warnings:
        logger.warning(w)
    for e in report.errors:
        logger.error(e)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
