"""Dataset validation helpers (YOLO format). Pure Python - no ultralytics needed."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import yaml

logger = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass
class DatasetReport:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    counts: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


def normalize_names(names: Union[Sequence[str], Dict[int, str]]) -> List[str]:
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def labels_dir_for(images_dir: Path) -> Path:
    """YOLO convention: .../images/<x> -> .../labels/<x>."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


def resolve_split_dirs(data_yaml: Path) -> Tuple[List[str], Dict[str, List[Path]]]:
    """Read a YOLO data.yaml and return (class names, {split: [image dirs]})."""
    data_yaml = Path(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    base = Path(cfg.get("path", "."))
    base = base if base.is_absolute() else (data_yaml.parent / base)
    out: Dict[str, List[Path]] = {}
    for split in SPLITS:
        entry = cfg.get(split)
        if not entry:
            continue
        entries = entry if isinstance(entry, list) else [entry]
        out[split] = [(Path(e) if Path(e).is_absolute() else base / e).resolve() for e in entries]
    return normalize_names(cfg.get("names", [])), out


def list_images(directory: Path) -> List[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if directory.is_dir() else []


def check_label_file(path: Path, num_classes: int) -> List[str]:
    """Return problems found in one YOLO label file."""
    problems = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:5]]
        except (ValueError, IndexError):
            problems.append(f"{path.name}:{n} malformed line")
            continue
        if len(parts) != 5:
            problems.append(f"{path.name}:{n} expected 5 values, got {len(parts)}")
        if not 0 <= cls < num_classes:
            problems.append(f"{path.name}:{n} class id {cls} out of range")
        if any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            problems.append(f"{path.name}:{n} coordinates not normalized/positive")
    return problems


def session_of(file_name: str) -> str:
    """Session id = text before '__' in the file name (e.g. camA_0930__frame_001.jpg -> camA_0930)."""
    return file_name.split("__", 1)[0] if "__" in file_name else ""


def validate_dataset(data_yaml: Path, expected_names: Sequence[str], max_label_problems: int = 20) -> DatasetReport:
    """Check data.yaml, folders, label syntax, class order and session leakage."""
    report = DatasetReport()
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        report.error(f"Dataset file not found: {data_yaml}")
        return report
    names, split_dirs = resolve_split_dirs(data_yaml)
    if names != list(expected_names):
        report.error(f"Class names/order in data.yaml {names} != expected {list(expected_names)}")
    for required in ("train", "val"):
        if required not in split_dirs:
            report.error(f"data.yaml has no '{required}' entry")
    sessions: Dict[str, set] = {}
    for split, dirs in split_dirs.items():
        images, empty, instances = 0, 0, [0] * len(expected_names)
        for d in dirs:
            if not d.is_dir():
                if split in ("train", "val"):
                    report.error(f"{split}: image folder missing: {d}")
                else:
                    report.warnings.append(f"{split}: image folder missing: {d}")
                continue
            lab_dir = labels_dir_for(d)
            for img in list_images(d):
                images += 1
                sess = session_of(img.name)
                if sess:
                    sessions.setdefault(sess, set()).add(split)
                lab = lab_dir / (img.stem + ".txt")
                if not lab.is_file():
                    empty += 1                      # missing label file = background image
                    continue
                for problem in check_label_file(lab, len(expected_names))[:max_label_problems]:
                    report.error(f"{split}: {problem}")
                for line in lab.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            instances[int(float(line.split()[0]))] += 1
                        except (ValueError, IndexError):
                            pass
        report.counts[split] = {"images": images, "background_images": empty,
                                "instances": dict(zip(expected_names, instances))}
        if split in ("train", "val") and images == 0 and not any("missing" in e and split in e for e in report.errors):
            report.error(f"{split}: no images found")
    leaked = sorted(s for s, where in sessions.items() if len(where) > 1)
    if leaked:
        report.error(f"Session leakage: sessions appear in several splits: {leaked[:10]}")
    if not sessions:
        report.warnings.append("No session ids ('<session>__file') in file names: cannot verify "
                               "camera/session separation between splits.")
    if "test" not in split_dirs or report.counts.get("test", {}).get("images", 0) == 0:
        report.warnings.append("No test images: a held-out test metric cannot be produced.")
    return report
