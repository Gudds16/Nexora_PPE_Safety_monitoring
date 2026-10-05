"""Hardware inspection and automatic batch-size / device selection."""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _ram_gb() -> float:
    try:
        import psutil  # optional
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
        except (ValueError, OSError, AttributeError):
            return 0.0


def inspect_hardware() -> Dict[str, Any]:
    """Report CPU / RAM / accelerator. Works even when torch is not installed."""
    info: Dict[str, Any] = {
        "accelerator": "cpu", "device_name": "cpu", "vram_gb": 0.0,
        "cpu_count": os.cpu_count() or 1, "ram_gb": _ram_gb(), "torch": None,
    }
    try:
        import torch
    except Exception:
        info["torch"] = "not installed"
        return info
    info["torch"] = torch.__version__
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(accelerator="cuda", device_name=props.name,
                        vram_gb=round(props.total_memory / 1024 ** 3, 1))
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info.update(accelerator="mps", device_name="Apple Silicon (MPS)")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Accelerator probe failed: %s", exc)
    return info


def select_device(requested: str, hw: Dict[str, Any]) -> str:
    """Translate config device ('auto', 'cpu', 'mps', '0') into an ultralytics device string."""
    if requested and requested != "auto":
        return str(requested)
    return {"cuda": "0", "mps": "mps"}.get(hw["accelerator"], "cpu")


def _model_scale(model_name: str) -> float:
    """Relative memory cost vs. the 's' model (n=small ... x=huge)."""
    match = re.search(r"yolo\d*([nsmlx])", model_name.lower())
    return {"n": 1.5, "s": 1.0, "m": 0.6, "l": 0.45, "x": 0.3}.get(match.group(1) if match else "s", 1.0)


def suggest_batch_size(hw: Dict[str, Any], imgsz: int = 640, model_name: str = "yolo11s.pt") -> int:
    """Conservative starting batch size from VRAM (CUDA) or RAM (MPS). Training halves it on OOM."""
    if hw["accelerator"] == "cuda":
        vram = hw["vram_gb"]
        table = [(3, 4), (5, 8), (7, 12), (10, 16), (14, 24), (20, 32), (32, 48)]
        base = next((b for limit, b in table if vram < limit), 64)
        scaled = base * _model_scale(model_name) * (640.0 / max(imgsz, 32)) ** 2
        return max(2, int(scaled))
    if hw["accelerator"] == "mps":
        ram = hw["ram_gb"]
        return 4 if ram < 12 else (8 if ram < 24 else 16)
    return 4
