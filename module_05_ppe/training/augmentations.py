"""Training augmentation presets (ultralytics hyper-parameters) and image degradations.

Design notes
* Hue shift is kept SMALL: vest/helmet colour carries meaning (hi-vis yellow/orange),
  a big hue shift could turn a yellow shirt-like pattern into a "vest" look.
* Brightness (hsv_v) is widened in the `cctv` preset to imitate low light / night shifts.
* ultralytics also applies Blur/MedianBlur/CLAHE automatically when `albumentations` is installed.
* Mosaic is switched off for the last `close_mosaic` epochs (standard YOLO practice).
"""
from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

PRESETS: Dict[str, Dict[str, Any]] = {
    "default": dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.5,
                    shear=2.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1,
                    close_mosaic=10),
    "cctv": dict(hsv_h=0.010, hsv_s=0.6, hsv_v=0.6, degrees=5.0, translate=0.1, scale=0.6,
                 shear=2.0, perspective=0.0005, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1,
                 close_mosaic=10),
    "none": dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
                 shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0,
                 close_mosaic=0),
}


def get_augmentation_params(preset: str = "cctv") -> Dict[str, Any]:
    """Return a copy of the ultralytics augmentation arguments for `preset`."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown augmentation preset '{preset}'. Choose from {sorted(PRESETS)}")
    return dict(PRESETS[preset])


def simulate_low_light(image: np.ndarray, gain: float = 0.3, noise_sigma: float = 6.0, seed: int = 0) -> np.ndarray:
    """Darken an image and add sensor noise (used for difficult-condition evaluation)."""
    rng = np.random.default_rng(seed)
    dark = image.astype(np.float32) * float(gain)
    dark += rng.normal(0.0, noise_sigma, image.shape)
    return np.clip(dark, 0, 255).astype(np.uint8)


def simulate_blur(image: np.ndarray, kernel: int = 9) -> np.ndarray:
    """Gaussian blur (kernel forced odd) imitating defocus / motion blur."""
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def simulate_low_resolution(image: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Downscale then upscale back, imitating low-quality CCTV."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


CONDITIONS = {"low_light": simulate_low_light, "blur": simulate_blur, "low_resolution": simulate_low_resolution}
