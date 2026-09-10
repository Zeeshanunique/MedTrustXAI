from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class AttributionResult:
    method: str
    heatmap: np.ndarray
    overlay: np.ndarray


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def _to_2d_heatmap(heatmap: np.ndarray) -> np.ndarray:
    hm = np.asarray(heatmap, dtype=np.float32)
    while hm.ndim > 2:
        hm = hm.mean(axis=0)
    if hm.ndim == 1:
        side = int(np.ceil(np.sqrt(hm.size)))
        padded = np.zeros(side * side, dtype=np.float32)
        padded[: hm.size] = hm.flatten()
        hm = padded.reshape(side, side)
    return hm


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    hm = _to_2d_heatmap(heatmap)
    return cv2.resize(hm, size, interpolation=cv2.INTER_CUBIC)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    hm = normalize_heatmap(heatmap)
    hm = resize_heatmap(hm, (rgb.shape[1], rgb.shape[0]))
    colored = cv2.applyColorMap((hm * 255).astype(np.uint8), colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (alpha * colored + (1 - alpha) * rgb).astype(np.uint8)
    return overlay


def save_attribution(path: str, overlay: np.ndarray) -> None:
    Image.fromarray(overlay).save(path)
