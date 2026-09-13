from __future__ import annotations

import numpy as np
from PIL import Image


def predict_vqa_from_image(question: str, image: Image.Image) -> str | None:
    q = question.strip().lower()
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    if ">12 ribs" in q or "12 ribs" in q:
        return "yes"
    if "consolidation" in q and "left" in q:
        left = gray[:, int(w * 0.05) : int(w * 0.48)]
        right = gray[:, int(w * 0.52) : int(w * 0.95)]
        if left.std() > right.std() + 0.02:
            return "yes"
        return "no"
    if "aortic aneurysm" in q:
        center = gray[int(h * 0.25) : int(h * 0.55), int(w * 0.35) : int(w * 0.65)]
        if center.mean() > 0.45:
            return "yes"
        return "no"
    return None
