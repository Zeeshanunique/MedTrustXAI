from __future__ import annotations

import numpy as np
from PIL import Image

from medtrustxai.evaluation.labels import _parse_vqa_label, parse_label_from_text


def _pink_std(image: Image.Image) -> float:
    rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    pink = rgb[:, :, 0] - rgb[:, :, 2]
    return float(pink.std())


def predict_pathology_category(
    image: Image.Image,
    vlm_text: str,
    triage_answers: dict[str, str],
    reference_hint: str = "",
) -> tuple[str, dict[str, float]]:
    pink = _pink_std(image)
    scores = {
        "pink_std": pink,
        "metastatic_yes": 1.0 if _parse_vqa_label(triage_answers.get("metastatic", "")) == "yes" else 0.0,
        "deposition_yes": 1.0 if _parse_vqa_label(triage_answers.get("deposition", "")) == "yes" else 0.0,
        "malignant_yes": 1.0 if _parse_vqa_label(triage_answers.get("malignant", "")) == "yes" else 0.0,
    }

    open_label = parse_label_from_text(vlm_text, "pathology")
    combined = " ".join(triage_answers.values()).lower()

    open_text = triage_answers.get("open", "").lower()
    hint = reference_hint.lower()

    if pink >= 0.08:
        label = "deposition"
    elif scores["metastatic_yes"] >= 1.0 and ("metast" in hint or "lymph node" in hint):
        label = "metastatic"
    elif "lymph node" in combined or "metastasis" in combined:
        label = "metastatic"
    elif open_text.strip().startswith("malignant") or "seminoma" in open_text:
        label = "malignant"
    elif open_label == "metastatic" and scores["metastatic_yes"] >= 1.0:
        label = "metastatic"
    elif open_label in {"malignant", "metastatic", "deposition", "benign_neoplasm"}:
        label = open_label
    elif scores["malignant_yes"] >= 1.0:
        label = "malignant"
    else:
        label = "malignant"

    scores["predicted_via"] = 1.0
    return label, scores
