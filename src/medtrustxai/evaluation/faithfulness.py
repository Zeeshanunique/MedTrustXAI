from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medtrustxai.modalities import Modality

RADIOLOGY_REGIONS = {
    "right upper lung": (0.55, 0.05, 0.95, 0.35),
    "left upper lung": (0.05, 0.05, 0.45, 0.35),
    "right lower lung": (0.55, 0.45, 0.95, 0.90),
    "left lower lung": (0.05, 0.45, 0.45, 0.90),
    "right lung": (0.50, 0.05, 0.95, 0.90),
    "left lung": (0.05, 0.05, 0.50, 0.90),
    "heart": (0.35, 0.35, 0.65, 0.70),
    "mediastinum": (0.35, 0.20, 0.65, 0.55),
    "pleura": (0.05, 0.05, 0.95, 0.90),
}

PATHOLOGY_REGIONS = {
    "epithelium": (0.05, 0.05, 0.95, 0.40),
    "glandular epithelium": (0.10, 0.10, 0.90, 0.45),
    "stroma": (0.05, 0.35, 0.95, 0.95),
    "connective tissue": (0.05, 0.35, 0.95, 0.95),
    "nuclei": (0.10, 0.10, 0.90, 0.90),
    "nuclear atypia": (0.15, 0.15, 0.85, 0.85),
    "necrosis": (0.30, 0.30, 0.70, 0.70),
    "inflammatory infiltrate": (0.05, 0.50, 0.95, 0.95),
    "tumor": (0.20, 0.20, 0.80, 0.80),
    "malignant": (0.20, 0.20, 0.80, 0.80),
    "lymphocytes": (0.05, 0.55, 0.95, 0.95),
    "lumen": (0.25, 0.25, 0.75, 0.75),
}

REGION_MAP: dict[Modality, dict[str, tuple[float, float, float, float]]] = {
    "radiology": RADIOLOGY_REGIONS,
    "pathology": PATHOLOGY_REGIONS,
}


@dataclass
class GroundingResult:
    modality: str
    mentioned_regions: list[str]
    predicted_region: str | None
    faithfulness_score: float


def _extract_terms(text: str, regions: dict[str, tuple[float, float, float, float]]) -> list[str]:
    text = text.lower()
    found: list[str] = []
    for region in sorted(regions, key=len, reverse=True):
        if region in text and region not in found:
            found.append(region)
    return found


def _region_from_heatmap(
    heatmap: np.ndarray,
    regions: dict[str, tuple[float, float, float, float]],
) -> str | None:
    hm = heatmap.astype(np.float64)
    hm = hm - hm.min()
    total = hm.sum()
    if total <= 0:
        return None
    ys, xs = np.indices(hm.shape)
    cy = (ys * hm).sum() / total / hm.shape[0]
    cx = (xs * hm).sum() / total / hm.shape[1]
    for region, (x1, y1, x2, y2) in regions.items():
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return region
    return None


def evaluate_grounding(
    explanation: str,
    heatmap: np.ndarray,
    modality: Modality = "radiology",
) -> GroundingResult:
    regions = REGION_MAP[modality]
    mentioned = _extract_terms(explanation, regions)
    predicted = _region_from_heatmap(heatmap, regions)

    if modality == "pathology" and not mentioned:
        mentioned = _extract_pathology_keywords(explanation)

    score = 1.0 if predicted and predicted in mentioned else 0.0
    if score == 0.0 and mentioned and predicted:
        score = 0.5 if _regions_overlap(mentioned, predicted, regions) else 0.0

    return GroundingResult(
        modality=modality,
        mentioned_regions=mentioned,
        predicted_region=predicted,
        faithfulness_score=score,
    )


def _extract_pathology_keywords(text: str) -> list[str]:
    keywords = [
        "epithelium",
        "stroma",
        "nuclei",
        "necrosis",
        "inflammatory",
        "tumor",
        "malignant",
        "gland",
        "lymphocyte",
    ]
    text = text.lower()
    return [kw for kw in keywords if kw in text]


def _regions_overlap(
    mentioned: list[str],
    predicted: str,
    regions: dict[str, tuple[float, float, float, float]],
) -> bool:
    if predicted not in regions:
        return False
    px1, py1, px2, py2 = regions[predicted]
    for term in mentioned:
        if term not in regions:
            continue
        x1, y1, x2, y2 = regions[term]
        if not (px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2):
            return True
    return False
