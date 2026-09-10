from __future__ import annotations

from typing import Literal

Modality = Literal["radiology", "pathology"]

MODALITIES: tuple[Modality, ...] = ("radiology", "pathology")

DEFAULT_MODALITY: Modality = "radiology"


def normalize_modality(value: str | None) -> Modality:
    if not value:
        return DEFAULT_MODALITY
    key = value.strip().lower()
    aliases = {
        "cxr": "radiology",
        "chest": "radiology",
        "radiology": "radiology",
        "pathology": "pathology",
        "histopathology": "pathology",
        "h&e": "pathology",
        "he": "pathology",
    }
    if key not in aliases:
        raise ValueError(f"Unknown modality '{value}'. Choose: {', '.join(MODALITIES)}")
    return aliases[key]
