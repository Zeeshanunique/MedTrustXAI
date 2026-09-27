from __future__ import annotations

from typing import Literal

Modality = Literal[
    "radiology",
    "pathology",
    "mri",
    "cbc",
    "blood_tests",
    "pulse_oximetry",
    "flu_testing",
]

MODALITIES: tuple[Modality, ...] = (
    "radiology",
    "pathology",
    "mri",
    "cbc",
    "blood_tests",
    "pulse_oximetry",
    "flu_testing",
)

DEFAULT_MODALITY: Modality = "radiology"


def normalize_modality(value: str | None) -> Modality:
    if not value:
        return DEFAULT_MODALITY
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cxr": "radiology",
        "chest": "radiology",
        "radiology": "radiology",
        "pathology": "pathology",
        "histopathology": "pathology",
        "h&e": "pathology",
        "he": "pathology",
        "mri": "mri",
        "magnetic_resonance": "mri",
        "cbc": "cbc",
        "complete_blood_count": "cbc",
        "blood": "blood_tests",
        "blood_test": "blood_tests",
        "blood_tests": "blood_tests",
        "blood_work": "blood_tests",
        "pulse": "pulse_oximetry",
        "oximetry": "pulse_oximetry",
        "pulse_oximetry": "pulse_oximetry",
        "spo2": "pulse_oximetry",
        "flu": "flu_testing",
        "flu_test": "flu_testing",
        "flu_testing": "flu_testing",
        "influenza": "flu_testing",
    }
    if key not in aliases:
        raise ValueError(f"Unknown modality '{value}'. Choose: {', '.join(MODALITIES)}")
    return aliases[key]

