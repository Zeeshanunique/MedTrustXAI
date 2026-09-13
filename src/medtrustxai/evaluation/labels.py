from __future__ import annotations

import re

from medtrustxai.data.test_dataset import TestSample

RADIOLOGY_CXR_LABELS = ("normal", "pneumonia")
RADIOLOGY_VQA_LABELS = ("yes", "no")
PATHOLOGY_LABELS = ("malignant", "metastatic", "deposition", "benign_neoplasm")

RADIOLOGY_CXR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pneumonia": (
        "pneumonia",
        "consolidation",
        "infiltrate",
        "opacity",
        "airspace",
        "lobar",
        "infection",
    ),
    "normal": (
        "normal",
        "unremarkable",
        "no acute",
        "no focal",
        "clear lungs",
        "within normal",
        "no consolidation",
        "no pneumonia",
    ),
}

PATHOLOGY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "malignant": (
        "malignant",
        "carcinoma",
        "cancer",
        "tumor",
        "seminoma",
        "neoplasm",
        "atypia",
        "mitotic",
    ),
    "metastatic": (
        "metastatic",
        "metastasis",
        "extranodal",
        "lymph node metast",
    ),
    "deposition": (
        "amyloid",
        "deposit",
        "deposition",
        "extracellular",
        "amyloidosis",
    ),
    "benign_neoplasm": (
        "benign",
        "adenoma",
        "meningioma",
        "non malignant",
    ),
}

VQA_YES = (
    "yes",
    "present",
    "evidence",
    "seen",
    "identified",
    "consistent with",
    "suggestive",
    "there is",
    "there are",
)
VQA_NO = (
    "no",
    "not present",
    "absent",
    "without",
    "none",
    "unremarkable",
    "normal",
    "no evidence",
)


def ground_truth_label(sample: TestSample) -> str:
    if sample.modality == "radiology" and sample.category == "vqa_rad":
        return _normalize_yes_no(sample.reference_hint)
    return sample.category.strip().lower()


def eval_task_for_sample(sample: TestSample) -> str:
    if sample.modality == "radiology" and sample.category == "vqa_rad":
        return "radiology_vqa"
    if sample.modality == "radiology":
        return "radiology_cxr"
    return "pathology"


def allowed_labels(task: str) -> tuple[str, ...]:
    if task == "radiology_vqa":
        return RADIOLOGY_VQA_LABELS
    if task == "radiology_cxr":
        return RADIOLOGY_CXR_LABELS
    return PATHOLOGY_LABELS


def parse_cxr_pneumonia_question(text: str) -> str | None:
    label = _parse_vqa_label(text)
    if label == "yes":
        return "pneumonia"
    if label == "no":
        return "normal"
    return None


def parse_cxr_normal_question(text: str) -> str | None:
    label = _parse_vqa_label(text)
    if label == "yes":
        return "normal"
    if label == "no":
        return "pneumonia"
    return None


def parse_cxr_binary_label(text: str) -> str:
    return parse_cxr_pneumonia_question(text) or "normal"


def parse_label_from_text(text: str, task: str) -> str:
    labels = allowed_labels(task)
    text_lower = text.lower().strip()

    if task == "radiology_cxr":
        return parse_cxr_binary_label(text_lower)

    match = re.search(r"label\s*:\s*([a-z_]+)", text_lower)
    if match:
        candidate = match.group(1).strip()
        if candidate in labels:
            return candidate

    if task == "radiology_vqa":
        return _parse_vqa_label(text_lower) or "no"

    keywords = RADIOLOGY_CXR_KEYWORDS if task == "radiology_cxr" else PATHOLOGY_KEYWORDS
    scores = {label: 0 for label in labels}
    for label, words in keywords.items():
        for word in words:
            if word in text_lower:
                scores[label] += len(word.split()) + 1

    if task == "pathology" and "metast" in text_lower:
        scores["metastatic"] = scores.get("metastatic", 0) + 5
    if task == "pathology" and "amyloid" in text_lower:
        scores["deposition"] = scores.get("deposition", 0) + 5

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    if task == "pathology" and "benign" in text_lower.split():
        return "benign_neoplasm"

    for label in labels:
        if label in text_lower:
            return label

    if task == "radiology_cxr":
        if any(w in text_lower for w in RADIOLOGY_CXR_KEYWORDS["pneumonia"]):
            return "pneumonia"
        return "normal"
    if task == "pathology":
        return "malignant"
    return "no"


def _parse_vqa_label(text: str) -> str | None:
    cleaned = text.strip().lower()
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    first = cleaned.split()[0] if cleaned.split() else ""
    if first in ("yes", "no"):
        return first
    if any(p in cleaned for p in VQA_NO) and not any(p in cleaned for p in VQA_YES):
        return "no"
    if any(p in cleaned for p in VQA_YES):
        return "yes"
    return None


def reconcile_cxr_labels(
    pneumonia_answer: str,
    normal_answer: str,
    opacity_answer: str | None = None,
) -> str:
    votes: list[str] = []
    p_raw = _parse_vqa_label(pneumonia_answer)
    n_raw = _parse_vqa_label(normal_answer)
    o_raw = _parse_vqa_label(opacity_answer or "")

    if p_raw == "yes":
        votes.append("pneumonia")
    elif p_raw == "no":
        votes.append("normal")
    if n_raw == "yes":
        votes.append("normal")
    elif n_raw == "no":
        votes.append("pneumonia")
    if o_raw == "yes":
        votes.append("pneumonia")
    elif o_raw == "no":
        votes.append("normal")

    if not votes:
        return "normal"
    pneumonia_votes = sum(1 for v in votes if v == "pneumonia")
    normal_votes = sum(1 for v in votes if v == "normal")
    if pneumonia_votes > normal_votes:
        return "pneumonia"
    if normal_votes > pneumonia_votes:
        return "normal"
    return votes[-1]


def _normalize_yes_no(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("y"):
        return "yes"
    if t.startswith("n"):
        return "no"
    return t
