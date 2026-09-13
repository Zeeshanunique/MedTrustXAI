from __future__ import annotations

from medtrustxai.evaluation.labels import _parse_vqa_label


def reconcile_pathology_labels(answers: dict[str, str]) -> str:
    if _parse_vqa_label(answers.get("metastatic", "")) == "yes":
        return "metastatic"
    if _parse_vqa_label(answers.get("deposition", "")) == "yes":
        return "deposition"
    if _parse_vqa_label(answers.get("malignant", "")) == "yes":
        return "malignant"
    if _parse_vqa_label(answers.get("benign", "")) == "yes":
        return "benign_neoplasm"
    return "malignant"


PATHOLOGY_QUESTIONS = {
    "metastatic": "Is metastatic carcinoma present in this H&E image? Answer only yes or no.",
    "deposition": "Is amyloid or extracellular deposition present? Answer only yes or no.",
    "malignant": "Is malignant neoplasm present? Answer only yes or no.",
    "benign": "Is this a benign neoplasm without malignancy? Answer only yes or no.",
}
