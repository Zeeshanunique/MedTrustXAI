from __future__ import annotations

from medtrustxai.data.test_dataset import TestSample
from medtrustxai.evaluation.labels import eval_task_for_sample


def build_sample_prompt(sample: TestSample, modality_cfg: dict) -> str:
    task = eval_task_for_sample(sample)
    if task == "radiology_vqa":
        question = sample.vqa_questions[0] if sample.vqa_questions else "Is there an abnormality?"
        template = modality_cfg.get(
            "vqa_prompt",
            "Answer with ONLY yes or no.\nQuestion: {question}",
        )
        return template.format(question=question)
    if task == "radiology_cxr":
        return str(
            modality_cfg.get(
                "cxr_binary_prompt",
                "Is pneumonia or focal airspace consolidation present on this chest X-ray? "
                "Answer with exactly one word: yes or no.",
            )
        )
    return str(
        modality_cfg.get(
            "pathology_prompt",
            modality_cfg.get(
                "classification_prompt",
                "What is the best diagnosis category for this H&E slide? "
                "Answer with exactly one word from: malignant, metastatic, deposition, benign.",
            ),
        )
    )


def radiology_cxr_confirm_normal_prompt() -> str:
    return (
        "Are the lungs normal without focal consolidation on this chest X-ray? "
        "Answer with exactly one word: yes or no."
    )


def radiology_cxr_opacity_prompt() -> str:
    return (
        "Is there increased opacity, infiltrate, or consolidation on this chest X-ray? "
        "Answer with exactly one word: yes or no."
    )


def classifier_system_prompt(modality_cfg: dict, fallback: str | None = None) -> str | None:
    return modality_cfg.get("classifier_system_prompt") or fallback
