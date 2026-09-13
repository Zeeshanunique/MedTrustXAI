from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from medtrustxai.data.test_dataset import TestDataset
from medtrustxai.evaluation.confusion_matrix import classify_from_record


@dataclass
class SampleMetrics:
    sample_id: str
    modality: str
    category: str
    reference: str
    prediction: str
    vqa_match: float
    token_f1: float
    faithfulness: float | None = None
    classification_correct: float | None = None


@dataclass
class EvaluationSummary:
    modality: str
    num_samples: int
    mean_vqa_match: float
    mean_token_f1: float
    mean_faithfulness: float | None
    mean_classification_accuracy: float | None
    samples: list[SampleMetrics]


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def vqa_match(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not ref:
        return 0.0
    if pred == ref:
        return 1.0
    yes_no = {"yes", "no", "normal", "abnormal"}
    if ref in yes_no:
        if ref in pred.split():
            return 1.0
        if ref == "yes" and any(w in pred for w in ("yes", "present", "evidence", "seen")):
            return 1.0
        if ref == "no" and any(w in pred for w in ("no", "normal", "without", "absent")):
            return 1.0
    return 1.0 if ref in pred else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize_answer(prediction).split())
    ref_tokens = set(normalize_answer(reference).split())
    if not ref_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0
    overlap = pred_tokens & ref_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_prediction(results_dir: Path, sample_id: str, modality: str) -> dict | None:
    path = results_dir / f"{sample_id}_{modality}_result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_results(
    manifest_path: str | Path,
    results_dir: str | Path,
    modality: str,
) -> EvaluationSummary:
    dataset = TestDataset.from_manifest(manifest_path, modality=modality)
    results_dir = Path(results_dir)
    rows: list[SampleMetrics] = []

    for sample in dataset.samples:
        record = load_prediction(results_dir, sample.id, modality)
        if record is None:
            continue
        prediction = record.get("findings", "")
        reference = sample.reference_hint
        faith = record.get("faithfulness") or {}
        cls_row = classify_from_record(sample, record)
        rows.append(
            SampleMetrics(
                sample_id=sample.id,
                modality=modality,
                category=sample.category,
                reference=reference,
                prediction=prediction,
                vqa_match=vqa_match(prediction, reference),
                token_f1=token_f1(prediction, reference),
                faithfulness=faith.get("faithfulness_score"),
                classification_correct=1.0 if cls_row.correct else 0.0,
            )
        )

    if not rows:
        return EvaluationSummary(
            modality=modality,
            num_samples=0,
            mean_vqa_match=0.0,
            mean_token_f1=0.0,
            mean_faithfulness=None,
            mean_classification_accuracy=None,
            samples=[],
        )

    faith_vals = [r.faithfulness for r in rows if r.faithfulness is not None]
    cls_vals = [r.classification_correct for r in rows if r.classification_correct is not None]
    return EvaluationSummary(
        modality=modality,
        num_samples=len(rows),
        mean_vqa_match=sum(r.vqa_match for r in rows) / len(rows),
        mean_token_f1=sum(r.token_f1 for r in rows) / len(rows),
        mean_faithfulness=(sum(faith_vals) / len(faith_vals)) if faith_vals else None,
        mean_classification_accuracy=(sum(cls_vals) / len(cls_vals)) if cls_vals else None,
        samples=rows,
    )


def save_evaluation(summary: EvaluationSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "modality": summary.modality,
        "num_samples": summary.num_samples,
        "mean_vqa_match": summary.mean_vqa_match,
        "mean_token_f1": summary.mean_token_f1,
        "mean_faithfulness": summary.mean_faithfulness,
        "mean_classification_accuracy": summary.mean_classification_accuracy,
        "samples": [asdict(s) for s in summary.samples],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
