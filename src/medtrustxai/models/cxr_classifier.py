from __future__ import annotations

import numpy as np
from PIL import Image


def _heuristic_pneumonia_score(image: Image.Image) -> float:
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    left = gray[:, int(w * 0.05) : int(w * 0.45)]
    right = gray[:, int(w * 0.55) : int(w * 0.95)]
    return float(abs(left.std() - right.std()) + 0.5 * (left.std() + right.std()))


def _torchxrayvision_score(image: Image.Image) -> float | None:
    try:
        import torch
        import torchxrayvision as xrv

        model = _torchxrayvision_score._model  # type: ignore[attr-defined]
    except Exception:
        return None

    try:
        if model is None:
            import torchxrayvision as xrv

            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            model.eval()
            _torchxrayvision_score._model = model  # type: ignore[attr-defined]

        gray = np.array(image.convert("L"), dtype=np.float32)
        gray = xrv.datasets.normalize(gray, maxval=255.0)
        tensor = torch.tensor(gray).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            preds = model(tensor)[0].cpu().numpy()
        names = list(model.pathologies)
        scores = {name: float(preds[i]) for i, name in enumerate(names)}
        return max(
            scores.get("Pneumonia", 0.0),
            scores.get("Consolidation", 0.0),
            scores.get("Infiltration", 0.0),
            scores.get("Lung Opacity", 0.0),
        )
    except Exception:
        return None


_torchxrayvision_score._model = None  # type: ignore[attr-defined]


def predict_cxr_category(image: Image.Image, threshold: float = 0.205) -> tuple[str, dict[str, float]]:
    heuristic = _heuristic_pneumonia_score(image)
    xrv_score = _torchxrayvision_score(image)

    if xrv_score is not None:
        score = 0.6 * xrv_score + 0.4 * heuristic
        source = "hybrid"
    else:
        score = heuristic
        source = "heuristic"

    label = "pneumonia" if score >= threshold else "normal"
    return label, {
        "pneumonia_score": score,
        "heuristic_score": heuristic,
        "xrv_score": xrv_score if xrv_score is not None else -1.0,
        "classifier_source": source,
    }
