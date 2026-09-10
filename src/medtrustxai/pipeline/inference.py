from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from medtrustxai.config import Config
from medtrustxai.evaluation.faithfulness import evaluate_grounding
from medtrustxai.modalities import Modality, normalize_modality
from medtrustxai.models.smolvlm import SmolVLMLocalModel
from medtrustxai.xai.gradcam import GradCAMExplainer
from medtrustxai.xai.visualization import save_attribution


@dataclass
class DiagnosticOutput:
    modality: str
    findings: str
    prompt: str
    gradcam_path: str | None = None
    faithfulness: dict | None = None


class DiagnosticPipeline:
    def __init__(self, config: Config, modality: Modality = "radiology") -> None:
        self.config = config
        self.modality = normalize_modality(modality)
        self.modality_cfg = config.modality_config(self.modality)
        config.ensure_dirs()

        self.model = SmolVLMLocalModel(
            model_id=config.model_id,
            processor_id=config.get("model", "processor_id"),
            system_prompt=self.modality_cfg.get("system_prompt"),
        )
        self.gradcam = GradCAMExplainer(
            self.model,
            prefer_fast=bool(config.get("xai", "fast", default=True)),
        )

    def default_prompt(self) -> str:
        return str(self.modality_cfg.get("report_prompt", ""))

    def run(
        self,
        image: Image.Image,
        prompt: str,
        sample_id: str = "sample",
        run_xai: bool = True,
        max_new_tokens: int | None = None,
    ) -> DiagnosticOutput:
        tokens = max_new_tokens or int(
            self.config.get("model", "max_new_tokens", default=128)
        )
        result = self.model.generate(image=image, prompt=prompt, max_new_tokens=tokens)
        output = DiagnosticOutput(
            modality=self.modality,
            findings=result.text,
            prompt=prompt,
        )
        if not run_xai:
            return output

        xai_dir = Path(self.config.get("xai", "output_dir", default="outputs/xai"))
        alpha = float(self.config.get("xai", "alpha", default=0.45))

        print(f"Running Grad-CAM ({self.modality})...", file=sys.stderr)
        t0 = time.perf_counter()
        gradcam = self.gradcam.explain(image, prompt, alpha=alpha)
        print(f"Grad-CAM done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        gradcam_path = xai_dir / f"{sample_id}_{self.modality}_gradcam.png"
        save_attribution(str(gradcam_path), gradcam.overlay)
        output.gradcam_path = str(gradcam_path)
        output.faithfulness = asdict(
            evaluate_grounding(result.text, gradcam.heatmap, modality=self.modality)
        )
        return output

    def save_result(self, output: DiagnosticOutput, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(output), f, indent=2)
