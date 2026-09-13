from __future__ import annotations

import sys
import time
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from medtrustxai.config import setup_hf_auth
from medtrustxai.models.vlm_common import (
    GenerationResult,
    build_followup_messages,
    build_messages,
    generate_from_messages,
    prepare_inputs,
)

PROCESSOR_FOR_MODEL = {
    "BIOMEDICA/BMC-smolvlm1-256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "BIOMEDICA/BMC-smolvlm1-500M": "HuggingFaceTB/SmolVLM-500M-Instruct",
}


def _resolve_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


class SmolVLMLocalModel:
    def __init__(
        self,
        model_id: str = "BIOMEDICA/BMC-smolvlm1-256M",
        processor_id: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        setup_hf_auth()
        self.model_id = model_id
        proc = processor_id or PROCESSOR_FOR_MODEL.get(
            model_id, "HuggingFaceTB/SmolVLM-256M-Instruct"
        )
        self.system_prompt = system_prompt or "Expert medical imaging assistant. Research use only."
        self._device, self.dtype = _resolve_device()

        print(f"Loading SmolVLM: {model_id} on {self._device}", file=sys.stderr)
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(proc)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self._device)
        self.model.eval()
        print(f"Model ready in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    @property
    def device(self) -> torch.device:
        return self._device

    def prepare_inputs(self, image: Image.Image, prompt: str) -> tuple[dict[str, Any], int]:
        return prepare_inputs(
            self.processor, self.system_prompt, image, prompt, self.device, self.dtype
        )

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 128,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        print("Generating response...", file=sys.stderr)
        t0 = time.perf_counter()
        text = generate_from_messages(
            self.processor,
            self.model,
            build_messages(system_prompt or self.system_prompt, image, prompt),
            self.device,
            self.dtype,
            max_new_tokens,
        )
        print(f"Generation done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        return GenerationResult(text=text, prompt=prompt)

    @torch.inference_mode()
    def chat(
        self,
        image: Image.Image,
        initial_prompt: str,
        findings: str,
        history: list[tuple[str, str]],
        question: str,
        max_new_tokens: int = 128,
    ) -> GenerationResult:
        messages = build_followup_messages(
            self.system_prompt, image, initial_prompt, findings, history, question
        )
        text = generate_from_messages(
            self.processor, self.model, messages, self.device, self.dtype, max_new_tokens
        )
        return GenerationResult(text=text, prompt=question)

    def get_vision_module(self) -> torch.nn.Module:
        if hasattr(self.model, "model") and hasattr(self.model.model, "vision_model"):
            return self.model.model.vision_model
        if hasattr(self.model, "vision_model"):
            return self.model.vision_model
        raise AttributeError("Could not locate vision module in SmolVLM model.")
