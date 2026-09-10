from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class GenerationResult:
    text: str
    prompt: str


def build_messages(system_prompt: str, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        },
    ]


def build_followup_messages(
    system_prompt: str,
    image: Image.Image,
    initial_prompt: str,
    findings: str,
    history: list[tuple[str, str]],
    question: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": initial_prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": findings}]},
    ]
    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
    return messages


def move_inputs(
    inputs: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point() and dtype is not None:
            moved[key] = value.to(device, dtype=dtype)
        else:
            moved[key] = value.to(device)
    return moved


def _encode(processor, messages: list[dict[str, Any]], device: torch.device, dtype: torch.dtype | None):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_inputs(inputs, device, dtype)
    return inputs, inputs["input_ids"].shape[-1]


def prepare_inputs(
    processor,
    system_prompt: str,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, Any], int]:
    return _encode(processor, build_messages(system_prompt, image, prompt), device, dtype)


def generate_from_messages(
    processor,
    model,
    messages: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_new_tokens: int = 128,
) -> str:
    inputs, input_len = _encode(processor, messages, device, dtype)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()
