from __future__ import annotations

import sys
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from medtrustxai.xai.visualization import AttributionResult, overlay_heatmap


class LocalVLM(Protocol):
    model: torch.nn.Module

    def prepare_inputs(self, image: Image.Image, prompt: str) -> tuple[dict[str, Any], int]: ...

    def get_vision_module(self) -> torch.nn.Module: ...


def _find_target_layer(vision: torch.nn.Module) -> torch.nn.Module:
    if hasattr(vision, "encoder") and hasattr(vision.encoder, "layers"):
        layers = vision.encoder.layers
        if len(layers) > 0:
            return layers[-1]
    if hasattr(vision, "vision_model"):
        inner = vision.vision_model
        if hasattr(inner, "encoder") and hasattr(inner.encoder, "layers") and len(inner.encoder.layers) > 0:
            return inner.encoder.layers[-1]
    for module in reversed(list(vision.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    return vision


def _compute_cam(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    if acts.shape != grads.shape:
        min_c = min(acts.shape[-1], grads.shape[-1])
        acts = acts[..., :min_c]
        grads = grads[..., :min_c]

    if acts.ndim == 4:
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
    elif acts.ndim == 3:
        weights = grads.mean(dim=1, keepdim=True)
        cam = (weights * acts).sum(dim=-1)
    else:
        cam = (acts * grads).mean(dim=-1)

    return F.relu(cam)


def _tokens_to_grid(cam_1d: np.ndarray) -> np.ndarray:
    if cam_1d.ndim != 1:
        return cam_1d
    n = cam_1d.size
    side = int(round(n**0.5))
    if side * side == n:
        return cam_1d.reshape(side, side)
    side = int(np.sqrt(n))
    if side * side < n:
        side += 1
    padded = np.zeros(side * side, dtype=cam_1d.dtype)
    padded[:n] = cam_1d
    return padded.reshape(side, side)


def _input_gradient_saliency(
    model_wrapper: LocalVLM,
    image: Image.Image,
    prompt: str,
    input_len: int,
    inputs: dict[str, Any],
    alpha: float,
) -> AttributionResult:
    pixel_key = next(
        (k for k in ("pixel_values", "images") if k in inputs and torch.is_tensor(inputs[k])),
        None,
    )
    if pixel_key is None:
        raise RuntimeError("No pixel tensor found for saliency fallback.")

    pixels = inputs[pixel_key].detach().clone().requires_grad_(True)
    step_inputs = dict(inputs)
    step_inputs[pixel_key] = pixels

    model_wrapper.model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outputs = model_wrapper.model(**step_inputs)
        score = outputs.logits[:, input_len - 1 : input_len, :].max()
        score.backward()

    grad = pixels.grad
    if grad is None:
        raise RuntimeError("Pixel gradients were not computed.")
    attr = grad.detach().abs()
    while attr.ndim > 2:
        attr = attr.mean(dim=0)
    attr = attr.float().cpu().numpy()
    overlay = overlay_heatmap(image, attr, alpha=alpha)
    return AttributionResult(method="input_gradient", heatmap=attr, overlay=overlay)


class GradCAMExplainer:
    def __init__(self, model: LocalVLM, prefer_fast: bool = False) -> None:
        self.model_wrapper = model
        self.prefer_fast = prefer_fast
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _register_hooks(self, layer: torch.nn.Module) -> None:
        def forward_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            self.gradients = grad.detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def clear_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def explain(
        self,
        image: Image.Image,
        prompt: str,
        target_token_idx: int = -1,
        alpha: float = 0.45,
    ) -> AttributionResult:
        device = getattr(self.model_wrapper, "device", None)
        use_fast = self.prefer_fast or (
            device is not None and device.type in {"mps", "cpu"}
        )
        if use_fast:
            print("Using fast input-gradient saliency (recommended on Mac).", file=sys.stderr)
            inputs, input_len = self.model_wrapper.prepare_inputs(image, prompt)
            return _input_gradient_saliency(
                self.model_wrapper, image, prompt, input_len, inputs, alpha
            )

        vision = self.model_wrapper.get_vision_module()
        layer = _find_target_layer(vision)
        self._register_hooks(layer)

        try:
            inputs, input_len = self.model_wrapper.prepare_inputs(image, prompt)
            self.model_wrapper.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                outputs = self.model_wrapper.model(**inputs)
                logits = outputs.logits[:, input_len - 1 : input_len, :]
                score = logits[0, 0, target_token_idx]
                score.backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            cam = _compute_cam(self.activations.float(), self.gradients.float())
            cam_np = cam[0].detach().cpu().numpy()
            if cam_np.ndim == 1:
                cam_np = _tokens_to_grid(cam_np)
            elif cam_np.ndim > 2:
                cam_np = cam_np.mean(axis=0)

            overlay = overlay_heatmap(image, cam_np, alpha=alpha)
            return AttributionResult(method="gradcam", heatmap=cam_np, overlay=overlay)
        except Exception as exc:
            print(f"Grad-CAM failed ({exc}); using input-gradient fallback.", file=sys.stderr)
            inputs, input_len = self.model_wrapper.prepare_inputs(image, prompt)
            return _input_gradient_saliency(
                self.model_wrapper, image, prompt, input_len, inputs, alpha
            )
        finally:
            self.clear_hooks()
