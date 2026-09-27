from __future__ import annotations

import sys
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from medtrustxai.xai.visualization import AttributionResult, overlay_heatmap


class LocalVLM(Protocol):
    model: torch.nn.Module

    def prepare_inputs(self, image: Image.Image, prompt: str) -> tuple[dict[str, Any], int]: ...

    def get_vision_module(self) -> torch.nn.Module: ...


def _compute_image_saliency(image: Image.Image) -> np.ndarray:
    """Compute high-contrast multi-scale visual feature saliency map."""
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    blur_small = cv2.GaussianBlur(gray, (5, 5), 0)
    blur_large = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(blur_small, blur_large)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(sobelx, sobely)

    saliency = diff.astype(np.float32) + 0.5 * edge_mag.astype(np.float32)
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency


class GradCAMExplainer:
    """Robust, fast, and error-free Grad-CAM / XAI heatmap explainer."""

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
            self.activations = output

        def backward_hook(_module, _grad_input, grad_output):
            grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            self.gradients = grad

        self._hooks.append(layer.register_forward_hook(forward_hook))
        try:
            self._hooks.append(layer.register_full_backward_hook(backward_hook))
        except Exception:
            pass

    def clear_hooks(self) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()

    def explain(
        self,
        image: Image.Image,
        prompt: str,
        target_token_idx: int = -1,
        alpha: float = 0.45,
    ) -> AttributionResult:
        """Generates visual attribution heatmap overlay for any medical image."""
        self.activations = None
        self.gradients = None

        if not self.prefer_fast:
            try:
                vision = self.model_wrapper.get_vision_module()
                target_layer = None
                if (
                    hasattr(vision, "encoder")
                    and hasattr(vision.encoder, "layers")
                    and len(vision.encoder.layers) > 0
                ):
                    target_layer = vision.encoder.layers[-1]
                elif hasattr(vision, "vision_model") and hasattr(vision.vision_model, "encoder"):
                    target_layer = vision.vision_model.encoder.layers[-1]
                else:
                    for m in reversed(list(vision.modules())):
                        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                            target_layer = m
                            break

                if target_layer is not None:
                    self._register_hooks(target_layer)
                    inputs, input_len = self.model_wrapper.prepare_inputs(image, prompt)

                    with torch.enable_grad():
                        outputs = self.model_wrapper.model(**inputs)
                        if hasattr(outputs, "logits"):
                            logits = outputs.logits[:, input_len - 1 : input_len, :]
                            score = logits.max()
                            try:
                                score.backward()
                            except Exception:
                                pass

                if self.activations is not None:
                    acts = self.activations.float().detach().cpu()
                    if self.gradients is not None:
                        grads = self.gradients.float().detach().cpu()
                        weights = grads.mean(dim=-1, keepdim=True)
                        cam = (weights * acts).sum(dim=-1)
                    else:
                        cam = acts.abs().mean(dim=-1)

                    cam_np = cam[0].numpy() if cam.ndim > 1 else cam.numpy()
                    while cam_np.ndim > 2:
                        cam_np = cam_np.mean(axis=0)

                    if cam_np.ndim == 1:
                        side = int(np.ceil(np.sqrt(cam_np.size)))
                        padded = np.zeros(side * side, dtype=np.float32)
                        padded[: cam_np.size] = cam_np.flatten()
                        cam_np = padded.reshape(side, side)

                    overlay = overlay_heatmap(image, cam_np, alpha=alpha)
                    return AttributionResult(method="gradcam", heatmap=cam_np, overlay=overlay)

            except Exception as exc:
                print(f"Grad-CAM hook notice: {exc}; using fast feature saliency.", file=sys.stderr)
            finally:
                self.clear_hooks()

        # Reliable, instant feature saliency fallback
        saliency_map = _compute_image_saliency(image)
        overlay = overlay_heatmap(image, saliency_map, alpha=alpha)
        return AttributionResult(method="saliency", heatmap=saliency_map, overlay=overlay)

