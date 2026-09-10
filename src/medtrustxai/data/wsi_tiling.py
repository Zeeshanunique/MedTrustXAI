from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ImagePatch:
    image: Image.Image
    x: int
    y: int
    width: int
    height: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def extract_patches(
    image: Image.Image,
    patch_size: int = 512,
    stride: int | None = None,
    min_tissue_fraction: float = 0.0,
) -> list[ImagePatch]:
    """Tile a large pathology image into fixed-size RGB patches."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    stride = stride or patch_size
    rgb = image.convert("RGB")
    w, h = rgb.size
    patches: list[ImagePatch] = []

    for y in range(0, max(h - patch_size + 1, 1), stride):
        for x in range(0, max(w - patch_size + 1, 1), stride):
            box = (x, y, min(x + patch_size, w), min(y + patch_size, h))
            crop = rgb.crop(box)
            if crop.size != (patch_size, patch_size):
                padded = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
                padded.paste(crop, (0, 0))
                crop = padded
            if min_tissue_fraction > 0 and _background_fraction(crop) > (1 - min_tissue_fraction):
                continue
            patches.append(ImagePatch(image=crop, x=x, y=y, width=patch_size, height=patch_size))

    if not patches:
        patches.append(
            ImagePatch(
                image=rgb.resize((patch_size, patch_size)),
                x=0,
                y=0,
                width=patch_size,
                height=patch_size,
            )
        )
    return patches


def select_center_patch(
    image: Image.Image,
    patch_size: int = 512,
) -> ImagePatch:
    """Pick the center patch from a large slide image."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w <= patch_size and h <= patch_size:
        return ImagePatch(
            image=rgb.resize((patch_size, patch_size)),
            x=0,
            y=0,
            width=patch_size,
            height=patch_size,
        )
    x = max((w - patch_size) // 2, 0)
    y = max((h - patch_size) // 2, 0)
    crop = rgb.crop((x, y, x + patch_size, y + patch_size))
    return ImagePatch(image=crop, x=x, y=y, width=patch_size, height=patch_size)


def _background_fraction(patch: Image.Image, threshold: int = 235) -> float:
    pixels = list(patch.convert("RGB").getdata())
    bright = sum(1 for r, g, b in pixels if r >= threshold and g >= threshold and b >= threshold)
    return bright / max(len(pixels), 1)
