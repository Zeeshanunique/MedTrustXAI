#!/usr/bin/env python3
"""Download public-domain radiology and pathology images for local testing."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_UA = "MedTrustXAI/0.1 (research)"

RADIOLOGY_DIR = ROOT / "data" / "test_cxr" / "images"
RADIOLOGY_MANIFEST = ROOT / "data" / "test_cxr" / "manifest.json"
PATHOLOGY_DIR = ROOT / "data" / "test_pathology" / "images"
PATHOLOGY_MANIFEST = ROOT / "data" / "test_pathology" / "manifest.json"

WIKIMEDIA_CXR: list[dict] = [
    {
        "id": "cxr_001_normal_pa",
        "filename": "cxr_001_normal_pa.jpg",
        "title": "File:Normal posteroanterior (PA) chest radiograph (X-ray).jpg",
        "category": "normal",
        "reference_hint": "Normal PA chest radiograph",
        "vqa_questions": ["Are the lungs normal appearing?", "Is there any focal consolidation?"],
    },
    {
        "id": "cxr_002_lobar_pneumonia",
        "filename": "cxr_002_lobar_pneumonia.jpg",
        "title": "File:X-ray of lobar pneumonia.jpg",
        "category": "pneumonia",
        "reference_hint": "Lobar pneumonia, right middle lobe",
        "vqa_questions": ["What abnormality is present in this image?", "Is there pneumonia?"],
    },
    {
        "id": "cxr_003_pneumonia_ap",
        "filename": "cxr_003_pneumonia_ap.jpg",
        "title": "File:Pneumonia x ray.jpg",
        "category": "pneumonia",
        "reference_hint": "Pneumonia with right upper lobe opacity",
        "vqa_questions": ["Is there increased opacity in the lungs?", "Which lobe shows abnormality?"],
    },
    {
        "id": "cxr_004_normal",
        "filename": "cxr_004_normal.png",
        "title": "File:Chest.png",
        "category": "normal",
        "reference_hint": "Normal heart and lungs",
        "vqa_questions": ["Are the lungs normal appearing?", "Is the heart size normal?"],
    },
]

WIKIMEDIA_PATHOLOGY: list[dict] = [
    {
        "id": "path_001_seminoma",
        "filename": "path_001_seminoma.png",
        "title": "File:Histopathology of seminoma.png",
        "category": "malignant",
        "reference_hint": "Classical seminoma H&E",
        "vqa_questions": [
            "Describe the tumor cell morphology.",
            "Are there features of malignancy?",
        ],
    },
    {
        "id": "path_002_renal_carcinoma",
        "filename": "path_002_renal_carcinoma.jpg",
        "title": "File:Histopathology of clear cell renal cell carcinoma, grade 1, high magnification.jpg",
        "category": "malignant",
        "reference_hint": "Clear cell renal cell carcinoma grade 1",
        "vqa_questions": [
            "What malignant features are visible?",
            "Describe the nuclear morphology.",
        ],
    },
    {
        "id": "path_003_ductal_carcinoma",
        "filename": "path_003_ductal_carcinoma.jpg",
        "title": "File:Histopathology of a lymph node with metastatic invasive ductal carcinoma from the breast.jpg",
        "category": "metastatic",
        "reference_hint": "Lymph node metastasis from breast ductal carcinoma",
        "vqa_questions": [
            "Is there evidence of metastatic carcinoma?",
            "Describe the stromal and epithelial features.",
        ],
    },
    {
        "id": "path_004_hepatic_amyloid",
        "filename": "path_004_hepatic_amyloid.jpg",
        "title": "File:Hepatic amyloidosis - high mag.jpg",
        "category": "deposition",
        "reference_hint": "Hepatic amyloidosis H&E",
        "vqa_questions": [
            "What extracellular deposits are visible?",
            "Describe the tissue architecture.",
        ],
    },
    {
        "id": "path_005_meningioma",
        "filename": "path_005_meningioma.png",
        "title": "File:Histopathology of meningioma.png",
        "category": "benign_neoplasm",
        "reference_hint": "Meningioma H&E",
        "vqa_questions": [
            "What type of neoplasm is suggested?",
            "Describe the cellular pattern.",
        ],
    },
]


def _open(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout)


def _wikimedia_download_url(title: str) -> str:
    api = (
        "https://commons.wikimedia.org/w/api.php?"
        + urllib.parse.urlencode(
            {
                "action": "query",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json",
            }
        )
    )
    with _open(api, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    pages = payload["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise FileNotFoundError(f"Wikimedia file not found: {title}")
    return page["imageinfo"][0]["url"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _open(url, timeout=120) as resp, dest.open("wb") as f:
        f.write(resp.read())


def _download_specs(
    specs: list[dict],
    out_dir: Path,
    modality: str,
) -> list[dict]:
    manifest: list[dict] = []
    for spec in specs:
        dest = out_dir / spec["filename"]
        print(f"Downloading {spec['id']} …")
        try:
            url = _wikimedia_download_url(spec["title"])
            _download(url, dest)
        except Exception as exc:
            print(f"  SKIP {spec['id']}: {exc}")
            continue
        manifest.append(
            {
                "id": spec["id"],
                "filename": spec["filename"],
                "path": str(dest.relative_to(ROOT)),
                "modality": modality,
                "category": spec["category"],
                "source": "Wikimedia Commons",
                "reference_hint": spec["reference_hint"],
                "vqa_questions": spec["vqa_questions"],
                "license": "CC0 / Public Domain",
            }
        )
        print(f"  -> {dest}")
        time.sleep(1.5)
    return manifest


def download_radiology() -> int:
    manifest = _download_specs(WIKIMEDIA_CXR, RADIOLOGY_DIR, "radiology")
    RADIOLOGY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RADIOLOGY_MANIFEST.write_text(
        json.dumps({"modality": "radiology", "samples": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nRadiology: {len(manifest)} samples -> {RADIOLOGY_MANIFEST}")
    return len(manifest)


def download_pathology() -> int:
    manifest = _download_specs(WIKIMEDIA_PATHOLOGY, PATHOLOGY_DIR, "pathology")
    PATHOLOGY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PATHOLOGY_MANIFEST.write_text(
        json.dumps({"modality": "pathology", "samples": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nPathology: {len(manifest)} samples -> {PATHOLOGY_MANIFEST}")
    return len(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download test radiology/pathology datasets")
    parser.add_argument(
        "--modality",
        choices=["radiology", "pathology", "all"],
        default="all",
        help="Which modality dataset to download",
    )
    args = parser.parse_args()

    total = 0
    if args.modality in {"radiology", "all"}:
        total += download_radiology()
    if args.modality in {"pathology", "all"}:
        total += download_pathology()
    print(f"\nDone. {total} total samples downloaded.")


if __name__ == "__main__":
    main()
