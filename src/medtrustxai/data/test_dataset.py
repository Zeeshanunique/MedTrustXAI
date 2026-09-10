from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from medtrustxai.modalities import Modality, normalize_modality


@dataclass(frozen=True)
class TestSample:
    id: str
    path: Path
    modality: Modality
    category: str
    reference_hint: str
    vqa_questions: list[str]
    source: str = ""
    license: str = ""


@dataclass
class TestDataset:
    root: Path
    modality: Modality
    samples: list[TestSample]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, sample: TestSample) -> Image.Image:
        return Image.open(sample.path).convert("RGB")

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        project_root: Path | str | None = None,
        modality: str | None = None,
    ) -> TestDataset:
        manifest_path = Path(manifest_path).resolve()
        dataset_root = manifest_path.parent
        proj = Path(project_root).resolve() if project_root else dataset_root.parent.parent
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        default_modality = normalize_modality(
            modality or data.get("modality") or _modality_from_path(manifest_path)
        )
        samples: list[TestSample] = []
        for row in data.get("samples", []):
            sample_modality = normalize_modality(row.get("modality") or default_modality)
            rel = row.get("path") or f"data/test_{sample_modality}/images/{row['filename']}"
            candidates = [
                Path(rel),
                dataset_root / "images" / row["filename"],
                proj / rel,
            ]
            path = next((p.resolve() for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Missing image for {row['id']}: tried {[str(c) for c in candidates]}")
            samples.append(
                TestSample(
                    id=row["id"],
                    path=path,
                    modality=sample_modality,
                    category=row.get("category", "unknown"),
                    reference_hint=row.get("reference_hint", ""),
                    vqa_questions=list(row.get("vqa_questions") or []),
                    source=row.get("source", ""),
                    license=row.get("license", ""),
                )
            )
        return cls(root=dataset_root, modality=default_modality, samples=samples)


def _modality_from_path(manifest_path: Path) -> str:
    name = manifest_path.parent.name.lower()
    if "pathology" in name or "path" in name:
        return "pathology"
    return "radiology"


def load_test_dataset(modality: Modality, project_root: Path | str | None = None) -> TestDataset:
    root = Path(project_root or Path.cwd())
    folder = "test_cxr" if modality == "radiology" else "test_pathology"
    manifest = root / "data" / folder / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Test dataset not found at {manifest}. Run: python scripts/download_test_dataset.py --modality {modality}"
        )
    return TestDataset.from_manifest(manifest, project_root=root, modality=modality)
