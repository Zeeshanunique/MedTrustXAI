from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        config_path = Path(path or "config/default.yaml")
        with config_path.open("r", encoding="utf-8") as f:
            return cls(raw=yaml.safe_load(f) or {})

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def model_id(self) -> str:
        return self.get("model", "local_id", default="BIOMEDICA/BMC-smolvlm1-256M")

    @property
    def output_dir(self) -> Path:
        return Path(self.get("paths", "outputs", default="outputs"))

    def modality_config(self, modality: str) -> dict[str, Any]:
        cfg = self.get("modalities", modality)
        if not isinstance(cfg, dict):
            raise KeyError(f"Unknown modality: {modality}")
        return cfg

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        Path(self.get("xai", "output_dir", default="outputs/xai")).mkdir(parents=True, exist_ok=True)


def setup_hf_auth() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
