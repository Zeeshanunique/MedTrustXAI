# MedTrustXAI

Explainable deep learning for **multi-modal medical imaging** — trustworthy AI-assisted diagnosis in **radiology** (chest X-ray) and **digital pathology** (H&E).

Built on [BIOMEDICA/BMC-smolvlm1-256M](https://huggingface.co/BIOMEDICA/BMC-smolvlm1-256M) with Grad-CAM, modality-specific faithfulness scoring, and follow-up chat.

**Research use only. Not for clinical diagnosis.**

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ui]"
```

## Download test data

```bash
python scripts/download_test_dataset.py --modality all
```

## Usage

```bash
# Radiology
medtrustxai infer --modality radiology --image data/test_cxr/images/cxr_001_normal_pa.jpg

# Digital pathology
medtrustxai infer --modality pathology --image data/test_pathology/images/path_001_normal_mucosa.jpg

# Batch (per-modality manifest)
medtrustxai batch --modality radiology
medtrustxai batch --modality pathology

# Web UI (modality selector)
medtrustxai serve --port 7860
```

## Config

`config/default.yaml` — per-modality prompts, XAI settings, manifest paths.
