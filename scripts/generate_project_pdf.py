#!/usr/bin/env python3
"""Generate academic project documentation PDF for MedTrustXAI."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "MedTrustXAI_Project_Documentation.pdf"

TITLE = (
    "Explainable Deep Learning for Multi-Modal Medical Imaging:\n"
    "Toward Trustworthy AI-Assisted Diagnosis in Radiology and Digital Pathology"
)
SUBTITLE = "MedTrustXAI - Multi-Modal Research Prototype & Implementation Report"
VERSION = "Version 0.2.0  |  M.Tech Research Project  |  September 2026"


class DocPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, "MedTrustXAI - Explainable Multi-Modal Medical Imaging", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def chapter(self, num: str, title: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(12, 64, 108)
        self.multi_cell(0, 7, f"{num}  {title}")
        self.ln(1)
        self.set_draw_color(12, 64, 108)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def section(self, title: str) -> None:
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, title)
        self.ln(1)

    def _ensure_space(self, height: float = 20) -> None:
        if self.get_y() + height > self.h - self.b_margin:
            self.add_page()

    def body(self, text: str) -> None:
        self._ensure_space(15)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(35, 35, 35)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.2, text)
        self.ln(2)

    def bullets(self, items: list[str]) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(35, 35, 35)
        for item in items:
            self._ensure_space(10)
            self.set_x(self.l_margin)
            self.multi_cell(0, 5.2, f"- {item}")
        self.ln(2)

    def mono(self, text: str) -> None:
        self._ensure_space(20)
        self.set_font("Courier", "", 8.2)
        self.set_fill_color(248, 249, 251)
        self.set_text_color(25, 25, 25)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4.3, text, fill=True)
        self.ln(3)


def _cover(pdf: DocPDF) -> None:
    pdf.add_page()
    pdf.set_fill_color(12, 64, 108)
    pdf.rect(0, 0, 210, 52, style="F")
    pdf.ln(18)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, 8, "Explainable Deep Learning for\nMulti-Modal Medical Imaging", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "Toward Trustworthy AI-Assisted Diagnosis\nin Radiology and Digital Pathology",
        align="C",
    )
    pdf.ln(28)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "MedTrustXAI", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, SUBTITLE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(16)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, VERSION, align="C", new_x="LMARGIN", new_y="NEXT")


def build_pdf() -> None:
    pdf = DocPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    _cover(pdf)

    pdf.add_page()
    pdf.chapter("", "Abstract")
    pdf.body(
        "This project presents MedTrustXAI, an explainable deep learning framework for "
        "multi-modal medical imaging that supports both radiology (chest X-ray) and digital "
        "pathology (H&E histopathology). The system integrates a biomedical Vision-Language "
        "Model (BMC-SmolVLM-256M) with visual attribution (Grad-CAM / input-gradient saliency), "
        "modality-specific faithfulness evaluation, WSI patch tiling for large pathology slides, "
        "and quantitative VQA/report metrics. A Gradio interface and CLI provide interactive "
        "demonstration, batch inference, and evaluation on curated test datasets."
    )

    pdf.chapter("1", "Introduction")
    pdf.body(
        "Trustworthy AI-assisted diagnosis requires models that not only predict accurately but "
        "also explain their reasoning in clinically inspectable form. Radiology and digital "
        "pathology are complementary imaging modalities: radiographs reveal macroscopic anatomy "
        "while histopathology reveals cellular and tissue architecture. MedTrustXAI unifies both "
        "under a single explainable VLM pipeline with modality-aware prompts, region maps, and metrics."
    )

    pdf.chapter("2", "Implemented Modalities")
    pdf.section("2.1 Radiology (Chest X-ray)")
    pdf.bullets(
        [
            "8-sample test dataset (normal, pneumonia, VQA-RAD chest)",
            "Anatomy-based faithfulness regions (lungs, heart, pleura, mediastinum)",
            "Prompt presets: full report, findings, diagnosis, VQA questions",
        ]
    )
    pdf.section("2.2 Digital Pathology (H&E)")
    pdf.bullets(
        [
            "4-sample test dataset (seminoma, renal carcinoma, metastatic ductal carcinoma, hepatic amyloidosis)",
            "Tissue-based faithfulness regions (epithelium, stroma, nuclei, tumor, necrosis)",
            "WSI patch tiling: center patch or grid extraction from large slides",
        ]
    )

    pdf.chapter("3", "System Architecture")
    pdf.mono(
        "   RADIOLOGY (CXR)  |  PATHOLOGY (H&E / WSI patch)\n"
        "                    |\n"
        "                    v\n"
        "         +----------------------+\n"
        "         |   Gradio UI / CLI    |\n"
        "         | infer|batch|evaluate|\n"
        "         +----------------------+\n"
        "                    |\n"
        "                    v\n"
        "         +----------------------+\n"
        "         | DiagnosticPipeline   |\n"
        "         | (modality-aware)     |\n"
        "         +----------------------+\n"
        "              /            \\\n"
        "             v              v\n"
        "   +----------------+  +-------------------+\n"
        "   | SmolVLMLocal   |  | GradCAMExplainer  |\n"
        "   | Model (256M)   |  | + Faithfulness    |\n"
        "   +----------------+  +-------------------+\n"
        "             |                |\n"
        "             v                v\n"
        "      Findings / VQA     Heatmap PNG\n"
        "             \\                /\n"
        "              v              v\n"
        "         +----------------------+\n"
        "         | outputs/ JSON + PNG  |\n"
        "         +----------------------+"
    )

    pdf.chapter("4", "Explainability and Trust")
    pdf.bullets(
        [
            "Grad-CAM on CUDA; fast input-gradient saliency on Mac MPS/CPU",
            "Modality-specific faithfulness: anatomy (radiology) vs tissue regions (pathology)",
            "Follow-up chat in Gradio for clinician-style questioning",
            "Saved JSON artifacts for reproducibility and guide review",
        ]
    )

    pdf.chapter("5", "WSI Patch Tiling")
    pdf.body(
        "Large whole-slide images are tiled into fixed-size patches (default 512x512). "
        "The center-patch mode analyzes the slide center; grid mode extracts overlapping "
        "patches for future multi-patch aggregation. Background-heavy patches can be "
        "filtered using a tissue fraction threshold."
    )
    pdf.mono(
        "medtrustxai infer --modality pathology \\\n"
        "  --image slide.png --tile-mode center --patch-size 512"
    )

    pdf.chapter("6", "Evaluation Metrics")
    pdf.bullets(
        [
            "VQA match: normalized yes/no and keyword overlap against reference_hint",
            "Token F1: word-level overlap between prediction and reference",
            "Faithfulness score: heatmap region vs mentioned anatomy/tissue terms",
            "CLI: medtrustxai evaluate --modality radiology",
        ]
    )

    pdf.chapter("7", "Usage")
    pdf.mono(
        "pip install -e \".[ui]\"\n"
        "python scripts/download_test_dataset.py --modality all\n"
        "medtrustxai batch --modality radiology --no-xai\n"
        "medtrustxai batch --modality pathology --no-xai\n"
        "medtrustxai evaluate --modality radiology\n"
        "medtrustxai serve --port 7860"
    )

    pdf.chapter("8", "Future Work")
    pdf.bullets(
        [
            "Multi-patch WSI aggregation with attention-weighted fusion",
            "Clinical text + imaging multi-modal fusion",
            "Expert radiologist/pathologist review studies",
            "Larger benchmarks: MIMIC-CXR, Camelyon, PatchCamelyon",
        ]
    )

    pdf.chapter("9", "Conclusion")
    pdf.body(
        "MedTrustXAI delivers a working multi-modal explainable AI prototype aligned with the "
        "research theme of trustworthy AI-assisted diagnosis in radiology and digital pathology. "
        "Both modalities are implemented with shared VLM and XAI infrastructure, modality-specific "
        "trust evaluation, WSI tiling, and quantitative metrics suitable for academic demonstration."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Generated: {OUT}")


if __name__ == "__main__":
    build_pdf()
