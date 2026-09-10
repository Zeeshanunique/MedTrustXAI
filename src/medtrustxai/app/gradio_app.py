from __future__ import annotations

import time
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from medtrustxai.config import Config
from medtrustxai.modalities import MODALITIES, normalize_modality
from medtrustxai.pipeline.inference import DiagnosticPipeline

PROMPT_PRESETS = {
    "radiology": {
        "Full report": (
            "Analyze this chest X-ray. Provide: (1) key findings, (2) likely diagnosis, "
            "(3) brief explanation of visual evidence."
        ),
        "Findings only": "List the key radiological findings visible in this chest X-ray.",
        "Diagnosis only": "What is the most likely diagnosis based on this chest X-ray?",
        "VQA — lungs normal?": "Are the lungs normal appearing?",
        "VQA — abnormality": "What abnormality is present in this image?",
        "Custom question": "",
    },
    "pathology": {
        "Full report": (
            "Analyze this histopathology image. Provide: (1) tissue architecture and cellular "
            "features, (2) likely diagnosis or differential, (3) explanation of visual evidence."
        ),
        "Morphology only": "Describe the tissue architecture and cellular morphology in this H&E slide.",
        "Diagnosis only": "What is the most likely diagnosis based on this histopathology image?",
        "VQA — malignancy": "Are there features suggestive of malignancy?",
        "VQA — inflammation": "Is there evidence of inflammatory infiltrate?",
        "Custom question": "",
    },
}

_pipelines: dict[str, DiagnosticPipeline] = {}


def _get_pipeline(config_path: str, modality: str) -> DiagnosticPipeline:
    key = normalize_modality(modality)
    if key not in _pipelines:
        _pipelines[key] = DiagnosticPipeline(Config.load(config_path), modality=key)
    return _pipelines[key]


def _to_pil(image: np.ndarray | Image.Image | None) -> Image.Image | None:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return None


def _faithfulness(data: dict | None) -> str:
    if not data:
        return "Enable Grad-CAM for faithfulness score."
    return (
        f"Score {data.get('faithfulness_score', 0):.2f} · "
        f"modality: {data.get('modality', '—')} · "
        f"regions: {', '.join(data.get('mentioned_regions') or []) or 'none'} · "
        f"focus: {data.get('predicted_region') or '—'}"
    )


def _chat_pairs(messages: list[dict]) -> list[tuple[str, str]]:
    return [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(0, len(messages) - 1, 2)
        if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant"
    ]


def _findings_message(text: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": text}] if text.strip() else []


def _example_paths(config: Config, modality: str) -> list[str]:
    cfg = config.modality_config(normalize_modality(modality))
    example_dir = Path(cfg.get("example_dir", f"data/test_{modality}/images"))
    if not example_dir.is_dir():
        return []
    return [
        str(p)
        for p in sorted(example_dir.glob("*"))
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]


def analyze(image, modality, preset, custom_prompt, run_xai, max_tokens, config_path):
    empty = [], "", "", _faithfulness(None)
    pil = _to_pil(image)
    if pil is None:
        return empty + ("Waiting for image…",)

    mod = normalize_modality(modality)
    presets = PROMPT_PRESETS[mod]
    prompt = presets.get(preset, preset)
    if preset == "Custom question":
        prompt = (custom_prompt or "").strip()
        if not prompt:
            return empty + ("Enter a custom question.",)

    t0 = time.perf_counter()
    try:
        out = _get_pipeline(config_path, mod).run(
            image=pil,
            prompt=prompt,
            sample_id=f"ui_{uuid.uuid4().hex[:8]}",
            run_xai=run_xai,
            max_new_tokens=int(max_tokens),
        )
    except Exception as exc:
        return empty + (f"Error ({time.perf_counter() - t0:.1f}s): {exc}",)

    gradcam = Image.open(out.gradcam_path).convert("RGB") if out.gradcam_path else None
    return (
        _findings_message(out.findings),
        gradcam,
        out.findings,
        prompt,
        f"Done in {time.perf_counter() - t0:.1f}s",
        _faithfulness(out.faithfulness),
    )


def chat_followup(image, modality, findings, initial_prompt, chat_history, user_message, max_tokens, config_path):
    message = (user_message or "").strip()
    if not message:
        return chat_history, "", "Type a question below."
    if not (findings or "").strip():
        return chat_history, "", "Click Analyze first."
    pil = _to_pil(image)
    if pil is None:
        return chat_history, "", "Upload an image first."

    t0 = time.perf_counter()
    try:
        reply = _get_pipeline(config_path, modality).model.chat(
            pil, initial_prompt, findings, _chat_pairs(chat_history), message, int(max_tokens)
        ).text
    except Exception as exc:
        return chat_history, "", f"Chat failed: {exc}"

    return (
        chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}],
        "",
        f"Replied in {time.perf_counter() - t0:.1f}s",
    )


def build_app(config_path: str = "config/default.yaml") -> gr.Blocks:
    load_dotenv()
    config = Config.load(config_path)
    modality_labels = {
        m: config.modality_config(m).get("label", m) for m in MODALITIES
    }

    with gr.Blocks(title="MedTrustXAI") as demo:
        gr.Markdown(
            f"# MedTrustXAI\n"
            f"Explainable multi-modal medical imaging — radiology & digital pathology\n\n"
            f"*{config.get('disclaimer', 'Research use only.')}*"
        )
        findings_state = gr.State("")
        prompt_state = gr.State("")

        with gr.Row():
            with gr.Column():
                modality = gr.Dropdown(
                    choices=[(modality_labels[m], m) for m in MODALITIES],
                    value="radiology",
                    label="Modality",
                )
                image_in = gr.Image(label="Medical image", type="numpy", height=360)
                preset = gr.Dropdown(
                    list(PROMPT_PRESETS["radiology"].keys()),
                    value="Full report",
                    label="Preset",
                )
                custom_prompt = gr.Textbox(label="Custom question", visible=False, lines=2)
                run_xai = gr.Checkbox(label="Run Grad-CAM (~2–3 min on Mac)", value=False)
                max_tokens = gr.Slider(
                    32, 256, value=int(config.get("model", "max_new_tokens", default=128)), step=16
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                chatbot = gr.Chatbot(label="Findings & follow-up", height=420)
                chat_input = gr.Textbox(
                    placeholder="Ask a follow-up question about the report…",
                    show_label=False,
                    lines=2,
                )
                chat_btn = gr.Button("Send", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
                gradcam_out = gr.Image(label="Grad-CAM", type="pil", height=220)
                faithfulness = gr.Textbox(label="Faithfulness", interactive=False)

        examples = gr.Examples(
            examples=[[p] for p in _example_paths(config, "radiology")[:6]],
            inputs=[image_in],
            label="Test samples (radiology)",
        )

        def on_modality_change(mod):
            mod = normalize_modality(mod)
            presets = list(PROMPT_PRESETS[mod].keys())
            paths = _example_paths(config, mod)[:6]
            return (
                gr.update(choices=presets, value=presets[0]),
                gr.update(visible=False),
                gr.update(samples=[[p] for p in paths] if paths else []),
            )

        modality.change(
            on_modality_change,
            modality,
            [preset, custom_prompt, examples.dataset],
        )
        preset.change(lambda p: gr.update(visible=p == "Custom question"), preset, custom_prompt)

        analyze_btn.click(
            analyze,
            [image_in, modality, preset, custom_prompt, run_xai, max_tokens, gr.State(config_path)],
            [chatbot, gradcam_out, findings_state, prompt_state, status, faithfulness],
        )

        chat_args = [
            image_in, modality, findings_state, prompt_state, chatbot, chat_input, max_tokens,
            gr.State(config_path),
        ]
        chat_btn.click(chat_followup, chat_args, [chatbot, chat_input, status])
        chat_input.submit(chat_followup, chat_args, [chatbot, chat_input, status])
        chatbot.clear(
            lambda f: (_findings_message(f), ""),
            inputs=[findings_state],
            outputs=[chatbot, chat_input],
        )

        def _preload():
            for m in MODALITIES:
                _get_pipeline(config_path, m)
            return f"Ready · {config.model_id}"

        demo.load(_preload, outputs=status)

    return demo


def launch(
    config_path: str = "config/default.yaml",
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
):
    build_app(config_path).launch(server_name=server_name, server_port=server_port, share=share)
