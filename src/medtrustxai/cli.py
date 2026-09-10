from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from medtrustxai.config import Config
from medtrustxai.data.test_dataset import TestDataset
from medtrustxai.modalities import MODALITIES, normalize_modality
from medtrustxai.pipeline.inference import DiagnosticPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explainable multi-modal medical imaging (radiology + pathology)"
    )
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer", help="Run diagnosis + XAI on one image")
    infer.add_argument("--modality", choices=MODALITIES, default="radiology")
    infer.add_argument("--image", required=True, help="Path to medical image")
    infer.add_argument("--prompt", default=None, help="Custom prompt")
    infer.add_argument("--sample-id", default="demo")
    infer.add_argument("--no-xai", action="store_true")
    infer.add_argument("--max-tokens", type=int, default=None)

    batch = sub.add_parser("batch", help="Run inference on a test dataset manifest")
    batch.add_argument("--modality", choices=MODALITIES, default="radiology")
    batch.add_argument("--manifest", default=None, help="Path to manifest.json")
    batch.add_argument("--prompt", default=None, help="Custom prompt (overrides VQA questions)")
    batch.add_argument("--no-xai", action="store_true")
    batch.add_argument("--max-tokens", type=int, default=None)
    batch.add_argument("--limit", type=int, default=None)

    serve = sub.add_parser("serve", help="Launch Gradio web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    config = Config.load(args.config)

    if args.command == "infer":
        modality = normalize_modality(args.modality)
        image = Image.open(args.image).convert("RGB")
        pipeline = DiagnosticPipeline(config, modality=modality)
        prompt = args.prompt or pipeline.default_prompt()
        output = pipeline.run(
            image=image,
            prompt=prompt,
            sample_id=args.sample_id,
            run_xai=not args.no_xai,
            max_new_tokens=args.max_tokens,
        )
        out_path = Path(config.output_dir) / f"{args.sample_id}_{modality}_result.json"
        pipeline.save_result(output, out_path)
        print(output.findings)
        if output.faithfulness:
            print(f"Faithfulness: {output.faithfulness['faithfulness_score']:.3f}")
        print(f"Saved: {out_path}")

    elif args.command == "batch":
        modality = normalize_modality(args.modality)
        modality_cfg = config.modality_config(modality)
        manifest = args.manifest or modality_cfg.get("default_manifest")
        dataset = TestDataset.from_manifest(manifest, modality=modality)
        samples = dataset.samples[: args.limit] if args.limit else dataset.samples
        if not samples:
            raise SystemExit(
                f"No samples in {manifest}. Run: python scripts/download_test_dataset.py --modality {modality}"
            )

        pipeline = DiagnosticPipeline(config, modality=modality)
        default_prompt = args.prompt or pipeline.default_prompt()
        print(f"Running {modality} batch on {len(samples)} sample(s)…\n")

        for sample in samples:
            image = dataset.load_image(sample)
            prompt = default_prompt
            if args.prompt is None and sample.vqa_questions:
                prompt = sample.vqa_questions[0]
            output = pipeline.run(
                image=image,
                prompt=prompt,
                sample_id=sample.id,
                run_xai=not args.no_xai,
                max_new_tokens=args.max_tokens,
            )
            out_path = Path(config.output_dir) / f"{sample.id}_{modality}_result.json"
            pipeline.save_result(output, out_path)
            print(f"[{sample.id}] {sample.category} — {sample.reference_hint}")
            print(output.findings[:200] + ("…" if len(output.findings) > 200 else ""))
            print(f"  saved: {out_path}\n")

    elif args.command == "serve":
        from medtrustxai.app.gradio_app import launch

        launch(
            config_path=args.config,
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )


if __name__ == "__main__":
    main()
