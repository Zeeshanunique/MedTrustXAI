from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from medtrustxai.config import Config
from medtrustxai.data.test_dataset import TestDataset
from medtrustxai.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    save_confusion_matrix_json,
)
from medtrustxai.evaluation.metrics import evaluate_results, save_evaluation
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
    infer.add_argument(
        "--tile-mode",
        choices=["none", "center", "grid"],
        default="center",
        help="Pathology WSI tiling: center patch or grid preview",
    )
    infer.add_argument("--patch-size", type=int, default=512)
    infer.add_argument("--stride", type=int, default=None)

    batch = sub.add_parser("batch", help="Run inference on a test dataset manifest")
    batch.add_argument("--modality", choices=MODALITIES, default="radiology")
    batch.add_argument("--manifest", default=None, help="Path to manifest.json")
    batch.add_argument("--prompt", default=None, help="Custom prompt (overrides VQA questions)")
    batch.add_argument("--no-xai", action="store_true")
    batch.add_argument("--max-tokens", type=int, default=None)
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--tile-mode", choices=["none", "center", "grid"], default="center")
    batch.add_argument("--patch-size", type=int, default=512)
    batch.add_argument("--stride", type=int, default=None)
    batch.add_argument(
        "--mode",
        choices=["classification", "report"],
        default="classification",
        help="classification: tuned labels for accuracy/confusion matrix; report: free-text",
    )

    evaluate = sub.add_parser("evaluate", help="Compute metrics and confusion matrix from results")
    evaluate.add_argument("--modality", choices=MODALITIES, default="radiology")
    evaluate.add_argument("--manifest", default=None)
    evaluate.add_argument("--results-dir", default="outputs")
    evaluate.add_argument("--output", default=None, help="JSON summary path")

    serve = sub.add_parser("serve", help="Launch Gradio web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--share", action="store_true")
    return parser


def _run_sample(
    pipeline: DiagnosticPipeline,
    image: Image.Image,
    prompt: str,
    sample_id: str,
    modality: str,
    run_xai: bool,
    max_tokens: int | None,
    tile_mode: str,
    patch_size: int,
    stride: int | None,
):
    if modality == "pathology" and tile_mode != "none":
        return pipeline.run_pathology_wsi(
            image=image,
            prompt=prompt,
            sample_id=sample_id,
            run_xai=run_xai,
            max_new_tokens=max_tokens,
            tile_mode=tile_mode,
            patch_size=patch_size,
            stride=stride,
        )
    return pipeline.run(
        image=image,
        prompt=prompt,
        sample_id=sample_id,
        run_xai=run_xai,
        max_new_tokens=max_tokens,
    )


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    config = Config.load(args.config)

    if args.command == "infer":
        modality = normalize_modality(args.modality)
        image = Image.open(args.image).convert("RGB")
        pipeline = DiagnosticPipeline(config, modality=modality)
        prompt = args.prompt or pipeline.default_prompt()
        output = _run_sample(
            pipeline,
            image,
            prompt,
            args.sample_id,
            modality,
            not args.no_xai,
            args.max_tokens,
            args.tile_mode,
            args.patch_size,
            args.stride,
        )
        out_path = Path(config.output_dir) / f"{args.sample_id}_{modality}_result.json"
        pipeline.save_result(output, out_path)
        print(output.findings)
        if output.patch_bbox:
            print(f"Patch bbox: {output.patch_bbox} (from {output.num_patches} patch(es))")
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
            if args.mode == "classification":
                output = pipeline.run_classification(
                    sample=sample,
                    image=image,
                    run_xai=not args.no_xai,
                    max_new_tokens=args.max_tokens,
                    tile_mode=args.tile_mode,
                    patch_size=args.patch_size,
                    stride=args.stride,
                )
            else:
                prompt = default_prompt
                if args.prompt is None and sample.vqa_questions:
                    prompt = sample.vqa_questions[0]
                output = _run_sample(
                    pipeline,
                    image,
                    prompt,
                    sample.id,
                    modality,
                    not args.no_xai,
                    args.max_tokens,
                    args.tile_mode,
                    args.patch_size,
                    args.stride,
                )
            out_path = Path(config.output_dir) / f"{sample.id}_{modality}_result.json"
            pipeline.save_result(output, out_path)
            print(f"[{sample.id}] {sample.category} — {sample.reference_hint}")
            if output.predicted_label and output.true_label:
                mark = "OK" if output.predicted_label == output.true_label else "MISS"
                print(f"  label: {output.true_label} -> {output.predicted_label} [{mark}]")
            print(output.findings[:200] + ("…" if len(output.findings) > 200 else ""))
            print(f"  saved: {out_path}\n")

    elif args.command == "evaluate":
        modality = normalize_modality(args.modality)
        modality_cfg = config.modality_config(modality)
        manifest = args.manifest or modality_cfg.get("default_manifest")
        summary = evaluate_results(manifest, args.results_dir, modality)
        out = Path(args.output or config.output_dir) / f"evaluation_{modality}.json"
        save_evaluation(summary, out)
        print(f"Modality: {modality}")
        print(f"Samples evaluated: {summary.num_samples}")
        print(f"Mean VQA match: {summary.mean_vqa_match:.3f}")
        print(f"Mean token F1: {summary.mean_token_f1:.3f}")
        if summary.mean_faithfulness is not None:
            print(f"Mean faithfulness: {summary.mean_faithfulness:.3f}")
        if summary.mean_classification_accuracy is not None:
            print(f"Classification accuracy: {summary.mean_classification_accuracy:.3f}")
        print(f"Saved: {out}")

        cm_results = compute_confusion_matrix(manifest, args.results_dir, modality)
        if cm_results:
            cm_json = Path(config.output_dir) / f"confusion_matrix_{modality}.json"
            save_confusion_matrix_json(cm_results, cm_json)
            for cm in cm_results:
                png = Path(config.output_dir) / f"confusion_matrix_{modality}_{cm.task}.png"
                plot_confusion_matrix(cm, png)
                print(f"\n{cm.task}: accuracy={cm.accuracy:.2%} ({len(cm.rows)} samples)")
                for label, recall in cm.per_class_recall.items():
                    print(f"  recall[{label}]: {recall:.2%}")
                print(f"  matrix PNG: {png}")
            print(f"Confusion matrix JSON: {cm_json}")
        else:
            print("No result files found for confusion matrix. Run: medtrustxai batch --modality", modality)

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
