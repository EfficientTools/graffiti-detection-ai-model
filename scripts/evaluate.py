#!/usr/bin/env python
"""
Evaluation script for graffiti detection model.
"""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graffiti_detection.evaluation.report import (  # noqa: E402
    EvaluationGates,
    build_evaluation_report,
    normalize_metrics,
    normalize_numeric_mapping,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate graffiti detection model")

    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights")

    parser.add_argument(
        "--data",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration file",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on",
    )

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")

    parser.add_argument("--img-size", type=int, default=640, help="Input image size")

    parser.add_argument(
        "--conf-threshold", type=float, default=0.25, help="Confidence threshold for predictions"
    )

    parser.add_argument("--iou-threshold", type=float, default=0.7, help="IoU threshold for NMS")

    parser.add_argument("--device", type=str, default="0", help="Device to use (0, 1, ..., or cpu)")

    parser.add_argument("--save-json", action="store_true", help="Save results in COCO JSON format")

    parser.add_argument("--save-txt", action="store_true", help="Save predictions in YOLO format")

    parser.add_argument(
        "--save-conf", action="store_true", help="Save confidence scores in txt files"
    )

    parser.add_argument("--project", type=str, default="runs/evaluate", help="Project directory")

    parser.add_argument("--name", type=str, default="exp", help="Experiment name")

    parser.add_argument(
        "--exist-ok", action="store_true", help="Allow overwriting existing project/name"
    )

    parser.add_argument("--plots", action="store_true", help="Generate evaluation plots")

    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    parser.add_argument("--min-map50", type=float, help="Fail below this mAP@0.5")
    parser.add_argument("--min-map50-95", type=float, help="Fail below this mAP@0.5:0.95")
    parser.add_argument("--min-precision", type=float, help="Fail below this precision")
    parser.add_argument("--min-recall", type=float, help="Fail below this recall")
    parser.add_argument(
        "--max-inference-ms",
        type=float,
        help="Fail when mean inference latency per image exceeds this value",
    )

    return parser.parse_args()


def main() -> int:
    """Main evaluation function."""
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")

    gates = EvaluationGates(
        min_map50=args.min_map50,
        min_map50_95=args.min_map50_95,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        max_inference_ms=args.max_inference_ms,
    )

    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")
        args.device = "cpu"

    # Load model
    print(f"Loading model from {args.model}")
    model = YOLO(str(model_path))

    # Print evaluation configuration
    print("\n" + "=" * 50)
    print("Evaluation Configuration:")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Device: {args.device}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print("=" * 50 + "\n")

    # Run validation
    print(f"Evaluating on {args.split} set...")
    results = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch_size,
        imgsz=args.img_size,
        conf=args.conf_threshold,
        iou=args.iou_threshold,
        device=args.device,
        save_json=args.save_json,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=args.plots,
        verbose=args.verbose,
    )

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)

    raw_metrics = normalize_numeric_mapping(results.results_dict)
    metrics = normalize_metrics(raw_metrics)

    # Detection metrics
    if "map50" in metrics:
        print(f"mAP@0.5: {metrics['map50']:.4f}")
    if "map50_95" in metrics:
        print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    if "precision" in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
    if "recall" in metrics:
        print(f"Recall: {metrics['recall']:.4f}")

    # Calculate F1 score
    if "precision" in metrics and "recall" in metrics:
        precision = metrics["precision"]
        recall = metrics["recall"]
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            print(f"F1-Score: {f1_score:.4f}")

    print("=" * 50 + "\n")

    # Save metrics to JSON
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(raw_metrics, f, indent=2, sort_keys=True)

    speed_ms = normalize_numeric_mapping(getattr(results, "speed", {}))
    report = build_evaluation_report(
        model_path=model_path,
        data_path=data_path,
        metrics=metrics,
        speed_ms=speed_ms,
        evaluation_config={
            "split": args.split,
            "batch_size": args.batch_size,
            "image_size": args.img_size,
            "confidence_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "device": args.device,
        },
        gates=gates,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": importlib.metadata.version("ultralytics"),
        },
    )
    report_file = output_dir / "evaluation-report.json"
    with report_file.open("w") as file:
        json.dump(report, file, indent=2, sort_keys=True)

    print(f"Metrics saved to: {metrics_file}")
    print(f"Reproducible report saved to: {report_file}")
    print(f"Results saved to: {output_dir}")

    # Per-class metrics if available
    if hasattr(results, "ap_class_index"):
        print("\nPer-class metrics:")
        print("-" * 50)
        for i, class_idx in enumerate(results.ap_class_index):
            class_name = results.names[class_idx]
            ap50 = results.ap50[i] if hasattr(results, "ap50") else 0
            ap = results.ap[i] if hasattr(results, "ap") else 0
            print(f"Class '{class_name}':")
            print(f"  AP@0.5: {ap50:.4f}")
            print(f"  AP@0.5:0.95: {ap:.4f}")

    failures = report["quality_gates"]["failures"]
    if failures:
        print("\nEvaluation quality gates failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    if gates.is_configured:
        print("\nAll evaluation quality gates passed.")
    print("\nEvaluation completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
