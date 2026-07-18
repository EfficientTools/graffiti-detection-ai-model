#!/usr/bin/env python3
"""Export trained graffiti weights for the offline Apple client."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "best.pt"
DEFAULT_OUTPUT = (
    ROOT
    / "ios"
    / "GraffitiGuard"
    / "GraffitiGuard"
    / "Resources"
    / "Models"
    / "GraffitiDetector.mlpackage"
)


def validate_labels(labels: Sequence[str]) -> None:
    """Require the one-class output contract consumed by the Apple client."""
    normalized = [label.strip().casefold() for label in labels]
    if normalized != ["graffiti"]:
        raise ValueError(
            "The Apple client requires exactly one model class named 'graffiti'; "
            f"received {list(labels)!r}."
        )


def validate_image_size(image_size: int) -> None:
    if image_size <= 0 or image_size % 32 != 0:
        raise ValueError("Image size must be a positive multiple of 32.")


def export_model(weights: Path, output: Path, image_size: int, force: bool) -> Path:
    if not weights.is_file():
        raise FileNotFoundError(f"Trained weights not found: {weights}")
    if output.suffix != ".mlpackage":
        raise ValueError("Output must use the .mlpackage extension.")
    if output.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output}. Pass --force to replace it.")
    validate_image_size(image_size)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    labels = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    validate_labels(labels)
    if model.task != "detect":
        raise ValueError(f"Expected an object-detection model, received task {model.task!r}.")

    exported = Path(
        model.export(
            format="coreml",
            imgsz=image_size,
            nms=True,
            conf=0.10,
            iou=0.45,
            half=True,
            device="cpu",
        )
    ).resolve()
    if exported.suffix != ".mlpackage" or not exported.is_dir():
        raise RuntimeError(
            "Ultralytics did not produce an ML package. Review its export log and retry."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if exported != output:
        if output.exists():
            shutil.rmtree(output) if output.is_dir() else output.unlink()
        shutil.move(str(exported), str(output))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = export_model(
            weights=args.weights.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
            image_size=args.image_size,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Exported offline detector to {output}")
    print("Run: xcodegen generate --spec ios/GraffitiGuard/project.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
