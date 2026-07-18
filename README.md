# Graffiti Detection AI Model

<p align="center">
  <img src="https://raw.githubusercontent.com/EfficientTools/graffiti-detection-ai-model/main/assets/graffiti-detection-logo.svg" alt="Graffiti Detection AI logo" width="780" />
</p>

<p align="center">
  <a href="https://github.com/EfficientTools/graffiti-detection-ai-model/actions/workflows/ci.yml"><img src="https://github.com/EfficientTools/graffiti-detection-ai-model/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/graffiti-detection-ai-model/"><img src="https://img.shields.io/pypi/v/graffiti-detection-ai-model" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/graffiti-detection-ai-model/"><img src="https://img.shields.io/pypi/pyversions/graffiti-detection-ai-model" alt="Supported Python versions" /></a>
  <a href="https://github.com/EfficientTools/graffiti-detection-ai-model/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License" /></a>
</p>

YOLOv8-based graffiti detection for Python applications, model training, batch inference, camera monitoring, alerts, and API integration.

## App for iPhone and iPad

Prefer a ready-to-use mobile experience? Graffiti Guard runs private, offline detection directly on iPhone and iPad, with an inspection-focused iPad interface and shareable results.

<p align="center">
  <a href="https://apps.apple.com/app/id6792218806">
    <img src="https://developer.apple.com/assets/elements/badges/download-on-the-app-store.svg" alt="Download Graffiti Guard on the App Store" height="54" />
  </a>
</p>

<p align="center"><sub>Version 1.0 is awaiting App Review. The link becomes active when Apple publishes the app.</sub></p>

## Why I Built It

I built this project because I hate seeing Melbourne and other cities damaged by unwanted graffiti. Earlier image review can help councils and maintenance teams prioritize work before cleanup costs and repeated surface damage escalate.

## What Makes It Useful

- A small Python API for image inference
- End-to-end scripts for dataset preparation, training, evaluation, and video inference
- Multi-camera monitoring with email, SMS, webhook, Discord, Slack, and OneSignal alerts
- FastAPI and Docker deployment options
- Private, offline Core ML inference on iPhone and iPad
- Shareable, human-reviewable inspection summaries on iPad
- Street-scene augmentation for poor lighting, weather, perspective, and CCTV artifacts

Training weights are not bundled. The Apple client includes an MIT-licensed Core ML detector for offline use; see [Third-Party Notices](https://github.com/EfficientTools/graffiti-detection-ai-model/blob/main/THIRD_PARTY_NOTICES.md).

## Installation

Python 3.9 or newer is required.

```bash
pip install graffiti-detection-ai-model
```

Install every runtime feature for training, alerts, and the API:

```bash
pip install "graffiti-detection-ai-model[all]"
```

For development from source:

```bash
git clone https://github.com/EfficientTools/graffiti-detection-ai-model.git
cd graffiti-detection-ai-model
python -m venv venv
source venv/bin/activate
python -m pip install -e ".[dev]"
```

## Python Usage

Run a trained model against the included example image:

<p align="center">
  <img src="https://raw.githubusercontent.com/EfficientTools/graffiti-detection-ai-model/main/assets/examples/graffiti-wall.jpg" alt="Graffiti painted on a city wall" width="760" />
</p>

```python
from graffiti_detection import GraffitiDetector

detector = GraffitiDetector("models/best.pt", conf_threshold=0.25)
example_image = (
    "https://raw.githubusercontent.com/EfficientTools/"
    "graffiti-detection-ai-model/main/assets/examples/graffiti-wall.jpg"
)
detections = detector.predict(example_image)

for detection in detections:
    print(
        f"{detection['class_name']}: {detection['confidence']:.1%} "
        f"at {detection['box']}"
    )
```

Replace `example_image` with a local image path for your own detection. Each result contains `class_id`, `class_name`, `confidence`, and an `xyxy` box.

## Command Line

```bash
# Prepare a YOLO-format dataset
python scripts/prepare_dataset.py --data-dir data/raw --output-dir data --validate --copy

# Train and evaluate
python scripts/train.py --data configs/dataset.yaml --model yolov8n --epochs 100
python scripts/evaluate.py --model models/best.pt --data configs/dataset.yaml --split test

# Run image, directory, video, or webcam inference
python scripts/inference.py --model models/best.pt --source image.jpg
```

For camera monitoring, copy the example camera and alert files, insert your own credentials locally, then run:

```bash
python scripts/multi_camera_surveillance.py \
  --model models/best.pt \
  --cameras configs/cameras.json \
  --alert-config configs/alerts.json
```

Start the API with:

```bash
MODEL_PATH=models/best.pt uvicorn api.graffiti_detector:app --host 0.0.0.0 --port 8000
```

See [DEPLOYMENT.md](https://github.com/EfficientTools/graffiti-detection-ai-model/blob/main/DEPLOYMENT.md) for Docker instructions.

## iPhone and iPad

The universal SwiftUI app in [`ios/GraffitiGuard`](https://github.com/EfficientTools/graffiti-detection-ai-model/tree/main/ios/GraffitiGuard) runs its bundled Core ML detector entirely on-device and overlays likely graffiti regions. Its iPad workflow supports drag-and-drop intake, adjustable confidence, and shareable inspection summaries. To replace it with your own trained weights, export the model and regenerate the Xcode project:

```bash
python -m pip install -e ".[apple]"
python scripts/export_coreml.py --weights models/best.pt
xcodegen generate --spec ios/GraffitiGuard/project.yml
```

See the app [README](https://github.com/EfficientTools/graffiti-detection-ai-model/blob/main/ios/GraffitiGuard/README.md) for Xcode and TestFlight instructions.

## Development

```bash
ruff check .
pytest tests/ -m "not gpu"
python -m build
twine check dist/*
```

## Author

<div align="center">

[![Pierre-Henry Soria](https://s.gravatar.com/avatar/a210fe61253c43c869d71eaed0e90149?s=160 "Pierre-Henry Soria")](https://pierrehenry.dev)

**Pierre-Henry Soria**

AI software engineer and consultant building practical computer vision systems for real-world problems.

[Work with Pierre-Henry](https://pierrehenry.dev) | [GitHub](https://github.com/pH-7) | [Bluesky](https://bsky.app/profile/pierrehenry.dev) | [X](https://x.com/phenrysay) | [Support](https://ko-fi.com/phenry)

</div>

## License

Distributed under the [MIT License](https://github.com/EfficientTools/graffiti-detection-ai-model/blob/main/LICENSE).

Apple, the Apple logo, App Store, iPhone, and iPad are trademarks of Apple Inc., registered in the U.S. and other countries.

## Responsible Use

Comply with local privacy, surveillance, and data-retention laws. Human review should remain part of any enforcement or response process.
