# Graffiti Detection Model Card

## Purpose

Graffiti Guard performs first-pass, single-class detection of likely graffiti regions in street images. It supports human review and maintenance triage; it is not an enforcement system and does not predict future vandalism.

## Model Artifacts

- Python users supply their own YOLO-compatible training weights.
- The Apple app bundles an MIT-licensed, one-class FP16 Core ML detector with non-maximum suppression.
- The Apple model accepts a 640-pixel letterboxed image and returns graffiti boxes with confidence scores.
- Provenance, checksum, and license details are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Evidence

Every Python evaluation can produce `evaluation-report.json` containing:

- SHA-256 identities for the exact model and dataset configuration
- mAP@0.5, mAP@0.5:0.95, precision, and recall
- preprocessing, inference, and postprocessing latency
- Python, platform, PyTorch, and Ultralytics versions
- pass/fail results for deployment-specific quality gates

Apple CI loads the model shipped inside the app, runs the bundled sample at the default 25% threshold, checks for the expected detection, validates confidence and box bounds, then applies conservative model-preparation, first-inference, and repeat-inference regression limits.

No universal accuracy claim is published because this repository does not include a representative, held-out deployment dataset. Compare models only on the same versioned dataset and hardware.

## Evaluation

Set thresholds for the intended deployment, then make regressions fail the release:

```bash
python scripts/evaluate.py \
  --model models/best.pt \
  --data configs/dataset.yaml \
  --split test \
  --min-map50 0.80 \
  --min-map50-95 0.50 \
  --min-precision 0.75 \
  --min-recall 0.75 \
  --max-inference-ms 50
```

Measure end-to-end Python inference separately on target hardware with `GraffitiDetector.benchmark(...)`.

## Limitations

- Legal street art and unwanted graffiti may look identical to the model.
- Small, distant, occluded, reflective, blurred, or poorly lit markings are harder to detect.
- Murals, signage, textured walls, and painted vehicles may cause false positives.
- A result below the selected threshold is not proof that graffiti is absent.
- Performance must be validated against local camera positions, weather, surfaces, and graffiti styles.

Human verification is required before maintenance, reporting, or enforcement decisions.
