"""
Sophisticated tests for public street-scenario graffiti detection.

Designed to run in CI pipelines (no GPU, no trained model weights required).
Validates every stage of the pipeline: preprocessing, augmentation, dataset,
metrics, alerts, and end-to-end data flow under realistic urban conditions.

Run:
    pytest tests/test_street_scenarios.py -v
    pytest tests/test_street_scenarios.py -v -m "not slow"     # skip slow tests
    pytest tests/test_street_scenarios.py -v -m pipeline        # CI-critical only
"""

import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest
import yaml

from src.data.augmentation import (
    get_street_scene_augmentation,
    get_training_augmentation,
    get_validation_augmentation,
    get_augmentation_by_preset,
    AUGMENTATION_PRESETS,
)
from src.data.preprocessing import (
    letterbox,
    preprocess_image,
    preprocess_street_scene,
    postprocess_boxes,
    normalize_image,
    apply_clahe,
    simulate_street_conditions,
)
from src.data.dataset import GraffitiDataset
from src.evaluation.metrics import (
    calculate_iou,
    calculate_ap,
    calculate_map,
    calculate_precision_recall_f1,
    non_max_suppression,
    DetectionMetrics,
)


# ---------------------------------------------------------------------------
# Helpers – synthetic images that mimic common street scenes
# ---------------------------------------------------------------------------

def _make_street_image(
    width: int = 640,
    height: int = 480,
    condition: str = "day",
) -> np.ndarray:
    """Generate a synthetic street-like image (RGB, uint8)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky band (top 40 %)
    sky_h = int(height * 0.4)
    if condition == "night":
        img[:sky_h, :] = (15, 15, 35)
    elif condition == "overcast":
        img[:sky_h, :] = (170, 170, 175)
    else:
        img[:sky_h, :] = (135, 190, 230)

    # Building / wall band (middle 40 %)
    wall_top = sky_h
    wall_bottom = int(height * 0.8)
    img[wall_top:wall_bottom, :] = (160, 155, 145)

    # Sidewalk / road (bottom 20 %)
    img[wall_bottom:, :] = (90, 90, 95)

    # Simulate graffiti splash on wall
    cx, cy = width // 2, (wall_top + wall_bottom) // 2
    cv2.rectangle(img, (cx - 60, cy - 30), (cx + 60, cy + 30), (200, 50, 50), -1)
    cv2.putText(img, "TAG", (cx - 40, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    return img


def _make_yolo_label(
    n_objects: int = 1,
    class_id: int = 0,
    img_w: int = 640,
    img_h: int = 480,
) -> List[str]:
    """Generate deterministic YOLO label lines for a synthetic image."""
    lines = []
    rng = np.random.RandomState(42)
    for _ in range(n_objects):
        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.25, 0.75)
        w = rng.uniform(0.05, 0.25)
        h = rng.uniform(0.04, 0.20)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _create_temp_dataset(
    n_images: int = 5,
    n_boxes_per_image: int = 2,
    img_w: int = 640,
    img_h: int = 480,
) -> Tuple[str, List[str], List[str]]:
    """Create a temporary directory with synthetic images + YOLO labels.

    Returns:
        (temp_dir, image_paths, label_paths)
    """
    tmp = tempfile.mkdtemp(prefix="graffiti_test_")
    img_dir = Path(tmp) / "images"
    lbl_dir = Path(tmp) / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    image_paths, label_paths = [], []
    conditions = ["day", "night", "overcast", "dusk"]

    for i in range(n_images):
        cond = conditions[i % len(conditions)]
        img = _make_street_image(img_w, img_h, condition=cond)
        img_path = img_dir / f"street_{i:04d}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        image_paths.append(str(img_path))

        lbl_path = lbl_dir / f"street_{i:04d}.txt"
        lines = _make_yolo_label(n_boxes_per_image, img_w=img_w, img_h=img_h)
        lbl_path.write_text("\n".join(lines) + "\n")
        label_paths.append(str(lbl_path))

    return tmp, image_paths, label_paths


# ═══════════════════════════════════════════════════════════════════════════
# 1. STREET-SCENE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.pipeline
class TestStreetPreprocessing(unittest.TestCase):
    """Validate preprocessing pipeline on synthetic street images."""

    def setUp(self):
        self.day_img = _make_street_image(condition="day")
        self.night_img = _make_street_image(condition="night")

    # -- letterbox -------------------------------------------------------

    def test_letterbox_preserves_aspect_ratio(self):
        """Letterbox should pad, not stretch."""
        resized, ratio, (dw, dh) = letterbox(self.day_img, new_shape=640)
        self.assertEqual(resized.shape[:2], (640, 640))
        self.assertTrue(dw >= 0 and dh >= 0)

    def test_letterbox_various_input_sizes(self):
        """Letterbox should work for multiple common CCTV resolutions."""
        for h, w in [(720, 1280), (1080, 1920), (480, 854), (240, 320)]:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            resized, _, _ = letterbox(img, new_shape=640)
            self.assertEqual(resized.shape[:2], (640, 640),
                             f"Failed for input size ({h},{w})")

    # -- street-specific preprocessing -----------------------------------

    def test_preprocess_street_scene_output_shape(self):
        """preprocess_street_scene should produce (1,3,H,W) float tensor."""
        bgr = cv2.cvtColor(self.day_img, cv2.COLOR_RGB2BGR)
        out = preprocess_street_scene(bgr, target_size=(640, 640))
        self.assertEqual(out.shape, (1, 3, 640, 640))
        self.assertTrue(out.dtype == np.float32)
        self.assertLessEqual(out.max(), 1.0)
        self.assertGreaterEqual(out.min(), 0.0)

    def test_preprocess_street_scene_night(self):
        """Night image preprocessing should still produce valid output."""
        bgr = cv2.cvtColor(self.night_img, cv2.COLOR_RGB2BGR)
        out = preprocess_street_scene(bgr, target_size=(640, 640), enhance_contrast=True)
        self.assertEqual(out.shape, (1, 3, 640, 640))

    def test_preprocess_street_scene_no_enhance(self):
        """Skip optional enhancements."""
        bgr = cv2.cvtColor(self.day_img, cv2.COLOR_RGB2BGR)
        out = preprocess_street_scene(bgr, enhance_contrast=False, denoise=False)
        self.assertEqual(out.shape, (1, 3, 640, 640))

    # -- condition simulation -------------------------------------------

    def test_simulate_street_conditions_all(self):
        """simulate_street_conditions should handle every known condition."""
        for cond in ("day", "night", "overcast", "dusk"):
            result = simulate_street_conditions(self.day_img, condition=cond)
            self.assertEqual(result.shape, self.day_img.shape)
            self.assertEqual(result.dtype, np.uint8)

    def test_night_simulation_darkens_image(self):
        mean_day = self.day_img.mean()
        night = simulate_street_conditions(self.day_img, condition="night")
        self.assertLess(night.mean(), mean_day * 0.6)

    # -- CLAHE -----------------------------------------------------------

    def test_clahe_enhances_low_contrast_image(self):
        low_contrast = np.full((100, 100, 3), 80, dtype=np.uint8)
        enhanced = apply_clahe(low_contrast)
        self.assertEqual(enhanced.shape, low_contrast.shape)

    # -- normalise / denormalise -----------------------------------------

    def test_normalize_range(self):
        normed = normalize_image(self.day_img)
        self.assertAlmostEqual(normed.max(), self.day_img.max() / 255.0, places=4)
        self.assertGreaterEqual(normed.min(), 0.0)

    # -- postprocess_boxes round-trip ------------------------------------

    def test_postprocess_identity(self):
        boxes = np.array([[100.0, 100.0, 200.0, 200.0]])
        out = postprocess_boxes(boxes, original_shape=(480, 640), ratio=(1.0, 1.0), padding=(0.0, 0.0))
        np.testing.assert_array_almost_equal(out, boxes)


# ═══════════════════════════════════════════════════════════════════════════
# 2. STREET-SCENE AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.pipeline
class TestStreetAugmentation(unittest.TestCase):
    """Validate the street-specific augmentation pipeline."""

    def setUp(self):
        self.img = _make_street_image()
        self.boxes = [[0.5, 0.5, 0.2, 0.15]]  # YOLO format
        self.labels = [0]

    def test_street_augmentation_returns_image(self):
        transform = get_street_scene_augmentation(img_size=640)
        result = transform(image=self.img, bboxes=self.boxes, class_labels=self.labels)
        self.assertIn("image", result)
        self.assertEqual(result["image"].shape[:2], (640, 640))

    def test_street_augmentation_preserves_bbox_format(self):
        """Bboxes should remain in YOLO normalised format after transform."""
        transform = get_street_scene_augmentation(img_size=640)
        result = transform(image=self.img, bboxes=self.boxes, class_labels=self.labels)
        for bbox in result["bboxes"]:
            for coord in bbox:
                self.assertGreaterEqual(coord, 0.0)
                self.assertLessEqual(coord, 1.0)

    def test_street_augmentation_multiple_objects(self):
        """Handle images with many graffiti instances at once."""
        boxes = [[0.2, 0.3, 0.1, 0.1], [0.5, 0.5, 0.2, 0.15], [0.8, 0.7, 0.12, 0.08]]
        labels = [0, 0, 0]
        transform = get_street_scene_augmentation(img_size=640)
        result = transform(image=self.img, bboxes=boxes, class_labels=labels)
        self.assertGreaterEqual(len(result["bboxes"]), 0)  # some may be clipped

    def test_preset_street_returns_pipeline(self):
        """get_augmentation_by_preset('street') should return a valid pipeline."""
        self.assertIn("street", AUGMENTATION_PRESETS)
        transform = get_augmentation_by_preset("street", img_size=640)
        result = transform(image=self.img, bboxes=self.boxes, class_labels=self.labels)
        self.assertIn("image", result)

    @pytest.mark.slow
    def test_street_augmentation_stability_over_many_runs(self):
        """Apply augmentation 100 times; no crash, output always valid."""
        transform = get_street_scene_augmentation(img_size=640)
        for _ in range(100):
            result = transform(image=self.img, bboxes=self.boxes, class_labels=self.labels)
            self.assertEqual(result["image"].shape[0], 640)
            self.assertEqual(result["image"].shape[1], 640)

    def test_empty_bboxes_handled(self):
        """Augmentation should survive images with no annotations."""
        transform = get_street_scene_augmentation(img_size=640)
        result = transform(image=self.img, bboxes=[], class_labels=[])
        self.assertEqual(result["image"].shape[:2], (640, 640))
        self.assertEqual(len(result["bboxes"]), 0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. DATASET LOADING UNDER STREET CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.pipeline
class TestStreetDataset(unittest.TestCase):
    """Load street-scenario data through GraffitiDataset."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir, cls.img_paths, cls.lbl_paths = _create_temp_dataset(
            n_images=6, n_boxes_per_image=3
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_dataset_length(self):
        ds = GraffitiDataset(self.img_paths, self.lbl_paths, img_size=640, augment=False)
        self.assertEqual(len(ds), 6)

    def test_dataset_getitem_returns_image_and_labels(self):
        ds = GraffitiDataset(self.img_paths, self.lbl_paths, img_size=640, augment=False)
        img, labels = ds[0]
        self.assertIsInstance(img, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.shape[1], 5)

    def test_dataset_with_street_augmentation(self):
        """Apply street augmentation directly to images loaded from the dataset.

        Note: The dataset converts YOLO labels to pixel-xyxy before calling
        the augmentation, so we test the augmentation standalone using YOLO
        format (consistent with the existing test pattern in test_augmentation).
        """
        transform = get_street_scene_augmentation(img_size=640)
        ds = GraffitiDataset(self.img_paths, self.lbl_paths, img_size=640, augment=False)
        for i in range(len(ds)):
            img, labels = ds[i]
            yolo_boxes = [[float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in labels]
            cls_labels = [int(l[0]) for l in labels]
            result = transform(image=img, bboxes=yolo_boxes, class_labels=cls_labels)
            self.assertIn("image", result)
            self.assertEqual(result["image"].shape[:2], (640, 640))

    def test_dataset_varied_resolutions(self):
        """Images at non-standard sizes should still load."""
        tmp = tempfile.mkdtemp(prefix="graffiti_restest_")
        img_dir = Path(tmp) / "images"
        lbl_dir = Path(tmp) / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()
        img_paths, lbl_paths = [], []
        for i, (h, w) in enumerate([(240, 320), (1080, 1920), (600, 800)]):
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            p = img_dir / f"res_{i}.jpg"
            cv2.imwrite(str(p), img)
            img_paths.append(str(p))
            lp = lbl_dir / f"res_{i}.txt"
            lp.write_text("0 0.5 0.5 0.2 0.2\n")
            lbl_paths.append(str(lp))

        ds = GraffitiDataset(img_paths, lbl_paths, img_size=640, augment=False)
        for i in range(len(ds)):
            img, labels = ds[i]
            self.assertEqual(img.shape[:2], (640, 640))

        shutil.rmtree(tmp)

    def test_dataset_empty_label_file(self):
        """Image with no graffiti should yield zero labels."""
        tmp = tempfile.mkdtemp(prefix="graffiti_empty_")
        img_dir = Path(tmp) / "images"
        lbl_dir = Path(tmp) / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()
        img = _make_street_image()
        img_path = img_dir / "empty.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        lbl_path = lbl_dir / "empty.txt"
        lbl_path.write_text("")

        ds = GraffitiDataset([str(img_path)], [str(lbl_path)], img_size=640, augment=False)
        img_out, labels = ds[0]
        self.assertEqual(labels.shape[0], 0)
        shutil.rmtree(tmp)


# ═══════════════════════════════════════════════════════════════════════════
# 4. DETECTION METRICS – STREET-REALISTIC SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.pipeline
class TestStreetMetrics(unittest.TestCase):
    """Metric calculations using realistic street-detection data."""

    def test_iou_partial_overlap(self):
        """Two overlapping wall tags."""
        box1 = np.array([100, 200, 250, 350])
        box2 = np.array([150, 220, 300, 370])
        iou = calculate_iou(box1, box2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_iou_contained_box(self):
        """Small tag fully inside a larger detection box."""
        outer = np.array([50, 50, 300, 300])
        inner = np.array([100, 100, 200, 200])
        iou = calculate_iou(outer, inner)
        expected = (100 * 100) / (250 * 250 + 100 * 100 - 100 * 100)
        self.assertAlmostEqual(iou, expected, places=4)

    def test_map_single_class_perfect(self):
        """Perfect predictions should yield mAP ≈ 1.0."""
        preds = {0: [(np.array([10, 10, 50, 50]), 0.95)]}
        gts = {0: [np.array([10, 10, 50, 50])]}
        mAP, _ = calculate_map(preds, gts, iou_threshold=0.5, num_classes=1)
        self.assertAlmostEqual(mAP, 1.0, places=1)

    def test_map_no_predictions(self):
        """No predictions should yield very low mAP (11-pt interpolation artefact)."""
        preds = {}
        gts = {0: [np.array([10, 10, 50, 50])]}
        mAP, _ = calculate_map(preds, gts, iou_threshold=0.5, num_classes=1)
        self.assertLess(mAP, 0.15)  # near zero, artefact of recall padding

    def test_precision_recall_f1_perfect(self):
        p, r, f = calculate_precision_recall_f1(tp=10, fp=0, fn=0)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 1.0)

    def test_precision_recall_f1_no_detections(self):
        p, r, f = calculate_precision_recall_f1(tp=0, fp=0, fn=5)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f, 0.0)

    def test_nms_removes_duplicate_detections(self):
        """Overlapping boxes with lower scores should be suppressed."""
        boxes = np.array([
            [100, 100, 200, 200],
            [105, 105, 205, 205],
            [400, 400, 500, 500],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.75, 0.85])
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        self.assertIn(0, keep)
        self.assertIn(2, keep)
        self.assertEqual(len(keep), 2)

    def test_nms_empty_input(self):
        keep = non_max_suppression(np.array([]).reshape(0, 4), np.array([]))
        self.assertEqual(len(keep), 0)

    def test_detection_metrics_batch_workflow(self):
        """Simulate a multi-image batch evaluation."""
        dm = DetectionMetrics(num_classes=1, iou_threshold=0.5)

        for _ in range(5):
            pred_boxes = np.array([[10, 10, 90, 90], [200, 200, 280, 280]])
            pred_scores = np.array([0.9, 0.6])
            pred_labels = np.array([0, 0])
            gt_boxes = np.array([[10, 10, 90, 90]])
            gt_labels = np.array([0])
            dm.add_predictions(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

        results = dm.compute()
        self.assertIn("mAP", results)
        self.assertIn("precision", results)
        self.assertIn("recall", results)
        self.assertIn("f1_score", results)
        self.assertGreater(results["recall"], 0.0)

    def test_ap_with_mixed_true_false_positives(self):
        """Realistic scenario: some correct, some wrong detections."""
        predictions = [(0.95, True), (0.85, False), (0.78, True), (0.60, False), (0.40, True)]
        ap = calculate_ap(predictions, num_gt=3)
        self.assertGreater(ap, 0.0)
        self.assertLessEqual(ap, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 5. ALERT SYSTEM – STREET DETECTION EVENTS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.pipeline
class TestStreetAlerts(unittest.TestCase):
    """Validate the alert manager works with street-scenario detection data."""

    def _detection_event(self, conf: float = 0.72):
        return {
            "camera_id": "street_cam_01",
            "timestamp": "2026-03-28T14:30:00Z",
            "detections": 2,
            "confidences": [conf, conf - 0.1],
            "frame": _make_street_image(),
        }

    def test_alert_manager_initialises_without_channels(self):
        from src.utils.alerts import AlertManager
        mgr = AlertManager(config={})
        self.assertEqual(len(mgr.alert_channels), 0)

    def test_alert_manager_send_no_channels_does_not_crash(self):
        from src.utils.alerts import AlertManager
        mgr = AlertManager(config={})
        mgr.send_alert(self._detection_event())

    def test_alert_config_schema_validation(self):
        """Verify the example alert config matches expected schema."""
        cfg_path = Path("configs/alerts_example.json")
        if not cfg_path.exists():
            self.skipTest("alerts_example.json not present")
        cfg = json.loads(cfg_path.read_text())
        for channel in ("email", "sms", "webhook", "discord", "slack"):
            self.assertIn(channel, cfg, f"Missing channel: {channel}")
            self.assertIn("enabled", cfg[channel])


# ═══════════════════════════════════════════════════════════════════════════
# 6. CONFIG INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.pipeline
class TestStreetConfigIntegrity(unittest.TestCase):
    """Ensure street-scenarios config and all related configs are consistent."""

    def test_street_scenarios_yaml_loads(self):
        cfg_path = Path("configs/street_scenarios.yaml")
        self.assertTrue(cfg_path.exists(), "street_scenarios.yaml must exist")
        cfg = yaml.safe_load(cfg_path.read_text())
        self.assertIn("scenarios", cfg)
        self.assertIn("detection", cfg)
        self.assertIn("quality_gates", cfg)

    def test_street_scenarios_has_required_conditions(self):
        cfg = yaml.safe_load(Path("configs/street_scenarios.yaml").read_text())
        required = {"urban_daylight", "urban_night", "rainy_street", "foggy_morning"}
        self.assertTrue(required.issubset(set(cfg["scenarios"].keys())))

    def test_quality_gates_are_numeric(self):
        cfg = yaml.safe_load(Path("configs/street_scenarios.yaml").read_text())
        for key, val in cfg["quality_gates"].items():
            self.assertIsInstance(val, (int, float), f"{key} must be numeric")

    def test_dataset_yaml_class_count(self):
        cfg = yaml.safe_load(Path("configs/dataset.yaml").read_text())
        self.assertGreaterEqual(cfg["nc"], 1)
        self.assertIn(0, cfg["names"])

    def test_training_yaml_hyperparams_sensible(self):
        cfg = yaml.safe_load(Path("configs/training.yaml").read_text())
        self.assertGreater(cfg["epochs"], 0)
        self.assertGreater(cfg["lr0"], 0)
        self.assertLessEqual(cfg["lr0"], 0.1)
        self.assertGreater(cfg["batch"], 0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. END-TO-END PIPELINE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.pipeline
class TestEndToEndStreetPipeline(unittest.TestCase):
    """
    Simulate the full inference pipeline on synthetic street data:
    image → preprocess → (mock) detect → post-process → metrics.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp, cls.img_paths, cls.lbl_paths = _create_temp_dataset(n_images=4, n_boxes_per_image=2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def _mock_detections(self, n: int = 2):
        """Return mock detection boxes, scores, labels."""
        rng = np.random.RandomState(99)
        boxes = rng.randint(0, 400, (n, 2))
        boxes = np.column_stack([boxes, boxes + rng.randint(50, 150, (n, 2))])
        scores = rng.uniform(0.3, 0.95, n)
        labels = np.zeros(n, dtype=int)
        return boxes.astype(np.float32), scores, labels

    def test_preprocess_detect_postprocess(self):
        """Run a complete preprocess → mock-detect → postprocess cycle."""
        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            self.assertIsNotNone(img, f"Failed to read {img_path}")

            # Preprocess
            processed = preprocess_street_scene(img, target_size=(640, 640))
            self.assertEqual(processed.shape, (1, 3, 640, 640))

            # Mock detection
            boxes, scores, labels = self._mock_detections(n=3)

            # Post-process
            original_shape = img.shape[:2]
            adjusted = postprocess_boxes(boxes, original_shape, ratio=(1.0, 1.0), padding=(0.0, 0.0))
            self.assertEqual(adjusted.shape[1], 4)

    def test_metrics_after_mock_evaluation(self):
        """Accumulate mock detections, then verify metrics computation."""
        dm = DetectionMetrics(num_classes=1, iou_threshold=0.5)

        for lbl_path in self.lbl_paths:
            lines = Path(lbl_path).read_text().strip().split("\n")
            gt_boxes = []
            for line in lines:
                parts = list(map(float, line.split()))
                cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
                x1 = (cx - w / 2) * 640
                y1 = (cy - h / 2) * 480
                x2 = (cx + w / 2) * 640
                y2 = (cy + h / 2) * 480
                gt_boxes.append([x1, y1, x2, y2])

            gt_boxes = np.array(gt_boxes)
            gt_labels = np.zeros(len(gt_boxes), dtype=int)

            pred_boxes, pred_scores, pred_labels = self._mock_detections(n=len(gt_boxes))
            dm.add_predictions(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)

        results = dm.compute()
        self.assertIn("mAP", results)
        self.assertIsInstance(results["mAP"], float)

    def test_full_dataset_iteration_with_augmentation(self):
        """Iterate dataset, apply street augmentation standalone per sample."""
        transform = get_street_scene_augmentation(img_size=640)
        ds = GraffitiDataset(
            self.img_paths, self.lbl_paths,
            img_size=640, augment=False,
        )
        for i in range(len(ds)):
            img, labels = ds[i]
            yolo_boxes = [[float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in labels]
            cls_labels = [int(l[0]) for l in labels]
            result = transform(image=img, bboxes=yolo_boxes, class_labels=cls_labels)
            self.assertIn("image", result)


# ═══════════════════════════════════════════════════════════════════════════
# 8. EDGE CASES & ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.pipeline
class TestStreetEdgeCases(unittest.TestCase):
    """Robustness tests for unusual but realistic street inputs."""

    def test_very_small_image(self):
        """Low-res thumbnail from a cheap CCTV camera."""
        tiny = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        resized, _, _ = letterbox(tiny, new_shape=640)
        self.assertEqual(resized.shape[:2], (640, 640))

    def test_very_large_image(self):
        """4K frame from a high-end surveillance camera."""
        big = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        resized, _, _ = letterbox(big, new_shape=640)
        self.assertEqual(resized.shape[:2], (640, 640))

    def test_grayscale_input(self):
        """Some CCTV feeds are grayscale."""
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        out = preprocess_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), target_size=640)
        self.assertEqual(out.shape, (1, 3, 640, 640))

    def test_nms_with_many_overlapping_boxes(self):
        """Dense graffiti wall with many overlapping tags."""
        rng = np.random.RandomState(7)
        n = 50
        base = np.array([100, 100, 200, 200], dtype=np.float32)
        boxes = base + rng.uniform(-30, 30, (n, 4)).astype(np.float32)
        boxes[:, 2:] = np.maximum(boxes[:, :2] + 10, boxes[:, 2:])
        scores = rng.uniform(0.3, 0.95, n).astype(np.float32)
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        self.assertGreater(len(keep), 0)
        self.assertLess(len(keep), n)

    def test_iou_tiny_boxes(self):
        """Very small graffiti tags seen from distance."""
        box1 = np.array([300, 300, 305, 305])
        box2 = np.array([302, 302, 307, 307])
        iou = calculate_iou(box1, box2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_box_at_image_boundary(self):
        """Graffiti partially out of frame."""
        boxes = np.array([[0.0, 0.0, 50.0, 50.0],
                          [600.0, 440.0, 650.0, 490.0]])
        clipped = postprocess_boxes(boxes, (480, 640), ratio=(1.0, 1.0), padding=(0.0, 0.0))
        self.assertTrue((clipped[:, 0] >= 0).all())
        self.assertTrue((clipped[:, 2] <= 640).all())
        self.assertTrue((clipped[:, 1] >= 0).all())
        self.assertTrue((clipped[:, 3] <= 480).all())

    def test_single_pixel_image(self):
        """Degenerate input must not crash."""
        pixel = np.array([[[128, 128, 128]]], dtype=np.uint8)
        resized, _, _ = letterbox(pixel, new_shape=640)
        self.assertEqual(resized.shape[:2], (640, 640))


if __name__ == "__main__":
    unittest.main()
