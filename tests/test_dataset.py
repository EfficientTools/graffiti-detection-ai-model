"""
Unit tests for dataset loading and preprocessing
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from graffiti_detection.data.augmentation import get_validation_augmentation
from graffiti_detection.data.dataset import GraffitiDataset, create_dataloaders
from graffiti_detection.data.preprocessing import (
    letterbox,
    postprocess_boxes,
    preprocess_image,
)


class TestGraffitiDataset(unittest.TestCase):
    """Test dataset loading and processing"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.img_dir = Path(cls.temp_dir) / "images"
        cls.label_dir = Path(cls.temp_dir) / "labels"
        cls.img_dir.mkdir(parents=True)
        cls.label_dir.mkdir(parents=True)

        # Create dummy image
        cls.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cls.image_path = cls.img_dir / "test_001.jpg"
        cv2.imwrite(str(cls.image_path), cls.test_image)

        # Create dummy label (YOLO format)
        cls.label_path = cls.label_dir / "test_001.txt"
        with open(cls.label_path, "w") as f:
            f.write("0 0.5 0.5 0.3 0.4\n")  # class x_center y_center width height

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        shutil.rmtree(cls.temp_dir)

    def test_dataset_creation(self):
        """Test dataset initialization"""
        image_paths = [str(self.image_path)]
        label_paths = [str(self.label_path)]

        dataset = GraffitiDataset(
            image_paths=image_paths, label_paths=label_paths, img_size=640, augment=False
        )

        self.assertEqual(len(dataset), 1)
        self.assertIsNotNone(dataset)

    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        image_paths = [str(self.image_path)]
        label_paths = [str(self.label_path)]

        dataset = GraffitiDataset(
            image_paths=image_paths, label_paths=label_paths, img_size=640, augment=False
        )

        image, labels = dataset[0]

        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(image.shape[2], 3)  # RGB
        self.assertEqual(labels.shape[1], 5)  # class + bbox coords

    def test_dataset_uses_normalized_boxes_for_augmentation(self):
        dataset = GraffitiDataset(
            image_paths=[str(self.image_path)],
            label_paths=[str(self.label_path)],
            img_size=640,
            augmentation=get_validation_augmentation(640),
        )

        sample = dataset[0]

        self.assertEqual(sample["labels"].shape, (1, 5))
        self.assertTrue(np.all(sample["labels"].numpy()[:, 1:] <= 1.0))
        self.assertTrue(np.all(sample["labels"].numpy()[:, 1:] >= 0.0))

    def test_dataset_augments_images_without_boxes(self):
        empty_label_path = self.label_dir / "empty.txt"
        empty_label_path.write_text("")
        transform = A.Compose(
            [A.HorizontalFlip(p=1.0)],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
            ),
        )
        dataset = GraffitiDataset(
            image_paths=[str(self.image_path)],
            label_paths=[str(empty_label_path)],
            augmentation=transform,
        )

        sample = dataset[0]

        expected = cv2.cvtColor(cv2.imread(str(self.image_path)), cv2.COLOR_BGR2RGB)[:, ::-1]
        expected = cv2.resize(expected, (640, 640)).astype(np.float32) / 255.0
        expected = np.transpose(expected, (2, 0, 1))
        np.testing.assert_allclose(sample["image"].numpy(), expected)
        self.assertEqual(sample["labels"].shape, (0, 5))


class TestPreprocessing(unittest.TestCase):
    """Test image preprocessing functions"""

    def setUp(self):
        """Set up test data"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_letterbox_resize(self):
        """Test letterbox resizing"""
        resized, ratio, (dw, dh) = letterbox(self.test_image, new_shape=640)

        self.assertEqual(resized.shape[0], 640)
        self.assertEqual(resized.shape[1], 640)
        self.assertIsInstance(ratio, tuple)
        self.assertGreater(ratio[0], 0)
        self.assertGreater(ratio[1], 0)

    def test_preprocess_image(self):
        """Test complete image preprocessing"""
        processed = preprocess_image(self.test_image, target_size=640)

        self.assertEqual(processed.shape, (1, 3, 640, 640))
        self.assertLessEqual(processed.max(), 1.0)
        self.assertGreaterEqual(processed.min(), 0.0)

    def test_postprocess_boxes(self):
        """Test bounding box postprocessing"""
        # Mock detection boxes
        boxes = np.array([[100, 100, 200, 200]])  # x1, y1, x2, y2
        img_shape = (480, 640)
        orig_shape = (480, 640)

        processed = postprocess_boxes(boxes, img_shape, orig_shape)

        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape[1], 4)


class TestDataLoaders(unittest.TestCase):
    """Test dataloader creation"""

    def test_dataloader_creation(self):
        """Test dataloader factory function"""
        # This would require proper dataset setup
        # Simplified test
        self.assertTrue(callable(create_dataloaders))


if __name__ == "__main__":
    unittest.main()
