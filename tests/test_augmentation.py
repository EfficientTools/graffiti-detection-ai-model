"""
Unit tests for data augmentation
"""

import unittest
import numpy as np
import cv2

from src.data.augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
    get_inference_augmentation,
    AUGMENTATION_PRESETS
)


class TestAugmentation(unittest.TestCase):
    """Test augmentation pipelines"""
    
    def setUp(self):
        """Set up test data"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_boxes = np.array([[0.3, 0.3, 0.5, 0.5]])  # normalized
        self.test_labels = np.array([0])
    
    def test_training_augmentation(self):
        """Test training augmentation pipeline"""
        transform = get_training_augmentation(img_size=640)
        
        self.assertIsNotNone(transform)
        
        # Apply transformation
        transformed = transform(
            image=self.test_image,
            bboxes=self.test_boxes,
            class_labels=self.test_labels
        )
        
        self.assertIn('image', transformed)
        self.assertEqual(transformed['image'].shape[0], 640)
        self.assertEqual(transformed['image'].shape[1], 640)
    
    def test_validation_augmentation(self):
        """Test validation augmentation pipeline"""
        transform = get_validation_augmentation(img_size=640)
        
        self.assertIsNotNone(transform)
        
        transformed = transform(
            image=self.test_image,
            bboxes=self.test_boxes,
            class_labels=self.test_labels
        )
        
        self.assertIn('image', transformed)
        self.assertEqual(transformed['image'].shape, (640, 640, 3))
    
    def test_inference_augmentation(self):
        """Test inference augmentation pipeline"""
        transform = get_inference_augmentation(img_size=640)
        
        self.assertIsNotNone(transform)
        
        transformed = transform(image=self.test_image)
        
        self.assertIn('image', transformed)
    
    def test_augmentation_presets(self):
        """Test augmentation preset configurations"""
        self.assertIn('light', AUGMENTATION_PRESETS)
        self.assertIn('medium', AUGMENTATION_PRESETS)
        self.assertIn('heavy', AUGMENTATION_PRESETS)
        
        for preset_name, preset in AUGMENTATION_PRESETS.items():
            self.assertIn('hsv', preset)
            self.assertIn('blur', preset)
            self.assertIn('noise', preset)


if __name__ == '__main__':
    unittest.main()
