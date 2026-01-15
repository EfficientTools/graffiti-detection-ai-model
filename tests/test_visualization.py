"""
Unit tests for visualization utilities
"""

import unittest
import numpy as np
import cv2

from src.utils.visualization import (
    draw_boxes,
    visualize_detection,
    save_detection_crops
)


class TestVisualization(unittest.TestCase):
    """Test visualization functions"""
    
    def setUp(self):
        """Set up test data"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_boxes = np.array([[100, 100, 300, 300]])
        self.test_scores = np.array([0.85])
        self.test_labels = np.array([0])
        self.class_names = {0: 'graffiti'}
    
    def test_draw_boxes(self):
        """Test drawing bounding boxes"""
        image_with_boxes = draw_boxes(
            self.test_image.copy(),
            self.test_boxes,
            self.test_scores,
            self.test_labels,
            self.class_names
        )
        
        self.assertIsInstance(image_with_boxes, np.ndarray)
        self.assertEqual(image_with_boxes.shape, self.test_image.shape)
        
        # Image should be modified
        self.assertFalse(np.array_equal(image_with_boxes, self.test_image))
    
    def test_draw_boxes_empty(self):
        """Test drawing with no boxes"""
        empty_boxes = np.array([]).reshape(0, 4)
        empty_scores = np.array([])
        empty_labels = np.array([])
        
        image_with_boxes = draw_boxes(
            self.test_image.copy(),
            empty_boxes,
            empty_scores,
            empty_labels,
            self.class_names
        )
        
        # Image should be unchanged
        self.assertTrue(np.array_equal(image_with_boxes, self.test_image))
    
    def test_save_detection_crops(self):
        """Test saving detection crops"""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            save_detection_crops(
                self.test_image,
                self.test_boxes,
                self.test_scores,
                output_dir
            )
            
            # Check if crops were saved
            saved_files = list(output_dir.glob("*.jpg"))
            self.assertEqual(len(saved_files), len(self.test_boxes))


class TestVisualizationEdgeCases(unittest.TestCase):
    """Test edge cases in visualization"""
    
    def test_draw_boxes_out_of_bounds(self):
        """Test drawing boxes that extend outside image"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Box extends outside image
        boxes = np.array([[-50, -50, 150, 150]])
        scores = np.array([0.9])
        labels = np.array([0])
        class_names = {0: 'graffiti'}
        
        # Should not crash
        result = draw_boxes(test_image.copy(), boxes, scores, labels, class_names)
        self.assertIsNotNone(result)
    
    def test_visualize_with_high_confidence(self):
        """Test visualization with various confidence levels"""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200]])
        
        # High confidence
        scores = np.array([0.99])
        labels = np.array([0])
        class_names = {0: 'graffiti'}
        
        result = draw_boxes(test_image.copy(), boxes, scores, labels, class_names)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
