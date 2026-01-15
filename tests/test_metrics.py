"""
Unit tests for evaluation metrics
"""

import unittest
import numpy as np

from src.evaluation.metrics import (
    calculate_iou,
    calculate_ap,
    calculate_map,
    DetectionMetrics
)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics calculations"""
    
    def test_calculate_iou(self):
        """Test IoU calculation"""
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])
        
        iou = calculate_iou(box1, box2)
        
        self.assertIsInstance(iou, float)
        self.assertGreaterEqual(iou, 0.0)
        self.assertLessEqual(iou, 1.0)
        
        # Perfect overlap
        iou_perfect = calculate_iou(box1, box1)
        self.assertAlmostEqual(iou_perfect, 1.0)
        
        # No overlap
        box3 = np.array([200, 200, 300, 300])
        iou_none = calculate_iou(box1, box3)
        self.assertEqual(iou_none, 0.0)
    
    def test_calculate_ap(self):
        """Test Average Precision calculation"""
        # Mock predictions and ground truth
        pred_boxes = np.array([
            [10, 10, 90, 90],
            [15, 15, 95, 95],
            [200, 200, 280, 280]
        ])
        pred_scores = np.array([0.9, 0.8, 0.7])
        pred_labels = np.array([0, 0, 0])
        
        gt_boxes = np.array([
            [10, 10, 90, 90]
        ])
        gt_labels = np.array([0])
        
        ap = calculate_ap(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels,
            iou_threshold=0.5
        )
        
        self.assertIsInstance(ap, float)
        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)
    
    def test_detection_metrics_class(self):
        """Test DetectionMetrics class"""
        metrics = DetectionMetrics(num_classes=1, iou_threshold=0.5)
        
        # Add predictions
        pred_boxes = np.array([[10, 10, 90, 90]])
        pred_scores = np.array([0.9])
        pred_labels = np.array([0])
        
        gt_boxes = np.array([[10, 10, 90, 90]])
        gt_labels = np.array([0])
        
        metrics.add_predictions(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels
        )
        
        # Compute metrics
        results = metrics.compute()
        
        self.assertIn('mAP', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('f1_score', results)


class TestMetricEdgeCases(unittest.TestCase):
    """Test edge cases in metrics"""
    
    def test_empty_predictions(self):
        """Test metrics with no predictions"""
        metrics = DetectionMetrics(num_classes=1)
        results = metrics.compute()
        
        self.assertEqual(results['mAP'], 0.0)
    
    def test_iou_edge_cases(self):
        """Test IoU with edge cases"""
        box = np.array([0, 0, 100, 100])
        
        # Zero-area box
        zero_box = np.array([50, 50, 50, 50])
        iou = calculate_iou(box, zero_box)
        self.assertEqual(iou, 0.0)
        
        # Identical boxes
        iou_same = calculate_iou(box, box)
        self.assertAlmostEqual(iou_same, 1.0)


if __name__ == '__main__':
    unittest.main()
