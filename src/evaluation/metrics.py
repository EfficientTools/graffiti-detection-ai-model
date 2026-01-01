"""
Metrics calculation for object detection.
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_ap(
    predictions: List[Tuple[float, bool]],
    num_gt: int
) -> float:
    """
    Calculate Average Precision for a single class.
    
    Args:
        predictions: List of (confidence, is_tp) tuples sorted by confidence
        num_gt: Number of ground truth boxes
        
    Returns:
        Average Precision value
    """
    if num_gt == 0:
        return 0.0
    
    # Sort by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x[0], reverse=True)
    
    tp = np.array([int(p[1]) for p in predictions])
    fp = 1 - tp
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def calculate_map(
    predictions: Dict[int, List[Tuple[np.ndarray, float]]],
    ground_truths: Dict[int, List[np.ndarray]],
    iou_threshold: float = 0.5,
    num_classes: int = 1
) -> Tuple[float, Dict[int, float]]:
    """
    Calculate mean Average Precision (mAP) across all classes.
    
    Args:
        predictions: Dict mapping class_id to list of (box, confidence) tuples
        ground_truths: Dict mapping class_id to list of boxes
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
        
    Returns:
        Tuple of (mAP, per_class_AP_dict)
    """
    aps = {}
    
    for class_id in range(num_classes):
        class_preds = predictions.get(class_id, [])
        class_gts = ground_truths.get(class_id, [])
        
        if len(class_gts) == 0:
            continue
        
        # Match predictions to ground truths
        matched_gts = set()
        pred_results = []
        
        for pred_box, conf in sorted(class_preds, key=lambda x: x[1], reverse=True):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(class_gts):
                if gt_idx in matched_gts:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            is_tp = False
            if best_iou >= iou_threshold and best_gt_idx != -1:
                is_tp = True
                matched_gts.add(best_gt_idx)
            
            pred_results.append((conf, is_tp))
        
        # Calculate AP for this class
        ap = calculate_ap(pred_results, len(class_gts))
        aps[class_id] = ap
    
    # Calculate mAP
    if aps:
        mAP = np.mean(list(aps.values()))
    else:
        mAP = 0.0
    
    return mAP, aps


def calculate_precision_recall_f1(
    tp: int,
    fp: int,
    fn: int
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1-score.
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score


def calculate_confusion_matrix(
    predictions: List[int],
    ground_truths: List[int],
    num_classes: int
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        predictions: List of predicted class IDs
        ground_truths: List of ground truth class IDs
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for pred, gt in zip(predictions, ground_truths):
        if 0 <= pred < num_classes and 0 <= gt < num_classes:
            cm[gt, pred] += 1
    
    return cm


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression (NMS).
    
    Args:
        boxes: Array of boxes [x1, y1, x2, y2]
        scores: Array of confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([])
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)


class DetectionMetrics:
    """
    Class for tracking and computing detection metrics.
    """
    
    def __init__(self, num_classes: int = 1, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = defaultdict(list)
        self.ground_truths = defaultdict(list)
        self.tp = 0
        self.fp = 0
        self.fn = 0
    
    def add_batch(
        self,
        pred_boxes: List[np.ndarray],
        pred_scores: List[np.ndarray],
        pred_labels: List[np.ndarray],
        gt_boxes: List[np.ndarray],
        gt_labels: List[np.ndarray]
    ):
        """
        Add a batch of predictions and ground truths.
        
        Args:
            pred_boxes: List of predicted boxes for each image
            pred_scores: List of predicted scores for each image
            pred_labels: List of predicted labels for each image
            gt_boxes: List of ground truth boxes for each image
            gt_labels: List of ground truth labels for each image
        """
        for p_boxes, p_scores, p_labels, g_boxes, g_labels in zip(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
        ):
            # Add to class-wise storage
            for box, score, label in zip(p_boxes, p_scores, p_labels):
                self.predictions[label].append((box, score))
            
            for box, label in zip(g_boxes, g_labels):
                self.ground_truths[label].append(box)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate mAP
        mAP, class_aps = calculate_map(
            self.predictions,
            self.ground_truths,
            self.iou_threshold,
            self.num_classes
        )
        
        metrics = {
            'mAP': mAP,
        }
        
        # Add per-class APs
        for class_id, ap in class_aps.items():
            metrics[f'AP_class_{class_id}'] = ap
        
        return metrics
