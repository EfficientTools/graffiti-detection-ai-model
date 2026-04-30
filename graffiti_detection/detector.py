"""High-level detector interface for library users."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class GraffitiDetector:
    """Load a YOLO model and return normalized detection dictionaries."""

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLO(self.model_path)

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        **predict_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run detection and return class, confidence, and xyxy boxes."""
        threshold = self.conf_threshold if conf_threshold is None else conf_threshold
        kwargs = {
            "source": str(source) if isinstance(source, Path) else source,
            "conf": threshold,
            "verbose": False,
            **predict_kwargs,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        result = self.model.predict(**kwargs)[0]
        class_names = getattr(result, "names", {})
        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "box": [float(value) for value in box.xyxy[0].tolist()],
                }
            )

        return detections
