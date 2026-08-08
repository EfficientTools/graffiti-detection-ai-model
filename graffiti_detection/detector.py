"""High-level detector interface for library users."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

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
        self.conf_threshold = self._validated_threshold(conf_threshold)
        self.device = device
        self.model = YOLO(self.model_path)

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        **predict_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run detection and return class, confidence, and xyxy boxes."""
        threshold = self._validated_threshold(
            self.conf_threshold if conf_threshold is None else conf_threshold
        )
        kwargs = {
            "source": str(source) if isinstance(source, Path) else source,
            "conf": threshold,
            "verbose": False,
            **predict_kwargs,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)
        if not results:
            return []

        result = results[0]
        class_names = getattr(result, "names", {})
        detections = []

        for box in result.boxes or []:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            if isinstance(class_names, Mapping):
                class_name = class_names.get(class_id, str(class_id))
            elif 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = str(class_id)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [float(value) for value in box.xyxy[0].tolist()],
                }
            )

        return detections

    @staticmethod
    def _validated_threshold(value: float) -> float:
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("conf_threshold must be a finite value between 0 and 1.")
        return threshold

    def benchmark(
        self,
        sources: Iterable[Union[str, Path, np.ndarray]],
        *,
        warmup_runs: int = 1,
        measured_runs: int = 3,
        predict_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        """Benchmark steady-state single-image inference on this detector."""
        from graffiti_detection.benchmark import benchmark_detector

        return benchmark_detector(
            self,
            sources,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            predict_kwargs=predict_kwargs,
        )
