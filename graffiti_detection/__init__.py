"""Public API for the graffiti detection package."""

from graffiti_detection._version import __version__
from graffiti_detection.detector import GraffitiDetector

__all__ = ["GraffitiDetector", "__version__"]
