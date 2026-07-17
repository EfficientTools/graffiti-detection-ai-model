"""Public API for the graffiti detection package."""

from graffiti_detection.detector import GraffitiDetector
from graffiti_detection._version import __version__

__all__ = ["GraffitiDetector", "__version__"]
