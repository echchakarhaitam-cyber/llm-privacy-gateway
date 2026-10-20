"""PII detectors. Every detector implements :class:`privgate.detectors.base.Detector`."""

from privgate.detectors.base import Detector
from privgate.detectors.regex import RegexDetector

__all__ = ["Detector", "RegexDetector"]
