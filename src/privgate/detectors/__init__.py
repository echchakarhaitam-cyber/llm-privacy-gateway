"""PII detectors. Every detector implements :class:`privgate.detectors.base.Detector`."""

from privgate.detectors.base import Detector
from privgate.detectors.regex import RegexDetector
from privgate.detectors.spacy_ner import SpacyDetector

__all__ = ["Detector", "RegexDetector", "SpacyDetector"]
