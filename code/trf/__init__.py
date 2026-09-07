"""Independent type-related feature (TRF) module."""

from .offline.common import TRFError
from .target.common import TargetTRFError

__all__ = ["TRFError", "TargetTRFError"]
