"""
===========================================================
Base Detection Converter
===========================================================

Converts detector-specific outputs into the project's
canonical DetectionResult objects.
"""

from abc import ABC
from abc import abstractmethod


class BaseDetectionConverter(ABC):

    """
    Abstract interface for all detector converters.
    """

    @abstractmethod
    def convert(
        self,
        results,
        *,
        sensor_id,
        frame_id,
        timestamp,
    ):
        """
        Convert detector output into DetectionResult objects.
        """
        raise NotImplementedError