"""
========================================================================
Abstract Tracking Filter
========================================================================

Defines the interface implemented by every tracking filter.

Supported Filters

• Linear Kalman Filter
• Extended Kalman Filter (EKF)
• Unscented Kalman Filter (UKF)
• Interacting Multiple Model (IMM)
• Particle Filter

Architecture

    TrackManager
         │
         ▼
    BaseTrackingFilter
         │
         ├── KalmanFilter
         ├── ExtendedKalmanFilter
         ├── UnscentedKalmanFilter
         ├── IMMFilter
         └── ParticleFilter

========================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.detection_result import DetectionResult
from tracking.models.state import DynamicState


class BaseTrackingFilter(ABC):
    """
    Abstract interface for tracking filters.

    Filters are stateless with respect to individual Tracks.

    The current DynamicState is supplied by TrackManager for every
    prediction and measurement-update operation.
    """

    ####################################################################
    # Initialization
    ####################################################################

    @abstractmethod
    def initialize(
        self,
        detection: DetectionResult,
    ) -> DynamicState:
        """
        Initialize a dynamic state from a detection.
        """
        raise NotImplementedError

    ####################################################################
    # Prediction
    ####################################################################

    @abstractmethod
    def predict(
        self,
        state: DynamicState,
        dt: float,
    ) -> DynamicState:
        """
        Predict a dynamic state forward by dt seconds.

        Parameters
        ----------
        state:
            Current estimated dynamic state.

        dt:
            Prediction interval in seconds.

        Returns
        -------
        DynamicState
            Predicted state.
        """
        raise NotImplementedError

    ####################################################################
    # Measurement Update
    ####################################################################

    @abstractmethod
    def update(
        self,
        state: DynamicState,
        detection: DetectionResult,
    ) -> DynamicState:
        """
        Correct a predicted state using a detection.

        Parameters
        ----------
        state:
            Predicted dynamic state.

        detection:
            Measurement used for correction.

        Returns
        -------
        DynamicState
            Updated state.
        """
        raise NotImplementedError

    ####################################################################
    # Covariance
    ####################################################################

    @abstractmethod
    def covariance(
        self,
        state: DynamicState,
    ) -> np.ndarray:
        """
        Return the covariance matrix associated with a state.
        """
        raise NotImplementedError

    ####################################################################
    # Status
    ####################################################################

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """
        True if the filter has been initialized.
        """
        raise NotImplementedError

    ####################################################################
    # Reset
    ####################################################################

    @abstractmethod
    def reset(self) -> None:
        """
        Reset filter status.
        """
        raise NotImplementedError

    ####################################################################
    # Serialization
    ####################################################################

    @abstractmethod
    def as_dict(self) -> dict:
        """
        Serialize filter configuration/state.
        """
        raise NotImplementedError