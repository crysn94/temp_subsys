"""
========================================================================
Track Quality Evaluation
========================================================================

Evaluates the quality of a persistent Track.

The quality evaluator is intentionally read-only.

It does NOT modify:

    - Track
    - lifecycle
    - detection history
    - state vector
    - confidence
    - timestamps

Quality is represented as a normalized value:

    min_quality <= quality <= max_quality

The evaluator considers:

    1. Detection rate
    2. Continuity
    3. Stability
    4. Detection confidence

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import numpy as np
from tracking.models.lifecycle import TrackState
from tracking.models.track import Track


# ======================================================================
# Track Quality
# ======================================================================


@dataclass(slots=True)
class TrackQuality:
    """
    Evaluate the quality of a Track.

    Parameters
    ----------
    min_quality:
        Minimum allowed quality value.

    max_quality:
        Maximum allowed quality value.

    reliability_threshold:
        Quality value at or above which a track is considered reliable.
    """

    min_quality: float = 0.0
    max_quality: float = 1.0
    reliability_threshold: float = 0.6

    # ==================================================================
    # Initialization
    # ==================================================================

    def __post_init__(self) -> None:

        self.min_quality = float(
            self.min_quality
        )

        self.max_quality = float(
            self.max_quality
        )

        self.reliability_threshold = float(
            self.reliability_threshold
        )

        # --------------------------------------------------------------
        # Quality range
        # --------------------------------------------------------------

        if not isfinite(
            self.min_quality
        ):
            raise ValueError(
                "min_quality must be finite."
            )

        if not isfinite(
            self.max_quality
        ):
            raise ValueError(
                "max_quality must be finite."
            )

        if self.min_quality > self.max_quality:
            raise ValueError(
                "min_quality must not be greater "
                "than max_quality."
            )

        # --------------------------------------------------------------
        # Reliability threshold
        # --------------------------------------------------------------

        if not isfinite(
            self.reliability_threshold
        ):
            raise ValueError(
                "reliability_threshold must be finite."
            )

        if not (
            self.min_quality
            <= self.reliability_threshold
            <= self.max_quality
        ):
            raise ValueError(
                "reliability_threshold must lie "
                "within the quality range."
            )

    # ==================================================================
    # Calculate
    # ==================================================================

    def calculate(
        self,
        track: Track,
    ) -> float:
        """
        Calculate the overall track quality.

        The result is always bounded by:

            min_quality <= result <= max_quality

        The calculation is read-only.
        """

        self._validate_track(track)

        detection_rate = (
            self.detection_rate(track)
        )

        continuity = (
            self.continuity_score(track)
        )

        stability = (
            self.stability_score(track)
        )

        confidence = self._confidence_score(
            track
        )

        # --------------------------------------------------------------
        # Weighted quality
        # --------------------------------------------------------------
        #
        # Detection history is the strongest indicator.
        # Continuity and stability describe persistence.
        # Confidence contributes sensor/classifier reliability.
        #

        normalized = (
            0.35 * detection_rate
            + 0.25 * continuity
            + 0.25 * stability
            + 0.15 * confidence
        )

        return self._scale_quality(
            normalized
        )

    # ==================================================================
    # Detection Rate
    # ==================================================================

    def detection_rate(
        self,
        track: Track,
    ) -> float:
        """
        Calculate the fraction of track cycles containing detections.

        Formula:

            detection_rate = hits / age

        For a brand-new track with age == 0:

            detection_rate = 0.0

        The result is bounded to [0, 1].
        """

        self._validate_track(track)

        age = max(
            0,
            int(
                getattr(
                    track,
                    "age",
                    0,
                )
            ),
        )

        hits = max(
            0,
            int(
                getattr(
                    track,
                    "hits",
                    0,
                )
            ),
        )

        if age <= 0:
            return 0.0

        rate = (
            float(hits)
            / float(age)
        )

        return self._clamp_unit(
            rate
        )

    # ==================================================================
    # Continuity
    # ==================================================================

    def continuity_score(
        self,
        track: Track,
    ) -> float:
        """
        Calculate track continuity.

        Consecutive successful detections produce a stronger
        continuity score.

        The score is normalized to [0, 1].
        """

        self._validate_track(track)

        age = max(
            0,
            int(
                getattr(
                    track,
                    "age",
                    0,
                )
            ),
        )

        consecutive_hits = max(
            0,
            int(
                getattr(
                    track,
                    "consecutive_hits",
                    0,
                )
            ),
        )

        if age <= 0:
            return 0.0

        score = (
            float(consecutive_hits)
            / float(age)
        )

        return self._clamp_unit(
            score
        )

    # ==================================================================
    # Stability
    # ==================================================================

    def stability_score(
        self,
        track: Track,
    ) -> float:
        """
        Calculate track stability.

        Stability decreases as consecutive misses increase.

        A continuously detected track has a stability of 1.0.

        A track with no history has a stability of 0.0.
        """

        self._validate_track(track)

        age = max(
            0,
            int(
                getattr(
                    track,
                    "age",
                    0,
                )
            ),
        )

        consecutive_misses = max(
            0,
            int(
                getattr(
                    track,
                    "consecutive_misses",
                    0,
                )
            ),
        )

        if age <= 0:
            return 0.0

        # No current miss streak -> maximally stable.
        if consecutive_misses <= 0:
            return 1.0

        score = (
            1.0
            - (
                float(consecutive_misses)
                / float(age)
            )
        )

        return self._clamp_unit(
            score
        )

    # ==================================================================
    # Reliability
    # ==================================================================

    def is_reliable(
            self,
            track: Track,
            threshold: float | None = None,
    ) -> bool:
        """
        Determine whether a track is sufficiently reliable.

        Parameters
        ----------
        track:
            Track whose quality is being evaluated.

        threshold:
            Optional reliability threshold in the range [0, 1].

            If omitted, the configured default reliability threshold
            is used.

        Returns
        -------
        bool
            True when the calculated quality score is greater than or
            equal to the selected threshold.

        Raises
        ------
        TypeError
            If threshold is not numeric.

        ValueError
            If threshold is not finite or is outside [0, 1].
        """

        if not isinstance(track, Track):
            raise TypeError(
                "track must be a Track instance."
            )

        # --------------------------------------------------------------
        # Select threshold
        # --------------------------------------------------------------

        if threshold is None:
            threshold_value = self.reliability_threshold
        else:
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "threshold must be a numeric value."
                ) from exc

        # --------------------------------------------------------------
        # Validate threshold
        # --------------------------------------------------------------

        if not np.isfinite(threshold_value):
            raise ValueError(
                "threshold must be finite."
            )

        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0."
            )

        # --------------------------------------------------------------
        # Calculate quality
        # --------------------------------------------------------------

        quality = self.calculate(track)

        return quality >= threshold_value

    # ==================================================================
    # Confidence
    # ==================================================================

    @staticmethod
    def _confidence_score(
        track: Track,
    ) -> float:
        """
        Return current detection confidence.

        Track.score already provides the current detection
        confidence and returns zero when no detection exists.
        """

        try:
            confidence = float(
                track.score
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            return 0.0

        return TrackQuality._clamp_unit(
            confidence
        )

    # ==================================================================
    # Scaling
    # ==================================================================

    def _scale_quality(
        self,
        normalized: float,
    ) -> float:
        """
        Scale normalized [0, 1] quality into the configured
        [min_quality, max_quality] interval.
        """

        normalized = (
            self._clamp_unit(
                normalized
            )
        )

        if (
            self.max_quality
            == self.min_quality
        ):
            return self.min_quality

        quality = (
            self.min_quality
            + normalized
            * (
                self.max_quality
                - self.min_quality
            )
        )

        return float(
            max(
                self.min_quality,
                min(
                    self.max_quality,
                    quality,
                ),
            )
        )

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_track(
        track: Track,
    ) -> None:

        if not isinstance(
            track,
            Track,
        ):
            raise TypeError(
                "track must be a Track instance."
            )

    # ==================================================================
    # Utility
    # ==================================================================

    @staticmethod
    def _clamp_unit(
        value: float,
    ) -> float:

        if not isfinite(value):
            return 0.0

        return float(
            max(
                0.0,
                min(
                    1.0,
                    value,
                ),
            )
        )

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_config(self) -> dict[str, float]:
        """
        Return serializable configuration.
        """

        return {
            "min_quality":
                self.min_quality,

            "max_quality":
                self.max_quality,

            "reliability_threshold":
                self.reliability_threshold,
        }

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            "TrackQuality("
            f"min_quality={self.min_quality}, "
            f"max_quality={self.max_quality}, "
            f"reliability_threshold="
            f"{self.reliability_threshold}"
            ")"
        )