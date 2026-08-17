"""
========================================================================
Association Cost Matrix
========================================================================

Builds the Track × Detection cost matrix used by association
algorithms in the C-UAS tracking framework.

The actual cost computation is delegated to BaseMetric implementations.

Supported metrics include:

• Euclidean
• Mahalanobis
• IoU
• GIoU
• Motion
• Appearance
• Hybrid

Supported association algorithms include:

• Hungarian
• Nearest Neighbor
• JPDA
• MHT

Convention
----------

Lower cost
    ↓
Better association

Higher cost
    ↓
Worse association

The CostMatrixBuilder is intentionally independent of the actual
association algorithm.
========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.detection_result import DetectionResult
from tracking.models.track import Track
from tracking.similarity.base_metric import BaseMetric


# ======================================================================
# Cost Matrix Builder
# ======================================================================

@dataclass(slots=True)
class CostMatrixBuilder:
    """
    Builds the Track × Detection association cost matrix.

    Parameters
    ----------
    metric:
        Metric used to calculate the association cost.

    invalid_cost:
        Cost assigned when a Track and Detection cannot be compared.

    Notes
    -----
    The metric's own ``safe_compute()`` method is used so that:

    • invalid inputs
    • missing measurements
    • NaN
    • infinity
    • numerical failures

    do not corrupt the association matrix.
    """

    metric: BaseMetric

    invalid_cost: float = 1e9

    # ==================================================================
    # Initialization
    # ==================================================================

    def __post_init__(self) -> None:

        if not isinstance(
            self.metric,
            BaseMetric,
        ):
            raise TypeError(
                "metric must be an instance of BaseMetric."
            )

        if self.invalid_cost <= 0:
            raise ValueError(
                "invalid_cost must be greater than zero."
            )

        self.invalid_cost = float(
            self.invalid_cost
        )

    # ==================================================================
    # Build
    # ==================================================================

    def build(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> np.ndarray:
        """
        Build Track × Detection cost matrix.

        Parameters
        ----------
        tracks:
            Existing tracks.

        detections:
            New sensor detections.

        Returns
        -------
        numpy.ndarray
            Matrix with shape:

                (number_of_tracks, number_of_detections)

        Example
        -------

        If there are:

            3 tracks
            4 detections

        the returned matrix has shape:

            (3, 4)

        where:

            matrix[i, j]

        represents the cost of assigning detection ``j`` to
        track ``i``.
        """

        # --------------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------------

        if tracks is None:
            raise ValueError(
                "tracks cannot be None."
            )

        if detections is None:
            raise ValueError(
                "detections cannot be None."
            )

        # --------------------------------------------------------------
        # Empty cases
        # --------------------------------------------------------------

        if not tracks or not detections:

            return np.empty(
                (
                    len(tracks),
                    len(detections),
                ),
                dtype=float,
            )

        # --------------------------------------------------------------
        # Initialize matrix
        # --------------------------------------------------------------

        matrix = np.full(
            (
                len(tracks),
                len(detections),
            ),
            self.invalid_cost,
            dtype=float,
        )

        # --------------------------------------------------------------
        # Calculate costs
        # --------------------------------------------------------------

        for track_index, track in enumerate(tracks):

            for detection_index, detection in enumerate(
                detections
            ):

                cost = self.metric.safe_compute(
                    track,
                    detection,
                )

                # ------------------------------------------------------
                # Final numerical protection
                # ------------------------------------------------------

                if not np.isfinite(cost):
                    cost = self.invalid_cost

                if cost < 0:
                    cost = self.invalid_cost

                # ------------------------------------------------------
                # Protect against metrics using a different invalid
                # cost than the builder.
                # ------------------------------------------------------

                if cost >= self.metric.invalid_cost:
                    cost = self.invalid_cost

                matrix[
                    track_index,
                    detection_index,
                ] = cost

        return matrix

    # ==================================================================
    # Convenience
    # ==================================================================

    @property
    def metric_name(self) -> str:
        """
        Return the active metric name.
        """

        return self.metric.name

    # ==================================================================

    @property
    def shape_description(self) -> str:
        """
        Describe the matrix orientation.
        """

        return (
            "rows=tracks, "
            "columns=detections"
        )

    # ==================================================================
    # Validity
    # ==================================================================

    def valid_mask(
        self,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Return a boolean matrix identifying valid associations.

        True
            Valid association cost.

        False
            Invalid/gated association.
        """

        matrix = np.asarray(
            matrix,
            dtype=float,
        )

        return (
            np.isfinite(matrix)
            &
            (matrix < self.invalid_cost)
        )

    # ==================================================================
    # Invalid Mask
    # ==================================================================

    def invalid_mask(
        self,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Return a boolean matrix identifying invalid associations.
        """

        return ~self.valid_mask(matrix)

    # ==================================================================
    # Minimum Cost
    # ==================================================================

    def minimum_cost(
        self,
        matrix: np.ndarray,
    ) -> float | None:
        """
        Return the minimum valid cost.

        Returns None when no valid association exists.
        """

        valid = self.valid_mask(matrix)

        if not np.any(valid):
            return None

        return float(
            np.min(
                matrix[valid]
            )
        )

    # ==================================================================
    # Maximum Valid Cost
    # ==================================================================

    def maximum_cost(
        self,
        matrix: np.ndarray,
    ) -> float | None:
        """
        Return the maximum valid cost.

        Returns None when no valid association exists.
        """

        valid = self.valid_mask(matrix)

        if not np.any(valid):
            return None

        return float(
            np.max(
                matrix[valid]
            )
        )

    # ==================================================================
    # Statistics
    # ==================================================================

    def statistics(
        self,
        matrix: np.ndarray,
    ) -> dict:
        """
        Return basic cost-matrix statistics.
        """

        matrix = np.asarray(
            matrix,
            dtype=float,
        )

        valid = self.valid_mask(matrix)

        valid_costs = matrix[valid]

        if valid_costs.size == 0:

            return {
                "shape": tuple(matrix.shape),
                "metric": self.metric_name,
                "total_entries": int(matrix.size),
                "valid_entries": 0,
                "invalid_entries": int(matrix.size),
                "minimum": None,
                "maximum": None,
                "mean": None,
            }

        return {
            "shape": tuple(matrix.shape),
            "metric": self.metric_name,
            "total_entries": int(matrix.size),
            "valid_entries": int(
                valid_costs.size
            ),
            "invalid_entries": int(
                matrix.size - valid_costs.size
            ),
            "minimum": float(
                np.min(valid_costs)
            ),
            "maximum": float(
                np.max(valid_costs)
            ),
            "mean": float(
                np.mean(valid_costs)
            ),
        }

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_config(self) -> dict:
        """
        Return serializable configuration.
        """

        return {
            "metric": self.metric.get_config(),
            "invalid_cost": self.invalid_cost,
        }

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            "CostMatrixBuilder("
            f"metric={self.metric_name}, "
            f"invalid_cost={self.invalid_cost}"
            ")"
        )