"""
========================================================================
Nearest Neighbor Association
========================================================================

Canonical greedy nearest-neighbor data association.

Responsibilities
----------------
• Build the Track × Detection cost matrix
• Select the lowest-cost valid association
• Enforce one-to-one assignment
• Apply association gating
• Return matched tracks and detections
• Return unmatched tracks
• Return unmatched detections

The actual association cost is provided by CostMatrixBuilder.

Supported metrics include:

• Euclidean
• Mahalanobis
• IoU
• GIoU
• Motion
• Hybrid

The association algorithm is independent of the metric.

Algorithm
---------

    Track × Detection cost matrix
                │
                ▼
        Find minimum cost
                │
                ▼
          Accept pair
                │
                ▼
      Remove Track + Detection
                │
                ▼
          Repeat

Lower cost
    ↓
Better association

Higher cost
    ↓
Worse association

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.detection_result import DetectionResult
from tracking.association.cost_matrix import CostMatrixBuilder
from tracking.models.track import Track


# ======================================================================
# Association Result
# ======================================================================


@dataclass(slots=True)
class AssociationResult:
    """
    Result of a nearest-neighbor association operation.

    Indices refer to the original input lists supplied to
    ``associate()``.
    """

    matches: list[tuple[int, int]]

    unmatched_tracks: list[int]

    unmatched_detections: list[int]

    cost_matrix: np.ndarray

    # ------------------------------------------------------------------

    @property
    def num_matches(self) -> int:
        return len(self.matches)

    # ------------------------------------------------------------------

    @property
    def num_unmatched_tracks(self) -> int:
        return len(self.unmatched_tracks)

    # ------------------------------------------------------------------

    @property
    def num_unmatched_detections(self) -> int:
        return len(self.unmatched_detections)

    # ------------------------------------------------------------------

    @property
    def has_matches(self) -> bool:
        return bool(self.matches)

    # ------------------------------------------------------------------

    def get_match_cost(
        self,
        track_index: int,
        detection_index: int,
    ) -> float:
        """
        Return the association cost for a matched pair.
        """

        return float(
            self.cost_matrix[
                track_index,
                detection_index,
            ]
        )

    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        """
        Return a serializable representation.
        """

        return {
            "matches": [
                {
                    "track_index": track_index,
                    "detection_index": detection_index,
                    "cost": float(
                        self.cost_matrix[
                            track_index,
                            detection_index,
                        ]
                    ),
                }
                for track_index, detection_index
                in self.matches
            ],
            "unmatched_tracks":
                list(self.unmatched_tracks),

            "unmatched_detections":
                list(self.unmatched_detections),

            "num_matches":
                self.num_matches,

            "num_unmatched_tracks":
                self.num_unmatched_tracks,

            "num_unmatched_detections":
                self.num_unmatched_detections,
        }


# ======================================================================
# Nearest Neighbor Associator
# ======================================================================


class NearestNeighborAssociator:
    """
    Greedy global nearest-neighbor association.

    The algorithm repeatedly selects the globally lowest-cost
    Track × Detection pair.

    Once a track or detection has been assigned, it cannot be
    assigned again.

    Parameters
    ----------
    cost_builder:
        CostMatrixBuilder used to calculate Track × Detection costs.

    max_cost:
        Maximum accepted association cost.

        A pair with:

            cost > max_cost

        is rejected.

    invalid_cost:
        Cost representing an invalid association.
    """

    def __init__(
        self,
        cost_builder: CostMatrixBuilder,
        max_cost: float | None = None,
        invalid_cost: float = 1e9,
    ) -> None:

        if max_cost is not None and max_cost < 0:
            raise ValueError(
                "max_cost must be non-negative."
            )

        if invalid_cost <= 0:
            raise ValueError(
                "invalid_cost must be greater than zero."
            )

        self.cost_builder = cost_builder

        self.max_cost = (
            float(max_cost)
            if max_cost is not None
            else None
        )

        self.invalid_cost = float(
            invalid_cost
        )

    # ==================================================================
    # Associate
    # ==================================================================

    def associate(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> AssociationResult:
        """
        Perform greedy global nearest-neighbor association.

        Parameters
        ----------
        tracks:
            Existing tracks.

        detections:
            New detections.

        Returns
        -------
        AssociationResult
            Contains:

            • matched pairs
            • unmatched tracks
            • unmatched detections
            • complete cost matrix
        """

        num_tracks = len(tracks)

        num_detections = len(detections)

        # --------------------------------------------------------------
        # Empty track set
        # --------------------------------------------------------------

        if num_tracks == 0:

            return AssociationResult(
                matches=[],
                unmatched_tracks=[],
                unmatched_detections=list(
                    range(num_detections)
                ),
                cost_matrix=np.empty(
                    (
                        0,
                        num_detections,
                    ),
                    dtype=float,
                ),
            )

        # --------------------------------------------------------------
        # Empty detection set
        # --------------------------------------------------------------

        if num_detections == 0:

            return AssociationResult(
                matches=[],
                unmatched_tracks=list(
                    range(num_tracks)
                ),
                unmatched_detections=[],
                cost_matrix=np.empty(
                    (
                        num_tracks,
                        0,
                    ),
                    dtype=float,
                ),
            )

        # --------------------------------------------------------------
        # Build cost matrix
        # --------------------------------------------------------------

        cost_matrix = self.cost_builder.build(
            tracks,
            detections,
        )

        cost_matrix = np.asarray(
            cost_matrix,
            dtype=float,
        )

        # --------------------------------------------------------------
        # Numerical safety
        # --------------------------------------------------------------

        cost_matrix = np.nan_to_num(
            cost_matrix,
            nan=self.invalid_cost,
            posinf=self.invalid_cost,
            neginf=self.invalid_cost,
        )

        # --------------------------------------------------------------
        # Work on a copy.
        #
        # The original cost matrix is preserved in AssociationResult.
        # --------------------------------------------------------------

        working_matrix = cost_matrix.copy()

        matches: list[tuple[int, int]] = []

        assigned_tracks: set[int] = set()

        assigned_detections: set[int] = set()

        # --------------------------------------------------------------
        # Greedy global nearest-neighbor search
        # --------------------------------------------------------------

        while True:

            # ----------------------------------------------------------
            # Find globally minimum remaining cost
            # ----------------------------------------------------------

            flat_index = int(
                np.argmin(working_matrix)
            )

            row, col = np.unravel_index(
                flat_index,
                working_matrix.shape,
            )

            row = int(row)
            col = int(col)

            cost = float(
                working_matrix[row, col]
            )

            # ----------------------------------------------------------
            # No valid association remains
            # ----------------------------------------------------------

            if cost >= self.invalid_cost:
                break

            # ----------------------------------------------------------
            # Apply gating
            # ----------------------------------------------------------

            if (
                self.max_cost is not None
                and cost > self.max_cost
            ):
                break

            # ----------------------------------------------------------
            # Accept association
            # ----------------------------------------------------------

            matches.append(
                (
                    row,
                    col,
                )
            )

            assigned_tracks.add(row)

            assigned_detections.add(col)

            # ----------------------------------------------------------
            # Remove the selected track and detection.
            #
            # Using invalid_cost prevents either from participating
            # in another association.
            # ----------------------------------------------------------

            working_matrix[row, :] = (
                self.invalid_cost
            )

            working_matrix[:, col] = (
                self.invalid_cost
            )

        # --------------------------------------------------------------
        # Unmatched tracks
        # --------------------------------------------------------------

        unmatched_tracks = [
            index
            for index in range(num_tracks)
            if index not in assigned_tracks
        ]

        # --------------------------------------------------------------
        # Unmatched detections
        # --------------------------------------------------------------

        unmatched_detections = [
            index
            for index in range(num_detections)
            if index not in assigned_detections
        ]

        # --------------------------------------------------------------
        # Return result
        # --------------------------------------------------------------

        return AssociationResult(
            matches=matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
            cost_matrix=cost_matrix,
        )

    # ==================================================================
    # Convenience Alias
    # ==================================================================

    def associate_objects(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> AssociationResult:
        """
        Alias for associate().
        """

        return self.associate(
            tracks,
            detections,
        )

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def metric_name(self) -> str:
        return self.cost_builder.metric_name

    # ==================================================================

    def get_config(self) -> dict:
        """
        Return serializable configuration.
        """

        return {
            "algorithm": "nearest_neighbor",
            "strategy": "greedy_global",
            "metric": self.metric_name,
            "max_cost": self.max_cost,
            "invalid_cost": self.invalid_cost,
        }

    # ==================================================================

    def __repr__(self) -> str:

        return (
            "NearestNeighborAssociator("
            f"metric={self.metric_name}, "
            f"max_cost={self.max_cost}, "
            f"invalid_cost={self.invalid_cost}"
            ")"
        )