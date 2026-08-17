"""
Hungarian Association

Canonical Hungarian / Linear Assignment association algorithm.

Responsibilities
----------------
• Build the Track × Detection cost matrix
• Solve the minimum-cost assignment problem
• Reject assignments above the configured cost threshold
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

The Hungarian algorithm itself is independent of the metric.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.detection_result import DetectionResult
from tracking.association.base_associator import (
    AssociationResult,
    BaseAssociator,
)
from tracking.association.cost_matrix import CostMatrixBuilder
from tracking.models.track import Track


# ======================================================================
# Hungarian Associator
# ======================================================================


class HungarianAssociator(BaseAssociator):
    """
    Hungarian minimum-cost data association.

    Parameters
    ----------
    cost_builder:
        CostMatrixBuilder responsible for generating the
        Track × Detection cost matrix.

    max_cost:
        Maximum acceptable association cost.

        Assignments with cost > max_cost are rejected.

    invalid_cost:
        Cost representing an invalid association.
    """

    def __init__(
        self,
        cost_builder: CostMatrixBuilder,
        max_cost: float | None = None,
        invalid_cost: float = 1e9,
        gating_threshold: float | None = None,
    ) -> None:

        if cost_builder is None:
            raise ValueError(
                "cost_builder must not be None."
            )

        if max_cost is not None and max_cost < 0:
            raise ValueError(
                "max_cost must be non-negative."
            )

        if invalid_cost <= 0:
            raise ValueError(
                "invalid_cost must be greater than zero."
            )

        super().__init__(
            cost_matrix_builder=cost_builder,
            gating_threshold=gating_threshold,
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
    # Association
    # ==================================================================

    def _associate(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> AssociationResult:
        """
        Perform Hungarian / linear assignment.

        This method is called by BaseAssociator.associate()
        after input validation and empty-input handling.
        """

        num_tracks = len(tracks)
        num_detections = len(detections)

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
        # Validate cost matrix shape
        # --------------------------------------------------------------

        expected_shape = (
            num_tracks,
            num_detections,
        )

        if cost_matrix.shape != expected_shape:
            raise ValueError(
                "Cost matrix has invalid shape "
                f"{cost_matrix.shape}; "
                f"expected {expected_shape}."
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
        # Apply max-cost gating before assignment.
        #
        # Invalid assignments are represented by invalid_cost.
        # --------------------------------------------------------------

        assignment_matrix = cost_matrix.copy()

        if self.max_cost is not None:

            assignment_matrix[
                assignment_matrix > self.max_cost
            ] = self.invalid_cost

        # --------------------------------------------------------------
        # Protect invalid associations.
        # --------------------------------------------------------------

        assignment_matrix[
            assignment_matrix >= self.invalid_cost
        ] = self.invalid_cost

        # --------------------------------------------------------------
        # Hungarian assignment
        # --------------------------------------------------------------

        row_indices, col_indices = (
            linear_sum_assignment(
                assignment_matrix
            )
        )

        # --------------------------------------------------------------
        # Process assignments
        # --------------------------------------------------------------

        matches: list[
            tuple[int, int]
        ] = []

        costs: dict[
            tuple[int, int],
            float,
        ] = {}

        assigned_tracks: set[int] = set()

        assigned_detections: set[int] = set()

        for row, col in zip(
            row_indices,
            col_indices,
        ):

            row = int(row)
            col = int(col)

            cost = float(
                cost_matrix[
                    row,
                    col,
                ]
            )

            # ----------------------------------------------------------
            # Reject invalid assignments
            # ----------------------------------------------------------

            if cost >= self.invalid_cost:
                continue

            # ----------------------------------------------------------
            # Reject assignments above max_cost
            # ----------------------------------------------------------

            if (
                self.max_cost is not None
                and cost > self.max_cost
            ):
                continue

            # ----------------------------------------------------------
            # One-to-one protection
            # ----------------------------------------------------------

            if row in assigned_tracks:
                continue

            if col in assigned_detections:
                continue

            # ----------------------------------------------------------
            # Store valid match
            # ----------------------------------------------------------

            matches.append(
                (
                    row,
                    col,
                )
            )

            costs[
                (
                    row,
                    col,
                )
            ] = cost

            assigned_tracks.add(row)
            assigned_detections.add(col)

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
        # Canonical association result
        # --------------------------------------------------------------

        result = AssociationResult(
            matches=matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
            costs=costs,
            metadata={
                "algorithm": "hungarian",
                "metric": self.metric_name,
                "max_cost": self.max_cost,
                "invalid_cost": self.invalid_cost,
            },
            cost_matrix=cost_matrix,
        )

        return result

    # ==================================================================
    # Public Convenience API
    # ==================================================================

    def associate_objects(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> AssociationResult:
        """
        Alias for associate().

        Kept for readability in higher-level tracking code.
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
        """
        Return the configured cost metric name.
        """

        return str(
            getattr(
                self.cost_builder,
                "metric_name",
                "unknown",
            )
        )

    # ==================================================================

    def get_config(self) -> dict:
        """
        Return serializable Hungarian configuration.
        """

        return {
            "algorithm": "hungarian",
            "metric": self.metric_name,
            "max_cost": self.max_cost,
            "invalid_cost": self.invalid_cost,
            "gating_threshold": self.gating_threshold,
        }

    # ==================================================================

    def __repr__(self) -> str:

        return (
            "HungarianAssociator("
            f"metric={self.metric_name}, "
            f"max_cost={self.max_cost}, "
            f"invalid_cost={self.invalid_cost}, "
            f"gating_threshold="
            f"{self.gating_threshold}"
            ")"
        )