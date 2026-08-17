"""
Base Data Associator

Defines the canonical interface for track-to-detection association.

Supported association algorithms include:

    • Hungarian
    • Nearest Neighbor
    • JPDA
    • MHT

Canonical flow:

    Tracks
       │
       ▼
    BaseAssociator
       │
       ▼
    CostMatrixBuilder
       │
       ▼
    Association Algorithm
       │
       ├── Matches
       ├── Unmatched Tracks
       └── Unmatched Detections

The base class contains common validation and result handling but does
not implement a specific assignment algorithm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.detection_result import DetectionResult
from tracking.models.track import Track


# ======================================================================
# Association Result
# ======================================================================


@dataclass(slots=True)
class AssociationResult:
    """
    Canonical result returned by an association algorithm.

    Parameters
    ----------
    matches:
        List of (track_index, detection_index) pairs.

    unmatched_tracks:
        Indices of tracks that were not assigned a detection.

    unmatched_detections:
        Indices of detections that were not assigned to a track.

    costs:
        Association cost for each matched pair.

    metadata:
        Optional algorithm-specific information.

    cost_matrix:
        Complete Track × Detection association cost matrix.

        Shape:

            (number_of_tracks, number_of_detections)

        This matrix is preserved so that downstream tracking modules
        can inspect the complete association problem.
    """

    matches: list[tuple[int, int]] = field(
        default_factory=list
    )

    unmatched_tracks: list[int] = field(
        default_factory=list
    )

    unmatched_detections: list[int] = field(
        default_factory=list
    )

    costs: dict[tuple[int, int], float] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    cost_matrix: np.ndarray = field(
        default_factory=lambda: np.empty(
            (0, 0),
            dtype=float,
        )
    )

    # ==================================================================
    # Initialization / normalization
    # ==================================================================

    def __post_init__(self) -> None:
        """
        Normalize the cost matrix to a NumPy array.

        Empty/default AssociationResult objects are allowed and receive
        a (0, 0) cost matrix.
        """

        self.cost_matrix = np.asarray(
            self.cost_matrix,
            dtype=float,
        )

        if self.cost_matrix.ndim != 2:
            raise ValueError(
                "AssociationResult cost_matrix must be "
                "a 2-dimensional array."
            )

    # ==================================================================
    # Convenience
    # ==================================================================

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

    @property
    def num_tracks(self) -> int:
        """
        Number of tracks represented by the cost matrix.
        """

        return int(
            self.cost_matrix.shape[0]
        )

    # ------------------------------------------------------------------

    @property
    def num_detections(self) -> int:
        """
        Number of detections represented by the cost matrix.
        """

        return int(
            self.cost_matrix.shape[1]
        )

    # ==================================================================
    # Cost Access
    # ==================================================================

    def get_cost(
        self,
        track_index: int,
        detection_index: int,
    ) -> float | None:
        """
        Return the cost for a track/detection pair.

        Returns None when the requested pair is outside the matrix.
        """

        if not (
            0 <= track_index < self.num_tracks
        ):
            return None

        if not (
            0 <= detection_index < self.num_detections
        ):
            return None

        return float(
            self.cost_matrix[
                track_index,
                detection_index,
            ]
        )

    # ==================================================================
    # Serialization
    # ==================================================================

    def as_dict(self) -> dict[str, Any]:
        """
        Serialize the association result.
        """

        return {
            "matches": [
                {
                    "track_index": track_index,
                    "detection_index": detection_index,
                    "cost": self.costs.get(
                        (
                            track_index,
                            detection_index,
                        )
                    ),
                }
                for (
                    track_index,
                    detection_index,
                ) in self.matches
            ],

            "unmatched_tracks":
                list(self.unmatched_tracks),

            "unmatched_detections":
                list(self.unmatched_detections),

            "cost_matrix":
                self.cost_matrix.tolist(),

            "num_matches":
                self.num_matches,

            "num_unmatched_tracks":
                self.num_unmatched_tracks,

            "num_unmatched_detections":
                self.num_unmatched_detections,

            "metadata":
                self.metadata,
        }


# ======================================================================
# Base Associator
# ======================================================================


class BaseAssociator(ABC):
    """
    Abstract base class for all track-to-detection association
    algorithms.

    Examples
    --------
        HungarianAssociator
        NearestNeighborAssociator
        JPDAAssociator
        MHTAssociator
    """

    def __init__(
        self,
        cost_matrix_builder=None,
        gating_threshold: float | None = None,
    ) -> None:

        self.cost_matrix_builder = (
            cost_matrix_builder
        )

        if gating_threshold is not None:

            if gating_threshold <= 0:
                raise ValueError(
                    "gating_threshold must be "
                    "greater than zero."
                )

        self.gating_threshold = (
            gating_threshold
        )

    # ==================================================================
    # Public Association API
    # ==================================================================

    def associate(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> AssociationResult:
        """
        Perform track-to-detection association.

        Common validation and empty-input handling are performed here.
        The actual assignment is delegated to ``_associate``.
        """

        self._validate_inputs(
            tracks,
            detections,
        )

        num_tracks = len(tracks)
        num_detections = len(detections)

        # --------------------------------------------------------------
        # Empty input handling
        # --------------------------------------------------------------

        if num_tracks == 0:

            return AssociationResult(
                matches=[],
                unmatched_tracks=[],
                unmatched_detections=list(
                    range(num_detections)
                ),
                costs={},
                metadata={
                    "algorithm": self.name,
                    "empty_input": True,
                },
                cost_matrix=np.empty(
                    (
                        0,
                        num_detections,
                    ),
                    dtype=float,
                ),
            )

        if num_detections == 0:

            return AssociationResult(
                matches=[],
                unmatched_tracks=list(
                    range(num_tracks)
                ),
                unmatched_detections=[],
                costs={},
                metadata={
                    "algorithm": self.name,
                    "empty_input": True,
                },
                cost_matrix=np.empty(
                    (
                        num_tracks,
                        0,
                    ),
                    dtype=float,
                ),
            )

        # --------------------------------------------------------------
        # Algorithm-specific association
        # --------------------------------------------------------------

        result = self._associate(
            tracks,
            detections,
        )

        return self._normalize_result(
            result,
            tracks,
            detections,
        )

    # ==================================================================
    # Algorithm-specific implementation
    # ==================================================================

    @abstractmethod
    def _associate(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> AssociationResult:
        """
        Implement the actual association algorithm.

        Subclasses must override this method.
        """

        raise NotImplementedError

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_inputs(
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> None:

        if tracks is None:
            raise ValueError(
                "tracks must not be None."
            )

        if detections is None:
            raise ValueError(
                "detections must not be None."
            )

        for track in tracks:

            if not isinstance(
                track,
                Track,
            ):
                raise TypeError(
                    "All tracks must be Track instances."
                )

        for detection in detections:

            if not isinstance(
                detection,
                DetectionResult,
            ):
                raise TypeError(
                    "All detections must be "
                    "DetectionResult instances."
                )

    # ==================================================================
    # Result Normalization
    # ==================================================================

    @staticmethod
    def _normalize_result(
            result: AssociationResult,
            tracks: list[Track],
            detections: list[DetectionResult],
    ) -> AssociationResult:
        """
        Validate and normalize an association result.

        The canonical AssociationResult contains a complete
        Track × Detection cost matrix.

        For backward compatibility, an algorithm may omit the matrix.
        In that case the matrix is reconstructed from ``result.costs``.

        This is useful for lightweight association algorithms and for
        tests that only provide matched-pair costs.

        Explicitly supplied matrices with an invalid non-empty shape
        are rejected.
        """

        # --------------------------------------------------------------
        # Result type
        # --------------------------------------------------------------

        if not isinstance(
                result,
                AssociationResult,
        ):
            raise TypeError(
                "_associate() must return "
                "AssociationResult."
            )

        num_tracks = len(tracks)
        num_detections = len(detections)

        expected_shape = (
            num_tracks,
            num_detections,
        )

        # --------------------------------------------------------------
        # Cost matrix
        # --------------------------------------------------------------

        cost_matrix = np.asarray(
            result.cost_matrix,
            dtype=float,
        )

        # --------------------------------------------------------------
        # Backward compatibility
        #
        # AssociationResult() without a cost_matrix creates:
        #
        #     shape == (0, 0)
        #
        # Treat that as "matrix not supplied".
        #
        # Reconstruct it from the explicit costs dictionary.
        # --------------------------------------------------------------

        if (
                cost_matrix.shape == (0, 0)
                and expected_shape != (0, 0)
        ):

            # Start with infinity so unspecified associations are
            # considered unavailable / invalid.
            cost_matrix = np.full(
                expected_shape,
                np.inf,
                dtype=float,
            )

            for key, value in result.costs.items():

                # ------------------------------------------------------
                # Validate dictionary key.
                # ------------------------------------------------------

                if not isinstance(
                        key,
                        (tuple, list),
                ):
                    continue

                if len(key) != 2:
                    continue

                track_index, detection_index = key

                # ------------------------------------------------------
                # Reject booleans because bool is an int subclass.
                # ------------------------------------------------------

                if isinstance(
                        track_index,
                        bool,
                ):
                    continue

                if isinstance(
                        detection_index,
                        bool,
                ):
                    continue

                if not isinstance(
                        track_index,
                        (int, np.integer),
                ):
                    continue

                if not isinstance(
                        detection_index,
                        (int, np.integer),
                ):
                    continue

                track_index = int(
                    track_index
                )

                detection_index = int(
                    detection_index
                )

                # ------------------------------------------------------
                # Ignore out-of-range cost entries.
                # ------------------------------------------------------

                if not (
                        0 <= track_index < num_tracks
                ):
                    continue

                if not (
                        0 <= detection_index
                        < num_detections
                ):
                    continue

                # ------------------------------------------------------
                # Convert cost.
                # ------------------------------------------------------

                try:

                    cost = float(value)

                except (
                        TypeError,
                        ValueError,
                ):

                    continue

                # ------------------------------------------------------
                # Preserve only finite costs.
                # ------------------------------------------------------

                if not np.isfinite(cost):
                    continue

                cost_matrix[
                    track_index,
                    detection_index,
                ] = cost

        # --------------------------------------------------------------
        # Explicit matrix supplied.
        #
        # If it is non-empty, its shape must exactly match the current
        # Track × Detection problem.
        # --------------------------------------------------------------

        elif cost_matrix.shape != expected_shape:

            raise ValueError(
                "AssociationResult cost_matrix "
                f"has invalid shape "
                f"{cost_matrix.shape}; "
                f"expected {expected_shape}."
            )

        # --------------------------------------------------------------
        # Numerical safety
        # --------------------------------------------------------------

        cost_matrix = np.nan_to_num(
            cost_matrix,
            nan=np.inf,
            posinf=np.inf,
            neginf=np.inf,
        )

        # --------------------------------------------------------------
        # Validate matches
        # --------------------------------------------------------------

        valid_matches: list[
            tuple[int, int]
        ] = []

        used_tracks: set[int] = set()

        used_detections: set[int] = set()

        for pair in result.matches:

            # ----------------------------------------------------------
            # Pair structure
            # ----------------------------------------------------------

            if not isinstance(
                    pair,
                    (tuple, list),
            ):
                continue

            if len(pair) != 2:
                continue

            track_index, detection_index = pair

            # ----------------------------------------------------------
            # Reject bool because bool is an int subclass.
            # ----------------------------------------------------------

            if isinstance(
                    track_index,
                    bool,
            ):
                continue

            if isinstance(
                    detection_index,
                    bool,
            ):
                continue

            # ----------------------------------------------------------
            # Integer validation
            # ----------------------------------------------------------

            if not isinstance(
                    track_index,
                    (int, np.integer),
            ):
                continue

            if not isinstance(
                    detection_index,
                    (int, np.integer),
            ):
                continue

            track_index = int(
                track_index
            )

            detection_index = int(
                detection_index
            )

            # ----------------------------------------------------------
            # Bounds validation
            # ----------------------------------------------------------

            if not (
                    0 <= track_index < num_tracks
            ):
                continue

            if not (
                    0 <= detection_index
                    < num_detections
            ):
                continue

            # ----------------------------------------------------------
            # One-to-one protection
            # ----------------------------------------------------------

            if track_index in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            used_tracks.add(
                track_index
            )

            used_detections.add(
                detection_index
            )

            valid_matches.append(
                (
                    track_index,
                    detection_index,
                )
            )

        # --------------------------------------------------------------
        # Recalculate unmatched tracks
        # --------------------------------------------------------------

        unmatched_tracks = [
            index
            for index in range(
                num_tracks
            )
            if index not in used_tracks
        ]

        # --------------------------------------------------------------
        # Recalculate unmatched detections
        # --------------------------------------------------------------

        unmatched_detections = [
            index
            for index in range(
                num_detections
            )
            if index not in used_detections
        ]

        # --------------------------------------------------------------
        # Preserve only valid costs belonging to valid matches
        # --------------------------------------------------------------

        valid_costs: dict[
            tuple[int, int],
            float,
        ] = {}

        for (
                track_index,
                detection_index,
        ) in valid_matches:

            key = (
                track_index,
                detection_index,
            )

            # ----------------------------------------------------------
            # Prefer explicitly supplied pair cost.
            # ----------------------------------------------------------

            if key in result.costs:

                try:

                    cost = float(
                        result.costs[key]
                    )

                except (
                        TypeError,
                        ValueError,
                ):

                    continue

            # ----------------------------------------------------------
            # Otherwise obtain cost from the matrix.
            # ----------------------------------------------------------

            else:

                cost = float(
                    cost_matrix[
                        track_index,
                        detection_index,
                    ]
                )

            # ----------------------------------------------------------
            # Only preserve finite costs.
            # ----------------------------------------------------------

            if not np.isfinite(cost):
                continue

            valid_costs[key] = cost

        # --------------------------------------------------------------
        # Preserve metadata
        # --------------------------------------------------------------

        metadata = dict(
            result.metadata
        )

        # --------------------------------------------------------------
        # Return canonical normalized result
        # --------------------------------------------------------------

        return AssociationResult(
            matches=valid_matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
            costs=valid_costs,
            metadata=metadata,
            cost_matrix=cost_matrix,
        )

        # --------------------------------------------------------------
        # Recalculate unmatched tracks
        # --------------------------------------------------------------

        unmatched_tracks = [
            index
            for index in range(
                num_tracks
            )
            if index not in used_tracks
        ]

        # --------------------------------------------------------------
        # Recalculate unmatched detections
        # --------------------------------------------------------------

        unmatched_detections = [
            index
            for index in range(
                num_detections
            )
            if index not in used_detections
        ]

        # --------------------------------------------------------------
        # Preserve only valid costs
        # --------------------------------------------------------------

        valid_costs: dict[
            tuple[int, int],
            float,
        ] = {}

        for (
            track_index,
            detection_index,
        ) in valid_matches:

            key = (
                track_index,
                detection_index,
            )

            # ----------------------------------------------------------
            # Prefer explicit pair cost when supplied.
            # ----------------------------------------------------------

            if key in result.costs:

                try:

                    cost = float(
                        result.costs[key]
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

            # ----------------------------------------------------------
            # Otherwise derive it from the cost matrix.
            # ----------------------------------------------------------

            else:

                cost = float(
                    cost_matrix[
                        track_index,
                        detection_index,
                    ]
                )

            # ----------------------------------------------------------
            # Reject non-finite costs.
            # ----------------------------------------------------------

            if not np.isfinite(cost):
                continue

            valid_costs[key] = cost

        # --------------------------------------------------------------
        # Preserve metadata
        # --------------------------------------------------------------

        metadata = dict(
            result.metadata
        )

        # --------------------------------------------------------------
        # Return canonical normalized result
        # --------------------------------------------------------------

        return AssociationResult(
            matches=valid_matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
            costs=valid_costs,
            metadata=metadata,
            cost_matrix=cost_matrix,
        )

    # ==================================================================
    # Configuration
    # ==================================================================

    @property
    def name(self) -> str:
        """
        Human-readable name of the association algorithm.
        """

        return self.__class__.__name__

    # ------------------------------------------------------------------

    @property
    def metric_name(self) -> str | None:
        """
        Return the configured cost metric name.
        """

        if self.cost_matrix_builder is None:
            return None

        return getattr(
            self.cost_matrix_builder,
            "metric_name",
            None,
        )

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            f"{self.name}("
            f"metric={self.metric_name!r}, "
            f"gating_threshold="
            f"{self.gating_threshold!r})"
        )