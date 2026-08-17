"""
========================================================================
Joint Probabilistic Data Association
========================================================================

Probabilistic data-association layer for the C-UAS tracking framework.

JPDA estimates the probability that each DetectionResult belongs to
each existing Track.

The metric layer remains independent from the association algorithm.

Architecture
------------

    Track
      +
    DetectionResult
      |
      v
    CostMatrixBuilder
      |
      v
    Track x Detection Cost Matrix
      |
      v
    JPDA Probability Calculation
      |
      +--------------------+
      |                    |
      v                    v
Association Probabilities  Miss Probability
      |
      v
Probabilistic Association Result

Lower metric cost
        |
        v
Higher association likelihood

This implementation provides a practical JPDA-style probability
calculation suitable for the current tracking architecture.

It does not enumerate the complete joint hypothesis space. Full
hypothesis enumeration can be added later when required by the
tracker.

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.detection_result import DetectionResult

from tracking.association.cost_matrix import CostMatrixBuilder
from tracking.models.track import Track


# ======================================================================
# JPDA Result
# ======================================================================


@dataclass(slots=True)
class JPDAResult:
    """
    Result produced by JPDA association.

    Parameters
    ----------
    probabilities:
        Track x Detection association probability matrix.

    missed_detection_probabilities:
        Probability associated with each track receiving no detection.

    matches:
        Optional hard one-to-one associations derived from the
        probability matrix.

    unmatched_tracks:
        Tracks without a selected hard association.

    unmatched_detections:
        Detections without a selected hard association.

    cost_matrix:
        Original Track x Detection cost matrix.
    """

    probabilities: np.ndarray

    missed_detection_probabilities: np.ndarray

    matches: list[tuple[int, int]]

    unmatched_tracks: list[int]

    unmatched_detections: list[int]

    cost_matrix: np.ndarray

    # ------------------------------------------------------------------

    @property
    def num_tracks(self) -> int:
        return int(
            self.probabilities.shape[0]
        )

    # ------------------------------------------------------------------

    @property
    def num_detections(self) -> int:
        return int(
            self.probabilities.shape[1]
        )

    # ------------------------------------------------------------------

    @property
    def num_matches(self) -> int:
        return len(self.matches)

    # ------------------------------------------------------------------

    def probability(
        self,
        track_index: int,
        detection_index: int,
    ) -> float:
        """
        Return P(detection | track).
        """

        return float(
            self.probabilities[
                track_index,
                detection_index,
            ]
        )

    # ------------------------------------------------------------------

    def best_detection(
        self,
        track_index: int,
    ) -> int | None:
        """
        Return the highest-probability detection for a track.

        Returns None when the track has no detections.
        """

        if self.num_detections == 0:
            return None

        probabilities = self.probabilities[
            track_index
        ]

        if probabilities.size == 0:
            return None

        index = int(
            np.argmax(probabilities)
        )

        return index

    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        """
        Serialize the JPDA result.
        """

        return {
            "probabilities":
                self.probabilities.tolist(),

            "missed_detection_probabilities":
                self.missed_detection_probabilities.tolist(),

            "matches": [
                {
                    "track_index": track_index,
                    "detection_index": detection_index,
                    "probability":
                        self.probability(
                            track_index,
                            detection_index,
                        ),
                    "cost":
                        float(
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
        }


# ======================================================================
# JPDA Associator
# ======================================================================


class JPDAAssociator:
    """
    Joint Probabilistic Data Association.

    Parameters
    ----------
    cost_builder:
        CostMatrixBuilder used to calculate Track x Detection costs.

    max_cost:
        Maximum cost considered valid for association.

    probability_threshold:
        Minimum probability required when converting the probabilistic
        result into hard one-to-one matches.

    detection_probability:
        Probability that a real target generates a detection.

        Typical values are between 0 and 1.

    clutter_probability:
        Background/clutter probability.

        Larger values make ambiguous detections less likely to be
        assigned to tracks.

    invalid_cost:
        Cost used for invalid associations.
    """

    def __init__(
        self,
        cost_builder: CostMatrixBuilder,
        max_cost: float | None = None,
        probability_threshold: float = 0.05,
        detection_probability: float = 0.90,
        clutter_probability: float = 0.10,
        invalid_cost: float = 1e9,
    ) -> None:

        if max_cost is not None and max_cost < 0:
            raise ValueError(
                "max_cost must be non-negative."
            )

        if not 0.0 <= probability_threshold <= 1.0:
            raise ValueError(
                "probability_threshold must be between 0 and 1."
            )

        if not 0.0 < detection_probability <= 1.0:
            raise ValueError(
                "detection_probability must be in (0, 1]."
            )

        if not 0.0 < clutter_probability <= 1.0:
            raise ValueError(
                "clutter_probability must be in (0, 1]."
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

        self.probability_threshold = float(
            probability_threshold
        )

        self.detection_probability = float(
            detection_probability
        )

        self.clutter_probability = float(
            clutter_probability
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
    ) -> JPDAResult:
        """
        Calculate JPDA-style association probabilities.

        Returns
        -------
        JPDAResult
            Contains the complete probability matrix as well as
            optional hard one-to-one associations.
        """

        num_tracks = len(tracks)

        num_detections = len(detections)

        # --------------------------------------------------------------
        # Empty case
        # --------------------------------------------------------------

        if (
            num_tracks == 0
            or num_detections == 0
        ):

            probabilities = np.zeros(
                (
                    num_tracks,
                    num_detections,
                ),
                dtype=float,
            )

            missed = np.ones(
                num_tracks,
                dtype=float,
            )

            return JPDAResult(
                probabilities=probabilities,
                missed_detection_probabilities=missed,
                matches=[],
                unmatched_tracks=list(
                    range(num_tracks)
                ),
                unmatched_detections=list(
                    range(num_detections)
                ),
                cost_matrix=np.empty(
                    (
                        num_tracks,
                        num_detections,
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
        # Valid association mask
        # --------------------------------------------------------------

        valid = cost_matrix < self.invalid_cost

        if self.max_cost is not None:

            valid &= (
                cost_matrix <= self.max_cost
            )

        # --------------------------------------------------------------
        # Convert metric cost into likelihood.
        #
        # For distance-based metrics:
        #
        #       likelihood ~ exp(-0.5 * cost²)
        #
        # This is particularly appropriate for Mahalanobis distance.
        #
        # For metrics such as IoU/GIoU the corresponding metric should
        # ideally provide a likelihood transformation. The generic
        # conversion below gives the JPDA layer a common interface.
        # --------------------------------------------------------------

        likelihood = np.zeros_like(
            cost_matrix,
            dtype=float,
        )

        valid_costs = cost_matrix[valid]

        if valid_costs.size > 0:

            likelihood[valid] = np.exp(
                -0.5
                * np.square(
                    valid_costs
                )
            )

        # --------------------------------------------------------------
        # Apply detection/clutter model
        # --------------------------------------------------------------

        likelihood[valid] *= (
            self.detection_probability
            / self.clutter_probability
        )

        # --------------------------------------------------------------
        # Calculate probabilities independently for each track.
        #
        # The missed-detection hypothesis is included explicitly.
        # --------------------------------------------------------------

        probabilities = np.zeros_like(
            likelihood,
            dtype=float,
        )

        missed = np.zeros(
            num_tracks,
            dtype=float,
        )

        for track_index in range(
            num_tracks
        ):

            row = likelihood[
                track_index
            ]

            total = (
                float(np.sum(row))
                + (
                    1.0
                    - self.detection_probability
                )
            )

            if total <= 0.0:

                missed[
                    track_index
                ] = 1.0

                continue

            probabilities[
                track_index
            ] = row / total

            missed[
                track_index
            ] = (
                1.0
                - self.detection_probability
            ) / total

        # --------------------------------------------------------------
        # Convert probabilities to hard one-to-one associations.
        #
        # This is NOT the probabilistic JPDA result itself.
        # It is only a compatibility representation for downstream
        # modules that currently expect hard matches.
        # --------------------------------------------------------------

        matches = self._extract_hard_matches(
            probabilities
        )

        assigned_tracks = {
            track_index
            for track_index, _
            in matches
        }

        assigned_detections = {
            detection_index
            for _, detection_index
            in matches
        }

        unmatched_tracks = [
            index
            for index in range(num_tracks)
            if index not in assigned_tracks
        ]

        unmatched_detections = [
            index
            for index in range(num_detections)
            if index not in assigned_detections
        ]

        return JPDAResult(
            probabilities=probabilities,
            missed_detection_probabilities=missed,
            matches=matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
            cost_matrix=cost_matrix,
        )

    # ==================================================================
    # Hard Match Extraction
    # ==================================================================

    def _extract_hard_matches(
        self,
        probabilities: np.ndarray,
    ) -> list[tuple[int, int]]:
        """
        Convert association probabilities into one-to-one matches.

        The highest probability pair is selected first.

        This method exists for compatibility with modules that require
        hard Track-Detection assignments.

        The full JPDA probability matrix remains available to callers.
        """

        if probabilities.size == 0:
            return []

        working = probabilities.copy()

        matches: list[
            tuple[int, int]
        ] = []

        num_tracks, num_detections = (
            working.shape
        )

        while True:

            flat_index = int(
                np.argmax(working)
            )

            track_index, detection_index = (
                np.unravel_index(
                    flat_index,
                    working.shape,
                )
            )

            track_index = int(
                track_index
            )

            detection_index = int(
                detection_index
            )

            probability = float(
                working[
                    track_index,
                    detection_index,
                ]
            )

            # ----------------------------------------------------------
            # No sufficiently probable association remains.
            # ----------------------------------------------------------

            if (
                probability
                < self.probability_threshold
            ):
                break

            matches.append(
                (
                    track_index,
                    detection_index,
                )
            )

            # ----------------------------------------------------------
            # Enforce one-to-one matching.
            # ----------------------------------------------------------

            working[
                track_index,
                :,
            ] = 0.0

            working[
                :,
                detection_index,
            ] = 0.0

            if len(matches) >= min(
                num_tracks,
                num_detections,
            ):
                break

        return matches

    # ==================================================================
    # Convenience
    # ==================================================================

    def associate_objects(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> JPDAResult:
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
            "algorithm": "jpda",
            "implementation": "jpda_style",
            "metric": self.metric_name,
            "max_cost": self.max_cost,
            "probability_threshold":
                self.probability_threshold,
            "detection_probability":
                self.detection_probability,
            "clutter_probability":
                self.clutter_probability,
            "invalid_cost":
                self.invalid_cost,
        }

    # ==================================================================

    def __repr__(self) -> str:

        return (
            "JPDAAssociator("
            f"metric={self.metric_name}, "
            f"max_cost={self.max_cost}, "
            f"probability_threshold="
            f"{self.probability_threshold})"
        )