"""
========================================================================
Multiple Hypothesis Tracking Association
========================================================================

MHT association layer for the C-UAS tracking framework.

MHT maintains multiple possible Track <-> Detection association
hypotheses instead of immediately committing to a single association.

Architecture

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
    Hypothesis Generation
      |
      v
    Hypothesis Scoring
      |
      v
    Hypothesis Pruning
      |
      v
    Ranked MHT Hypotheses

Lower total cost
    ->
Better hypothesis

The implementation intentionally uses bounded hypothesis generation.
It is therefore suitable as the foundation for a production MHT
implementation without requiring exhaustive enumeration of the complete
hypothesis tree.

Constraints

    • One detection can belong to at most one track.
    • One track can receive at most one detection.
    • Tracks may remain unmatched.
    • Detections may remain unmatched.
    • Invalid / gated associations are not included.
    • Only the best ``max_hypotheses`` hypotheses are retained.

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.detection_result import DetectionResult
from tracking.association.cost_matrix import CostMatrixBuilder
from tracking.models.track import Track


# ======================================================================
# MHT Hypothesis
# ======================================================================

@dataclass(slots=True)
class MHTHypothesis:
    """
    Represents one possible global Track <-> Detection association.

    Parameters
    ----------
    matches:
        List of ``(track_index, detection_index)`` pairs.

    unmatched_tracks:
        Track indices without an assigned detection.

    unmatched_detections:
        Detection indices without an assigned track.

    total_cost:
        Total hypothesis cost.

    score:
        Normalized hypothesis score.

    """

    matches: list[tuple[int, int]]

    unmatched_tracks: list[int]

    unmatched_detections: list[int]

    total_cost: float

    score: float

    # ------------------------------------------------------------------

    @property
    def num_matches(self) -> int:
        return len(self.matches)

    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "matches": [
                {
                    "track_index": track_index,
                    "detection_index": detection_index,
                }
                for track_index, detection_index in self.matches
            ],
            "unmatched_tracks": list(
                self.unmatched_tracks
            ),
            "unmatched_detections": list(
                self.unmatched_detections
            ),
            "num_matches": self.num_matches,
            "total_cost": float(
                self.total_cost
            ),
            "score": float(
                self.score
            ),
        }


# ======================================================================
# MHT Result
# ======================================================================

@dataclass(slots=True)
class MHTResult:
    """
    Result returned by ``MHTAssociator``.

    ``hypotheses[0]`` is always the best surviving hypothesis when
    hypotheses are available.
    """

    hypotheses: list[MHTHypothesis]

    cost_matrix: np.ndarray

    # ------------------------------------------------------------------

    @property
    def best_hypothesis(self) -> MHTHypothesis | None:
        if not self.hypotheses:
            return None

        return self.hypotheses[0]

    # ------------------------------------------------------------------

    @property
    def num_hypotheses(self) -> int:
        return len(self.hypotheses)

    # ------------------------------------------------------------------

    @property
    def matches(self) -> list[tuple[int, int]]:
        hypothesis = self.best_hypothesis

        if hypothesis is None:
            return []

        return list(
            hypothesis.matches
        )

    # ------------------------------------------------------------------

    @property
    def unmatched_tracks(self) -> list[int]:
        hypothesis = self.best_hypothesis

        if hypothesis is None:
            return []

        return list(
            hypothesis.unmatched_tracks
        )

    # ------------------------------------------------------------------

    @property
    def unmatched_detections(self) -> list[int]:
        hypothesis = self.best_hypothesis

        if hypothesis is None:
            return []

        return list(
            hypothesis.unmatched_detections
        )

    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "num_hypotheses":
                self.num_hypotheses,

            "best_hypothesis":
                (
                    self.best_hypothesis.as_dict()
                    if self.best_hypothesis
                    else None
                ),

            "hypotheses": [
                hypothesis.as_dict()
                for hypothesis in self.hypotheses
            ],
        }


# ======================================================================
# MHT Associator
# ======================================================================

class MHTAssociator:
    """
    Multiple Hypothesis Tracking associator.

    Parameters
    ----------
    cost_builder:
        CostMatrixBuilder used to calculate Track x Detection costs.

    max_hypotheses:
        Maximum number of hypotheses retained.

    max_cost:
        Maximum valid association cost.

    miss_cost:
        Cost assigned when a track receives no detection.

    clutter_cost:
        Cost assigned when a detection remains unmatched.

    """

    def __init__(
        self,
        cost_builder: CostMatrixBuilder,
        max_hypotheses: int = 20,
        max_cost: float | None = None,
        miss_cost: float = 5.0,
        clutter_cost: float = 5.0,
    ) -> None:

        if max_hypotheses <= 0:
            raise ValueError(
                "max_hypotheses must be greater than zero."
            )

        if max_cost is not None and max_cost < 0:
            raise ValueError(
                "max_cost must be non-negative."
            )

        if miss_cost < 0:
            raise ValueError(
                "miss_cost must be non-negative."
            )

        if clutter_cost < 0:
            raise ValueError(
                "clutter_cost must be non-negative."
            )

        self.cost_builder = cost_builder

        self.max_hypotheses = int(
            max_hypotheses
        )

        self.max_cost = (
            float(max_cost)
            if max_cost is not None
            else None
        )

        self.miss_cost = float(
            miss_cost
        )

        self.clutter_cost = float(
            clutter_cost
        )

    # ==================================================================
    # Associate
    # ==================================================================

    def associate(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> MHTResult:
        """
        Generate and rank multiple association hypotheses.

        No Track or DetectionResult is modified.
        """

        num_tracks = len(tracks)
        num_detections = len(detections)

        # --------------------------------------------------------------
        # Empty case
        # --------------------------------------------------------------

        if (
            num_tracks == 0
            and num_detections == 0
        ):
            return MHTResult(
                hypotheses=[],
                cost_matrix=np.empty(
                    (0, 0),
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

        cost_matrix = np.nan_to_num(
            cost_matrix,
            nan=np.inf,
            posinf=np.inf,
            neginf=np.inf,
        )

        # --------------------------------------------------------------
        # Generate hypotheses
        # --------------------------------------------------------------

        candidates: list[
            tuple[
                list[tuple[int, int]],
                float,
            ]
        ] = []

        self._generate_hypotheses(
            track_index=0,
            num_tracks=num_tracks,
            num_detections=num_detections,
            cost_matrix=cost_matrix,
            used_detections=set(),
            matches=[],
            total_cost=0.0,
            candidates=candidates,
        )

        # --------------------------------------------------------------
        # Convert candidates into MHTHypothesis objects
        # --------------------------------------------------------------

        hypotheses: list[MHTHypothesis] = []

        for matches, total_cost in candidates:

            matched_tracks = {
                track_index
                for track_index, _
                in matches
            }

            matched_detections = {
                detection_index
                for _, detection_index
                in matches
            }

            unmatched_tracks = [
                index
                for index in range(num_tracks)
                if index not in matched_tracks
            ]

            unmatched_detections = [
                index
                for index in range(num_detections)
                if index not in matched_detections
            ]

            score = self._calculate_score(
                total_cost
            )

            hypotheses.append(
                MHTHypothesis(
                    matches=list(matches),
                    unmatched_tracks=unmatched_tracks,
                    unmatched_detections=unmatched_detections,
                    total_cost=float(total_cost),
                    score=float(score),
                )
            )

        # --------------------------------------------------------------
        # Sort by cost
        # --------------------------------------------------------------

        hypotheses.sort(
            key=lambda hypothesis:
                hypothesis.total_cost
        )

        # --------------------------------------------------------------
        # Keep best N hypotheses
        # --------------------------------------------------------------

        hypotheses = hypotheses[
            :self.max_hypotheses
        ]

        return MHTResult(
            hypotheses=hypotheses,
            cost_matrix=cost_matrix,
        )

    # ==================================================================
    # Hypothesis Generation
    # ==================================================================

    def _generate_hypotheses(
        self,
        track_index: int,
        num_tracks: int,
        num_detections: int,
        cost_matrix: np.ndarray,
        used_detections: set[int],
        matches: list[tuple[int, int]],
        total_cost: float,
        candidates: list[
            tuple[
                list[tuple[int, int]],
                float,
            ]
        ],
    ) -> None:
        """
        Recursively generate feasible Track-Detection hypotheses.

        Each track has two types of choices:

            1. Match it with one unused valid detection.
            2. Leave the track unmatched.

        This guarantees one-to-one association.
        """

        # --------------------------------------------------------------
        # All tracks processed
        # --------------------------------------------------------------

        if track_index >= num_tracks:

            final_cost = (
                total_cost
                + (
                    self.clutter_cost
                    * (
                        num_detections
                        - len(used_detections)
                    )
                )
            )

            candidates.append(
                (
                    list(matches),
                    float(final_cost),
                )
            )

            return

        # --------------------------------------------------------------
        # Option 1:
        # Leave this track unmatched.
        # --------------------------------------------------------------

        self._generate_hypotheses(
            track_index=track_index + 1,
            num_tracks=num_tracks,
            num_detections=num_detections,
            cost_matrix=cost_matrix,
            used_detections=used_detections,
            matches=matches,
            total_cost=(
                total_cost
                + self.miss_cost
            ),
            candidates=candidates,
        )

        # --------------------------------------------------------------
        # Option 2:
        # Match this track with a valid detection.
        # --------------------------------------------------------------

        for detection_index in range(
            num_detections
        ):

            if detection_index in used_detections:
                continue

            cost = float(
                cost_matrix[
                    track_index,
                    detection_index,
                ]
            )

            if not np.isfinite(cost):
                continue

            if cost >= 1e9:
                continue

            if (
                self.max_cost is not None
                and cost > self.max_cost
            ):
                continue

            used_detections.add(
                detection_index
            )

            matches.append(
                (
                    track_index,
                    detection_index,
                )
            )

            self._generate_hypotheses(
                track_index=track_index + 1,
                num_tracks=num_tracks,
                num_detections=num_detections,
                cost_matrix=cost_matrix,
                used_detections=used_detections,
                matches=matches,
                total_cost=(
                    total_cost
                    + cost
                ),
                candidates=candidates,
            )

            matches.pop()

            used_detections.remove(
                detection_index
            )

    # ==================================================================
    # Score
    # ==================================================================

    @staticmethod
    def _calculate_score(
        total_cost: float,
    ) -> float:
        """
        Convert total hypothesis cost into a bounded score.

        Lower cost -> higher score.

        The exponential transformation prevents large costs from
        dominating downstream ranking.
        """

        if not np.isfinite(total_cost):
            return 0.0

        return float(
            np.exp(
                -0.5 * total_cost
            )
        )

    # ==================================================================
    # Convenience
    # ==================================================================

    def associate_objects(
        self,
        tracks: list[Track],
        detections: list[DetectionResult],
    ) -> MHTResult:
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
            "algorithm": "mht",
            "implementation": "bounded_hypothesis_generation",
            "metric": self.metric_name,
            "max_hypotheses": self.max_hypotheses,
            "max_cost": self.max_cost,
            "miss_cost": self.miss_cost,
            "clutter_cost": self.clutter_cost,
        }

    # ==================================================================

    def __repr__(self) -> str:

        return (
            "MHTAssociator("
            f"metric={self.metric_name}, "
            f"max_hypotheses={self.max_hypotheses}, "
            f"max_cost={self.max_cost})"
        )