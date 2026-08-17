"""
========================================================================
2D IoU Association Metric
========================================================================

Computes Intersection over Union (IoU) between the Track bounding box
and the DetectionResult bounding box.

IoU is an IMAGE-SPACE association metric.

It is therefore independent of the 3D tracking state.

IoU:

    intersection_area
    -----------------
        union_area

Range:

    0.0 <= IoU <= 1.0

Association cost:

    cost = 1.0 - IoU

Therefore:

    IoU = 1.0
        -> cost = 0.0
        -> perfect overlap

    IoU = 0.0
        -> cost = 1.0
        -> no overlap

Lower cost
    -> better association

========================================================================
"""

from __future__ import annotations

from core.detection_result import DetectionResult
from core.geometry import BoundingBox2D

from tracking.models.track import Track
from tracking.similarity.base_metric import BaseMetric


class IoUMetric(BaseMetric):

    """
    Intersection over Union association metric.

    This metric compares 2D image-space bounding boxes.

    It does NOT use:

        • 3D position
        • velocity
        • acceleration
        • covariance

    Those quantities belong to other tracking metrics.
    """

    # ==================================================================
    # Constructor
    # ==================================================================

    def __init__(
        self,
        invalid_cost: float = 1e9,
        min_iou: float = 0.0,
    ) -> None:

        super().__init__(
            invalid_cost=invalid_cost
        )

        if not 0.0 <= min_iou <= 1.0:
            raise ValueError(
                "min_iou must be between 0 and 1."
            )

        self.min_iou = float(
            min_iou
        )

    # ==================================================================
    # Bounding Box Extraction
    # ==================================================================

    @staticmethod
    def _get_track_bbox(
        track: Track,
    ) -> BoundingBox2D | None:
        """
        Obtain the latest 2D image bounding box from a Track.

        The Track is expected to expose a ``bbox`` property.

        The fallback to ``current_detection.bbox`` is retained so
        that the metric remains compatible while the Track model
        is being finalized.
        """

        bbox = getattr(
            track,
            "bbox",
            None,
        )

        if isinstance(
            bbox,
            BoundingBox2D,
        ):
            return bbox

        detection = getattr(
            track,
            "current_detection",
            None,
        )

        if detection is None:
            return None

        bbox = getattr(
            detection,
            "bbox",
            None,
        )

        if isinstance(
            bbox,
            BoundingBox2D,
        ):
            return bbox

        return None

    # ==================================================================

    @staticmethod
    def _get_detection_bbox(
        detection: DetectionResult,
    ) -> BoundingBox2D | None:
        """
        Obtain the 2D image bounding box from DetectionResult.

        DetectionResult should eventually expose:

            bbox: BoundingBox2D | None

        Returning None here allows the association layer to fail
        safely until that field is available.
        """

        bbox = getattr(
            detection,
            "bbox",
            None,
        )

        if isinstance(
            bbox,
            BoundingBox2D,
        ):
            return bbox

        return None

    # ==================================================================
    # IoU Calculation
    # ==================================================================

    @staticmethod
    def _calculate_iou(
        track_bbox: BoundingBox2D,
        detection_bbox: BoundingBox2D,
    ) -> float:
        """
        Calculate IoU between two bounding boxes.
        """

        if not track_bbox.is_valid:
            return 0.0

        if not detection_bbox.is_valid:
            return 0.0

        intersection = (
            track_bbox.intersection(
                detection_bbox
            )
        )

        union = (
            track_bbox.union(
                detection_bbox
            )
        )

        if union <= 0.0:
            return 0.0

        iou = (
            intersection
            / union
        )

        # Numerical protection.
        return max(
            0.0,
            min(
                1.0,
                float(iou),
            ),
        )

    # ==================================================================
    # Compute
    # ==================================================================

    def compute(
        self,
        track: Track,
        detection: DetectionResult,
    ) -> float:
        """
        Compute IoU-based association cost.

        Returns:

            1 - IoU

        Lower value means better association.
        """

        # --------------------------------------------------------------
        # Track bounding box
        # --------------------------------------------------------------

        track_bbox = (
            self._get_track_bbox(
                track
            )
        )

        if track_bbox is None:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Detection bounding box
        # --------------------------------------------------------------

        detection_bbox = (
            self._get_detection_bbox(
                detection
            )
        )

        if detection_bbox is None:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Calculate IoU
        # --------------------------------------------------------------

        iou = self._calculate_iou(
            track_bbox,
            detection_bbox,
        )

        # --------------------------------------------------------------
        # Optional gating
        # --------------------------------------------------------------

        if iou < self.min_iou:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Convert similarity to cost
        # --------------------------------------------------------------

        cost = 1.0 - iou

        return float(
            cost
        )

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def requires_position(self) -> bool:
        """
        IoU does not require 3D position.
        """

        return False

    # ==================================================================

    @property
    def requires_bbox(self) -> bool:
        """
        IoU requires 2D image bounding boxes.
        """

        return True

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_config(self) -> dict:

        config = super().get_config()

        config.update(
            {
                "distance_type": "iou",

                "similarity": "intersection_over_union",

                "association_cost":
                    "1_minus_iou",

                "min_iou":
                    self.min_iou,

                "dimensions":
                    2,

                "coordinate_space":
                    "image",
            }
        )

        return config

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            "IoUMetric("
            f"min_iou={self.min_iou}, "
            f"invalid_cost={self.invalid_cost})"
        )