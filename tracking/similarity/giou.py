"""
========================================================================
Generalized IoU Association Metric
========================================================================

Computes the Generalized Intersection over Union (GIoU) between a
tracked object's latest image bounding box and a detection bounding box.

GIoU is an extension of IoU that provides a meaningful penalty when
two bounding boxes do not overlap.

Association convention:

    Higher GIoU  -> better geometric match
    Lower GIoU   -> worse geometric match

Since the tracking association framework uses:

    lower cost = better match

the metric returns:

    cost = 1 - GIoU

Therefore:

    GIoU =  1.0  -> cost = 0.0
    GIoU =  0.0  -> cost = 1.0
    GIoU <  0.0  -> cost > 1.0

========================================================================
"""

from __future__ import annotations

from core.detection_result import DetectionResult
from tracking.models.track import Track
from tracking.similarity.base_metric import BaseMetric


class GIoUMetric(BaseMetric):
    """
    Generalized IoU association metric.

    Compares:

        Track bounding box
            vs
        Detection bounding box

    The returned value is an association cost.

    Lower cost = better association.
    """

    # ==================================================================
    # Constructor
    # ==================================================================

    def __init__(
        self,
        invalid_cost: float = 1e9,
    ) -> None:

        super().__init__(
            invalid_cost=invalid_cost
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
        Compute GIoU-based association cost.

        Returns
        -------
        float
            ``1 - GIoU``

        Invalid or unavailable bounding boxes return
        ``self.invalid_cost``.
        """

        # --------------------------------------------------------------
        # Validate track bounding box
        # --------------------------------------------------------------

        track_bbox = track.bbox

        if track_bbox is None:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Validate detection bounding box
        # --------------------------------------------------------------

        detection_bbox = detection.bbox

        if detection_bbox is None:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Validate geometry
        # --------------------------------------------------------------

        if not track_bbox.is_valid:
            return self.invalid_cost

        if not detection_bbox.is_valid:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Areas
        # --------------------------------------------------------------

        track_area = track_bbox.area
        detection_area = detection_bbox.area

        if track_area <= 0.0:
            return self.invalid_cost

        if detection_area <= 0.0:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Intersection
        # --------------------------------------------------------------

        intersection = track_bbox.intersection(
            detection_bbox
        )

        # --------------------------------------------------------------
        # Union
        # --------------------------------------------------------------

        union = (
            track_area
            + detection_area
            - intersection
        )

        if union <= 0.0:
            return self.invalid_cost

        # --------------------------------------------------------------
        # IoU
        # --------------------------------------------------------------

        iou = intersection / union

        # --------------------------------------------------------------
        # Smallest enclosing bounding box
        # --------------------------------------------------------------

        enclosing_x1 = min(
            track_bbox.x1,
            detection_bbox.x1,
        )

        enclosing_y1 = min(
            track_bbox.y1,
            detection_bbox.y1,
        )

        enclosing_x2 = max(
            track_bbox.x2,
            detection_bbox.x2,
        )

        enclosing_y2 = max(
            track_bbox.y2,
            detection_bbox.y2,
        )

        enclosing_width = (
            enclosing_x2
            - enclosing_x1
        )

        enclosing_height = (
            enclosing_y2
            - enclosing_y1
        )

        enclosing_area = (
            enclosing_width
            * enclosing_height
        )

        if enclosing_area <= 0.0:
            return self.invalid_cost

        # --------------------------------------------------------------
        # Generalized IoU
        #
        # GIoU = IoU - (C - U) / C
        #
        # where:
        #
        # C = area of smallest enclosing box
        # U = union area
        # --------------------------------------------------------------

        giou = (
            iou
            -
            (
                enclosing_area
                - union
            )
            / enclosing_area
        )

        # --------------------------------------------------------------
        # Convert similarity to cost
        # --------------------------------------------------------------

        cost = 1.0 - giou

        # --------------------------------------------------------------
        # Numerical safety
        # --------------------------------------------------------------

        if cost != cost:
            return self.invalid_cost

        if cost == float("inf"):
            return self.invalid_cost

        if cost == float("-inf"):
            return self.invalid_cost

        if cost < 0.0:
            return self.invalid_cost

        return float(cost)

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def requires_position(self) -> bool:
        """
        GIoU does not require 3D position.
        """

        return False

    @property
    def requires_bbox(self) -> bool:
        """
        GIoU requires image bounding boxes.
        """

        return True

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_config(self) -> dict:

        config = super().get_config()

        config.update(
            {
                "dimensions": 2,
                "metric_type": "GIoU",
                "output_type": "cost",
            }
        )

        return config