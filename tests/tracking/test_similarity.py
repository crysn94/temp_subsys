from core.detection_result import DetectionResult
from core.geometry.point import Point3D
from tracking.similarity.euclidean import EuclideanMetric
from tracking.models.track import Track


def test_euclidean_metric_requires_position():

    metric = EuclideanMetric()

    assert metric.requires_position is True


def test_euclidean_metric_without_positions():

    metric = EuclideanMetric()

    track = Track()

    detection = DetectionResult(
        sensor=None,
        class_id=0,
        class_name="drone",
        confidence=0.9,
    )

    cost = metric.safe_compute(
        track,
        detection,
    )

    assert cost == metric.invalid_cost


def test_euclidean_metric_3d():

    metric = EuclideanMetric()

    detection = DetectionResult(
        sensor=None,
        class_id=0,
        class_name="drone",
        confidence=0.9,
        position=Point3D(
            x=103.0,
            y=204.0,
            z=55.0,
        ),
    )

    track = Track(
        current_detection=detection,
    )

    # Track and detection have the same position.
    cost = metric.safe_compute(
        track,
        detection,
    )

    assert cost == 0.0