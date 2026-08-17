"""
========================================================================
Hybrid Association Metric Tests
========================================================================

Tests the canonical HybridMetric used by the C-UAS tracking system.

The HybridMetric combines:

    • Mahalanobis position distance
    • Motion distance
    • IoU
    • GIoU

The tests verify:

    • initialization
    • weight normalization
    • good association
    • poor association
    • partial measurements
    • invalid measurements
    • configuration
    • component metrics
    • numerical safety

========================================================================
"""

from __future__ import annotations

import numpy as np

from core.detection_result import DetectionResult
from core.geometry.bbox import BoundingBox2D
from core.geometry.point import Point3D
from core.geometry.velocity import Velocity3D
from core.sensor_identifier import (
    SensorIdentifier,
    SensorCategory,
)
from tracking.similarity import MahalanobisMetric
from tracking.models.state_vector import StateVector
from tracking.models.track import Track
from core.timestamps import Timestamp
from tracking.similarity.hybrid import HybridMetric


# ======================================================================
# Helpers
# ======================================================================


def create_state_vector(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> StateVector:
    """
    Create a 3D constant-velocity state vector.

    State:

        [x, y, z, vx, vy, vz]

    Covariance:

        6x6

    A timestamp is mandatory because a dynamic state estimate
    is meaningful only at a specific point in time.
    """

    state = np.array(
        [
            x,
            y,
            z,
            vx,
            vy,
            vz,
        ],
        dtype=float,
    )

    covariance = np.eye(
        6,
        dtype=float,
    )

    return StateVector(
        state=state,
        covariance=covariance,
        timestamp=Timestamp.now(),
    )


def create_sensor() -> SensorIdentifier:
    """
    Create a deterministic sensor identifier for testing.
    """

    return SensorIdentifier(
        sensor_id="test_sensor",
        name="Test Sensor",
        category=SensorCategory.EO,
    )


def create_detection(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    bbox: BoundingBox2D | None = None,
) -> DetectionResult:
    """
    Create a standard 3D DetectionResult.
    """

    return DetectionResult(
        sensor=create_sensor(),

        class_id=0,

        class_name="drone",

        confidence=0.95,

        position=Point3D(
            x=x,
            y=y,
            z=z,
        ),

        velocity=Velocity3D(
            vx=vx,
            vy=vy,
            vz=vz,
        ),

        bbox=bbox,
    )


def create_track(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    bbox: BoundingBox2D | None = None,
) -> Track:
    """
    Create a Track with a 3D state vector.

    The optional bbox is attached through current_detection so
    IoU/GIoU can operate on the Track.
    """

    track = Track(
        state_vector=create_state_vector(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
        )
    )

    if bbox is not None:

        track.current_detection = create_detection(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            bbox=bbox,
        )

    return track


# ======================================================================
# Initialization
# ======================================================================


def test_hybrid_metric_initialization():
    """
    Verify HybridMetric can be constructed with default parameters.
    """

    metric = HybridMetric()

    assert metric is not None

    assert metric.position_weight > 0.0

    assert metric.motion_weight > 0.0

    assert metric.iou_weight > 0.0

    assert metric.giou_weight > 0.0


# ======================================================================
# Weight Normalization
# ======================================================================


def test_weights_are_normalized():
    """
    Verify that arbitrary positive weights are normalized to 1.
    """

    metric = HybridMetric(
        position_weight=4.0,
        motion_weight=2.0,
        iou_weight=3.0,
        giou_weight=1.0,
    )

    total = (
        metric.position_weight
        + metric.motion_weight
        + metric.iou_weight
        + metric.giou_weight
    )

    assert np.isclose(
        total,
        1.0,
    )


# ======================================================================
# Invalid Weight
# ======================================================================


def test_negative_weight_is_rejected():
    """
    Negative association weights are not meaningful.
    """

    try:

        HybridMetric(
            position_weight=-1.0,
        )

    except ValueError:

        return

    raise AssertionError(
        "HybridMetric should reject negative weights."
    )


def test_all_zero_weights_are_rejected():
    """
    At least one metric must have a non-zero weight.
    """

    try:

        HybridMetric(
            position_weight=0.0,
            motion_weight=0.0,
            iou_weight=0.0,
            giou_weight=0.0,
        )

    except ValueError:

        return

    raise AssertionError(
        "HybridMetric should reject all-zero weights."
    )


# ======================================================================
# Good Association
# ======================================================================


def test_good_association_has_low_cost():
    """
    A detection close to the predicted track state should produce
    a relatively low association cost.
    """

    metric = HybridMetric()

    bbox = BoundingBox2D.from_xyxy(
        100.0,
        100.0,
        200.0,
        200.0,
    )

    track = create_track(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=5.0,
        vy=2.0,
        vz=1.0,
        bbox=bbox,
    )

    detection = create_detection(
        x=10.1,
        y=20.1,
        z=30.1,
        vx=5.0,
        vy=2.0,
        vz=1.0,
        bbox=bbox,
    )

    cost = metric.safe_compute(
        track,
        detection,
    )

    assert np.isfinite(cost)

    assert cost < metric.invalid_cost


# ======================================================================
# Poor Association
# ======================================================================


def test_poor_association_has_higher_cost():
    """
    A detection far away from the predicted track should have a
    substantially worse association cost than a nearby detection.
    """

    metric = HybridMetric()

    bbox = BoundingBox2D.from_xyxy(
        100.0,
        100.0,
        200.0,
        200.0,
    )

    track = create_track(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=5.0,
        vy=2.0,
        vz=1.0,
        bbox=bbox,
    )

    good_detection = create_detection(
        x=10.1,
        y=20.1,
        z=30.1,
        vx=5.0,
        vy=2.0,
        vz=1.0,
        bbox=bbox,
    )

    bad_detection = create_detection(
        x=1000.0,
        y=1000.0,
        z=1000.0,
        vx=-100.0,
        vy=-100.0,
        vz=-100.0,
        bbox=BoundingBox2D.from_xyxy(
            500.0,
            500.0,
            600.0,
            600.0,
        ),
    )

    good_cost = metric.safe_compute(
        track,
        good_detection,
    )

    bad_cost = metric.safe_compute(
        track,
        bad_detection,
    )

    assert good_cost < bad_cost


# ======================================================================
# Missing Position
# ======================================================================


def test_missing_position_does_not_crash():
    """
    A detection without position should not crash HybridMetric.

    Other available components may still be evaluated.
    """

    metric = HybridMetric(
        position_weight=0.5,
        motion_weight=0.5,
        iou_weight=0.0,
        giou_weight=0.0,
    )

    track = create_track(
        x=10.0,
        y=10.0,
        z=10.0,
        vx=1.0,
        vy=1.0,
        vz=1.0,
    )

    detection = DetectionResult(
        sensor=create_sensor(),

        class_id=0,

        class_name="drone",

        confidence=0.9,

        position=None,

        velocity=Velocity3D(
            1.0,
            1.0,
            1.0,
        ),
    )

    cost = metric.safe_compute(
        track,
        detection,
    )

    assert np.isfinite(cost)

    assert cost < metric.invalid_cost


# ======================================================================
# Missing Velocity
# ======================================================================


def test_missing_velocity_does_not_crash():
    """
    Position-only detections should remain usable by HybridMetric.
    """

    metric = HybridMetric(
        position_weight=1.0,
        motion_weight=0.0,
        iou_weight=0.0,
        giou_weight=0.0,
    )

    track = create_track(
        x=10.0,
        y=10.0,
        z=10.0,
        vx=1.0,
        vy=1.0,
        vz=1.0,
    )

    detection = DetectionResult(
        sensor=create_sensor(),

        class_id=0,

        class_name="drone",

        confidence=0.9,

        position=Point3D(
            10.1,
            10.1,
            10.1,
        ),

        velocity=None,
    )

    cost = metric.safe_compute(
        track,
        detection,
    )

    assert np.isfinite(cost)

    assert cost < metric.invalid_cost


# ======================================================================
# Missing Bounding Box
# ======================================================================


def test_missing_bbox_does_not_crash():
    """
    A 3D sensor detection without an image bounding box should still
    be associable through position/motion metrics.
    """

    metric = HybridMetric(
        position_weight=0.6,
        motion_weight=0.4,
        iou_weight=0.0,
        giou_weight=0.0,
    )

    track = create_track(
        x=10.0,
        y=20.0,
        z=30.0,
        vx=5.0,
        vy=2.0,
        vz=1.0,
    )

    detection = create_detection(
        x=10.1,
        y=20.1,
        z=30.1,
        vx=5.0,
        vy=2.0,
        vz=1.0,
        bbox=None,
    )

    cost = metric.safe_compute(
        track,
        detection,
    )

    assert np.isfinite(cost)

    assert cost < metric.invalid_cost


# ======================================================================
# Bounding Box Association
# ======================================================================


def test_bbox_metrics_are_used_when_available():
    """
    Verify that HybridMetric can operate with IoU/GIoU information.
    """

    metric = HybridMetric(
        position_weight=0.0,
        motion_weight=0.0,
        iou_weight=0.6,
        giou_weight=0.4,
    )

    track_bbox = BoundingBox2D.from_xyxy(
        100.0,
        100.0,
        200.0,
        200.0,
    )

    detection_bbox = BoundingBox2D.from_xyxy(
        105.0,
        105.0,
        205.0,
        205.0,
    )

    track = create_track(
        bbox=track_bbox,
    )

    detection = create_detection(
        bbox=detection_bbox,
    )

    cost = metric.safe_compute(
        track,
        detection,
    )

    assert np.isfinite(cost)

    assert cost < metric.invalid_cost


# ======================================================================
# No Usable Measurement
# ======================================================================


def test_no_usable_measurement_returns_invalid_cost():
    """
    If all configured components are unusable, HybridMetric must
    return invalid_cost rather than raising an exception.
    """

    metric = HybridMetric(
        position_weight=1.0,
        motion_weight=0.0,
        iou_weight=0.0,
        giou_weight=0.0,
    )

    track = Track()

    detection = DetectionResult(
        sensor=create_sensor(),

        class_id=0,

        class_name="drone",

        confidence=0.9,

        position=None,

        velocity=None,
    )

    cost = metric.safe_compute(
        track,
        detection,
    )

    assert cost == metric.invalid_cost


# ======================================================================
# Configuration
# ======================================================================


def test_get_config():
    """
    Verify HybridMetric produces a serializable configuration.
    """

    metric = HybridMetric()

    config = metric.get_config()

    assert isinstance(
        config,
        dict,
    )

    assert "name" in config

    assert "invalid_cost" in config

    assert "weights" in config

    assert "components" in config

    assert config["weights"]["position"] > 0.0

    assert config["weights"]["motion"] > 0.0

    assert config["weights"]["iou"] > 0.0

    assert config["weights"]["giou"] > 0.0


# ======================================================================
# Component Metrics
# ======================================================================


def test_component_metrics():
    """
    Verify all configured component metrics are exposed.
    """

    metric = HybridMetric()

    components = metric.metrics

    assert "position" in components

    assert "motion" in components

    assert "iou" in components

    assert "giou" in components

    assert isinstance(
        components["position"],
        MahalanobisMetric,
    )