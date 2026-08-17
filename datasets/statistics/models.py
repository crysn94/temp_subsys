from dataclasses import dataclass, field


@dataclass
class ImageStatistics:

    total_images: int = 0

    average_width: float = 0

    average_height: float = 0

    minimum_width: int = 0

    maximum_width: int = 0

    minimum_height: int = 0

    maximum_height: int = 0

    average_aspect_ratio: float = 0


@dataclass
class AnnotationStatistics:

    total_labels: int = 0

    total_annotations: int = 0

    negative_images: int = 0

    orphan_labels: int = 0

    missing_labels: int = 0

    invalid_labels: int = 0

    invalid_class_ids: int = 0

    malformed_annotations: int = 0

    average_objects_per_image: float = 0.0

    minimum_objects_per_image: int = 0

    maximum_objects_per_image: int = 0

    class_distribution: dict[int, int] = field(default_factory=dict)


@dataclass
class BoundingBoxStatistics:

    total_boxes: int = 0

    average_width: float = 0.0
    average_height: float = 0.0

    average_area: float = 0.0
    average_aspect_ratio: float = 0.0

    minimum_width: float = 0.0
    maximum_width: float = 0.0

    minimum_height: float = 0.0
    maximum_height: float = 0.0

    smallest_area: float = 0.0
    largest_area: float = 0.0

    average_occupancy: float = 0.0

    small_objects: int = 0
    medium_objects: int = 0
    large_objects: int = 0

    edge_objects: int = 0

    center_x_distribution: dict[str, int] = field(default_factory=dict)
    center_y_distribution: dict[str, int] = field(default_factory=dict)

@dataclass
class QualityStatistics:

    overall_score: float = 0.0

    passed: bool = False

    image_score: float = 0.0

    annotation_score: float = 0.0

    bbox_score: float = 0.0

    balance_score: float = 0.0

    integrity_score: float = 0.0

    warnings: int = 0

    errors: int = 0

    recommendations: list[str] = field(default_factory=list)

@dataclass
class RecommendationStatistics:

    # YOLO Model
    model: str = "yolo11s.pt"

    image_size: int = 640

    batch_size: int = 16

    epochs: int = 100

    workers: int = 8

    optimizer: str = "AdamW"

    learning_rate: float = 0.001

    cache: bool = True

    amp: bool = True

    multi_scale: bool = False

    mosaic: bool = True

    mixup: bool = False

    copy_paste: bool = False

    hsv_h: float = 0.015

    hsv_s: float = 0.7

    hsv_v: float = 0.4

    degrees: float = 10.0

    translate: float = 0.10

    scale: float = 0.50

    shear: float = 2.0

    perspective: float = 0.0

    flipud: float = 0.0

    fliplr: float = 0.50

    close_mosaic: int = 10

    notes: list[str] = field(default_factory=list)

# ===============================================================
# Object Size / Anchor Statistics
# ===============================================================

@dataclass
class AnchorStatistics:

    total_boxes: int = 0

    average_width: float = 0.0
    average_height: float = 0.0

    average_area: float = 0.0

    average_aspect_ratio: float = 0.0

    median_width: float = 0.0
    median_height: float = 0.0

    median_area: float = 0.0

    small_objects: int = 0
    medium_objects: int = 0
    large_objects: int = 0

    width_percentiles: dict[str, float] = field(default_factory=dict)

    height_percentiles: dict[str, float] = field(default_factory=dict)

    area_percentiles: dict[str, float] = field(default_factory=dict)

    recommendations: list[str] = field(default_factory=list)

@dataclass
class DatasetStatisticsModel:

    images: ImageStatistics = field(
        default_factory=ImageStatistics
    )

    annotations: AnnotationStatistics = field(
        default_factory=AnnotationStatistics
    )

    bounding_boxes: BoundingBoxStatistics = field(
        default_factory=BoundingBoxStatistics
    )

    quality: QualityStatistics = field(
        default_factory=QualityStatistics
    )

    recommendations: RecommendationStatistics = field(
        default_factory=RecommendationStatistics
    )

    anchors: AnchorStatistics = field(
        default_factory=AnchorStatistics
    )