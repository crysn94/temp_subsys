from datasets.dataset_manager import DatasetManager
from datasets.statistics.models import DatasetStatisticsModel
from datasets.statistics.analyzers.image_analyzer import ImageAnalyzer
from datasets.statistics.analyzers.annotation_analyzer import AnnotationAnalyzer
from datasets.statistics.analyzers.bbox_analyzer import BoundingBoxAnalyzer
from datasets.statistics.analyzers.quality_analyzer import QualityAnalyzer
from datasets.statistics.analyzers.recommendation_analyzer import RecommendationAnalyzer
from datasets.statistics.analyzers.anchor_analyzer import AnchorAnalyzer

class DatasetStatistics:

    def __init__(self, dataset_root):

        self.dataset = DatasetManager(dataset_root)

        self.stats = DatasetStatisticsModel()

        self.analyzers = [

            ImageAnalyzer(),
            AnnotationAnalyzer(),
            BoundingBoxAnalyzer(),
            QualityAnalyzer(),
            RecommendationAnalyzer(),
            AnchorAnalyzer(),

        ]

    ##################################################################

    def analyze(self):

        for analyzer in self.analyzers:

            analyzer.analyze(
                self.dataset,
                self.stats
            )

        return self.stats

    ##################################################################

    def print_summary(self):

        img = self.stats.images

        print()

        print("=" * 70)
        print("IMAGE STATISTICS")
        print("=" * 70)

        print(f"Total Images            : {img.total_images:,}")
        print(f"Corrupted Images        : {img.corrupted_images:,}")

        print()

        print(f"Average Width           : {img.average_width:.2f}")
        print(f"Average Height          : {img.average_height:.2f}")
        print(f"Average Aspect Ratio    : {img.average_aspect_ratio:.3f}")

        print()

        print(
            f"Minimum Resolution      : "
            f"{img.minimum_width} x {img.minimum_height}"
        )

        print(
            f"Maximum Resolution      : "
            f"{img.maximum_width} x {img.maximum_height}"
        )

        print()

        print("Resolution Distribution")

        for resolution, count in sorted(
            img.resolution_distribution.items()
        ):
            print(f"  {resolution:<12} : {count:,}")

        print()

        print("Image Formats")

        for fmt, count in sorted(img.image_formats.items()):
            print(f"  {fmt:<10} : {count:,}")

        print()

        print("Color Modes")

        for mode, count in sorted(img.color_modes.items()):
            print(f"  {mode:<10} : {count:,}")

        print("=" * 70)

        ann = self.stats.annotations

        print()
        print("=" * 70)
        print("ANNOTATION STATISTICS")
        print("=" * 70)

        print(f"Label Files            : {ann.total_labels:,}")

        print(f"Objects                : {ann.total_annotations:,}")

        print(f"Negative Images        : {ann.negative_images:,}")

        print(f"Missing Labels         : {ann.missing_labels:,}")

        print(f"Orphan Labels          : {ann.orphan_labels:,}")

        print(f"Malformed Labels       : {ann.malformed_annotations:,}")

        print(f"Invalid Class IDs      : {ann.invalid_class_ids:,}")

        print()

        print(f"Objects/Image Avg      : {ann.average_objects_per_image:.2f}")

        print(f"Objects/Image Min      : {ann.minimum_objects_per_image}")

        print(f"Objects/Image Max      : {ann.maximum_objects_per_image}")

        print()

        print("Class Distribution")

        for cls, count in sorted(
                ann.class_distribution.items()
        ):
            print(f"  Class {cls:<3} : {count:,}")

        bbox = self.stats.bounding_boxes

        print()
        print("=" * 70)
        print("BOUNDING BOX STATISTICS")
        print("=" * 70)

        print(f"Total Boxes            : {bbox.total_boxes:,}")

        print()

        print(f"Average Width          : {bbox.average_width:.4f}")

        print(f"Average Height         : {bbox.average_height:.4f}")

        print(f"Average Area           : {bbox.average_area:.6f}")

        print(f"Average Aspect Ratio   : {bbox.average_aspect_ratio:.3f}")

        print()

        print(f"Small Objects          : {bbox.small_objects:,}")

        print(f"Medium Objects         : {bbox.medium_objects:,}")

        print(f"Large Objects          : {bbox.large_objects:,}")

        print()

        print(f"Edge Objects           : {bbox.edge_objects:,}")

        quality = self.stats.quality

        print()
        print("=" * 70)
        print("DATASET QUALITY")
        print("=" * 70)

        print(f"Overall Score        : {quality.overall_score:.2f}/100")

        print(f"Passed              : {quality.passed}")

        print()

        print(f"Image Score         : {quality.image_score:.2f}")

        print(f"Annotation Score    : {quality.annotation_score:.2f}")

        print(f"Bounding Box Score  : {quality.bbox_score:.2f}")

        print(f"Balance Score       : {quality.balance_score:.2f}")

        print(f"Integrity Score     : {quality.integrity_score:.2f}")

        print()

        if quality.recommendations:

            print("Recommendations")

            for recommendation in quality.recommendations:
                print(f"  • {recommendation}")

        else:

            print("No recommendations.")

        rec = self.stats.recommendations

        print()
        print("=" * 70)
        print("YOLO TRAINING RECOMMENDATIONS")
        print("=" * 70)

        print(f"Model              : {rec.model}")
        print(f"Image Size         : {rec.image_size}")
        print(f"Batch Size         : {rec.batch_size}")
        print(f"Epochs             : {rec.epochs}")
        print(f"Workers            : {rec.workers}")
        print(f"Optimizer          : {rec.optimizer}")
        print(f"Learning Rate      : {rec.learning_rate}")
        print(f"Cache              : {rec.cache}")
        print(f"AMP                : {rec.amp}")
        print(f"Multi Scale        : {rec.multi_scale}")
        print(f"Mosaic             : {rec.mosaic}")
        print(f"MixUp              : {rec.mixup}")
        print(f"Copy Paste         : {rec.copy_paste}")

        if rec.notes:
            print("\nRecommendations:")
            for note in rec.notes:
                print(f"  • {note}")

        print("=" * 70)

        anchor = self.stats.anchors

        print()
        print("=" * 70)
        print("OBJECT SIZE ANALYSIS")
        print("=" * 70)

        print(f"Total Objects         : {anchor.total_boxes:,}")

        print()

        print(f"Average Width         : {anchor.average_width:.4f}")

        print(f"Average Height        : {anchor.average_height:.4f}")

        print(f"Average Area          : {anchor.average_area:.6f}")

        print()

        print(f"Median Width          : {anchor.median_width:.4f}")

        print(f"Median Height         : {anchor.median_height:.4f}")

        print(f"Median Area           : {anchor.median_area:.6f}")

        print()

        print(f"Small Objects         : {anchor.small_objects:,}")

        print(f"Medium Objects        : {anchor.medium_objects:,}")

        print(f"Large Objects         : {anchor.large_objects:,}")

        print()

        print("Area Percentiles")

        for p, value in anchor.area_percentiles.items():
            print(f"  P{p:<3}: {value:.6f}")

        print()

        if anchor.recommendations:

            print("Recommendations")

            for r in anchor.recommendations:
                print(f"  • {r}")