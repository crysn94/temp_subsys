"""
======================================================================
Dataset Quality Analyzer
======================================================================
"""

from datasets.statistics.analyzers.base_analyzer import BaseAnalyzer


class QualityAnalyzer(BaseAnalyzer):

    ##################################################################

    def analyze(self, dataset, statistics):

        self.log("Computing dataset quality...")

        quality = statistics.quality

        image = statistics.images

        annotation = statistics.annotations

        bbox = statistics.bounding_boxes

        ##############################################################
        # Image Quality
        ##############################################################

        image_score = 100.0

        if image.corrupted_images > 0:

            image_score -= min(
                30,
                image.corrupted_images * 2
            )

        ##############################################################
        # Annotation Quality
        ##############################################################

        annotation_score = 100.0

        annotation_score -= annotation.malformed_annotations * 5

        annotation_score -= annotation.invalid_class_ids * 5

        annotation_score -= annotation.orphan_labels * 2

        annotation_score -= annotation.missing_labels * 2

        annotation_score = max(annotation_score, 0)

        ##############################################################
        # Bounding Box Quality
        ##############################################################

        bbox_score = 100.0

        if bbox.total_boxes == 0:

            bbox_score = 0

        ##############################################################
        # Class Balance
        ##############################################################

        balance_score = 100.0

        classes = annotation.class_distribution

        if len(classes) > 1:

            values = list(classes.values())

            ratio = min(values) / max(values)

            balance_score = ratio * 100

        ##############################################################
        # Dataset Integrity
        ##############################################################

        integrity = 100.0

        integrity -= image.corrupted_images

        integrity -= annotation.orphan_labels

        integrity -= annotation.missing_labels

        integrity = max(integrity, 0)

        ##############################################################
        # Final Score
        ##############################################################

        overall = (

            image_score +

            annotation_score +

            bbox_score +

            balance_score +

            integrity

        ) / 5

        quality.image_score = image_score

        quality.annotation_score = annotation_score

        quality.bbox_score = bbox_score

        quality.balance_score = balance_score

        quality.integrity_score = integrity

        quality.overall_score = overall

        quality.passed = overall >= 80

        ##############################################################
        # Recommendations
        ##############################################################

        if image.corrupted_images:

            quality.recommendations.append(
                "Remove corrupted images."
            )

        if annotation.missing_labels:

            quality.recommendations.append(
                "Generate missing label files."
            )

        if annotation.orphan_labels:

            quality.recommendations.append(
                "Remove orphan label files."
            )

        if annotation.invalid_class_ids:

            quality.recommendations.append(
                "Fix invalid class IDs."
            )

        if balance_score < 80:

            quality.recommendations.append(
                "Improve class balance."
            )

        if bbox.small_objects > bbox.large_objects * 20:

            quality.recommendations.append(
                "Dataset contains many small objects. Consider higher image size."
            )

        quality.warnings = len(
            quality.recommendations
        )

        self.log(
            f"Overall Dataset Score : {overall:.2f}"
        )