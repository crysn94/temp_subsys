"""
======================================================================
Annotation Analyzer
======================================================================
"""

from collections import Counter

from datasets.statistics.analyzers.base_analyzer import BaseAnalyzer


class AnnotationAnalyzer(BaseAnalyzer):

    ##################################################################

    def analyze(self, dataset, statistics):

        self.log("Starting annotation analysis...")

        ann = statistics.annotations

        class_counter = Counter()

        objects_per_image = []

        total_labels = 0
        total_annotations = 0

        negative_images = 0
        orphan_labels = 0
        missing_labels = 0
        invalid_class_ids = 0
        malformed_annotations = 0

        ##############################################################

        for split in dataset.splits:

            images = dataset.get_images(split)

            image_stems = {
                img.stem for img in images
            }

            labels = dataset.get_labels(split)

            label_stems = {
                lbl.stem for lbl in labels
            }

            ##########################################################
            # Missing Labels
            ##########################################################

            for stem in image_stems:

                if stem not in label_stems:

                    missing_labels += 1

            ##########################################################
            # Read Labels
            ##########################################################

            for label_path in labels:

                total_labels += 1

                if label_path.stem not in image_stems:

                    orphan_labels += 1
                    continue

                with open(label_path, "r") as f:

                    lines = [
                        line.strip()
                        for line in f.readlines()
                        if line.strip()
                    ]

                if len(lines) == 0:

                    negative_images += 1

                    objects_per_image.append(0)

                    continue

                objects = 0

                for line in lines:

                    parts = line.split()

                    if len(parts) < 5:

                        malformed_annotations += 1
                        continue

                    try:

                        cls = int(parts[0])

                    except Exception:

                        malformed_annotations += 1
                        continue

                    if cls < 0:

                        invalid_class_ids += 1
                        continue

                    class_counter[cls] += 1

                    total_annotations += 1

                    objects += 1

                objects_per_image.append(objects)

        ##############################################################

        ann.total_labels = total_labels

        ann.total_annotations = total_annotations

        ann.negative_images = negative_images

        ann.orphan_labels = orphan_labels

        ann.missing_labels = missing_labels

        ann.invalid_class_ids = invalid_class_ids

        ann.malformed_annotations = malformed_annotations

        ann.class_distribution = dict(class_counter)

        ##############################################################

        if len(objects_per_image):

            ann.average_objects_per_image = (

                sum(objects_per_image)

                / len(objects_per_image)

            )

            ann.minimum_objects_per_image = (

                min(objects_per_image)

            )

            ann.maximum_objects_per_image = (

                max(objects_per_image)

            )

        self.log("Annotation analysis complete.")