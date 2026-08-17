"""
YOLO Recommendation Analyzer
"""

from datasets.statistics.analyzers.base_analyzer import BaseAnalyzer


class RecommendationAnalyzer(BaseAnalyzer):

    ###########################################################

    def analyze(self, dataset, statistics):

        self.log("Generating YOLO recommendations...")

        rec = statistics.recommendations

        img = statistics.images

        ann = statistics.annotations

        bbox = statistics.bounding_boxes

        #######################################################
        # Model Selection
        #######################################################

        if img.total_images < 10000:

            rec.model = "yolo11n.pt"

        elif img.total_images < 50000:

            rec.model = "yolo11s.pt"

        elif img.total_images < 150000:

            rec.model = "yolo11m.pt"

        else:

            rec.model = "yolo11l.pt"

        #######################################################
        # Image Size
        #######################################################

        if bbox.average_area < 0.005:

            rec.image_size = 960

            rec.multi_scale = True

            rec.notes.append(
                "Very small objects detected."
            )

        elif bbox.average_area < 0.015:

            rec.image_size = 768

        else:

            rec.image_size = 640

        #######################################################
        # Batch Size
        #######################################################

        if rec.image_size >= 960:

            rec.batch_size = 8

        elif rec.image_size >= 768:

            rec.batch_size = 16

        else:

            rec.batch_size = 32

        #######################################################
        # Epochs
        #######################################################

        if img.total_images > 50000:

            rec.epochs = 100

        else:

            rec.epochs = 150

        #######################################################
        # Mosaic
        #######################################################

        if bbox.small_objects > bbox.large_objects:

            rec.mosaic = True

            rec.notes.append(
                "Enable Mosaic for small object detection."
            )

        #######################################################
        # Copy Paste
        #######################################################

        if ann.average_objects_per_image < 2:

            rec.copy_paste = True

            rec.notes.append(
                "Sparse annotations detected."
            )

        #######################################################
        # MixUp
        #######################################################

        if len(ann.class_distribution) > 2:

            rec.mixup = True

        #######################################################
        # Workers
        #######################################################

        if img.total_images > 50000:

            rec.workers = 16

        else:

            rec.workers = 8

        #######################################################
        # Cache
        #######################################################

        rec.cache = True

        #######################################################
        # Optimizer
        #######################################################

        rec.optimizer = "AdamW"

        #######################################################

        self.log("Recommendation generation complete.")