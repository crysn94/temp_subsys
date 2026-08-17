"""
===========================================================
Model Factory
===========================================================
"""

from training.models.yolo.detector import YOLODetector


class ModelFactory:

    @staticmethod
    def create(config):

        if config.model.startswith("yolo"):

            return YOLODetector(config)

        raise ValueError(
            f"Unsupported model: {config.model}"
        )