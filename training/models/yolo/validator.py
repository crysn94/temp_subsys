"""
===========================================================
YOLO Validator
===========================================================
"""

from training.models.yolo.detector import YOLODetector


class YOLOValidator:

    def __init__(self, detector):

        self.detector = detector

    ############################################################

    def validate(self, **kwargs):

        if self.detector.model is None:

            self.detector.load()

        results = self.detector.model.val(**kwargs)

        return results