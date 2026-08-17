"""
===========================================================
Training Evaluator
===========================================================
"""

from training.metrics import DetectionMetrics
from training.models.yolo.validator import YOLOValidator


class Evaluator:

    def __init__(

        self,

        detector,

        config,

    ):

        self.detector = detector

        self.config = config

    ############################################################

    def evaluate(self):

        validator = YOLOValidator(

            self.detector

        )

        results = validator.validate(

            data=str(self.config.data_yaml),

            imgsz=self.config.image_size,

            batch=self.config.batch_size,

        )

        metrics = DetectionMetrics()

        ########################################################

        metrics.precision = results.box.mp

        metrics.recall = results.box.mr

        metrics.map50 = results.box.map50

        metrics.map5095 = results.box.map

        metrics.fitness = results.fitness

        ########################################################

        metrics.preprocess_time = (

            results.speed["preprocess"]

        )

        metrics.inference_time = (

            results.speed["inference"]

        )

        metrics.postprocess_time = (

            results.speed["postprocess"]

        )

        return metrics