"""
YOLO Detector
"""

from ultralytics import YOLO

from training.models.base_model import BaseModel


class YOLODetector(BaseModel):

    def __init__(self, config):

        super().__init__(config)

        self.model = None

    ###########################################################

    def load(self):

        self.model = YOLO(
            self.config.model
        )

        return self.model

    ###########################################################

    def summary(self):

        print()

        print("=" * 60)

        print("YOLO MODEL")

        print("=" * 60)

        print("Weights :", self.config.model)

        print("Loaded  :", self.model is not None)


    ###########################################################

    def validate(self):

        raise NotImplementedError

    ###########################################################

    def predict(self):

        raise NotImplementedError

    ###########################################################

    def export(self):

        raise NotImplementedError