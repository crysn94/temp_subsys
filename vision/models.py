from pathlib import Path

from ultralytics import YOLO


class RuntimeModel:

    def __init__(self, weights: Path):

        self.weights = Path(weights)

        self.model = None

    ############################################################

    def load(self):

        self.model = YOLO(self.weights)

        return self.model

    ############################################################

    def unload(self):

        self.model = None