"""
===========================================================
Training Pipeline
===========================================================
"""

from training.models.model_factory import ModelFactory
from training.models.yolo.trainer import YOLOTrainer
from datasets.data_yaml import DataYamlGenerator

class Trainer:

    def __init__(self, config):

        self.config = config

        self.detector = ModelFactory.create(config)

    ############################################################

    def run(self):
        generator = DataYamlGenerator(
            self.config.dataset_root
        )

        self.config.data_yaml = generator.generate()

        trainer = YOLOTrainer(
            self.detector,
            self.config
        )

        return trainer.train()