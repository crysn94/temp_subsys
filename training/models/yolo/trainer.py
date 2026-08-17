"""
===========================================================
YOLO Trainer
===========================================================
"""

from pathlib import Path
import yaml
from training.config import TrainingConfig
from training.models.yolo.detector import YOLODetector
from training.utils.device import DeviceManager

class YOLOTrainer:

    """
    Adapter around Ultralytics model.train().
    """

    def __init__(
        self,
        detector: YOLODetector,
        config: TrainingConfig,
    ):

        self.detector = detector
        self.config = config

    ############################################################

    def train(self):

        if self.detector.model is None:
            self.detector.load()

        print()

        print("=" * 70)
        print("STARTING YOLO TRAINING")
        print("=" * 70)

        device = self.config.device

        if device in ("", "auto"):
            device = DeviceManager.get_device()

        print(f"Training Device : {device}")

        results = self.detector.model.train(

            data=str(self.config.data_yaml),

            epochs=self.config.epochs,

            imgsz=self.config.image_size,

            batch=self.config.batch_size,

            workers=self.config.workers,

            device=device,

            optimizer=self.config.optimizer,

            lr0=self.config.learning_rate,

            weight_decay=self.config.weight_decay,

            momentum=self.config.momentum,

            project=str(self.config.project),

            name=self.config.experiment_name,

            exist_ok=self.config.overwrite,

            cache=self.config.cache,

            amp=self.config.amp,

            pretrained=self.config.pretrained,

            mosaic=1.0 if self.config.mosaic else 0.0,

            mixup=0.2 if self.config.mixup else 0.0,

            copy_paste=0.2 if self.config.copy_paste else 0.0,

            scale=self.config.scale,

            translate=self.config.translate,

            degrees=self.config.degrees,

            perspective=self.config.perspective,

            shear=self.config.shear,

            fliplr=self.config.fliplr,

            flipud=self.config.flipud,

            hsv_h=self.config.hsv_h,

            hsv_s=self.config.hsv_s,

            hsv_v=self.config.hsv_v,

            cos_lr=self.config.cosine_lr,

            warmup_epochs=self.config.warmup_epochs,

            close_mosaic=self.config.close_mosaic,

            patience=self.config.patience,

            val=self.config.run_validation,

            save=True,

            save_period=self.config.save_period,

            verbose=True,

        )

        return results