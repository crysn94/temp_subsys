"""
======================================================================
Training Configuration
======================================================================

Central configuration used by the entire YOLOv11 training pipeline.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

@dataclass(slots=True)
class TrainingConfig:

    ##################################################################
    # Dataset
    ##################################################################
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    dataset_root: Path = PROJECT_ROOT / "Dataset"

    data_yaml: Path = PROJECT_ROOT / "Dataset" / "data.yaml"

    project: Path = PROJECT_ROOT / "training" / "runs"

    ##################################################################
    # Model
    ##################################################################

    model: str = "yolo11s.pt"
    pretrained: bool = True

    ##################################################################
    # Training
    ##################################################################

    epochs: int = 10
    batch_size: int = 16
    image_size: int = 640

    device: str = ""

    workers: int = 8

    cache: bool = True

    amp: bool = True

    deterministic: bool = False

    seed: int = 42

    ##################################################################
    # Optimizer
    ##################################################################

    optimizer: str = "AdamW"

    learning_rate: float = 0.001

    weight_decay: float = 0.0005

    momentum: float = 0.937

    ##################################################################
    # Scheduler
    ##################################################################

    cosine_lr: bool = True

    warmup_epochs: int = 3

    ##################################################################
    # Augmentation
    ##################################################################

    mosaic: bool = True

    mixup: bool = False

    copy_paste: bool = False

    multi_scale: bool = False

    hsv_h: float = 0.015

    hsv_s: float = 0.70

    hsv_v: float = 0.40

    degrees: float = 10.0

    translate: float = 0.10

    scale: float = 0.50

    shear: float = 2.0

    perspective: float = 0.0

    flipud: float = 0.0

    fliplr: float = 0.50

    close_mosaic: int = 10

    ##################################################################
    # Validation
    ##################################################################

    run_validation: bool = True

    save_best: bool = True

    save_period: int = 10

    patience: int = 50

    ##################################################################
    # Output
    ##################################################################

    project: Path = Path("training/runs")

    experiment_name: str = "bird_drone"

    overwrite: bool = False

    ##################################################################
    # Export
    ##################################################################

    export_onnx: bool = False

    export_tensorrt: bool = False

    export_openvino: bool = False

    ##################################################################
    # Metadata
    ##################################################################

    notes: list[str] = field(default_factory=list)

    ####################################################################

    def validate(self):

        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self.image_size < 320:
            raise ValueError("image_size too small")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: {self.data_yaml}"
            )

    ####################################################################

    @classmethod
    def from_statistics(cls, stats: Any) -> "TrainingConfig":

        config = cls()

        rec = stats.recommendations

        config.model = rec.model
        config.image_size = rec.image_size
        config.batch_size = rec.batch_size
        config.epochs = rec.epochs
        config.workers = rec.workers
        config.mosaic = rec.mosaic
        config.mixup = rec.mixup
        config.copy_paste = rec.copy_paste
        config.multi_scale = rec.multi_scale

        return config

    ####################################################################

    def to_dict(self) -> dict:

        return asdict(self)