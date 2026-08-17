from enum import Enum


class Device(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Optimizer(str, Enum):
    SGD = "SGD"
    ADAM = "Adam"
    ADAMW = "AdamW"
    RMSPROP = "RMSProp"


class YOLOModel(str, Enum):
    YOLO11N = "yolo11n.pt"
    YOLO11S = "yolo11s.pt"
    YOLO11M = "yolo11m.pt"
    YOLO11L = "yolo11l.pt"
    YOLO11X = "yolo11x.pt"