from dataclasses import dataclass


@dataclass(slots=True)
class DetectionMetrics:

    precision: float = 0.0

    recall: float = 0.0

    map50: float = 0.0

    map5095: float = 0.0

    fitness: float = 0.0

    inference_time: float = 0.0

    preprocess_time: float = 0.0

    postprocess_time: float = 0.0