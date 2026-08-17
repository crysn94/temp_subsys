from core.geometry.bbox import BoundingBox2D
from core.payloads import EOIRPayload
from core.sensor_identifier import (
    SensorCategory,
    SensorIdentifier,
)

from core.detection_result import DetectionResult

sensor = SensorIdentifier(

    sensor_id="EO_FRONT",

    name="Front Camera",

    category=SensorCategory.EO,

)

payload = EOIRPayload(

    bbox=BoundingBox2D.from_xyxy(
        100,
        120,
        300,
        350,
    ),

    confidence=0.97,

)

det = DetectionResult(

    sensor=sensor,

    class_id=1,

    class_name="Drone",

    confidence=0.97,

    payload=payload,

)

print("="*60)
print(det)
print("="*60)

print()

print(det.as_dict())