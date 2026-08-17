"""
YOLO Detection Converter
"""

from __future__ import annotations

import uuid
from datetime import datetime

from vision.converters.base_converter import BaseDetectionConverter


class YOLODetectionConverter(BaseDetectionConverter):

    def convert(

        self,

        results,

        *,

        sensor_id,

        frame_id,

        timestamp=None,

    ):

        if timestamp is None:
            timestamp = datetime.utcnow()

        detections = []

        #######################################################

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                detections.append(

                    self._convert_box(

                        box,

                        sensor_id,

                        frame_id,

                        timestamp,

                    )

                )

        return detections

    ###########################################################

    def _convert_box(

        self,

        box,

        sensor_id,

        frame_id,

        timestamp,

    ):

        raise NotImplementedError