"""
====================================================================
Sensor Factory
====================================================================

Creates reusable SensorIdentifier objects for testing.

====================================================================
"""

from core.sensor_identifier import (
    SensorIdentifier,
    SensorCategory,
)


class SensorFactory:

    _counter = 0

    @classmethod
    def _next_id(cls) -> str:
        cls._counter += 1
        return f"SENSOR_{cls._counter:04d}"

    # --------------------------------------------------------------
    # EO
    # --------------------------------------------------------------

    @classmethod
    def eo(cls):

        return SensorIdentifier(
            sensor_id=cls._next_id(),
            name="EO Camera",
            category=SensorCategory.EO,
        )

    # --------------------------------------------------------------

    @classmethod
    def ir(cls):

        return SensorIdentifier(
            sensor_id=cls._next_id(),
            name="IR Camera",
            category=SensorCategory.IR,
        )

    # --------------------------------------------------------------

    @classmethod
    def radar(cls):

        return SensorIdentifier(
            sensor_id=cls._next_id(),
            name="Radar",
            category=SensorCategory.RADAR,
        )

    # --------------------------------------------------------------

    @classmethod
    def rf(cls):

        return SensorIdentifier(
            sensor_id=cls._next_id(),
            name="RF Receiver",
            category=SensorCategory.RF,
        )

    # --------------------------------------------------------------

    @classmethod
    def acoustic(cls):

        return SensorIdentifier(
            sensor_id=cls._next_id(),
            name="Acoustic Array",
            category=SensorCategory.ACOUSTIC,
        )