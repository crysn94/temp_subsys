"""
========================================================================
Timestamp Utilities
========================================================================

Canonical timestamp representation used throughout the C-UAS framework.

Features

• UTC timezone aware
• ISO-8601 serialization
• Unix timestamp conversion
• Time arithmetic
• Comparison operators

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True, slots=True, order=True)
class Timestamp:

    value: datetime

    ####################################################################
    # Constructors
    ####################################################################

    @classmethod
    def now(cls) -> "Timestamp":

        return cls(datetime.now(UTC))

    @classmethod
    def from_datetime(
        cls,
        dt: datetime,
    ) -> "Timestamp":

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        return cls(dt.astimezone(UTC))

    @classmethod
    def from_unix(
        cls,
        seconds: float,
    ) -> "Timestamp":

        return cls(
            datetime.fromtimestamp(
                seconds,
                UTC,
            )
        )

    ####################################################################
    # Conversion
    ####################################################################

    def to_datetime(self) -> datetime:

        return self.value

    def to_unix(self) -> float:

        return self.value.timestamp()

    def isoformat(self) -> str:

        return self.value.isoformat()

    ####################################################################
    # Time Arithmetic
    ####################################################################

    def age(self) -> timedelta:

        return datetime.now(UTC) - self.value

    def elapsed(
        self,
        other: "Timestamp",
    ) -> timedelta:

        return self.value - other.value

    def add_seconds(
        self,
        seconds: float,
    ) -> "Timestamp":

        return Timestamp(
            self.value +
            timedelta(seconds=seconds)
        )

    def add_milliseconds(
        self,
        milliseconds: float,
    ) -> "Timestamp":

        return Timestamp(
            self.value +
            timedelta(milliseconds=milliseconds)
        )

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self):

        return {

            "iso": self.isoformat(),

            "unix": self.to_unix(),

        }

    ####################################################################

    def __str__(self):

        return self.isoformat()