"""
========================================================================
Canonical Track Object
========================================================================

Represents one persistent tracked object inside the C-UAS framework.

A Track is the canonical object shared by

• Kalman Filter
• Data Association
• Sensor Fusion
• Threat Assessment
• Prediction
• Mitigation

Architecture

    DetectionResult
          │
          ▼
        Track
          │
          ├── lifecycle
          │
          ├── DynamicState
          │      └── StateVector (3D CV)
          │
          ├── detection history
          │
          ├── confidence
          │
          └── metadata

3D Constant Velocity state:

    [x, y, z, vx, vy, vz]

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4
import numpy as np

from core.confidence import Confidence
from core.detection_result import DetectionResult
from tracking.models.state_vector import StateVector
from core.timestamps import Timestamp

from tracking.models.lifecycle import TrackState
from tracking.models.state import DynamicState


@dataclass(slots=True)
class Track:

    ####################################################################
    # Identity
    ####################################################################

    track_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    ####################################################################
    # Lifecycle
    ####################################################################

    lifecycle: TrackState = TrackState.NEW

    ####################################################################
    # Dynamic State
    ####################################################################

    state_vector: DynamicState | None = None

    ####################################################################
    # Measurements
    ####################################################################

    current_detection: DetectionResult | None = None

    detections: list[DetectionResult] = field(
        default_factory=list
    )

    ####################################################################
    # Confidence
    ####################################################################

    confidence: Confidence = field(
        default_factory=Confidence
    )

    ####################################################################
    # Timing
    ####################################################################

    created: Timestamp = field(
        default_factory=Timestamp.now
    )

    updated: Timestamp = field(
        default_factory=Timestamp.now
    )

    ####################################################################
    # Statistics
    ####################################################################

    age: int = 0

    hits: int = 0

    misses: int = 0

    consecutive_hits: int = 0

    consecutive_misses: int = 0

    ####################################################################
    # Metadata
    ####################################################################

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    ####################################################################
    # Detection Management
    ####################################################################

    def add_detection(
            self,
            detection: DetectionResult,
    ) -> None:
        """
        Add a detection to the track and update the track state.
        """

        if not isinstance(
                detection,
                DetectionResult,
        ):
            raise TypeError(
                "detection must be a DetectionResult instance."
            )

        # --------------------------------------------------------------
        # Store detection
        # --------------------------------------------------------------

        self.current_detection = detection

        self.detections.append(
            detection
        )

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        self.age += 1
        self.hits += 1

        self.consecutive_hits += 1
        self.consecutive_misses = 0

        # --------------------------------------------------------------
        # Update state from measurement
        # --------------------------------------------------------------

        self.update_state_from_detection(
            detection
        )

        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        self.touch()

    ####################################################################
    # State Management
    ####################################################################

    def update_state(
            self,
            state: DynamicState,
    ) -> None:
        """
        Update the dynamic state estimate of the track.

        Supports all DynamicState implementations, including:

        • StateVector
        • StateCA
        """

        self.state_vector = state
        self.touch()

    def update_state_vector(
            self,
            state: DynamicState,
    ) -> None:
        """
        Backward-compatible alias for update_state().
        """

        self.update_state(state)

    ####################################################################
    # Timing
    ####################################################################

    def touch(self) -> None:
        """
        Update the track's modification timestamp.
        """

        self.updated = Timestamp.now()

    ####################################################################
    # Miss Handling
    ####################################################################

    def mark_missed(self) -> None:
        """
        Register a missed measurement.

        Lifecycle transition is intentionally handled by the
        lifecycle manager rather than directly here.
        """

        self.age += 1

        self.misses += 1

        self.consecutive_hits = 0

        self.consecutive_misses += 1

        self.touch()

    ####################################################################
    # Lifecycle
    ####################################################################

    def confirm(self) -> None:
        self.lifecycle = TrackState.CONFIRMED
        self.touch()

    def coast(self) -> None:
        self.lifecycle = TrackState.COASTING
        self.touch()

    def lose(self) -> None:
        self.lifecycle = TrackState.LOST
        self.touch()

    def delete(self) -> None:
        self.lifecycle = TrackState.DELETED
        self.touch()

    ####################################################################
    # Convenience
    ####################################################################

    @property
    def is_active(self) -> bool:
        return self.lifecycle.is_active

    @property
    def is_confirmed(self) -> bool:
        return self.lifecycle.is_confirmed

    @property
    def is_terminal(self) -> bool:
        return self.lifecycle.is_terminal

    @property
    def can_receive_updates(self) -> bool:
        return self.lifecycle.can_receive_updates

    # ==================================================================
    # Detection
    # ==================================================================

    @property
    def has_detection(self) -> bool:
        """
        Return True when the track has received at least one detection.
        """

        return self.current_detection is not None

    @property
    def last_detection(self) -> DetectionResult | None:
        """
        Return the most recent detection.
        """

        return self.current_detection

    @property
    def detection_history(self) -> list[DetectionResult]:
        """
        Return the complete detection history.

        The internal history remains owned by Track.
        A copy is returned to prevent external modification.
        """

        return list(self.detections)

    @property
    def num_detections(self) -> int:
        """
        Return the number of detections received by this track.
        """

        return len(self.detections)

    ####################################################################
    # Position
    ####################################################################

    @property
    def latest_position(self):

        if self.state_vector is not None:
            return self.state_vector.position

        if self.current_detection is not None:
            return self.current_detection.position

        return None

    @property
    def position(self):
        return self.latest_position

    @property
    def estimated_position(self):

        if self.state_vector is None:
            return None

        return self.state_vector.position



    ####################################################################
    # Velocity
    ####################################################################

    @property
    def latest_velocity(self):

        if self.state_vector is not None:
            return self.state_vector.velocity

        if self.current_detection is not None:
            return self.current_detection.velocity

        return None

    @property
    def velocity(self):
        return self.latest_velocity

    @property
    def estimated_velocity(self):

        state = self.state_vector

        if state is None:
            return None

        return state.velocity

    @property
    def speed(self):

        state = self.state_vector

        if state is None:
            return 0.0

        return state.velocity.speed

    @property
    def acceleration(self):
        """
        Latest estimated 3D acceleration.

        Returns None when the current state model does not
        provide acceleration.
        """

        if self.state_vector is None:
            return None

        return getattr(
            self.state_vector,
            "acceleration",
            None,
        )

    @property
    def state_model(self) -> str | None:
        """
        Name of the dynamic state model currently used.
        """

        if self.state_vector is None:
            return None

        return self.state_vector.model_name

    ####################################################################
    # Classification
    ####################################################################

    @property
    def class_id(self) -> int | None:

        if self.current_detection is None:
            return None

        return self.current_detection.class_id

    @property
    def class_name(self) -> str | None:

        if self.current_detection is None:
            return None

        return self.current_detection.class_name

    ####################################################################
    # Sensor
    ####################################################################

    @property
    def sensor(self):

        if self.current_detection is None:
            return None

        return self.current_detection.sensor

    ####################################################################
    # Bounding Box
    ####################################################################

    @property
    def bbox(self):

        if self.current_detection is None:
            return None

        return getattr(
            self.current_detection,
            "bbox",
            None,
        )

    ####################################################################
    # Detection Score
    ####################################################################

    @property
    def score(self) -> float:

        if self.current_detection is None:
            return 0.0

        return self.current_detection.confidence

    ####################################################################
    # State
    ####################################################################

    @property
    def has_state(self) -> bool:
        return self.state_vector is not None

    ####################################################################
    # Timing Information
    ####################################################################

    @property
    def age_seconds(self) -> float:

        return (
                self.updated.to_datetime()
                - self.created.to_datetime()
        ).total_seconds()

    ####################################################################
    # Statistics
    ####################################################################

    def reset_statistics(self) -> None:

        self.hits = 0
        self.misses = 0

        self.consecutive_hits = 0
        self.consecutive_misses = 0

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self) -> dict:

        return {
            "track_id": self.track_id,

            "lifecycle": self.lifecycle.name,

            "created": self.created.isoformat(),

            "updated": self.updated.isoformat(),

            "age": self.age,

            "age_seconds": self.age_seconds,

            "hits": self.hits,

            "misses": self.misses,

            "consecutive_hits": self.consecutive_hits,

            "consecutive_misses": self.consecutive_misses,

            "state_model": self.state_model,

            "state_vector": (
                self.state_vector.as_dict()
                if self.state_vector is not None
                else None
            ),

            "current_detection": (
                self.current_detection.as_dict()
                if self.current_detection is not None
                else None
            ),

            "confidence": self.confidence.as_dict(),

            "num_detections": len(self.detections),

            "metadata": self.metadata,
        }

    ####################################################################
    # State Update
    ####################################################################

    def update_state_from_detection(
            self,
            detection: DetectionResult,
    ) -> None:
        """
        Update the canonical track state from a detection.

        Supported detection position representations:

            Point3D-style:
                position.x
                position.y
                position.z

            NumPy array:
                np.array([x, y, z])

            List / tuple:
                [x, y, z]

        The Track remains the owner of its StateVector.
        """

        if not isinstance(
                detection,
                DetectionResult,
        ):
            raise TypeError(
                "detection must be a DetectionResult instance."
            )

        position = detection.position

        if position is None:
            raise ValueError(
                "Detection must contain a position."
            )

        # ==============================================================
        # Extract position
        # ==============================================================

        # --------------------------------------------------------------
        # Point3D-style object
        # --------------------------------------------------------------

        if all(
                hasattr(position, attribute)
                for attribute in ("x", "y", "z")
        ):
            try:
                x = float(position.x)
                y = float(position.y)
                z = float(position.z)

            except (
                    AttributeError,
                    TypeError,
                    ValueError,
            ) as exc:

                raise ValueError(
                    "Detection position must contain "
                    "three numeric values."
                ) from exc

        # --------------------------------------------------------------
        # Array-like position
        # --------------------------------------------------------------

        else:

            try:
                position_array = np.asarray(
                    position,
                    dtype=float,
                ).reshape(-1)

            except (
                    TypeError,
                    ValueError,
            ) as exc:

                raise ValueError(
                    "Detection position must contain "
                    "three numeric values."
                ) from exc

            if position_array.size != 3:
                raise ValueError(
                    "Detection position must contain "
                    "three numeric values."
                )

            x = float(position_array[0])
            y = float(position_array[1])
            z = float(position_array[2])

        # ==============================================================
        # Validate numerical values
        # ==============================================================

        if not np.all(
                np.isfinite(
                    [x, y, z]
                )
        ):
            raise ValueError(
                "Detection position must contain "
                "finite numeric values."
            )

        # ==============================================================
        # Preserve current velocity
        # ==============================================================

        current_velocity = self.state_vector.velocity

        vx = float(current_velocity.x)
        vy = float(current_velocity.y)
        vz = float(current_velocity.z)

        # ==============================================================
        # Timestamp
        # ==============================================================

        timestamp = getattr(
            detection,
            "timestamp",
            None,
        )

        if timestamp is None:
            timestamp = self.state_vector.timestamp

        # ==============================================================
        # Update StateVector
        # ==============================================================

        self.state_vector = StateVector.from_components(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            covariance=self.state_vector.covariance,
            timestamp=timestamp,
        )
    ####################################################################
    # Representation
    ####################################################################

    def __len__(self) -> int:
        return self.num_detections

    def __repr__(self) -> str:

        return (
            f"Track("
            f"id={self.track_id}, "
            f"state={self.lifecycle.name}, "
            f"hits={self.hits}, "
            f"misses={self.misses})"
        )