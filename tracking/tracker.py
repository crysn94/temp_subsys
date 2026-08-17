"""
Central Tracking Coordinator.

Tracker coordinates the complete track-management cycle:

    DetectionResult[]
          |
          v
       Tracker
          |
          +----------------------+
          |                      |
          v                      v
   Data Associator          TrackManager
          |                      |
          |                      v
          |                   Tracks
          |
          v
     Track Update
          |
          v
 TrackLifecycleManager
          |
          v
 Updated Tracks

Responsibilities
----------------
Tracker is responsible for coordinating:

- detection input
- data association
- track creation
- matched-track updates
- missed-track handling
- lifecycle management
- persistent track storage

Tracker does NOT own:

- Kalman filtering
- association mathematics
- lifecycle rules
- track state representation
- threat assessment
- sensor fusion

Those responsibilities remain in their respective modules.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import Any

from core.detection_result import DetectionResult
from tracking.similarity.euclidean import EuclideanMetric

from tracking.association.base_associator import (
    AssociationResult,
)

from tracking.association.cost_matrix import (
    CostMatrixBuilder,
)

from tracking.association.hungarian import (
    HungarianAssociator,
)

from tracking.management.track_lifecycle import (
    TrackLifecycleManager,
)

from tracking.management.track_manager import (
    TrackManager,
)

from tracking.models.state_vector import (
    StateVector,
)

from tracking.models.track import (
    Track,
)




# ======================================================================
# Tracker Result
# ======================================================================


@dataclass(slots=True)
class TrackerResult:
    """
    Result returned by one Tracker update cycle.

    Attributes
    ----------
    tracks:
        All currently managed tracks.

    active_tracks:
        Currently active tracks.

    association:
        Association result produced by the configured associator.

    created_track_ids:
        IDs of tracks created during this update.

    updated_track_ids:
        IDs of tracks updated by matched detections.

    missed_track_ids:
        IDs of tracks that received no detection.

    deleted_track_ids:
        IDs of tracks removed because their lifecycle reached
        the deleted state.

    metadata:
        Additional tracker information.
    """

    tracks: list[Track] = field(
        default_factory=list
    )

    active_tracks: list[Track] = field(
        default_factory=list
    )

    association: AssociationResult | None = None

    created_track_ids: list[str] = field(
        default_factory=list
    )

    updated_track_ids: list[str] = field(
        default_factory=list
    )

    missed_track_ids: list[str] = field(
        default_factory=list
    )

    deleted_track_ids: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def num_tracks(self) -> int:
        return len(self.tracks)

    @property
    def num_active_tracks(self) -> int:
        return len(self.active_tracks)

    @property
    def num_created_tracks(self) -> int:
        return len(self.created_track_ids)

    @property
    def num_updated_tracks(self) -> int:
        return len(self.updated_track_ids)

    @property
    def num_missed_tracks(self) -> int:
        return len(self.missed_track_ids)

    @property
    def num_deleted_tracks(self) -> int:
        return len(self.deleted_track_ids)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "tracks": [
                track.track_id
                for track in self.tracks
            ],
            "active_tracks": [
                track.track_id
                for track in self.active_tracks
            ],
            "created_track_ids":
                list(self.created_track_ids),
            "updated_track_ids":
                list(self.updated_track_ids),
            "missed_track_ids":
                list(self.missed_track_ids),
            "deleted_track_ids":
                list(self.deleted_track_ids),
            "metadata":
                dict(self.metadata),
        }


# ======================================================================
# Tracker
# ======================================================================


class Tracker:
    """
    Central coordinator for persistent multi-object tracking.

    Parameters
    ----------
    associator:
        Data-association algorithm.

        Supported implementations include:

        - HungarianAssociator
        - NearestNeighborAssociator
        - JPDAAssociator
        - MHTAssociator

    track_manager:
        Optional TrackManager instance.

    lifecycle_manager:
        Optional TrackLifecycleManager instance.
    """

    def __init__(
            self,
            associator=None,
            track_manager: TrackManager | None = None,
            lifecycle_manager: TrackLifecycleManager | None = None,
    ) -> None:

        # --------------------------------------------------------------
        # Default association configuration
        # --------------------------------------------------------------
        #
        # Tracker should work out-of-the-box.
        #
        # Default pipeline:
        #
        # Tracker
        #    |
        #    v
        # HungarianAssociator
        #    |
        #    v
        # CostMatrixBuilder
        #    |
        #    v
        # EuclideanMetric
        #
        # Users can still inject a different associator.

        if associator is None:
            metric = EuclideanMetric()

            cost_builder = CostMatrixBuilder(
                metric=metric,
            )

            associator = HungarianAssociator(
                cost_builder=cost_builder,
            )

        self.associator = associator

        # --------------------------------------------------------------
        # Track manager
        # --------------------------------------------------------------

        self.track_manager = (
            track_manager
            if track_manager is not None
            else TrackManager()
        )

        # --------------------------------------------------------------
        # Lifecycle manager
        # --------------------------------------------------------------

        self.lifecycle_manager = (
            lifecycle_manager
            if lifecycle_manager is not None
            else TrackLifecycleManager()
        )

        # --------------------------------------------------------------
        # Deterministic Track ID generation
        # --------------------------------------------------------------

        self._next_track_number = 1

    # ==================================================================
    # Track Collection
    # ==================================================================

    @property
    def tracks(self) -> list[Track]:
        """
        Return all managed tracks.
        """

        return self.track_manager.tracks

    # ------------------------------------------------------------------

    @property
    def active_tracks(self) -> list[Track]:
        """
        Return currently active tracks.
        """

        return self.track_manager.active_tracks

    # ------------------------------------------------------------------

    @property
    def num_tracks(self) -> int:
        return self.track_manager.num_tracks

    # ------------------------------------------------------------------

    @property
    def num_active_tracks(self) -> int:
        return self.track_manager.num_active_tracks

    # ==================================================================
    # Track Access
    # ==================================================================

    def add_track(
        self,
        track: Track,
    ) -> Track:
        """
        Add an externally created Track.
        """

        if not isinstance(track, Track):
            raise TypeError(
                "track must be a Track instance."
            )

        return self.track_manager.add_track(
            track
        )

    # ------------------------------------------------------------------

    def get_track(
        self,
        track_id: str,
    ) -> Track | None:
        return self.track_manager.get_track(
            track_id
        )

    # ------------------------------------------------------------------

    def remove_track(
        self,
        track_id: str,
    ) -> Track | None:
        return self.track_manager.remove_track(
            track_id
        )

    # ------------------------------------------------------------------

    def clear(self) -> None:
        self.track_manager.clear()

    # ==================================================================
    # Main Update
    # ==================================================================

    def update(
        self,
        detections: list[DetectionResult],
    ) -> TrackerResult:
        """
        Process one detection cycle.

        Processing order:

        1. Validate detections.
        2. Retrieve active tracks.
        3. Perform association.
        4. Update matched tracks.
        5. Mark unmatched tracks as missed.
        6. Create tracks for unmatched detections.
        7. Apply lifecycle transitions.
        8. Remove deleted tracks.
        9. Return TrackerResult.
        """

        self._validate_detections(
            detections
        )

        active_tracks = list(
            self.track_manager.active_tracks
        )

        # --------------------------------------------------------------
        # Empty detection cycle
        # --------------------------------------------------------------

        if not detections:

            association = self._empty_association(
                len(active_tracks)
            )

            missed_track_ids: list[str] = []

            for track in active_tracks:

                self._handle_missed_track(
                    track
                )

                missed_track_ids.append(
                    track.track_id
                )

            deleted_track_ids = (
                self._remove_deleted_tracks()
            )

            return self._build_result(
                association=association,
                created_track_ids=[],
                updated_track_ids=[],
                missed_track_ids=missed_track_ids,
                deleted_track_ids=deleted_track_ids,
                num_detections_processed=len(detections),
            )

        # --------------------------------------------------------------
        # No existing tracks
        # --------------------------------------------------------------

        if not active_tracks:

            created_track_ids = (
                self._create_tracks(
                    detections
                )
            )

            return self._build_result(
                association=self._empty_association(
                    0,
                    len(detections),
                ),
                created_track_ids=created_track_ids,
                updated_track_ids=[],
                missed_track_ids=[],
                deleted_track_ids=[],
                num_detections_processed=len(detections),
            )

        # --------------------------------------------------------------
        # Association
        # --------------------------------------------------------------

        if self.associator is None:
            raise ValueError(
                "Tracker requires an associator "
                "when active tracks exist."
            )

        association = self.associator.associate(
            active_tracks,
            detections,
        )

        if not isinstance(
            association,
            AssociationResult,
        ):
            raise TypeError(
                "associator.associate() must return "
                "AssociationResult."
            )

        # --------------------------------------------------------------
        # Matched tracks
        # --------------------------------------------------------------

        updated_track_ids: list[str] = []

        for (
            track_index,
            detection_index,
        ) in association.matches:

            if not (
                0 <= track_index
                < len(active_tracks)
            ):
                continue

            if not (
                0 <= detection_index
                < len(detections)
            ):
                continue

            track = active_tracks[
                track_index
            ]

            detection = detections[
                detection_index
            ]

            self._handle_detection(
                track,
                detection,
            )

            updated_track_ids.append(
                track.track_id
            )

        # --------------------------------------------------------------
        # Missed tracks
        # --------------------------------------------------------------

        missed_track_ids: list[str] = []

        for track_index in (
            association.unmatched_tracks
        ):

            if not (
                0 <= track_index
                < len(active_tracks)
            ):
                continue

            track = active_tracks[
                track_index
            ]

            self._handle_missed_track(
                track
            )

            missed_track_ids.append(
                track.track_id
            )

        # --------------------------------------------------------------
        # New tracks
        # --------------------------------------------------------------

        created_track_ids: list[str] = []

        for detection_index in (
            association.unmatched_detections
        ):

            if not (
                0 <= detection_index
                < len(detections)
            ):
                continue

            detection = detections[
                detection_index
            ]

            track = self._create_track(
                detection
            )

            created_track_ids.append(
                track.track_id
            )

        # --------------------------------------------------------------
        # Deleted tracks
        # --------------------------------------------------------------

        deleted_track_ids = (
            self._remove_deleted_tracks()
        )

        return self._build_result(
            association=association,
            created_track_ids=created_track_ids,
            updated_track_ids=updated_track_ids,
            missed_track_ids=missed_track_ids,
            deleted_track_ids=deleted_track_ids,
            num_detections_processed=len(detections),
        )

    # ==================================================================
    # Detection Handling
    # ==================================================================

    def _handle_detection(
        self,
        track: Track,
        detection: DetectionResult,
    ) -> None:
        """
        Apply a detection to an existing track.

        Track remains responsible for maintaining its own detection
        history and state.
        """

        if hasattr(
            track,
            "add_detection",
        ):
            track.add_detection(
                detection
            )

        self.lifecycle_manager.on_detection(
            track
        )

    # ------------------------------------------------------------------

    def _handle_missed_track(
        self,
        track: Track,
    ) -> None:
        """
        Handle a track that received no detection.
        """

        if hasattr(
            track,
            "mark_missed",
        ):
            track.mark_missed()

        self.lifecycle_manager.on_missed_detection(
            track
        )

    # ==================================================================
    # Track Creation
    # ==================================================================

    def _create_tracks(
        self,
        detections: list[DetectionResult],
    ) -> list[str]:

        created_track_ids: list[str] = []

        for detection in detections:

            track = self._create_track(
                detection
            )

            created_track_ids.append(
                track.track_id
            )

        return created_track_ids

    # ------------------------------------------------------------------

    def _create_track(
        self,
        detection: DetectionResult,
    ) -> Track:
        """
        Create a Track from a detection.

        The detection position is used to initialize the state vector.
        """

        state_vector = self._state_vector_from_detection(
            detection
        )

        track_id = self._generate_track_id()

        track = Track(
            track_id=track_id,
            state_vector=state_vector,
        )

        self.track_manager.add_track(
            track
        )

        # Register the initial detection.
        self._handle_detection(
            track,
            detection,
        )

        return track

    # ------------------------------------------------------------------

    def _state_vector_from_detection(
            self,
            detection: DetectionResult,
    ) -> StateVector:
        """
        Construct an initial StateVector from a DetectionResult.

        Supported position representations:

            Point3D:
                position.x
                position.y
                position.z

            NumPy array:
                np.array([x, y, z])

            List / tuple:
                [x, y, z]

        Velocity is optional and defaults to zero.
        """

        # ==============================================================
        # Position
        # ==============================================================

        position = getattr(
            detection,
            "position",
            None,
        )

        position = getattr(
            detection,
            "position",
            None,
        )

        print(
            "DEBUG position:",
            repr(position),
            "TYPE:",
            type(position),
        )

        if position is None:
            raise ValueError(
                "DetectionResult must contain position "
                "to create a track."
            )

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
                    TypeError,
                    ValueError,
                    AttributeError,
            ) as exc:

                raise ValueError(
                    "Detection position must contain "
                    "three numeric values."
                ) from exc

        # --------------------------------------------------------------
        # Array-like object
        # --------------------------------------------------------------

        else:

            try:
                position_array = np.asarray(
                    position,
                    dtype=float,
                )

            except (
                    TypeError,
                    ValueError,
            ) as exc:

                raise ValueError(
                    "Detection position must contain "
                    "three numeric values."
                ) from exc

            # Flatten [x,y,z] safely
            position_array = position_array.reshape(-1)

            if position_array.size != 3:
                raise ValueError(
                    "Detection position must contain "
                    "three numeric values."
                )

            x = float(position_array[0])
            y = float(position_array[1])
            z = float(position_array[2])

        # --------------------------------------------------------------
        # Finite validation
        # --------------------------------------------------------------

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
        # Velocity
        # ==============================================================

        velocity = getattr(
            detection,
            "velocity",
            None,
        )

        vx = 0.0
        vy = 0.0
        vz = 0.0

        if velocity is not None:

            # Point3D/Velocity3D-style object
            if all(
                    hasattr(velocity, attribute)
                    for attribute in ("x", "y", "z")
            ):

                try:
                    vx = float(velocity.x)
                    vy = float(velocity.y)
                    vz = float(velocity.z)

                except (
                        TypeError,
                        ValueError,
                        AttributeError,
                ):
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0

            # Array-like velocity
            else:

                try:
                    velocity_array = np.asarray(
                        velocity,
                        dtype=float,
                    ).reshape(-1)

                    if velocity_array.size == 3:
                        vx = float(
                            velocity_array[0]
                        )
                        vy = float(
                            velocity_array[1]
                        )
                        vz = float(
                            velocity_array[2]
                        )

                except (
                        TypeError,
                        ValueError,
                ):
                    vx = 0.0
                    vy = 0.0
                    vz = 0.0

        # --------------------------------------------------------------
        # Velocity safety
        # --------------------------------------------------------------

        if not np.all(
                np.isfinite(
                    [vx, vy, vz]
                )
        ):
            vx = 0.0
            vy = 0.0
            vz = 0.0

        # ==============================================================
        # Timestamp
        # ==============================================================

        timestamp = getattr(
            detection,
            "timestamp",
            None,
        )

        if timestamp is None:
            from core.timestamps import Timestamp

            timestamp = Timestamp.now()

        # ==============================================================
        # Create StateVector
        # ==============================================================

        return StateVector.from_components(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            covariance=None,
            timestamp=timestamp,
        )

    # ==================================================================
    # Track IDs
    # ==================================================================

    def _generate_track_id(
        self,
    ) -> str:
        """
        Generate a deterministic unique track identifier.
        """

        while True:

            track_id = (
                f"T{self._next_track_number:04d}"
            )

            self._next_track_number += 1

            if not self.track_manager.contains(
                track_id
            ):
                return track_id

    # ==================================================================
    # Deleted Tracks
    # ==================================================================

    def _remove_deleted_tracks(
        self,
    ) -> list[str]:
        """
        Remove tracks whose lifecycle is DELETED.
        """

        deleted_track_ids: list[str] = []

        for track in list(
            self.track_manager.tracks
        ):

            lifecycle = getattr(
                track,
                "lifecycle",
                None,
            )

            lifecycle_name = getattr(
                lifecycle,
                "name",
                None,
            )

            if lifecycle_name == "DELETED":

                track_id = track.track_id

                self.track_manager.remove_track(
                    track_id
                )

                deleted_track_ids.append(
                    track_id
                )

        return deleted_track_ids

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_detections(
        detections: list[DetectionResult],
    ) -> None:

        if detections is None:
            raise TypeError(
                "detections must not be None."
            )

        if not isinstance(
            detections,
            list,
        ):
            raise TypeError(
                "detections must be a list."
            )

        for detection in detections:

            if not isinstance(
                detection,
                DetectionResult,
            ):
                raise TypeError(
                    "All detections must be "
                    "DetectionResult instances."
                )

    # ==================================================================
    # Empty Association
    # ==================================================================

    @staticmethod
    def _empty_association(
        num_tracks: int,
        num_detections: int = 0,
    ) -> AssociationResult:
        """
        Construct an empty association result.

        Unmatched indices are explicitly represented so downstream
        logic can process them deterministically.
        """

        return AssociationResult(
            matches=[],
            unmatched_tracks=list(
                range(num_tracks)
            ),
            unmatched_detections=list(
                range(num_detections)
            ),
        )

    # ==================================================================
    # Result
    # ==================================================================

    def _build_result(
            self,
            association: AssociationResult | None,
            created_track_ids: list[str],
            updated_track_ids: list[str],
            missed_track_ids: list[str],
            deleted_track_ids: list[str],
            num_detections_processed: int = 0,
    ) -> TrackerResult:

        return TrackerResult(
            tracks=self.track_manager.tracks,
            active_tracks=self.track_manager.active_tracks,
            association=association,
            created_track_ids=list(
                created_track_ids
            ),
            updated_track_ids=list(
                updated_track_ids
            ),
            missed_track_ids=list(
                missed_track_ids
            ),
            deleted_track_ids=list(
                deleted_track_ids
            ),
            metadata={
                "num_detections_processed": int(
                    num_detections_processed
                ),
            },
        )

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        associator_name = (
            self.associator.__class__.__name__
            if self.associator is not None
            else None
        )

        return (
            "Tracker("
            f"num_tracks={self.num_tracks}, "
            f"num_active_tracks="
            f"{self.num_active_tracks}, "
            f"associator="
            f"{associator_name!r}"
            ")"
        )