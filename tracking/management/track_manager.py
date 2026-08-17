"""
Track Manager
=============

Central manager for persistent Track objects.

TrackManager is responsible for:

- storing active and inactive Track objects
- enforcing unique track identifiers
- adding tracks
- retrieving tracks
- removing tracks
- clearing all tracks
- filtering active tracks
- preserving deterministic insertion order

TrackManager does NOT own:

- Kalman filtering
- data association
- track state estimation
- lifecycle transitions
- threat assessment
- sensor fusion

Those responsibilities remain in their respective modules.

Canonical architecture
----------------------

Detection
    |
    v
Data Association
    |
    v
TrackManager
    |
    +---- Track
    |      |
    |      +-- StateVector
    |      +-- Lifecycle
    |      +-- Detection history
    |      +-- Confidence
    |
    +---- Track
    |
    +---- Track
"""

from __future__ import annotations

from tracking.models.track import Track


# ======================================================================
# Track Manager
# ======================================================================


class TrackManager:
    """
    Manage a collection of persistent Track objects.

    Tracks are stored by their ``track_id`` while insertion order is
    preserved by the underlying dictionary.

    The manager intentionally does not modify the lifecycle state of
    tracks. Lifecycle transitions remain the responsibility of Track
    and the future lifecycle-management layer.
    """

    def __init__(self) -> None:
        """
        Create an empty TrackManager.
        """

        self._tracks: dict[str, Track] = {}

    # ==================================================================
    # Track Collection
    # ==================================================================

    @property
    def tracks(self) -> list[Track]:
        """
        Return all tracks in insertion order.

        A new list is returned so callers cannot directly modify the
        manager's internal collection.
        """

        return list(self._tracks.values())

    # ------------------------------------------------------------------

    @property
    def active_tracks(self) -> list[Track]:
        """
        Return only currently active tracks.

        Track lifecycle state is determined by the Track object itself
        through its ``is_active`` property.
        """

        return [
            track
            for track in self._tracks.values()
            if track.is_active
        ]

    # ------------------------------------------------------------------

    @property
    def num_tracks(self) -> int:
        """
        Return the total number of managed tracks.
        """

        return len(self._tracks)

    # ------------------------------------------------------------------

    @property
    def num_active_tracks(self) -> int:
        """
        Return the number of active tracks.
        """

        return len(self.active_tracks)

    # ==================================================================
    # Add
    # ==================================================================

    def add_track(
        self,
        track: Track,
    ) -> Track:
        """
        Add a Track to the manager.

        Parameters
        ----------
        track:
            Track instance to add.

        Returns
        -------
        Track
            The same Track instance that was supplied.

        Raises
        ------
        TypeError
            If ``track`` is not a Track instance.

        ValueError
            If another track already uses the same track ID.
        """

        if not isinstance(track, Track):
            raise TypeError(
                "track must be a Track instance."
            )

        track_id = track.track_id

        if track_id in self._tracks:
            raise ValueError(
                f"Track with ID {track_id!r} "
                "already exists."
            )

        self._tracks[track_id] = track

        return track

    # ==================================================================
    # Retrieval
    # ==================================================================

    def get_track(
        self,
        track_id: str,
    ) -> Track | None:
        """
        Retrieve a track by ID.

        Returns ``None`` when the requested track does not exist.
        """

        return self._tracks.get(track_id)

    # ------------------------------------------------------------------

    def get_tracks(self) -> list[Track]:
        """
        Return all managed tracks in insertion order.
        """

        return list(self._tracks.values())

    # ==================================================================
    # Contains
    # ==================================================================

    def contains(
        self,
        track_id: str,
    ) -> bool:
        """
        Return True when a track with the given ID exists.
        """

        return track_id in self._tracks

    # ==================================================================
    # Remove
    # ==================================================================

    def remove_track(
        self,
        track_id: str,
    ) -> Track | None:
        """
        Remove and return a track by ID.

        Returns ``None`` when the track does not exist.
        """

        return self._tracks.pop(
            track_id,
            None,
        )

    # ==================================================================
    # Clear
    # ==================================================================

    def clear(self) -> None:
        """
        Remove all managed tracks.
        """

        self._tracks.clear()

    # ==================================================================
    # Representation
    # ==================================================================

    def __len__(self) -> int:
        """
        Return the number of managed tracks.
        """

        return self.num_tracks

    # ------------------------------------------------------------------

    def __contains__(
        self,
        track_id: str,
    ) -> bool:
        """
        Support:

            "T001" in manager
        """

        return self.contains(track_id)

    # ------------------------------------------------------------------

    def __iter__(self):
        """
        Iterate over tracks in insertion order.
        """

        return iter(self._tracks.values())

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "TrackManager("
            f"num_tracks={self.num_tracks}, "
            f"num_active_tracks={self.num_active_tracks}"
            ")"
        )