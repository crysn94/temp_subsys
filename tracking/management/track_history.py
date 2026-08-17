"""
Track History Management.

TrackHistory maintains historical snapshots of Track objects.

Responsibilities
----------------
- Record Track snapshots
- Maintain per-track chronological history
- Retrieve historical snapshots
- Retrieve the latest snapshot
- Maintain optional bounded history
- Clear individual or complete histories
- Preserve deterministic insertion order
- Keep historical snapshots isolated from current Track objects

TrackHistory does NOT own:

- Kalman filtering
- data association
- track lifecycle transitions
- track creation/removal
- threat assessment
- sensor fusion

The Track object remains the canonical owner of the current state.
"""

from __future__ import annotations

from copy import deepcopy

from tracking.models.track import Track


# ======================================================================
# Track History
# ======================================================================


class TrackHistory:
    """
    Maintain historical snapshots of Track objects.

    History is maintained independently for each track ID.

    Example
    -------

    ::

        T001
          |
          +-- snapshot 0
          +-- snapshot 1
          +-- snapshot 2

        T002
          |
          +-- snapshot 0
          +-- snapshot 1

    Parameters
    ----------
    max_history_size:
        Maximum number of historical snapshots retained per track.

        ``None`` means unlimited history.
    """

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        max_history_size: int | None = None,
    ) -> None:

        if (
            max_history_size is not None
            and max_history_size <= 0
        ):
            raise ValueError(
                "max_history_size must be greater than zero "
                "or None."
            )

        self._max_history_size = max_history_size

        # --------------------------------------------------------------
        # Track ID -> ordered list of Track snapshots
        # --------------------------------------------------------------

        self._history: dict[str, list[Track]] = {}

    # ==================================================================
    # Configuration
    # ==================================================================

    @property
    def max_history_size(self) -> int | None:
        """
        Return the configured maximum history size.
        """

        return self._max_history_size

    # ==================================================================
    # Recording
    # ==================================================================

    def record(
        self,
        track: Track,
    ) -> Track:
        """
        Record a snapshot of a Track.

        A deep copy is stored so that subsequent modifications to the
        live Track do not modify historical records.

        Parameters
        ----------
        track:
            Track instance to snapshot.

        Returns
        -------
        Track
            The stored historical snapshot.

        Raises
        ------
        TypeError
            If ``track`` is not a Track instance.
        """

        if not isinstance(track, Track):
            raise TypeError(
                "track must be a Track instance."
            )

        track_id = track.track_id

        # --------------------------------------------------------------
        # Create history bucket
        # --------------------------------------------------------------

        if track_id not in self._history:
            self._history[track_id] = []

        # --------------------------------------------------------------
        # Snapshot
        # --------------------------------------------------------------

        snapshot = deepcopy(track)

        self._history[track_id].append(
            snapshot
        )

        # --------------------------------------------------------------
        # Enforce bounded history
        # --------------------------------------------------------------

        if (
            self._max_history_size is not None
            and len(self._history[track_id])
            > self._max_history_size
        ):

            excess = (
                len(self._history[track_id])
                - self._max_history_size
            )

            del self._history[track_id][:excess]

        return snapshot

    # ==================================================================
    # Retrieval
    # ==================================================================

    def get_history(
        self,
        track_id: str,
    ) -> list[Track]:
        """
        Return historical snapshots for a track.

        A new list is returned so callers cannot modify the internal
        history container.

        The Track objects themselves are also copied to protect the
        stored historical state.
        """

        entries = self._history.get(
            track_id,
            [],
        )

        return deepcopy(entries)

    # ------------------------------------------------------------------

    def latest(
        self,
        track_id: str,
    ) -> Track | None:
        """
        Return the most recent historical snapshot.

        Returns ``None`` when no history exists for the requested track.
        """

        entries = self._history.get(
            track_id
        )

        if not entries:
            return None

        return deepcopy(
            entries[-1]
        )

    # ==================================================================
    # Counts
    # ==================================================================

    @property
    def num_tracks(self) -> int:
        """
        Return the number of tracks with stored history.
        """

        return len(self._history)

    # ------------------------------------------------------------------

    @property
    def num_entries(self) -> int:
        """
        Return the total number of historical snapshots.
        """

        return sum(
            len(entries)
            for entries in self._history.values()
        )

    # ------------------------------------------------------------------

    def num_entries_for(
        self,
        track_id: str,
    ) -> int:
        """
        Return the number of historical snapshots for a track.
        """

        return len(
            self._history.get(
                track_id,
                [],
            )
        )

    # ==================================================================
    # Contains
    # ==================================================================

    def contains(
        self,
        track_id: str,
    ) -> bool:
        """
        Return True if history exists for the specified track ID.
        """

        return track_id in self._history

    # ==================================================================
    # Clearing
    # ==================================================================

    def clear_track(
        self,
        track_id: str,
    ) -> None:
        """
        Remove all history for a specific track.

        Missing track IDs are ignored.
        """

        self._history.pop(
            track_id,
            None,
        )

    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        Remove all historical data.
        """

        self._history.clear()

    # ==================================================================
    # Representation
    # ==================================================================

    def __len__(self) -> int:
        """
        Return the total number of stored history entries.
        """

        return self.num_entries

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            "TrackHistory("
            f"num_tracks={self.num_tracks}, "
            f"num_entries={self.num_entries}, "
            f"max_history_size="
            f"{self.max_history_size!r}"
            ")"
        )