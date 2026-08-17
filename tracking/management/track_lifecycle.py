"""
Track Lifecycle Management.

Responsible for applying lifecycle transition rules to Track objects.

Canonical lifecycle:

    NEW
      |
      v
    CONFIRMED
      |
      v
    COASTING
      |
      v
    LOST
      |
      v
    DELETED

The Track object remains the canonical owner of:

- lifecycle state
- hit/miss counters
- consecutive hit/miss counters
- timestamps
- state vector

This manager coordinates lifecycle transitions and does not require a
DetectionResult for lifecycle bookkeeping.
"""

from __future__ import annotations

from tracking.models.lifecycle import TrackState
from tracking.models.track import Track


# ======================================================================
# Track Lifecycle Manager
# ======================================================================


class TrackLifecycleManager:
    """
    Manage Track lifecycle transitions.

    Parameters
    ----------
    confirmation_hits:
        Number of consecutive detections required to confirm a NEW track.

    coasting_misses:
        Number of consecutive missed detections after which a CONFIRMED
        track enters COASTING.

    lost_misses:
        Number of consecutive missed detections after which a track
        enters LOST.

    deletion_misses:
    Number of consecutive missed detections after which a LOST
    track enters DELETED.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
            self,
            confirmation_hits: int = 3,
            coasting_misses: int = 2,
            lost_misses: int = 5,
            deletion_misses: int = 8,
    ) -> None:

        if confirmation_hits <= 0:
            raise ValueError(
                "confirmation_hits must be greater than zero."
            )

        if coasting_misses <= 0:
            raise ValueError(
                "coasting_misses must be greater than zero."
            )

        if lost_misses <= 0:
            raise ValueError(
                "lost_misses must be greater than zero."
            )

        if deletion_misses <= 0:
            raise ValueError(
                "deletion_misses must be greater than zero."
            )

        if lost_misses < coasting_misses:
            raise ValueError(
                "lost_misses must not be less than "
                "coasting_misses."
            )

        if deletion_misses < lost_misses:
            raise ValueError(
                "deletion_misses must not be less than "
                "lost_misses."
            )

        self.confirmation_hits = int(
            confirmation_hits
        )

        self.coasting_misses = int(
            coasting_misses
        )

        self.lost_misses = int(
            lost_misses
        )

        self.deletion_misses = int(
            deletion_misses
        )

    # ==================================================================
    # Detection
    # ==================================================================

    def on_detection(
        self,
        track: Track,
    ) -> Track:
        """
        Process a successful detection for a track.

        A detection:

        - increments total hits
        - increments consecutive hits
        - resets consecutive misses
        - confirms NEW tracks after the configured threshold
        - restores COASTING tracks to CONFIRMED
        - restores LOST tracks to CONFIRMED
        - leaves CONFIRMED tracks confirmed
        - does nothing to DELETED tracks
        """

        self._validate_track(track)

        # --------------------------------------------------------------
        # Deleted tracks are terminal.
        # --------------------------------------------------------------

        if track.lifecycle == TrackState.DELETED:
            return track

        # --------------------------------------------------------------
        # Update lifecycle bookkeeping.
        #
        # We intentionally do not call Track.add_detection() here because
        # that method requires a DetectionResult. The lifecycle manager
        # receives only a Track and therefore maintains the lifecycle
        # counters directly.
        # --------------------------------------------------------------

        self._increment_hit_counters(track)

        # A successful detection breaks a miss streak.
        self._reset_consecutive_misses(track)

        # --------------------------------------------------------------
        # Lifecycle transitions
        # --------------------------------------------------------------

        if track.lifecycle == TrackState.NEW:

            if (
                self._consecutive_hits(track)
                >= self.confirmation_hits
            ):
                track.lifecycle = TrackState.CONFIRMED

        elif track.lifecycle == TrackState.COASTING:

            track.lifecycle = TrackState.CONFIRMED

        elif track.lifecycle == TrackState.LOST:

            track.lifecycle = TrackState.CONFIRMED

        # CONFIRMED remains CONFIRMED.

        return track

    # ==================================================================
    # Missed Detection
    # ==================================================================

    def on_missed_detection(
        self,
        track: Track,
    ) -> Track:
        """
        Process a missed detection for a track.

        A missed detection:

        - increments total misses
        - increments consecutive misses
        - resets consecutive hits
        - moves CONFIRMED tracks to COASTING after the threshold
        - moves tracks to LOST after the lost threshold

        Deleted tracks cannot be updated.
        """

        self._validate_track(track)

        # --------------------------------------------------------------
        # Deleted tracks are terminal and cannot be updated.
        # --------------------------------------------------------------

        if track.lifecycle == TrackState.DELETED:
            raise ValueError(
                "Deleted tracks cannot be updated."
            )

        # --------------------------------------------------------------
        # Update lifecycle bookkeeping.
        # --------------------------------------------------------------

        self._increment_miss_counters(track)

        self._reset_consecutive_hits(track)

        consecutive_misses = (
            self._consecutive_misses(track)
        )

        # --------------------------------------------------------------
        # LOST has priority over COASTING.
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # LOST -> DELETED
        # --------------------------------------------------------------

        if (
                consecutive_misses >= self.deletion_misses
        ):
            track.lifecycle = TrackState.DELETED

            return track

        # --------------------------------------------------------------
        # COASTING / LOST transition
        # --------------------------------------------------------------

        if (
                consecutive_misses >= self.lost_misses
        ):
            track.lifecycle = TrackState.LOST

            return track

        # --------------------------------------------------------------
        # CONFIRMED -> COASTING
        # --------------------------------------------------------------

        if (
                track.lifecycle == TrackState.CONFIRMED
                and consecutive_misses >= self.coasting_misses
        ):
            track.lifecycle = TrackState.COASTING

        return track

    # ==================================================================
    # Delete
    # ==================================================================

    def delete(
        self,
        track: Track,
    ) -> Track:
        """
        Permanently mark a track as DELETED.

        Deletion is terminal. Calling delete() repeatedly is harmless.
        """

        self._validate_track(track)

        track.lifecycle = TrackState.DELETED

        return track

    # ==================================================================
    # Validation
    # ==================================================================

    @staticmethod
    def _validate_track(
        track: Track,
    ) -> None:

        if not isinstance(track, Track):
            raise TypeError(
                "track must be a Track instance."
            )

    # ==================================================================
    # Counter Helpers
    # ==================================================================

    @staticmethod
    def _increment_hit_counters(
        track: Track,
    ) -> None:
        """
        Increment total and consecutive hit counters.

        Track owns these counters; this manager only applies the
        lifecycle bookkeeping required for a detection event.
        """

        track.hits += 1
        track.consecutive_hits += 1

    # ------------------------------------------------------------------

    @staticmethod
    def _increment_miss_counters(
        track: Track,
    ) -> None:
        """
        Increment total and consecutive miss counters.
        """

        track.misses += 1
        track.consecutive_misses += 1

    # ------------------------------------------------------------------

    @staticmethod
    def _reset_consecutive_hits(
        track: Track,
    ) -> None:

        track.consecutive_hits = 0

    # ------------------------------------------------------------------

    @staticmethod
    def _reset_consecutive_misses(
        track: Track,
    ) -> None:

        track.consecutive_misses = 0

    # ==================================================================
    # Counter Access
    # ==================================================================

    @staticmethod
    def _consecutive_hits(
        track: Track,
    ) -> int:

        return int(track.consecutive_hits)

    # ------------------------------------------------------------------

    @staticmethod
    def _consecutive_misses(
        track: Track,
    ) -> int:

        return int(track.consecutive_misses)

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_config(self) -> dict[str, int]:
        """
        Return lifecycle configuration.
        """

        return {
            "confirmation_hits": self.confirmation_hits,
            "coasting_misses": self.coasting_misses,
            "lost_misses": self.lost_misses,
            "deletion_misses": self.deletion_misses,
        }

    # ==================================================================
    # Representation
    # ==================================================================

    def __repr__(self) -> str:

        return (
            "TrackLifecycleManager("
            f"confirmation_hits="
            f"{self.confirmation_hits}, "
            f"coasting_misses="
            f"{self.coasting_misses}, "
            f"lost_misses="
            f"{self.lost_misses}, "
            f"deletion_misses="
            f"{self.deletion_misses}"
            ")"
        )