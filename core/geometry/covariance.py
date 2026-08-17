"""
========================================================================
Covariance Geometry
========================================================================
for covariance geometry/container only

Canonical covariance matrices used throughout the C-UAS framework.

Supports

• Covariance2D
• Covariance3D
• Determinant
• Trace
• Inverse
• Mahalanobis Distance
• Validation

========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .point import Point2D, Point3D


# ======================================================================
# Covariance2D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Covariance2D:

    xx: float
    xy: float
    yy: float

    ####################################################################
    # Matrix
    ####################################################################

    @property
    def matrix(self) -> np.ndarray:

        return np.array([
            [self.xx, self.xy],
            [self.xy, self.yy],
        ], dtype=float)

    ####################################################################
    # Properties
    ####################################################################

    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.matrix))

    @property
    def trace(self) -> float:
        return float(np.trace(self.matrix))

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    ####################################################################
    # Validation
    ####################################################################

    @property
    def is_symmetric(self) -> bool:
        return np.allclose(
            self.matrix,
            self.matrix.T,
        )

    @property
    def is_positive_definite(self) -> bool:

        try:
            np.linalg.cholesky(self.matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    ####################################################################
    # Distance
    ####################################################################

    def mahalanobis(
        self,
        p1: Point2D,
        p2: Point2D,
    ) -> float:

        d = np.array([
            p1.x - p2.x,
            p1.y - p2.y,
        ])

        return float(
            np.sqrt(
                d.T @ self.inverse @ d
            )
        )

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self):

        return {
            "xx": self.xx,
            "xy": self.xy,
            "yy": self.yy,
        }


# ======================================================================
# Covariance3D
# ======================================================================

@dataclass(frozen=True, slots=True)
class Covariance3D:

    xx: float
    xy: float
    xz: float

    yy: float
    yz: float

    zz: float

    ####################################################################
    # Matrix
    ####################################################################

    @property
    def matrix(self):

        return np.array([

            [self.xx, self.xy, self.xz],

            [self.xy, self.yy, self.yz],

            [self.xz, self.yz, self.zz],

        ], dtype=float)

    ####################################################################
    # Properties
    ####################################################################

    @property
    def determinant(self):
        return float(np.linalg.det(self.matrix))

    @property
    def trace(self):
        return float(np.trace(self.matrix))

    @property
    def inverse(self):
        return np.linalg.inv(self.matrix)

    ####################################################################
    # Validation
    ####################################################################

    @property
    def is_symmetric(self):
        return np.allclose(
            self.matrix,
            self.matrix.T,
        )

    @property
    def is_positive_definite(self):

        try:
            np.linalg.cholesky(self.matrix)
            return True

        except np.linalg.LinAlgError:
            return False

    ####################################################################
    # Mahalanobis
    ####################################################################

    def mahalanobis(
        self,
        p1: Point3D,
        p2: Point3D,
    ):

        d = np.array([
            p1.x - p2.x,
            p1.y - p2.y,
            p1.z - p2.z,
        ])

        return float(
            np.sqrt(
                d.T @ self.inverse @ d
            )
        )

    ####################################################################
    # Serialization
    ####################################################################

    def as_dict(self):

        return {

            "xx": self.xx,
            "xy": self.xy,
            "xz": self.xz,

            "yy": self.yy,
            "yz": self.yz,

            "zz": self.zz,

        }