from core.geometry.covariance import Covariance2D, Covariance3D
from core.geometry import Point2D, Point3D

print("=" * 60)
print("Covariance2D")
print("=" * 60)

cov = Covariance2D(
    xx=4,
    xy=1,
    yy=3,
)

print("Matrix")
print(cov.matrix)

print("Determinant :", cov.determinant)
print("Trace       :", cov.trace)
print("Symmetric   :", cov.is_symmetric)
print("PositiveDef :", cov.is_positive_definite)

p1 = Point2D(0, 0)
p2 = Point2D(2, 1)

print("Mahalanobis :", cov.mahalanobis(p1, p2))

print()

print("=" * 60)
print("Covariance3D")
print("=" * 60)

cov3 = Covariance3D(
    xx=3,
    xy=0.5,
    xz=0.2,
    yy=2,
    yz=0.1,
    zz=4,
)

print(cov3.matrix)

print("Determinant :", cov3.determinant)
print("Trace       :", cov3.trace)
print("PositiveDef :", cov3.is_positive_definite)

a = Point3D(0, 0, 0)
b = Point3D(1, 2, 3)

print("Mahalanobis :", cov3.mahalanobis(a, b))