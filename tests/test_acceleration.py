from core.geometry.acceleration import (
    Acceleration2D,
    Acceleration3D,
)

print("=" * 60)
print("ACCELERATION TEST")
print("=" * 60)

a = Acceleration2D(6, 8)

print("Acceleration:", a)
print("Magnitude:", a.magnitude)
print("Direction:", a.direction)
print("Normalized:", a.normalize())
print("Scaled:", a.scale(2))
print("Rotated:", a.rotate(90))

print()

a3 = Acceleration3D(3, 4, 12)

print("Acceleration3D:", a3)
print("Magnitude:", a3.magnitude)
print("Normalized:", a3.normalize())

other = Acceleration3D(0, 1, 0)

print("Dot:", a3.dot(other))
print("Cross:", a3.cross(other))