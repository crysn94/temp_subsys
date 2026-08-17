print("Testing geometry imports...")

try:
    from core.geometry.point import Point2D, Point3D
    print("✓ point.py")
except Exception as e:
    print("✗ point.py")
    print(e)

try:
    from core.geometry.bbox import BoundingBox2D, BoundingBox3D
    print("✓ bbox.py")
except Exception as e:
    print("✗ bbox.py")
    print(e)

try:
    from core.geometry.velocity import Velocity3D
    print("✓ velocity.py")
except Exception as e:
    print("✗ velocity.py")
    print(e)

try:
    from core.geometry.orientation import Orientation
    print("✓ orientation.py")
except Exception as e:
    print("✗ orientation.py")
    print(e)

try:
    from core.geometry.polygon import Polygon
    print("✓ polygon.py")
except Exception as e:
    print("✗ polygon.py")
    print(e)