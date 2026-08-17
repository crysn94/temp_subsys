from core.confidence import Confidence

print("=" * 60)
print("CONFIDENCE TEST")
print("=" * 60)

c = Confidence(

    detection=0.96,

    classification=0.91,

    localization=0.88,

    tracking=0.92,

    identification=0.81,

    fusion=0.95,

)

print(c)

print()

print("Overall")

print(c.overall)

print()

print("Minimum")

print(c.minimum)

print()

print("Maximum")

print(c.maximum)

print()

print("Reliable")

print(c.is_reliable())

print()

print(c.as_dict())