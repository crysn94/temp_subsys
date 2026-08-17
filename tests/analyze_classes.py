from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "Dataset"

stats = defaultdict(lambda: {0: 0, 1: 0})

for txt in DATASET.rglob("*.txt"):

    name = txt.stem

    if name.startswith("B"):
        prefix = "Bird"
    elif name.startswith("D"):
        prefix = "Drone"
    else:
        prefix = "Unknown"

    with open(txt, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            try:
                cls = int(float(values[0]))
                stats[prefix][cls] += 1
            except:
                pass

print("\nAnnotation Counts\n")

for group in sorted(stats):

    print(group)

    for cls in sorted(stats[group]):

        print(f"  Class {cls}: {stats[group][cls]}")

    print()