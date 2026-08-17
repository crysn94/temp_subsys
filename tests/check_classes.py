from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
dataset = PROJECT_ROOT / "Dataset"

print("Project Root :", PROJECT_ROOT)
print("Dataset Path :", dataset)
print("Exists       :", dataset.exists())

label_files = list(dataset.rglob("*.txt"))

print("Label files found:", len(label_files))

for f in label_files[:10]:
    print(f)

print("Current working directory:", Path.cwd())
print("Dataset path:", dataset.resolve())
print("Dataset exists:", dataset.exists())

counter = Counter()

for txt in label_files:

    with open(txt, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            try:
                class_id = int(float(values[0]))
                counter[class_id] += 1
            except Exception:
                print(f"Skipping malformed line in {txt.name}: {line[:60]}")
                continue

print("\nDetected Classes")

if not counter:
    print("No class IDs found.")
else:
    for cls, count in sorted(counter.items()):
        print(f"Class {cls}: {count}")