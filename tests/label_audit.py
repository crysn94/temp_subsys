from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "Dataset"

stats = defaultdict(lambda: defaultdict(int))

for split in ["train", "valid", "test"]:

    label_dir = DATASET / split / "labels"

    if not label_dir.exists():
        continue

    for txt in label_dir.glob("*.txt"):

        if txt.stat().st_size == 0:
            continue

        prefix = txt.stem[0].upper()

        with open(txt) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                cls = int(float(line.split()[0]))

                stats[(split, prefix)][cls] += 1

print()

for key in sorted(stats):

    split, prefix = key

    print(f"{split:5s} {prefix}")

    for cls in sorted(stats[key]):

        print(f"   Class {cls}: {stats[key][cls]}")

    print()