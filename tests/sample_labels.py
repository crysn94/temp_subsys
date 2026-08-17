from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "Dataset"

samples = {
    "Bird": None,
    "Drone": None
}

for txt in sorted(DATASET.rglob("*.txt")):
    stem = txt.stem

    if stem.startswith("B") and samples["Bird"] is None:
        samples["Bird"] = txt

    if stem.startswith("D") and samples["Drone"] is None:
        samples["Drone"] = txt

for name, path in samples.items():

    print("=" * 60)
    print(name)
    print(path)

    with open(path, "r") as f:

        print()

        for i, line in enumerate(f):

            print(line.strip())

            if i == 9:
                break