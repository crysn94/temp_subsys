"""
============================================================
YOLO Dataset Label Repair Utility

Supports:
    ✔ Detection Labels
    ✔ Segmentation Labels
    ✔ Mixed Datasets

Repairs class IDs based on filename prefix.

Author: CUAS Vision SDK
============================================================
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CLASS_MAPPING = {
    "B": 1,      # Bird
    "D": 0       # Drone
}


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


# ------------------------------------------------------------
# Repair Statistics
# ------------------------------------------------------------

@dataclass
class RepairStats:

    total_files: int = 0
    repaired_files: int = 0
    copied_images: int = 0

    total_annotations: int = 0
    repaired_annotations: int = 0

    empty_files: int = 0

    per_class = defaultdict(int)


# ------------------------------------------------------------
# Label Repair
# ------------------------------------------------------------

class LabelRepair:

    def __init__(self,
                 source_dataset: Path,
                 output_dataset: Path):

        self.source = Path(source_dataset)
        self.output = Path(output_dataset)

        self.stats = RepairStats()

        self.report_rows = []

    # --------------------------------------------------------

    def repair(self):

        print("=" * 70)
        print("REPAIRING DATASET")
        print("=" * 70)

        if self.output.exists():
            shutil.rmtree(self.output)

        self.output.mkdir(parents=True)

        for split in ["train", "valid", "test"]:

            self._repair_split(split)

        self._write_report()

        self.summary()

    # --------------------------------------------------------

    def _repair_split(self, split):

        print(f"\nProcessing {split}")

        src_split = self.source / split
        dst_split = self.output / split

        image_src = src_split / "images"
        label_src = src_split / "labels"

        image_dst = dst_split / "images"
        label_dst = dst_split / "labels"

        image_dst.mkdir(parents=True, exist_ok=True)
        label_dst.mkdir(parents=True, exist_ok=True)

        # Copy Images
        for image in image_src.iterdir():

            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            shutil.copy2(image, image_dst / image.name)

            self.stats.copied_images += 1

        # Repair Labels
        for label in label_src.glob("*.txt"):

            self._repair_label(label,
                               label_dst / label.name)

    # --------------------------------------------------------

    def _repair_label(self,
                      src_label: Path,
                      dst_label: Path):

        self.stats.total_files += 1

        prefix = src_label.stem[0].upper()

        expected_class = CLASS_MAPPING.get(prefix)

        if expected_class is None:

            shutil.copy2(src_label, dst_label)
            return

        if src_label.stat().st_size == 0:

            shutil.copy2(src_label, dst_label)

            self.stats.empty_files += 1
            return

        repaired = False
        output_lines = []

        with open(src_label, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                values = line.split()

                old_class = int(float(values[0]))

                values[0] = str(expected_class)

                output_lines.append(" ".join(values))

                self.stats.total_annotations += 1
                self.stats.per_class[expected_class] += 1

                if old_class != expected_class:

                    repaired = True

                    self.stats.repaired_annotations += 1

                    self.report_rows.append([
                        src_label.name,
                        old_class,
                        expected_class
                    ])

        with open(dst_label, "w") as f:

            f.write("\n".join(output_lines))

        if repaired:
            self.stats.repaired_files += 1

    # --------------------------------------------------------

    def _write_report(self):

        report = self.output / "repair_report.csv"

        with open(report,
                  "w",
                  newline="") as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow([
                "File",
                "OldClass",
                "NewClass"
            ])

            writer.writerows(self.report_rows)

    # --------------------------------------------------------

    def summary(self):

        print()

        print("=" * 70)
        print("REPAIR SUMMARY")
        print("=" * 70)

        print(f"Label Files           : {self.stats.total_files}")
        print(f"Repaired Files        : {self.stats.repaired_files}")
        print(f"Images Copied         : {self.stats.copied_images}")

        print()

        print(f"Annotations           : {self.stats.total_annotations}")
        print(f"Annotations Modified  : {self.stats.repaired_annotations}")

        print()

        print(f"Empty Label Files     : {self.stats.empty_files}")

        print()

        print("Class Distribution")

        for cls in sorted(self.stats.per_class):
            print(f"  Class {cls}: {self.stats.per_class[cls]}")

        print()

        print("Repair report written to:")

        print(self.output / "repair_report.csv")

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    SOURCE = PROJECT_ROOT / "Dataset"

    OUTPUT = PROJECT_ROOT / "Dataset_fixed"

    repair = LabelRepair(
        SOURCE,
        OUTPUT
    )

    repair.repair()