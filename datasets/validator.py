"""
Dataset Validator

Validates YOLO Detection Dataset

Dataset/
    train/
        images/
        labels/

    valid/
        images/
        labels/

    test/
        images/
        labels/
"""

from pathlib import Path
from collections import Counter

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


class DatasetValidator:

    def __init__(self, dataset_root):

        self.root = Path(dataset_root).resolve()

        self.splits = [
            "train",
            "valid",
            "test"
        ]

        self.errors = []
        self.warnings = []

        self.statistics = {
            "train": {},
            "valid": {},
            "test": {}
        }

        self.class_counter = Counter()

        self.total_images = 0
        self.total_labels = 0

        self.detection_annotations = 0

        self.segmentation_annotations = 0

        self.total_polygon_vertices = 0

        ########################################################
        # Negative Sample Statistics
        ########################################################

        self.negative_samples = {
            "train": [],
            "valid": [],
            "test": []
        }
    ########################################################

    def get_images(self, directory):

        images = []

        for ext in IMAGE_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))

        return sorted(images)

    ########################################################

    def validate(self):

        print("=" * 60)
        print("DATASET VALIDATION")
        print("=" * 60)

        print("Dataset Root :", self.root)
        print("Exists       :", self.root.exists())
        print()

        self.check_dataset_exists()

        self.check_directory_structure()

        self.check_images_labels()

        self.summary()

        return len(self.errors) == 0

    ########################################################

    def check_dataset_exists(self):

        if not self.root.exists():

            self.errors.append(
                f"Dataset not found : {self.root}"
            )

    ########################################################

    def check_directory_structure(self):

        for split in self.splits:

            split_dir = self.root / split

            if not split_dir.exists():

                self.errors.append(
                    f"Missing directory : {split_dir}"
                )

                continue

            image_dir = split_dir / "images"

            label_dir = split_dir / "labels"

            if not image_dir.exists():

                self.errors.append(
                    f"Missing images directory : {image_dir}"
                )

            if not label_dir.exists():

                self.errors.append(
                    f"Missing labels directory : {label_dir}"
                )

    ########################################################

    def check_images_labels(self):

        for split in self.splits:

            image_dir = self.root / split / "images"
            label_dir = self.root / split / "labels"

            if not image_dir.exists():
                continue

            images = self.get_images(image_dir)

            labels = list(label_dir.glob("*.txt"))

            self.statistics[split]["images"] = len(images)
            self.statistics[split]["labels"] = len(labels)

            self.total_images += len(images)
            self.total_labels += len(labels)

            image_names = {img.stem for img in images}
            label_names = {lbl.stem for lbl in labels}

            #
            # Missing labels
            #

            for img in images:

                label = label_dir / (img.stem + ".txt")

                if not label.exists():
                    self.warnings.append(
                        f"[{split}] Missing label: {img.name}"
                    )

            #
            # Orphan labels
            #

            for lbl in labels:

                if lbl.stem not in image_names:
                    self.warnings.append(
                        f"[{split}] Orphan label: {lbl.name}"
                    )

            #
            # Validate label contents
            #

            for lbl in labels:
                self.validate_label(lbl)

    ########################################################

    def validate_label(self, label_path):

        with open(label_path, "r") as f:
            lines = f.readlines()

        ########################################################
        # Empty Label File
        #
        # In YOLO object detection datasets, an empty label file
        # usually represents a valid negative/background image.
        ########################################################

        if len(lines) == 0:

            split = label_path.parent.parent.name

            image_found = False

            image_dir = label_path.parent.parent / "images"

            for ext in IMAGE_EXTENSIONS:

                image_path = image_dir / (label_path.stem + ext)

                if image_path.exists():
                    image_found = True
                    break

                image_path = image_dir / (label_path.stem + ext.upper())

                if image_path.exists():
                    image_found = True
                    break

            if image_found:

                self.negative_samples[split].append(label_path.stem)

            else:

                self.warnings.append(
                    f"Orphan empty label: {label_path.name}"
                )

            return

        for line_number, line in enumerate(lines, start=1):

            values = line.strip().split()

            if len(values) == 0:
                continue

            #
            # YOLO Detection
            #
            if len(values) == 5:

                self.validate_detection(
                    values,
                    label_path,
                    line_number
                )

            #
            # YOLO Segmentation
            #
            elif len(values) >= 7 and ((len(values) - 1) % 2 == 0):

                self.validate_segmentation(
                    values,
                    label_path,
                    line_number
                )

            else:

                self.errors.append(
                    f"{label_path.name} "
                    f"(line {line_number}) "
                    f"Unknown YOLO annotation format "
                    f"({len(values)} values)"
                )

    ########################################################

    def validate_detection(
            self,
            values,
            label_path,
            line_number
    ):

        try:

            class_id = int(values[0])

            x = float(values[1])
            y = float(values[2])
            w = float(values[3])
            h = float(values[4])

        except Exception:

            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) "
                f"Invalid numeric values"
            )

            return

        #
        # Count classes
        #

        self.class_counter[class_id] += 1

        #
        # Class validation
        #

        if class_id not in [0, 1]:
            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) "
                f"Invalid class id {class_id}"
            )

        #
        # Bounding boxes
        #

        if not (0 <= x <= 1):
            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) invalid x"
            )

        if not (0 <= y <= 1):
            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) invalid y"
            )

        if not (0 < w <= 1):
            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) invalid width"
            )

        if not (0 < h <= 1):
            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) invalid height"
            )

        #
        # Statistics
        #

        self.detection_annotations += 1

    ########################################################

    def validate_segmentation(
            self,
            values,
            label_path,
            line_number
    ):

        try:

            class_id = int(values[0])

        except Exception:

            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) "
                f"Invalid class id"
            )

            return

        self.class_counter[class_id] += 1

        if class_id not in [0, 1]:
            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) "
                f"Invalid class id {class_id}"
            )

        coords = []

        try:

            coords = list(
                map(float, values[1:])
            )

        except Exception:

            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) "
                f"Invalid polygon values"
            )

            return

        xs = coords[0::2]
        ys = coords[1::2]

        #
        # Polygon needs at least 3 points
        #

        if len(xs) < 3:
            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) "
                f"Polygon has fewer than 3 vertices"
            )

        #
        # Normalized coordinates
        #

        for x in xs:

            if not (0 <= x <= 1):
                self.errors.append(
                    f"{label_path.name} "
                    f"(line {line_number}) "
                    f"Invalid polygon x={x}"
                )

        for y in ys:

            if not (0 <= y <= 1):
                self.errors.append(
                    f"{label_path.name} "
                    f"(line {line_number}) "
                    f"Invalid polygon y={y}"
                )

        #
        # Bounding box
        #

        xmin = min(xs)
        xmax = max(xs)

        ymin = min(ys)
        ymax = max(ys)

        width = xmax - xmin
        height = ymax - ymin

        if width <= 0:
            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) "
                f"Invalid polygon width"
            )

        if height <= 0:
            self.errors.append(
                f"{label_path.name} "
                f"(line {line_number}) "
                f"Invalid polygon height"
            )

        #
        # Statistics
        #

        self.segmentation_annotations += 1

        self.total_polygon_vertices += len(xs)


    ########################################################

    def summary(self):

        print()

        print("=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)

        #
        # Split Statistics
        #

        for split in self.splits:
            stats = self.statistics.get(split, {})

            print(f"\n{split.upper()}")

            print(f"  Images : {stats.get('images', 0)}")
            print(f"  Labels : {stats.get('labels', 0)}")

        #
        # Overall Statistics
        #

        print("\n" + "=" * 70)
        print("OVERALL DATASET")

        print(f"Total Images : {self.total_images}")
        print(f"Total Labels : {self.total_labels}")

        #
        # Annotation Statistics
        #

        print("\n" + "=" * 70)
        print("ANNOTATION TYPES")

        print(f"Detection Annotations   : {self.detection_annotations}")
        print(f"Segmentation Annotations: {self.segmentation_annotations}")

        total_annotations = (
                self.detection_annotations +
                self.segmentation_annotations
        )

        print(f"Total Annotations       : {total_annotations}")

        if self.segmentation_annotations > 0:
            avg_vertices = (
                    self.total_polygon_vertices /
                    self.segmentation_annotations
            )

            print(
                f"Average Polygon Vertices : {avg_vertices:.2f}"
            )

        ########################################################
        # Negative Sample Analysis
        ########################################################

        print("\n" + "=" * 70)
        print("NEGATIVE SAMPLE ANALYSIS")

        total_negative = 0

        for split in self.splits:

            count = len(self.negative_samples[split])

            total_negative += count

            images = self.statistics[split].get("images", 0)

            ratio = 0.0

            if images > 0:
                ratio = (count / images) * 100

            print(f"\n{split.upper()}")

            print(f"Negative Images : {count}")
            print(f"Image Ratio     : {ratio:.2f}%")

        print("\nOVERALL")

        overall_ratio = 0.0

        if self.total_images > 0:
            overall_ratio = (
                                    total_negative /
                                    self.total_images
                            ) * 100

        print(f"Negative Samples : {total_negative}")
        print(f"Dataset Ratio    : {overall_ratio:.2f}%")

        #
        # Dataset Health
        #

        if overall_ratio < 5:

            health = "Excellent"

        elif overall_ratio < 15:

            health = "Good"

        elif overall_ratio < 30:

            health = "Acceptable"

        else:

            health = "Needs Review"

        print(f"Dataset Health   : {health}")

        #
        # Errors
        #

        print("\n" + "=" * 70)
        print(f"ERRORS ({len(self.errors)})")

        if len(self.errors) == 0:

            print("None")

        else:

            max_errors_to_show = 50

            for error in self.errors[:max_errors_to_show]:
                print(f"• {error}")

            if len(self.errors) > max_errors_to_show:
                print(
                    f"\n... {len(self.errors) - max_errors_to_show} "
                    f"more errors not shown ..."
                )

        #
        # Warnings
        #

        print("\n" + "=" * 70)
        print(f"DATASET WARNINGS ({len(self.warnings)})")
        if len(self.warnings) == 0:

            print("None")

        else:

            max_warnings_to_show = 50

            for warning in self.warnings[:max_warnings_to_show]:
                print(f"• {warning}")

            if len(self.warnings) > max_warnings_to_show:
                print(
                    f"\n... {len(self.warnings) - max_warnings_to_show} "
                    f"more warnings not shown ..."
                )

        ########################################################
        # Dataset Quality Score
        ########################################################

        print("\n" + "=" * 70)
        print("DATASET QUALITY")

        score = 100.0

        score -= len(self.errors) * 5
        score -= len(self.warnings) * 0.25

        score = max(score, 0)

        print(f"Quality Score : {score:.1f}/100")

        if score >= 95:
            quality = "Excellent"

        elif score >= 90:
            quality = "Very Good"

        elif score >= 80:
            quality = "Good"

        elif score >= 70:
            quality = "Fair"

        else:
            quality = "Poor"

        print(f"Assessment   : {quality}")

        #
        # Final Status
        #

        print("\n" + "=" * 70)

        if len(self.errors) == 0:

            if len(self.warnings) == 0:

                status = "PASSED"

            else:

                status = "PASSED WITH WARNINGS"

        else:

            status = "FAILED"

        print(f"VALIDATION STATUS : {status}")

        print("=" * 70)