"""
dataset_manager.py

Central dataset access layer for the C-UAS Vision Subsystem.

Supports YOLO dataset structure:

Dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
"""

from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


class DatasetManager:

    def __init__(self, dataset_root):

        self.root = Path(dataset_root)

        self.splits = [
            "train",
            "valid",
            "test"
        ]

    ##############################################################

    def exists(self):

        return self.root.exists()

    ##############################################################

    def get_split_directory(self, split):

        return self.root / split

    ##############################################################

    def get_image_directory(self, split):

        return self.root / split / "images"

    ##############################################################

    def get_label_directory(self, split):

        return self.root / split / "labels"

    ##############################################################

    def get_images(self, split):

        image_dir = self.get_image_directory(split)

        if not image_dir.exists():
            print(f"[WARNING] Image directory not found: {image_dir}")

            return []

        images = []

        #
        # Recursive search
        #

        for ext in IMAGE_EXTENSIONS:
            images.extend(image_dir.rglob(f"*{ext}"))

            images.extend(image_dir.rglob(f"*{ext.upper()}"))

        images = sorted(images)

        print(f"[INFO] {split.upper()} -> {len(images)} images")

        return images

    ##############################################################

    def get_labels(self, split):

        label_dir = self.get_label_directory(split)

        if not label_dir.exists():
            return []

        return sorted(label_dir.glob("*.txt"))

    ##############################################################

    def get_all_images(self):

        images = []

        for split in self.splits:

            images.extend(self.get_images(split))

        return images

    ##############################################################

    def get_all_labels(self):

        labels = []

        for split in self.splits:

            labels.extend(self.get_labels(split))

        return labels

    ##############################################################

    def image_to_label(self, image_path):

        image_path = Path(image_path)

        split = image_path.parent.parent.name

        return (
            self.get_label_directory(split)
            / (image_path.stem + ".txt")
        )

    ##############################################################

    def label_to_image(self, label_path):

        label_path = Path(label_path)

        split = label_path.parent.parent.name

        image_dir = self.get_image_directory(split)

        for ext in IMAGE_EXTENSIONS:

            candidate = image_dir / (label_path.stem + ext)

            if candidate.exists():
                return candidate

            candidate = image_dir / (label_path.stem + ext.upper())

            if candidate.exists():
                return candidate

        return None

    ##############################################################

    def dataset_summary(self):

        summary = {}

        for split in self.splits:

            summary[split] = {

                "images": len(
                    self.get_images(split)
                ),

                "labels": len(
                    self.get_labels(split)
                )

            }

        return summary

    ##############################################################

    def print_summary(self):

        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        total_images = 0
        total_labels = 0

        for split in self.splits:

            images = len(self.get_images(split))
            labels = len(self.get_labels(split))

            total_images += images
            total_labels += labels

            print(f"\n{split.upper()}")

            print(f"Images : {images}")
            print(f"Labels : {labels}")

        print("\n" + "=" * 60)

        print(f"Total Images : {total_images}")
        print(f"Total Labels : {total_labels}")

        print("=" * 60)