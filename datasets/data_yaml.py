"""
======================================================================
YOLO Dataset YAML Generator
======================================================================
"""

from pathlib import Path
import yaml


class DataYamlGenerator:
    """
    Generates a valid Ultralytics data.yaml file from the dataset.
    """

    ##################################################################

    def __init__(self, dataset_root: Path):

        self.dataset_root = Path(dataset_root).resolve()

        self.output_file = self.dataset_root / "data.yaml"

        self.class_names = {
            0: "bird",
            1: "drone"
        }

    ##################################################################

    def validate(self):

        required = [

            self.dataset_root / "train" / "images",

            self.dataset_root / "train" / "labels",

            self.dataset_root / "valid" / "images",

            self.dataset_root / "valid" / "labels",

            self.dataset_root / "test" / "images",

            self.dataset_root / "test" / "labels",

        ]

        missing = []

        for directory in required:

            if not directory.exists():

                missing.append(directory)

        if missing:

            raise FileNotFoundError(

                "Dataset structure is incomplete:\n"

                + "\n".join(str(x) for x in missing)

            )

    ##################################################################

    def build(self):

        return {

            "path": str(self.dataset_root),

            "train": "train/images",

            "val": "valid/images",

            "test": "test/images",

            "names": self.class_names

        }

    ##################################################################

    def write(self):

        data = self.build()

        with open(

                self.output_file,

                "w",

                encoding="utf-8"

        ) as file:

            yaml.safe_dump(

                data,

                file,

                sort_keys=False,

                allow_unicode=True

            )

        return self.output_file

    ##################################################################

    def generate(self):

        self.validate()

        return self.write()