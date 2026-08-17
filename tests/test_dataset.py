from pathlib import Path
from datasets.validator import DatasetValidator
import os
import datasets.validator

print("Validator module:", datasets.validator.__file__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "Dataset"

print("Project Root :", PROJECT_ROOT)
print("Dataset Path :", DATASET_PATH)
print("Dataset Exists :", DATASET_PATH.exists())
print("Current Working Directory :", os.getcwd())
print()

validator = DatasetValidator(DATASET_PATH)

is_valid = validator.validate()

if is_valid:
    print("\nDataset validation successful.")
else:
    print("\nDataset validation failed.")