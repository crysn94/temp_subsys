from pathlib import Path

from datasets.dataset_manager import DatasetManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "Dataset"

manager = DatasetManager(DATASET)

print("Dataset Exists :", manager.exists())

manager.print_summary()