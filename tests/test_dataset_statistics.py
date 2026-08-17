from pathlib import Path

from datasets.statistics.statistics import DatasetStatistics

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_PATH = PROJECT_ROOT / "Dataset"

print("Project Root :", PROJECT_ROOT)
print("Dataset Path :", DATASET_PATH)
print()

# Create statistics engine
stats_engine = DatasetStatistics(DATASET_PATH)

# Run all analyzers
stats_engine.analyze()

# Print summary
stats_engine.print_summary()