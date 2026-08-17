from pathlib import Path

from datasets.data_yaml import DataYamlGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "Dataset"

generator = DataYamlGenerator(DATASET)

yaml_file = generator.generate()

print()

print("=" * 70)

print("GENERATED DATA YAML")

print("=" * 70)

print(yaml_file)

print()

print(yaml_file.read_text())