from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJ_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATASET = PROCESSED_DATA_DIR / "final_dataset_with_all_features_v3.1.csv"

RUNS_DIR = PROJ_ROOT / "runs"
