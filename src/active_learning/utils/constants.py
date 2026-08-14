from pathlib import Path
from enum import StrEnum

INITIAL_DATA_SPLIT_SEED = 0
EXPERIMENTS_SEEDS = [0,1,2]

TEST_SPLIT_SIZE = 0.1
VALIDATION_SPLIT_SIZE_OUT_OF_TRAINING_POOL = 0.055
INITIAL_TRAINING_SPLIT_SIZE = 1000

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_MAIN_DIR_PATH = BASE_DIR / "data"
DATA_FOLDERS_PATH = DATA_MAIN_DIR_PATH / "eurosat" / "2750"
DATA_MAIN_CSV_PATH = DATA_MAIN_DIR_PATH / "data.csv"
DATA_SPLITS_DIR_PATH = DATA_MAIN_DIR_PATH / "data_splits"
BASE_DATA_SPLIT_CSV_PATH = DATA_SPLITS_DIR_PATH / "base_data_split.csv"


class BaseSplitNames(StrEnum):
    INITIAL_TRAIN = "initial_train"
    VALIDATION = "validation"
    TEST = "test"
    POOL = "pool"

class DataSplitColNames(StrEnum):
    RELATIVE_PATH = "relative_path"
    CLASS_NAME = "class_name"
    BASE_SPLIT = "base_split"

EUROSAT_CLASS_TO_IDX = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9,
}

EUROSAT_IDX_TO_CLASS = {
    idx: class_name
    for class_name, idx in EUROSAT_CLASS_TO_IDX.items()
}


if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")