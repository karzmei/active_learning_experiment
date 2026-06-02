from pathlib import Path

INITIAL_DATA_SPLIT_SEED = 0
EXPERIMENTS_SEEDS = [0,1,2]

TEST_SPLIT_SIZE = 0.1
VALIDATION_SPLIT_SIZE = 0.055
INITIAL_TRAINING_SPLIT_SIZE = 1000

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_MAIN_DIR_PATH = BASE_DIR / "data"
DATA_FOLDERS_PATH = DATA_MAIN_DIR_PATH / "eurosat" / "2750"
DATA_MAIN_CSV_PATH = DATA_MAIN_DIR_PATH / "data.csv"
DATA_SPLITS_DIR_PATH = DATA_MAIN_DIR_PATH / "data_splits"
BASE_DATA_SPLIT_CSV_PATH = DATA_SPLITS_DIR_PATH / "base_data_split.csv"


if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")