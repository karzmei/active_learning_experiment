from pathlib import Path

INITIAL_DATA_SPLIT_SEED = 0
EXPERIMENTS_SEEDS = [0,1,2]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_MAIN_DIR_PATH = BASE_DIR / "data"
DATA_FOLDERS_PATH = DATA_MAIN_DIR_PATH / "eurosat" / "2750"
DATA_MAIN_CSV_PATH = DATA_MAIN_DIR_PATH / "data.csv"


if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")