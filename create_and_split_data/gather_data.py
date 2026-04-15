from pathlib import Path
from torchvision.datasets import EuroSAT
import pandas as pd

from utils.constants import DATA_MAIN_DIR_PATH, DATA_FOLDERS_PATH, DATA_MAIN_CSV_PATH

def download_data(data_dir_path: Path = DATA_MAIN_DIR_PATH):
    dataset = EuroSAT(
        root=data_dir_path,
        download=True,
    )

def create_main_data_table(data_dir_path: Path = DATA_FOLDERS_PATH, path_to_save_csv: Path = DATA_MAIN_CSV_PATH):
    classes_dir_paths = data_dir_path.iterdir()

    paths = []
    labels = []
    for class_dir_path in classes_dir_paths:
        img_paths = list(class_dir_path.glob("*.jpg"))
        paths.extend(img_paths)
        labels.extend([class_dir_path.stem] * len(img_paths))

    df = pd.DataFrame({"image_path": paths, "label": labels})

    df.to_csv(path_to_save_csv, index=False)


if __name__ == "__main__":
    download_data(DATA_MAIN_DIR_PATH)
    create_main_data_table(DATA_FOLDERS_PATH, DATA_MAIN_CSV_PATH)