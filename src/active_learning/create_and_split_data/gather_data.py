from pathlib import Path
from torchvision.datasets import EuroSAT
import pandas as pd

from active_learning.utils.constants import DATA_MAIN_DIR_PATH, DATA_FOLDERS_PATH, DATA_MAIN_CSV_PATH, DataSplitColNames

def download_data(data_dir_path: Path = DATA_MAIN_DIR_PATH):
    dataset = EuroSAT(
        root=data_dir_path,
        download=True,
    )

def create_main_data_table(data_dir_path: Path = DATA_FOLDERS_PATH, path_to_save_csv: Path = DATA_MAIN_CSV_PATH):
    classes_dir_paths = sorted(
        path for path in data_dir_path.iterdir()
        if path.is_dir()
    )

    relative_paths = []
    labels = []
    for class_dir_path in classes_dir_paths:
        img_paths = sorted(class_dir_path.glob("*.jpg"))
        relative_img_paths = [img_path.relative_to(data_dir_path) for img_path in img_paths]
        relative_paths.extend(relative_img_paths)
        labels.extend([class_dir_path.stem] * len(relative_img_paths))

    df = pd.DataFrame({
        DataSplitColNames.RELATIVE_PATH: relative_paths,
        DataSplitColNames.CLASS_NAME: labels
    })

    path_to_save_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_to_save_csv, index=False)


if __name__ == "__main__":
    download_data(DATA_MAIN_DIR_PATH)
    create_main_data_table(DATA_FOLDERS_PATH, DATA_MAIN_CSV_PATH)
