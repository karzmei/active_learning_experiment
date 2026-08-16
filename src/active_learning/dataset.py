from torch.utils.data import Dataset
from PIL import Image

from active_learning.utils.constants import DataSplitColNames, EUROSAT_CLASS_TO_IDX
from active_learning.utils.image_transforms import model_preprocessing_transform


class EuroSATDataset(Dataset):
    def __init__(self, data_split_df, data_main_dir_path, transform=None):
        self.data_split_df = data_split_df
        self.data_main_dir_path = data_main_dir_path
        self.transform = transform

    def __len__(self):
        return len(self.data_split_df)

    def __getitem__(self, idx):
        row = self.data_split_df.iloc[idx]
        relative_path = row[DataSplitColNames.RELATIVE_PATH]
        class_name = row[DataSplitColNames.CLASS_NAME]
        if class_name not in EUROSAT_CLASS_TO_IDX:
            raise ValueError(f"Class name '{class_name}' not found in EUROSAT_CLASS_TO_IDX mapping.")
        
        label = EUROSAT_CLASS_TO_IDX[class_name]

        image_path = self.data_main_dir_path / relative_path
        image = self.load_image(image_path)

        image = image.resize((224, 224))

        if self.transform:
            image = self.transform(image)

        tensored_image = model_preprocessing_transform(image)

        return tensored_image, label

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image
