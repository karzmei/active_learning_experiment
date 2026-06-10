import torch
from torch.utils.data import DataLoader

from src.active_learning.dataset import EuroSATDataset
from src.active_learning.model import ResNet18Model
from src.active_learning.utils.constants import DATA_FOLDERS_PATH, DataSplitColNames
from src.active_learning.utils.image_transforms import train_augmentations


def train(model, train_dataloader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0.0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

def get_training_data(base_split_csv_path, init_training_col_name, additional_csv_path=None):
    import pandas as pd

    base_data_split_df = pd.read_csv(base_split_csv_path)
    train_split_df = base_data_split_df[base_data_split_df[DataSplitColNames.BASE_SPLIT] == init_training_col_name]

    if additional_csv_path:
        additional_df = pd.read_csv(additional_csv_path)
        train_split_df = pd.concat([train_split_df, additional_df], ignore_index=True)

    return train_split_df

def run_training_loop():
    from src.active_learning.utils.constants import BASE_DATA_SPLIT_CSV_PATH
    import pandas as pd

    train_split_df = get_training_data(BASE_DATA_SPLIT_CSV_PATH, init_training_col_name="initial_train")

    train_dataset = EuroSATDataset(train_split_df, DATA_FOLDERS_PATH, transform=train_augmentations)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ResNet18Model(pretrained=True, num_classes=10)
    train(model, train_dataloader, num_epochs=10)


if __name__ == "__main__":
    run_training_loop()