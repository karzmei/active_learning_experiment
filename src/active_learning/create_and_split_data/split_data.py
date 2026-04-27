from sklearn.model_selection import train_test_split
import pandas as pd


from src.active_learning.utils.constants import DATA_MAIN_CSV_PATH, INITIAL_REMAINING_POOL_SPLIT_SAVE_PATH, POOL_SPLIT_SAVE_PATH, TEST_SPLIT_SIZE, VALIDATION_SPLIT_SIZE, INITIAL_TRAINING_SPLIT_SAVE_PATH, TEST_SPLIT_SAVE_PATH, VALIDATION_SPLIT_SAVE_PATH
from src.active_learning.utils.constants import INITIAL_DATA_SPLIT_SEED

def split_csv_test_validation_pool(data_csv_path=DATA_MAIN_CSV_PATH, test_size=TEST_SPLIT_SIZE, val_size=VALIDATION_SPLIT_SIZE, random_state=INITIAL_DATA_SPLIT_SEED, test_split_save_path=TEST_SPLIT_SAVE_PATH, validation_split_save_path=VALIDATION_SPLIT_SAVE_PATH, pool_split_save_path=POOL_SPLIT_SAVE_PATH):
    data_df = pd.read_csv(data_csv_path)

    trainval_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state, stratify=data_df["label"])

    pool_df, val_df = train_test_split(trainval_df, test_size=val_size, random_state=random_state, stratify=trainval_df["label"])

<<<<<<< HEAD
    pool_split_save_path.parent.mkdir(parents=True, exist_ok=True)

=======
>>>>>>> 2c90e354c02c6d21e0f67acd009aeef6517d0a64
    pool_df.to_csv(pool_split_save_path, index=False)
    val_df.to_csv(validation_split_save_path, index=False)
    test_df.to_csv(test_split_save_path, index=False)


def split_csv_initial_training_set(pool_csv_path=POOL_SPLIT_SAVE_PATH, initial_training_size=1000, random_state=INITIAL_DATA_SPLIT_SEED, initial_train_split_save_path=INITIAL_TRAINING_SPLIT_SAVE_PATH, initial_remaining_pool_split_save_path=INITIAL_REMAINING_POOL_SPLIT_SAVE_PATH):
    pool_df = pd.read_csv(pool_csv_path)

    initial_train_df, remaining_pool_df = train_test_split(pool_df, test_size=(len(pool_df) - initial_training_size) / len(pool_df), random_state=random_state, stratify=pool_df["label"])

    initial_train_df.to_csv(initial_train_split_save_path, index=False)
    remaining_pool_df.to_csv(initial_remaining_pool_split_save_path, index=False)


if __name__ == "__main__":
    split_csv_test_validation_pool()
    split_csv_initial_training_set()