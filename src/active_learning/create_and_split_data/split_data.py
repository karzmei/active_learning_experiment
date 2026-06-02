from sklearn.model_selection import train_test_split
import pandas as pd


from src.active_learning.utils.constants import DATA_MAIN_CSV_PATH, TEST_SPLIT_SIZE, VALIDATION_SPLIT_SIZE, BASE_DATA_SPLIT_CSV_PATH
from src.active_learning.utils.constants import INITIAL_DATA_SPLIT_SEED

def split_csv_test_validation_pool(data_csv_path=DATA_MAIN_CSV_PATH, test_size=TEST_SPLIT_SIZE, val_size=VALIDATION_SPLIT_SIZE, random_state=INITIAL_DATA_SPLIT_SEED, base_data_split_save_path=BASE_DATA_SPLIT_CSV_PATH, ):
    data_df = pd.read_csv(data_csv_path)

    trainval_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state, stratify=data_df["label"])

    pool_df, val_df = train_test_split(trainval_df, test_size=val_size, random_state=random_state, stratify=trainval_df["label"])

    # create df with columns: relative_path, label, base_split (with values: "", "validation", "test")

    base_data_split_df = pd.concat([
        test_df.assign(base_split="test"),
        val_df.assign(base_split="validation"),
        pool_df.assign(base_split="pool")
    ])

    base_data_split_save_path.parent.mkdir(parents=True, exist_ok=True)
    base_data_split_df.to_csv(base_data_split_save_path, index=False)



def split_csv_initial_training_set(base_data_split_csv_path=BASE_DATA_SPLIT_CSV_PATH, initial_training_size=1000, random_state=INITIAL_DATA_SPLIT_SEED):
    df = pd.read_csv(base_data_split_csv_path)
    number_of_classes = df["label"].nunique()

    initial_training_split_indices = df[df['base_split']=="pool"].groupby("label", group_keys=False).sample(initial_training_size // number_of_classes, random_state=random_state).index

    df.loc[initial_training_split_indices, "base_split"] = "initial_train"
    df.to_csv(base_data_split_csv_path, index=False)

if __name__ == "__main__":
    split_csv_test_validation_pool()
    split_csv_initial_training_set()