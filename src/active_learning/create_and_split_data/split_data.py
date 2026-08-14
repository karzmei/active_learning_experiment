from sklearn.model_selection import train_test_split
import pandas as pd


from src.active_learning.utils.constants import DATA_MAIN_CSV_PATH, TEST_SPLIT_SIZE, BASE_DATA_SPLIT_CSV_PATH, VALIDATION_SPLIT_SIZE_OUT_OF_TRAINING_POOL
from src.active_learning.utils.constants import INITIAL_DATA_SPLIT_SEED, BaseSplitNames, DataSplitColNames

def split_csv_test_validation_pool(data_csv_path=DATA_MAIN_CSV_PATH, test_size=TEST_SPLIT_SIZE, val_size=VALIDATION_SPLIT_SIZE_OUT_OF_TRAINING_POOL, random_state=INITIAL_DATA_SPLIT_SEED, base_data_split_save_path=BASE_DATA_SPLIT_CSV_PATH):
    data_df = pd.read_csv(data_csv_path)

    trainval_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state, stratify=data_df[DataSplitColNames.CLASS_NAME])

    pool_df, val_df = train_test_split(trainval_df, test_size=val_size, random_state=random_state, stratify=trainval_df[DataSplitColNames.CLASS_NAME])

    base_data_split_df = pd.concat([
        test_df.assign(base_split=BaseSplitNames.TEST),
        val_df.assign(base_split=BaseSplitNames.VALIDATION),
        pool_df.assign(base_split=BaseSplitNames.POOL)
    ])

    base_data_split_save_path.parent.mkdir(parents=True, exist_ok=True)
    base_data_split_df.to_csv(base_data_split_save_path, index=False)



def split_csv_initial_training_set(base_data_split_csv_path=BASE_DATA_SPLIT_CSV_PATH, initial_training_size=1000, random_state=INITIAL_DATA_SPLIT_SEED):
    df = pd.read_csv(base_data_split_csv_path)
    number_of_classes = df[DataSplitColNames.CLASS_NAME].nunique()

    initial_training_split_indices = df[df[DataSplitColNames.BASE_SPLIT]==BaseSplitNames.POOL].groupby(DataSplitColNames.CLASS_NAME, group_keys=False).sample(initial_training_size // number_of_classes, random_state=random_state).index

    df.loc[initial_training_split_indices, DataSplitColNames.BASE_SPLIT] = BaseSplitNames.INITIAL_TRAIN
    df.to_csv(base_data_split_csv_path, index=False)

if __name__ == "__main__":
    split_csv_test_validation_pool()
    split_csv_initial_training_set()