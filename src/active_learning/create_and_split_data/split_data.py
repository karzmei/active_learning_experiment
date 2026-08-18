import warnings

from sklearn.model_selection import train_test_split
import pandas as pd


from active_learning.utils.constants import DATA_MAIN_CSV_PATH, TEST_SPLIT_SIZE, BASE_DATA_SPLIT_CSV_PATH, VALIDATION_SPLIT_SIZE_OUT_OF_TRAINING_POOL
from active_learning.utils.constants import INITIAL_DATA_SPLIT_SEED, BaseSplitNames, DataSplitColNames

def split_csv_test_validation_pool(data_csv_path=DATA_MAIN_CSV_PATH, test_size=TEST_SPLIT_SIZE, val_size_out_of_pool=VALIDATION_SPLIT_SIZE_OUT_OF_TRAINING_POOL, random_state=INITIAL_DATA_SPLIT_SEED, base_data_split_save_path=BASE_DATA_SPLIT_CSV_PATH):
    data_df = pd.read_csv(data_csv_path)

    trainval_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state, stratify=data_df[DataSplitColNames.CLASS_NAME])

    pool_df, val_df = train_test_split(trainval_df, test_size=val_size_out_of_pool, random_state=random_state, stratify=trainval_df[DataSplitColNames.CLASS_NAME])

    base_data_split_df = pd.concat([
        test_df.assign(base_split=BaseSplitNames.TEST),
        val_df.assign(base_split=BaseSplitNames.VALIDATION),
        pool_df.assign(base_split=BaseSplitNames.POOL)
    ])

    base_data_split_save_path.parent.mkdir(parents=True, exist_ok=True)
    base_data_split_df.to_csv(base_data_split_save_path, index=False)



def split_csv_initial_training_set(base_data_split_csv_path=BASE_DATA_SPLIT_CSV_PATH, initial_training_size=1000, random_state=INITIAL_DATA_SPLIT_SEED):
    df = pd.read_csv(base_data_split_csv_path)
    pool_df = df[df[DataSplitColNames.BASE_SPLIT]==BaseSplitNames.POOL]

    if pool_df.empty:
        raise ValueError("Cannot create an initial training set from an empty pool.")

    number_of_classes = pool_df[DataSplitColNames.CLASS_NAME].nunique()

    samples_per_class = initial_training_size // number_of_classes

    pool_class_sizes = pool_df[DataSplitColNames.CLASS_NAME].value_counts()

    if initial_training_size > len(pool_df):
        raise ValueError(
            f"Cannot select {initial_training_size} initial samples "
            f"from a pool containing only {len(pool_df)} samples."
        )
    if initial_training_size < number_of_classes:
        raise ValueError(
            "The initial training size must be at least the number of classes."
        )

    if samples_per_class > pool_class_sizes.min():
        raise ValueError(f"Not enough samples in the pool to select {samples_per_class} samples per class. The smallest class has {pool_class_sizes.min()} samples.")

    initial_training_split_indices = pool_df.groupby(DataSplitColNames.CLASS_NAME, group_keys=False).sample(samples_per_class, random_state=random_state).index

    num_missing_samples = initial_training_size - len(initial_training_split_indices)
    if num_missing_samples > 0:
        remaining_pool_indices = pool_df.index.difference(initial_training_split_indices)
        additional_indices = pool_df.loc[remaining_pool_indices].sample(num_missing_samples, random_state=random_state).index
        initial_training_split_indices = initial_training_split_indices.union(additional_indices)

    df.loc[initial_training_split_indices, DataSplitColNames.BASE_SPLIT] = BaseSplitNames.INITIAL_TRAIN

    sampled_class_counts = df.loc[
        initial_training_split_indices, 
        DataSplitColNames.CLASS_NAME,
        ].value_counts()

    mean_class_size = sampled_class_counts.mean()

    relative_deviation = (
        (sampled_class_counts - mean_class_size).abs()
        / mean_class_size
        )

    if (relative_deviation > 0.1).any():
        warnings.warn(
            "At least one class in the initial training split differs "
            "from the mean class size by more than 10%.",
        stacklevel=2,
        )

    df.to_csv(base_data_split_csv_path, index=False)


if __name__ == "__main__":
    split_csv_test_validation_pool()
    split_csv_initial_training_set()
