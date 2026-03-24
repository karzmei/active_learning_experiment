import pandas as pd

from utils.constants import INITIAL_DATA_SPLIT_SEED

def split_initial_test_train_validation(data_df_path, dir_save_path, test_percentage=0.2, validation_percentage=0.1):
    data_df = pd.read_csv(data_df_path)

    data_df = data_df.sample(frac=1, random_state=INITIAL_DATA_SPLIT_SEED).reset_index(drop=True)

    total_samples = len(data_df)
    test_samples = int(total_samples * test_percentage)
    validation_samples = int(total_samples * validation_percentage)

    test_df = data_df.iloc[:test_samples]
    validation_df = data_df.iloc[test_samples:test_samples + validation_samples]
    train_df = data_df.iloc[test_samples + validation_samples:]

    test_df.to_csv(f"{dir_save_path}/test.csv", index=False)
    validation_df.to_csv(f"{dir_save_path}/validation.csv", index=False)
    train_df.to_csv(f"{dir_save_path}/train.csv", index=False)


def sample_initial_labeled_data(train_df_path, dir_save_path, sample_percentage=0.1, sample_size=1000):
    train_df = pd.read_csv(train_df_path)

    if sample_size is not None:
        total_num_samples = len(train_df)
        if sample_size < total_num_samples:
            sample_percentage = sample_size / total_num_samples


    sampled_train_df = train_df.sample(frac=sample_percentage, random_state=INITIAL_DATA_SPLIT_SEED).reset_index(drop=True)

    sampled_train_df.to_csv(f"{dir_save_path}/initial_labeled_train_data.csv", index=False)