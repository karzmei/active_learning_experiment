import pandas as pd

from active_learning.create_and_split_data.split_data import (
    split_csv_initial_training_set,
    split_csv_test_validation_pool,
)
from active_learning.utils.constants import (
    BaseSplitNames,
    DataSplitColNames,
)


def create_dummy_data(num_classes=3, samples_per_class=100):
    rows = []

    for class_idx in range(num_classes):
        class_name = f"class_{class_idx}"

        for image_idx in range(samples_per_class):
            rows.append({
                DataSplitColNames.RELATIVE_PATH:
                    f"{class_name}/image_{image_idx}.jpg",
                DataSplitColNames.CLASS_NAME: class_name,
            })

    return pd.DataFrame(rows)


def test_initial_data_split(tmp_path):
    source_csv_path = tmp_path / "data.csv"
    split_csv_path = tmp_path / "base_data_split.csv"

    num_classes = 3
    initial_training_size=31
    samples_per_class = 30

    source_df = create_dummy_data(
        num_classes=num_classes,
        samples_per_class=samples_per_class,
    )
    source_df.to_csv(source_csv_path, index=False)

    split_csv_test_validation_pool(
        data_csv_path=source_csv_path,
        test_size=0.1,
        val_size_out_of_pool=0.1,
        random_state=0,
        base_data_split_save_path=split_csv_path,
    )

    split_csv_initial_training_set(
        base_data_split_csv_path=split_csv_path,
        initial_training_size=initial_training_size,
        random_state=0,
    )

    split_df = pd.read_csv(split_csv_path)

    # Every source image appears exactly once.
    assert len(split_df) == len(source_df)
    assert split_df[DataSplitColNames.RELATIVE_PATH].is_unique
    assert set(split_df[DataSplitColNames.RELATIVE_PATH]) == set(
        source_df[DataSplitColNames.RELATIVE_PATH]
    )

    # Every row belongs to a recognized split.
    assert set(split_df[DataSplitColNames.BASE_SPLIT]) == {
        BaseSplitNames.INITIAL_TRAIN,
        BaseSplitNames.POOL,
        BaseSplitNames.VALIDATION,
        BaseSplitNames.TEST,
    }

    # The requested number of initial examples was selected.
    initial_train_df = split_df[
        split_df[DataSplitColNames.BASE_SPLIT]
        == BaseSplitNames.INITIAL_TRAIN
    ]
    assert len(initial_train_df) == initial_training_size

    # Every class is represented in every split.
    expected_classes = {f"class_{i}" for i in range(num_classes)}

    for split_name in BaseSplitNames:
        classes_in_split = set(
            split_df.loc[
                split_df[DataSplitColNames.BASE_SPLIT] == split_name,
                DataSplitColNames.CLASS_NAME,
            ]
        )
        assert classes_in_split == expected_classes