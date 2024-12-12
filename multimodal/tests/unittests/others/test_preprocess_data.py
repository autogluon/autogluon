import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from autogluon.multimodal.data import (
    CustomLabelEncoder,
    data_to_df,
    infer_column_types,
    infer_problem_type,
    init_df_preprocessor,
)
from autogluon.multimodal.utils import get_config, is_url

from ..utils import PetFinderDataset


@pytest.mark.parametrize(
    "labels,positive_class",
    [
        ([1, 2, 2], 2),
        ([1, 2, 2], 1),
        ([1, 0, 1, 0, 0], 0),
        ([1, 0, 1, 0, 0], None),
        (["a", "e", "e", "a"], "a"),
        (["b", "d", "b", "d"], "d"),
        (["a", "d", "e", "b"], "d"),
        ([3, 2, 1, 0], 2),
    ],
)
def test_label_encoder(labels, positive_class):
    label_encoder = CustomLabelEncoder(positive_class=positive_class)
    label_encoder.fit(labels)

    # test encoding positive class
    if positive_class:
        assert label_encoder.transform([positive_class]).item() == len(label_encoder.classes_) - 1
    else:
        assert label_encoder.transform([label_encoder.classes_[-1]]).item() == len(label_encoder.classes_) - 1

    # test encoding
    sklearn_le = LabelEncoder()
    sklearn_le.fit(labels)
    sk_encoded_labels = sklearn_le.transform(labels)
    our_encoded_labels = label_encoder.transform(labels)
    if positive_class:
        sk_pos_label = sklearn_le.transform([positive_class]).item()
    else:
        sk_pos_label = len(sklearn_le.classes_) - 1
    sk_encoded_labels[sk_encoded_labels == sk_pos_label] = len(sklearn_le.classes_)
    sk_encoded_labels[sk_encoded_labels > sk_pos_label] -= 1
    sk_encoded_labels[sk_encoded_labels < 0] = 0
    assert (sk_encoded_labels == our_encoded_labels).all()

    # test inverse encoding
    assert label_encoder.inverse_transform(our_encoded_labels).tolist() == labels


@pytest.mark.parametrize(
    "data,required_columns,all_columns,is_valid_input",
    [
        ([1, 2, 3], ["a"], ["a"], True),
        ([1, 2, 3], ["a"], ["a", "b"], False),
        ({"a": [1, 2, 3]}, ["a"], ["a", "b"], True),
        ({"a": [1, 2, 3]}, ["b"], ["a", "b"], False),
        ({"a": [1, 2, 3], "b": [4, 5, 6]}, ["b", "a"], ["a", "b"], True),
        ([(1, 2), (3, 4)], ["a", "b"], ["a", "b"], True),
        ([[1, 2], [3, 4]], ["a"], ["a", "b"], True),
        ([[1, 2], [3, 4]], ["a", "b"], ["a", "b", "c"], False),
        ([{"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}], ["a", "b", "c"], ["a", "b", "c"], True),
        ([{"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}], ["a", "c"], ["a", "b", "c"], True),
        ([{"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}], ["a", "b", "c", "d"], ["a", "b", "c", "d"], False),
        ([{"a": 1, "b": 2, "c": 3}, {"a": 10, "b": 20, "c": 30}], ["a", "b", "d"], ["a", "b", "c", "d"], False),
        (pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), ["b", "a"], ["b", "c", "a"], True),
    ],
)
def test_data_to_df(data, required_columns, all_columns, is_valid_input):
    if is_valid_input:
        df = data_to_df(data=data, required_columns=required_columns, all_columns=all_columns)
    else:
        with pytest.raises(ValueError):
            df = data_to_df(data=data, required_columns=required_columns, all_columns=all_columns)


@pytest.mark.parametrize(
    "path,is_valid_url",
    [
        ("/media/data/coco17/annotations/instances_val2017.json", False),
        ("This is a test.", False),
        ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", True),
        ("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar", True),
        (
            "https://automl-mm-bench.s3.amazonaws.com/voc_script/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth",
            True,
        ),
    ],
)
def test_is_url(path, is_valid_url):
    assert is_url(path) == is_valid_url


@pytest.mark.parametrize(
    "convert_categorical_to_text,convert_numerical_to_text",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_convert_to_text(convert_categorical_to_text, convert_numerical_to_text):
    overrides = {
        "data.categorical.convert_to_text": convert_categorical_to_text,
        "data.numerical.convert_to_text": convert_numerical_to_text,
    }
    dataset = PetFinderDataset()
    label_column = dataset.label_columns[0]
    train_data = dataset.train_df
    config = get_config(
        problem_type=dataset.problem_type,
        overrides=overrides,
    )
    column_types = infer_column_types(
        data=train_data,
        label_columns=label_column,
        problem_type=dataset.problem_type,
    )
    df_preprocessor = init_df_preprocessor(
        config=config,
        column_types=column_types,
        label_column=label_column,
        train_df_x=train_data.drop(columns=label_column),
        train_df_y=train_data[label_column],
    )
    if convert_categorical_to_text:
        assert len(df_preprocessor.categorical_feature_names) == 0
        assert len(df_preprocessor.text_feature_names) > 0
    if convert_numerical_to_text:
        assert len(df_preprocessor.numerical_feature_names) == 0
        assert len(df_preprocessor.text_feature_names) > 0
