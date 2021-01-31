import pytest
import numpy as np
import numpy.testing as npt
import tempfile
import pickle
import os
from sklearn.model_selection import train_test_split
from autogluon.core.utils.loaders import load_pd
from autogluon.text.text_prediction.mx.preprocessing import MultiModalTextFeatureProcessor, base_preprocess_cfg
from autogluon.text.text_prediction.infer_types import infer_column_problem_types


TEST_CASES = [
    ['melbourne_airbnb_sample',
     'https://autogluon-text-data.s3.amazonaws.com/test_cases/melbourne_airbnb_sample_1000.pq',
     'price_label']
]


def assert_dataset_match(lhs_dataset, rhs_dataset, threshold=1E-4):
    assert len(lhs_dataset) == len(rhs_dataset)
    for i in range(len(lhs_dataset)):
        for j in range(len(lhs_dataset[0])):
            npt.assert_allclose(lhs_dataset[i][j], rhs_dataset[i][j], threshold, threshold)


@pytest.mark.parametrize('dataset_name,url,label_column', TEST_CASES)
@pytest.mark.parametrize('backbone_name', ['google_electra_small',
                                           'google_albert_base_v2'])
@pytest.mark.parametrize('all_to_text', [False, True])
def test_preprocessor(dataset_name, url, label_column,
                      backbone_name, all_to_text):
    all_df = load_pd.load(url)
    feature_columns = [col for col in all_df.columns if col != label_column]
    train_df, valid_df = train_test_split(all_df, test_size=0.1,
                                          random_state=np.random.RandomState(100))
    column_types, problem_type = infer_column_problem_types(train_df, valid_df,
                                                            label_columns=label_column)
    cfg = base_preprocess_cfg()
    if all_to_text:
        cfg.categorical.convert_to_text = True
        cfg.numerical.convert_to_text = True
    preprocessor = MultiModalTextFeatureProcessor(column_types=column_types,
                                                  label_column=label_column,
                                                  tokenizer_name=backbone_name,
                                                  cfg=cfg)
    train_dataset = preprocessor.fit_transform(train_df[feature_columns], train_df[label_column])
    train_dataset_after_transform = preprocessor.transform(train_df[feature_columns], train_df[label_column])
    for i in range(len(train_dataset)):
        for j in range(len(train_dataset[0])):
            npt.assert_allclose(train_dataset[i][j],
                                train_dataset_after_transform[i][j],
                                1E-4, 1E-4)
    valid_dataset = preprocessor.transform(valid_df[feature_columns], valid_df[label_column])
    test_dataset = preprocessor.transform(valid_df[feature_columns])
    assert_dataset_match(train_dataset, train_dataset_after_transform)
    for i in range(len(test_dataset)):
        for j in range(len(test_dataset[0])):
            npt.assert_allclose(valid_dataset[i][j],
                                test_dataset[i][j],
                                1E-4, 1E-4)
    # Test for pickle dump and load
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        with open(os.path.join(tmp_dir_name, 'preprocessor.pkl'), 'wb') as out_f:
            pickle.dump(preprocessor, out_f)
        with open(os.path.join(tmp_dir_name, 'preprocessor.pkl'), 'rb') as in_f:
            preprocessor_loaded = pickle.load(in_f)
        valid_dataset_loaded = preprocessor_loaded.transform(valid_df[feature_columns],
                                                             valid_df[label_column])
        assert_dataset_match(valid_dataset_loaded, valid_dataset)
        test_dataset_loaded = preprocessor_loaded.transform(valid_df[feature_columns])
        assert_dataset_match(test_dataset_loaded, test_dataset)

