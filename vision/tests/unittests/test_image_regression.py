from autogluon.vision import ImagePredictor as Task
import autogluon.core as ag
import os
import pandas as pd
import numpy as np


def test_task():
    dataset, _, test_dataset = Task.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    model_list = Task.list_models()
    predictor = Task(problem_type='regression')
    predictor.fit(dataset, num_trials=2, hyperparameters={'epochs': 3, 'batch_size': 8})
    test_result = predictor.predict(test_dataset)
    single_test = predictor.predict(test_dataset.iloc[0]['image'])
    predictor.save('regressor.ag')
    predictor2 = Task.load('regressor.ag')
    fit_summary = predictor2.fit_summary()
    test_score = predictor2.evaluate(test_dataset)
    # raw dataframe
    df_test_dataset = pd.DataFrame(test_dataset)
    test_score = predictor2.evaluate(df_test_dataset)
    assert test_score < 2, f'{test_score} too bad'
    test_feature = predictor2.predict_feature(test_dataset)
    single_test2 = predictor2.predict(test_dataset.iloc[0]['image'])
    assert isinstance(single_test2, pd.Series)
    assert single_test2.equals(single_test)
    # to numpy
    test_feature_numpy = predictor2.predict_feature(test_dataset, as_pandas=False)
    single_test2_numpy = predictor2.predict(test_dataset.iloc[0]['image'], as_pandas=False)
    assert np.array_equal(single_test2.to_numpy(), single_test2_numpy)
