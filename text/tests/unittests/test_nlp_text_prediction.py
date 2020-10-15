import numpy as np
import pandas as pd
import pytest
from autogluon.core.utils.loaders import load_pd
from autogluon.text import TextPrediction as task

test_hyperparameters = {
    'models': {
        'BertForTextPredictionBasic': {
            'search_space': {
                'optimization.num_train_epochs': 1
            }
        }
    }
}


def test_sst():
    train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                              'glue/sst/train.parquet')
    dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                            'glue/sst/dev.parquet')
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    valid_perm = rng_state.permutation(len(dev_data))
    train_data = train_data.iloc[train_perm[:100]]
    dev_data = dev_data.iloc[valid_perm[:10]]
    predictor = task.fit(train_data, hyperparameters=test_hyperparameters,
                         label='label', num_trials=1,
                         ngpus_per_trial=0,
                         verbosity=4,
                         output_directory='./sst',
                         plot_results=False)
    dev_acc = predictor.evaluate(dev_data, metrics=['acc'])
    dev_prediction = predictor.predict(dev_data)
    dev_pred_prob = predictor.predict_proba(dev_data)


def test_mrpc():
    train_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/train.parquet')
    dev_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/dev.parquet')
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    valid_perm = rng_state.permutation(len(dev_data))
    train_data = train_data.iloc[train_perm[:100]]
    dev_data = dev_data.iloc[valid_perm[:10]]
    predictor = task.fit(train_data, hyperparameters=test_hyperparameters,
                         label='label', num_trials=1,
                         verbosity=4,
                         ngpus_per_trial=1,
                         output_directory='./mrpc',
                         plot_results=False)
    dev_acc = predictor.evaluate(dev_data, metrics=['acc'])
    dev_prediction = predictor.predict(dev_data)
    dev_pred_prob = predictor.predict_proba(dev_data)


def test_sts():
    train_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')
    dev_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet')
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    valid_perm = rng_state.permutation(len(dev_data))
    train_data = train_data.iloc[train_perm[:100]]
    dev_data = dev_data.iloc[valid_perm[:10]]
    predictor = task.fit(train_data, hyperparameters=test_hyperparameters,
                         label='score', num_trials=1,
                         verbosity=4,
                         ngpus_per_trial=1,
                         output_directory='./sts',
                         plot_results=False)
    dev_rmse = predictor.evaluate(dev_data, metrics=['rmse'])
    dev_prediction = predictor.predict(dev_data)


def test_no_text_column_raise():
    data = [('😁😁😁😁😁😁', 'grin')] * 20 + [('😃😃😃😃😃😃😃😃', 'smile')] * 50 + [
        ('😉😉😉', 'wink')] * 30

    df = pd.DataFrame(data, columns=['data', 'label'])
    with pytest.raises(NotImplementedError):
        predictor = task.fit(df, label='label',
                             verbosity=4)


def test_emoji():
    data = []
    for i in range(50):
        data.append(('😁' * (i + 1), 'grin'))

    for i in range(30):
        data.append(('😃' * (i + 1), 'smile'))

    for i in range(20):
        data.append(('😉' * (i + 1), 'wink'))
    df = pd.DataFrame(data, columns=['data', 'label'])

    predictor = task.fit(df, label='label',
                         verbosity=3)


def test_no_job_finished_raise():
    train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                              'glue/sst/train.parquet')
    dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                            'glue/sst/dev.parquet')
    with pytest.raises(RuntimeError):
        # Setting a very small time limits to trigger the bug
        predictor = task.fit(train_data, hyperparameters=test_hyperparameters,
                             label='label', num_trials=1,
                             ngpus_per_trial=0,
                             verbosity=4,
                             time_limits=10,
                             output_directory='./sst_raise',
                             plot_results=False)


def test_mixed_column_type():
    train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                              'glue/sts/train.parquet')
    dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                            'glue/sts/dev.parquet')
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    valid_perm = rng_state.permutation(len(dev_data))
    train_data = train_data.iloc[train_perm[:100]]
    dev_data = dev_data.iloc[valid_perm[:10]]

    # Add more columns as feature
    train_data = pd.DataFrame({'sentence1': train_data['sentence1'],
                               'sentence2': train_data['sentence2'],
                               'sentence3': train_data['sentence2'],
                               'categorical0': train_data['genre'],
                               'numerical0': train_data['score'],
                               'genre': train_data['genre'],
                               'score': train_data['score']})
    dev_data = pd.DataFrame({'sentence1': dev_data['sentence1'],
                             'sentence2': dev_data['sentence2'],
                             'sentence3': dev_data['sentence2'],
                             'categorical0': dev_data['genre'],
                             'numerical0': dev_data['score'],
                             'genre': dev_data['genre'],
                             'score': dev_data['score']})
    # Train Regression
    predictor1 = task.fit(train_data,
                          hyperparameters=test_hyperparameters,
                          label='score', num_trials=1,
                          verbosity=4,
                          ngpus_per_trial=1,
                          output_directory='./sts_score',
                          plot_results=False)
    dev_rmse = predictor1.evaluate(dev_data, metrics=['rmse'])
    dev_prediction = predictor1.predict(dev_data)

    # Tran Classification
    predictor2 = task.fit(train_data,
                          hyperparameters=test_hyperparameters,
                          label='genre', num_trials=1,
                          verbosity=4,
                          ngpus_per_trial=1,
                          output_directory='./sts_genre',
                          plot_results=False)
    dev_rmse = predictor2.evaluate(dev_data, metrics=['acc'])
    dev_prediction = predictor2.predict(dev_data)

    # Specify the feature column
    predictor3 = task.fit(train_data,
                          hyperparameters=test_hyperparameters,
                          feature_columns=['sentence1', 'sentence3', 'categorical0'],
                          label='score', num_trials=1,
                          verbosity=4,
                          ngpus_per_trial=1,
                          output_directory='./sts_score',
                          plot_results=False)
    dev_rmse = predictor1.evaluate(dev_data, metrics=['rmse'])
    dev_prediction = predictor1.predict(dev_data)
    model_path = 'saved_model'
    predictor1.save(model_path)
    loaded_predictor = task.load(model_path)
    loaded_predictions = loaded_predictor.predict(dev_data)
    np.testing.assert_array_almost_equal(dev_prediction, loaded_predictions)