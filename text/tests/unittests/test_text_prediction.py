import os
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import tempfile
from autogluon.core.utils.loaders import load_pd
from autogluon.text import TextPredictor


test_hyperparameters = {
    'models': {
        'BertForTextPredictionBasic': {
            'search_space': {
                'optimization.num_train_epochs': 1
            }
        }
    }
}


def verify_predictor_save_load(predictor, df, verify_proba=False,
                               verify_embedding=True):
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictions = predictor.predict(df)
        loaded_predictor = task.load(root)
        predictions2 = loaded_predictor.predict(df)
        npt.assert_equal(predictions, predictions2)
        if verify_proba:
            predictions_prob = predictor.predict_proba(df)
            predictions2_prob = loaded_predictor.predict_proba(df)
            npt.assert_equal(predictions_prob, predictions2_prob)
        if verify_embedding:
            embeddings = predictor.extract_embedding(df)
            assert embeddings.shape[0] == len(df)


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
                         ngpus_per_trial=1,
                         verbosity=4,
                         output_directory='./sst',
                         plot_results=False)
    dev_acc = predictor.evaluate(dev_data, metrics=['acc'])
    verify_predictor_save_load(predictor, dev_data, verify_proba=True)


def test_cpu_only_raise():
    train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                              'glue/sst/train.parquet')
    dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                            'glue/sst/dev.parquet')
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    valid_perm = rng_state.permutation(len(dev_data))
    train_data = train_data.iloc[train_perm[:100]]
    dev_data = dev_data.iloc[valid_perm[:10]]
    with pytest.raises(RuntimeError):
        predictor = task.fit(train_data, hyperparameters=test_hyperparameters,
                             label='label', num_trials=1,
                             ngpus_per_trial=0,
                             verbosity=4,
                             output_directory='./sst',
                             plot_results=False)
    os.environ['AUTOGLUON_TEXT_TRAIN_WITHOUT_GPU'] = '1'
    predictor = task.fit(train_data, hyperparameters=test_hyperparameters,
                         label='label', num_trials=1,
                         ngpus_per_trial=0,
                         verbosity=4,
                         output_directory='./sst',
                         plot_results=False)

    os.environ['AUTOGLUON_TEXT_TRAIN_WITHOUT_GPU'] = '0'
    with pytest.raises(RuntimeError):
        predictor = task.fit(train_data, hyperparameters=test_hyperparameters,
                             label='label', num_trials=1,
                             ngpus_per_trial=0,
                             verbosity=4,
                             output_directory='./sst',
                             plot_results=False)

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
    verify_predictor_save_load(predictor, dev_data, verify_proba=True)


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
    verify_predictor_save_load(predictor, dev_data)


# Test the case that the model should raise because there are no text columns in the model.
def test_no_text_column_raise():
    data = [('😁😁😁😁😁😁', 'grin')] * 20 + [('😃😃😃😃😃😃😃😃', 'smile')] * 50 + [
        ('😉😉😉', 'wink')] * 30

    df = pd.DataFrame(data, columns=['data', 'label'])
    with pytest.raises(AssertionError):
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
    verify_predictor_save_load(predictor, df)


def test_no_job_finished_raise():
    train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                              'glue/sst/train.parquet')
    with pytest.raises(RuntimeError):
        # Setting a very small time limits to trigger the bug
        predictor = task.fit(train_data, hyperparameters=test_hyperparameters,
                             label='label', num_trials=1,
                             ngpus_per_trial=1,
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
    verify_predictor_save_load(predictor1, dev_data)

    # Train Classification
    predictor2 = task.fit(train_data,
                          hyperparameters=test_hyperparameters,
                          label='genre', num_trials=1,
                          verbosity=4,
                          ngpus_per_trial=1,
                          output_directory='./sts_genre',
                          plot_results=False)
    dev_rmse = predictor2.evaluate(dev_data, metrics=['acc'])
    verify_predictor_save_load(predictor2, dev_data, verify_proba=True)

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
    verify_predictor_save_load(predictor3, dev_data)


def test_empty_text_item():
    train_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    train_data = train_data.iloc[train_perm[:100]]
    train_data.iat[0, 0] = None
    train_data.iat[10, 0] = None
    predictor = task.fit(train_data, hyperparameters=test_hyperparameters,
                         label='score', num_trials=1,
                         ngpus_per_trial=1,
                         verbosity=4,
                         output_directory='./sts_empty_text_item',
                         plot_results=False)
