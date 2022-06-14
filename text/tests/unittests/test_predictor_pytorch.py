import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import tempfile

from autogluon.core.utils.loaders import load_pd
from autogluon.text import TextPredictor

DATA_INFO = {
    'sst': {
        'train': 'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet',
        'dev': 'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet',
        'label': 'label',
        'metric': 'average_precision',
        'verify_proba': True,
    },
    'mrpc': {
        'train': 'https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/train.parquet',
        'dev': 'https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/dev.parquet',
        'label': 'label',
        'metric': 'acc',
        'verify_proba': True,
    },
    'sts': {
        'train': 'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet',
        'dev': 'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet',
        'label': 'score',
        'metric': 'rmse',
        'verify_proba': False,
    }
}

# Don't set hyperparameters as a global variable. Otherwise, one hyperparameters variable will be used
# across different pytest jobs. If hyperparameters is modified in one job, it will also be applied to
# later jobs, causing unexpected behaviors.


def get_test_hyperparameters():
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
    }
    return hyperparameters


def verify_predictor_save_load(predictor, df, verify_proba=False,
                               verify_embedding=True):
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictions = predictor.predict(df, as_pandas=False)
        loaded_predictor = TextPredictor.load(root)
        predictions2 = loaded_predictor.predict(df, as_pandas=False)
        predictions2_df = loaded_predictor.predict(df, as_pandas=True)
        npt.assert_equal(predictions, predictions2)
        npt.assert_equal(predictions2,
                         predictions2_df.to_numpy())
        if verify_proba:
            predictions_prob = predictor.predict_proba(df, as_pandas=False)
            predictions2_prob = loaded_predictor.predict_proba(df, as_pandas=False)
            predictions2_prob_df = loaded_predictor.predict_proba(df, as_pandas=True)
            npt.assert_equal(predictions_prob, predictions2_prob)
            npt.assert_equal(predictions2_prob, predictions2_prob_df.to_numpy())
        if verify_embedding:
            embeddings = predictor.extract_embedding(df)
            assert embeddings.shape[0] == len(df)


def test_mixed_column_type():
    train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                              'glue/sts/train.parquet')
    dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                            'glue/sts/dev.parquet')
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    valid_perm = rng_state.permutation(len(dev_data))
    train_data = train_data.iloc[train_perm[:1000]]
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
    predictor = TextPredictor(label='score', verbosity=4)
    predictor.fit(train_data,
                   hyperparameters=get_test_hyperparameters(),
                   time_limit=30,
                   seed=123)

    dev_rmse = predictor.evaluate(dev_data, metrics=['rmse'])
    verify_predictor_save_load(predictor, dev_data)

    # Train Classification
    predictor = TextPredictor(label='genre', verbosity=4)
    predictor.fit(train_data,
                   hyperparameters=get_test_hyperparameters(),
                   time_limit=30,
                   seed=123)

    dev_rmse = predictor.evaluate(dev_data, metrics=['acc'])
    verify_predictor_save_load(predictor, dev_data, verify_proba=True)

    # Specify the feature column
    predictor = TextPredictor(label='score', verbosity=4)
    predictor.fit(train_data[['sentence1', 'sentence3', 'categorical0', 'score']],
                   hyperparameters=get_test_hyperparameters(),
                   time_limit=30,
                   seed=123)
    dev_rmse = predictor.evaluate(dev_data, metrics=['rmse'])
    verify_predictor_save_load(predictor, dev_data)


@pytest.mark.parametrize('key', DATA_INFO.keys())
def test_predictor_fit(key):
    train_data = load_pd.load(DATA_INFO[key]['train'])
    dev_data = load_pd.load(DATA_INFO[key]['dev'])
    label = DATA_INFO[key]['label']
    eval_metric = DATA_INFO[key]['metric']
    verify_proba = DATA_INFO[key]['verify_proba']

    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    valid_perm = rng_state.permutation(len(dev_data))
    train_data = train_data.iloc[train_perm[:100]]
    dev_data = dev_data.iloc[valid_perm[:10]]
    predictor = TextPredictor(label=label, eval_metric=eval_metric)
    predictor.fit(train_data, hyperparameters=['optimization.max_epochs=1',
                                               'model.hf_text.checkpoint_name="google/electra-small-discriminator"'],
                  time_limit=30, seed=123)
    dev_score = predictor.evaluate(dev_data)
    verify_predictor_save_load(predictor, dev_data, verify_proba=verify_proba)

    # Test for continuous fit
    predictor.fit(train_data,
                  hyperparameters='optimization.max_epochs=1 '
                                  'model.hf_text.checkpoint_name="google/electra-small-discriminator"',
                  time_limit=30, seed=123)
    verify_predictor_save_load(predictor, dev_data, verify_proba=verify_proba)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictor = TextPredictor.load(root)
        predictor.fit(train_data, hyperparameters=get_test_hyperparameters(),
                      time_limit=30, seed=123)


def test_cpu_only_warning():
    train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                              'glue/sst/train.parquet')
    dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                            'glue/sst/dev.parquet')
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    valid_perm = rng_state.permutation(len(dev_data))
    train_data = train_data.iloc[train_perm[:100]]
    dev_data = dev_data.iloc[valid_perm[:10]]
    predictor = TextPredictor(label='label', eval_metric='acc')
    with pytest.warns(UserWarning):
        predictor.fit(train_data, hyperparameters=get_test_hyperparameters(),
                      num_gpus=0, seed=123)


def test_emoji():
    data = []
    for i in range(50 * 3):
        data.append(('😁' * (i + 1), 'grin'))

    for i in range(30 * 3):
        data.append(('😃' * (i + 1), 'smile'))

    for i in range(20 * 3):
        data.append(('😉' * (i + 1), 'wink'))
    df = pd.DataFrame(data, columns=['data', 'label'])
    predictor = TextPredictor(label='label', verbosity=3)
    predictor.fit(df,
                  hyperparameters=get_test_hyperparameters(),
                  time_limit=30,
                  seed=123)
    assert set(predictor.class_labels) == {'grin', 'smile', 'wink'}
    assert predictor.class_labels_internal == [0, 1, 2]
    verify_predictor_save_load(predictor, df)
    pred_labels = predictor.predict(df)
    for ele in pred_labels:
        assert ele in {'grin', 'smile', 'wink'}


def test_empty_text_item():
    train_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')
    rng_state = np.random.RandomState(123)
    train_perm = rng_state.permutation(len(train_data))
    train_data = train_data.iloc[train_perm[:100]]
    train_data.iat[0, 0] = None
    train_data.iat[10, 0] = None
    predictor = TextPredictor(label='score', verbosity=4)
    predictor.fit(train_data, hyperparameters=get_test_hyperparameters(), time_limit=30)


def test_standalone_with_emoji():
    import tempfile
    from unittest import mock

    requests_gag = mock.patch(
        'requests.Session.request',
        mock.Mock(side_effect=RuntimeError(
            'Please use the `responses` library to mock HTTP in your tests.'
        ))
    )

    data = []
    for i in range(50 * 3):
        data.append(('😁' * (i + 1), 'grin'))

    for i in range(30 * 3):
        data.append(('😃' * (i + 1), 'smile'))

    for i in range(20 * 3):
        data.append(('😉' * (i + 1), 'wink'))
    df = pd.DataFrame(data, columns=['data', 'label'])
    predictor = TextPredictor(label='label', verbosity=3)
    predictor.fit(
        df,
        hyperparameters=get_test_hyperparameters(),
        time_limit=5,
        seed=123,
    )

    predictions1 = predictor.predict(df, as_pandas=False)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root, standalone=True)
        with requests_gag:  # no internet connections
            offline_predictor = TextPredictor.load(root)
            predictions2 = offline_predictor.predict(df, as_pandas=False)

    npt.assert_equal(predictions1, predictions2)
