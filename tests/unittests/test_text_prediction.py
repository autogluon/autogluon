from autogluon import TextPrediction as task
from autogluon.utils.tabular.utils.loaders import load_pd


def test_sst():
    train_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                              'glue/sst/train.parquet')
    dev_data = load_pd.load('https://autogluon-text.s3-accelerate.amazonaws.com/'
                            'glue/sst/dev.parquet')
    train_data = train_data.iloc[:1000]
    dev_data = dev_data.iloc[:10]
    predictor = task.fit(train_data, label='label', num_trials=1)
    dev_acc = predictor.evaluate(dev_data, metrics=['acc'])
    dev_prediction = predictor.predict(dev_data)
    dev_pred_prob = predictor.predict_proba(dev_data)


def test_mrpc():
    train_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/train.parquet')
    dev_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/mrpc/dev.parquet')
    train_data = train_data.iloc[:1000]
    dev_data = dev_data.iloc[:10]
    predictor = task.fit(train_data, label='label', num_trials=1)
    dev_acc = predictor.evaluate(dev_data, metrics=['acc'])
    dev_prediction = predictor.predict(dev_data)
    dev_pred_prob = predictor.predict_proba(dev_data)


def test_sts():
    train_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/train.parquet')
    dev_data = load_pd.load(
        'https://autogluon-text.s3-accelerate.amazonaws.com/glue/sts/dev.parquet')
    train_data = train_data.iloc[:1000]
    dev_data = dev_data.iloc[:10]
    predictor = task.fit(train_data, label='score', num_trials=1)
    dev_rmse = predictor.evaluate(dev_data, metrics=['rmse'])
    dev_prediction = predictor.predict(dev_data)
