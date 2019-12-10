# Documents for Kaggle Benchmark

## How to build

1. `pip uninstall autogluon`
2. `python setup.py develop`

## Prepare the dataset

1. Assume your instance can visit S3.

2. `aws s3 cp s3://zhiz-data/datasets/a.zip /your_path/`

3. `unzip a.zip -d dataset`

## Train the dataset with fit function:

1. ```dataset = 'dogs-vs-cats-redux-kernels-edition/'```

2. `python autogluon/examples/image_classification/benchmark.py`

3. `kaggle competitions submit -c dogs-vs-cats-redux-kernels-edition -f submission.csv -m "Message"`





