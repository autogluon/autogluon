# Documents for Kaggle Benchmark

## Prepare the dataset by shell


1. Assume your instance can visit S3.

2. `aws s3 cp s3://zhiz-data/datasets/a.zip /your_path/`

3. `unzip a.zip -d dataset`


## Train the dataset with fit function:

`python autogluon/examples/image_classification/benchmark.py --data-dir /your_path/ --dataset ()`

```()
dataset = 'dogs-vs-cats-redux-kernels-edition'
dataset = 'aerial-cactus-identification'
dataset = 'plant-seedlings-classification'
dataset = 'fisheries_Monitoring'
dataset = 'dog-breed-identification'
dataset = 'shopee-iet-machine-learning-competition'
```

3. `kaggle competitions submit -c dogs-vs-cats-redux-kernels-edition -f submission.csv -m "Message"`





