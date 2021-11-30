# Cloud Training and Preidcting with Cloud Predictor
:label:`cloud_predictor`

In the previous tutorials, you've seen how to train AutoGluon Models :ref:`cloud_aws_sagemaker_fit` and deploy the trained models :ref:`cloud_aws_sagemaker_deploy` manually. In this section, we will learn how to use the Cloud Predictor, which wraps the similar functionality for you.

## Pre-requisites
Before starting ensure that the latest version of sagemaker python API is installed via (`pip install --upgrade sagemaker`). 
This is required to ensure the information about newly released containers is available.

Also, you need to have an AWS account with the credentials being setup. For more information, please refer to [Set up AWS Credentials and Region for Development](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html).

## Training with Cloud Predictor
To train AutoGluon models with Cloud Predictor, we specify the init and fit arguments first. These arguments are the same arguments that you would use for a Tabular/Text Predictor.
For example:
```{.python}
time_limit = 60
predictor_init_args = dict(
    label='class',
    eval_metric='roc_auc'
)
predictor_fit_args = dict(
    train_data='train.csv',  # train data can be both a file in local path or s3
    time_limit=time_limit,
    presets='best_quality'
)
```
Then we init the cloud predictor with the type of task we want('tabular' in the following example) and call `fit` on it:
```{.python}
from autogluon.cloud import CloudPredictor
cloud_predictor = CloudPredictor('tabular').fit(
    predictor_init_args,
    predictor_fit_args,
    framework_version='latest',
    instance_type='ml.m5.large',
)
```

## Predicting with Cloud Predictor
We can use the trained cloud predictor to predict on new data with either real-time inference or batch transformation.

### Predicting with Real-time Inference
Real-time inference is ideal for inference workloads where you have real-time, interactive, low latency requirements. To be noticed, real-time inference would require us to deploy an endpoint first.

Deploy an endpoint with the previously trained cloud predictor is simple:
```{.python}
cloud_predictor.deploy(
    instance_type='ml.m5.large',
    framework_version='latest',
    )
```

Once the model is deployed, we can call `predict_real_time()` on it:
```{.python}
test_data = 'test.csv'
cloud_predictor.predict_real_time(test_data)
```

### Predicting with Batch Transformation
If the goal is to generate predictions from a trained model on a large dataset where minimizing latency isn’t a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.

**Important: SageMaker batch transformation does not support passing data with headers. Therefore, we will need to pass the test data without the headers and making sure columns are in the same order as the training data.**

Batch transformation with CloudPredictor is simple too:
```{.python}
cloud_predictor.predict('test_no_header.csv')
# The results will be stored in s3, cloud predictor can download the result for you
cloud_predictor.download_predict_results()
```

## To a Local Predictor
Cloud Predictor does not require installation of AutoGluon locally. However, if you have it installed, you can convert a cloud predictor to be a local preidctor.

**Important: make sure the local AutoGluon has the same version as the framework version being used during `fit()`**
```{.python}
local_predictor = cloud_predictor.to_local_predictor()
local_predictor.leaderboard()
```
