# Cloud Training and Prediction with CloudPredictor
:label:`cloud_predictor`

In the previous tutorials, you've seen how to train AutoGluon models using :ref:`cloud_aws_sagemaker_fit` and deploy the trained models by :ref:`cloud_aws_sagemaker_deploy` manually. In this section, we will learn how to use the CloudPredictor, which wraps the similar functionality for you.

## Pre-requisites
Before starting ensure that the latest version of sagemaker python API is installed via (`pip install --upgrade sagemaker`). 
This is required to ensure the information about newly released containers is available.

Also, you need to have an AWS account with the credentials being setup. For more information, please refer to [How do I create and activate a new AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/) and [Set up AWS Credentials and Region for Development](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html).

## Training with CloudPredictor
To train AutoGluon models with CloudPredictor, we specify the init and fit arguments first. These arguments are the same arguments that you would use for a Tabular/Text Predictor. For a quick start: checkout :ref:`sec_tabularquick` and :ref:`sec_textprediction_beginner`
For example, for a TabularCloudPredictor:
```{.python}
time_limit = 60
predictor_init_args = dict(
    label='class',
    eval_metric='roc_auc'
)
predictor_fit_args = dict(
    train_data='train.csv',  # train data can be a file in local path, a file in s3, or a DataFrame in-memory
    time_limit=time_limit,
    presets='best_quality'
)
```
Then we init the TabularCloudPredictor and call `fit` on it:
```{.python}
from autogluon.cloud import TabularCloudPredictor
tabular_cloud_predictor = TabularCloudPredictor().fit(
    predictor_init_args,
    predictor_fit_args,
    framework_version='latest',
    #  ml.m5.large is an instance with 2 vCPU and 8 GiB of memory. It will cost $0.115/hour for training.
    #  For a list of all instances' specs and pricing, check out: https://aws.amazon.com/sagemaker/pricing/
    instance_type='ml.m5.large',
)
```
TextCloudPredictor works just like the TabularCloudPredictor. We will focus on the TabularCloudPredictor for simplicity in this tutorial.

### Reattach to an existing training job
It is possible to lose access to the local running script, which is training the cloud predictor. The training job will continue if it is already started in SageMaker. To recover the training job:
```{.python}
cloud_predictor = TabularCloudPredictor().attach_job(JobName)  # you can get the job name from previous run's log or from SageMaker console
```
To be notice, reattaching to a job is a blocking call and won't give live log stream. Logs will be printed out once the training job is finished.

## Predicting with CloudPredictor
We can use the trained CloudPredictor to predict on new data with either real-time inference or batch transformation.

### Predicting with Real-time Inference
Real-time inference is ideal for inference workloads where you have real-time, interactive, low latency requirements. To be noticed, real-time inference would require us to deploy an endpoint first. An Amazon SageMaker endpoint is a fully managed service that allows you to make real-time inferences via API calls.

Deploy an endpoint with the previously trained CloudPredictor is simple:
```{.python}
tabular_cloud_predictor.deploy(
    instance_type='ml.m5.large',
    framework_version='latest',
    )
```

Once the model is deployed, we can call `predict_real_time()` on it:
```{.python}
test_data = 'test.csv'
tabular_cloud_predictor.predict_real_time(test_data)
import pandas as pd
tabular_cloud_predictor.predict_real_time(pd.read_csv(test_data))  # passing a dataframe
```

Remember to shut down the deployed endpoint:
```{.python}
tabular_cloud_predictor.cleanup_deployment()
```

### Predicting with Batch Transformation
If the goal is to generate predictions from a trained model on a large dataset where where maximizing bulk throughput of predictions is important, then the batch transform functionality may be easier, more scalable, and more appropriate.

Batch transformation with CloudPredictor is simple too:
```{.python}
tabular_cloud_predictor.predict('test.csv')
# The results will be stored in s3, CloudPredictor can download the result for you
tabular_cloud_predictor.download_predict_results()
```

## To a Local Predictor
CloudPredictor does not require installation of AutoGluon locally. However, if you have it installed, you can convert a CloudPredictor to be a local predictor.

**Important: make sure the local AutoGluon has the same version as the framework version being used during `fit()`**
You can check the framework version used to train the CloudPredictor by
```{.python}
info = tabular_cloud_predictor.info()
print(info['fit_job']['framework_version'])
```
You can check the local AutoGluon version installed by
```{.bash}
pip freeze | grep autogluon
```
Once you've made sure the local AutoGluon version matches the CloudPredictor `fit()` framework version. Get a local predictor is easy:
```{.python}
local_predictor = tabular_cloud_predictor.to_local_predictor()
local_predictor.leaderboard()
```

## Save/Load CloudPredictor
Of course you can save the CloudPredictor and reuse it later on.
```{.python}
cloud_predictor.save()
loaded_predictor = TabularCloudPredictor.load(SavedPredictorPath)  # You can get the SavedPredictorPath from the console output of the `save()` method
loaded_predictor.info()
```