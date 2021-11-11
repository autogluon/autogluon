# Deploying AutoGluon Models using AWS SageMaker
:label:`cloud_aws_sagemaker_deploy`

After learning how to train a model using AWS SageMaker :ref:`cloud_aws_sagemaker_fit`, in this section we will learn how to deploy 
trained models using AWS SageMaker and Deep Learning Containers. 

The full end-to-end example is available in [amazon-sagemaker-examples](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/autogluon-tabular-containers) repository.

## Pre-requisites
Before starting ensure that the latest version of sagemaker python API is installed via (`pip install --upgrade sagemaker`). 
This is required to ensure the information about newly released containers is available.

## Endpoint Deployment - Inference Script

To start using the containers, a user training script and the [wrapper classes](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/autogluon-tabular-containers/ag_model.py) are required.
When authoring an inference [scripts](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/autogluon-tabular-containers/scripts/), 
please refer to SageMaker [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html).

Here is one of the possible inference scripts. The `model_fn` function is responsible for loading your model. It takes a `model_dir` 
argument that specifies where the model is stored. The `input_fn` function is responsible for deserializing your input data so that 
it can be passed to your model. It takes input data and content type as parameters, and returns deserialized data. 
The SageMaker inference toolkit provides a default implementation that deserializes the following content types: JSON, CSV Numpy array, NPZ.

```{.python}
from autogluon.tabular import TabularPredictor
import os
import json
from io import StringIO
import pandas as pd
import numpy as np


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir)
    globals()["column_names"] = model.feature_metadata_in.get_features()
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):

    if input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf, header=None)
        num_cols = len(data.columns)

        if num_cols != len(column_names):
            raise Exception(
                f"Invalid data format. Input data has {num_cols} while the model expects {len(column_names)}"
            )

        else:
            data.columns = column_names

    else:
        raise Exception(f"{input_content_type} content type not supported")

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1).values

    return json.dumps(prediction.tolist()), output_content_type
```

## Deployment as an inference endpoint

To deploy AutoGluon model as a SageMaker inference endpoint:

```{.python}
import sagemaker

# Helper wrappers referred earlier
from ag_model import (
    AutoGluonTraining,
    AutoGluonInferenceModel,
    AutoGluonTabularPredictor,
)
from sagemaker import utils

role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.session.Session()
region = sagemaker_session._region_name

bucket = sagemaker_session.default_bucket()
s3_prefix = f"autogluon_sm/{utils.sagemaker_timestamp()}"
output_path = f"s3://{bucket}/{s3_prefix}/output/"
```

Upload the model archive trained earlier (if you trained AutoGluon model locally, then zip archive of the model output directory should be used):

```{.python}
endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-autogluon-serving-trained-model")

model_data = sagemaker_session.upload_data(
    path=os.path.join(".", "model.tar.gz"), key_prefix=f"{endpoint_name}/models"
)
```

Deploy:

```{.python}
instance_type = "ml.m5.2xlarge"

model = AutoGluonInferenceModel(
    model_data=model_data,
    role=role,
    region=region,
    framework_version="0.3.1",
    instance_type=instance_type,
    source_dir="scripts",
    entry_point="tabular_serve.py",
)

predictor = model.deploy(
    initial_instance_count=1, serializer=CSVSerializer(), instance_type=instance_type
)
```

Once the predictor is deployed, it can be used for inference in the following way:

```{.python}
predictions = predictor.predict(data)
```

## Using SageMaker batch transform for offline processing

Deploying a trained model to a hosted endpoint has been available in SageMaker since launch and is a great way to provide real-time 
predictions to a service like a website or mobile app. But, if the goal is to generate predictions from a trained model on a large 
dataset where minimizing latency isn’t a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.

[Read more about Batch Transform.](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)

Upload the model archive trained earlier (if you trained AutoGluon model locally, then zip archive of the model output directory should be used):

```{.python}
endpoint_name = sagemaker.utils.unique_name_from_base(
    "sagemaker-autogluon-batch_transform-trained-model"
)

model_data = sagemaker_session.upload_data(
    path=os.path.join(".", "model.tar.gz"), key_prefix=f"{endpoint_name}/models"
)
```

Prepare transform job:

```{.python}
instance_type = "ml.m5.2xlarge"

model = AutoGluonInferenceModel(
    model_data=model_data,
    role=role,
    region=region,
    framework_version="0.3.1",
    instance_type=instance_type,
    entry_point="tabular_serve-batch.py",
    source_dir="scripts",
    predictor_cls=AutoGluonTabularPredictor,
)

transformer = model.transformer(
    instance_count=1,
    instance_type=instance_type,
    strategy="MultiRecord",
    max_payload=6,
    max_concurrent_transforms=1,
    output_path=output_path,
    accept="application/json",
    assemble_with="Line",
)
```

Batch transform accepts CSV without header and index column - we need to remove them before sending to the transform job.

```{.python}
output_file_name = "test_no_header.csv"

pd.read_csv(f"data/test.csv")[:100].to_csv(f"data/{output_file_name}", header=False, index=False)

test_input = transformer.sagemaker_session.upload_data(
    path=os.path.join("data", "test_no_header.csv"), key_prefix=s3_prefix
)
```

Run transform job.

When making predictions on a large dataset, you can exclude attributes that aren't needed for prediction. After the predictions have been made, you can 
associate some of the excluded attributes with those predictions or with other input data in your report. By using batch transform to perform these data 
processing steps, you can often eliminate additional preprocessing or postprocessing. You can use input files in JSON and CSV format only. 
More details on how to use filters are available here: [Associate Prediction Results with Input Records](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html)

```{.python}
transformer.transform(
    test_input,
    input_filter="$[:14]",  # filter-out target variable
    split_type="Line",
    content_type="text/csv",
    output_filter="$['class']",  # keep only prediction class in the output
)

transformer.wait()

output_s3_location = f"{transformer.output_path[:-1]}/{output_file_name}"
```

The output file will be in `output_s3_location` variable.

In this tutorial we explored a few options how to deploy AutoGluon models using SageMaker. To explore more, refer to 
[SageMaker inference](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html) documentation.
