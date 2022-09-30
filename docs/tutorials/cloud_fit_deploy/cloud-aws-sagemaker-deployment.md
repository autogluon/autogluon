# Deploying AutoGluon Models with AWS SageMaker
:label:`cloud_aws_sagemaker_deploy`

After learning how to train a model using AWS SageMaker :ref:`cloud_aws_sagemaker_fit`, in this section we will learn how to deploy 
trained models using AWS SageMaker and Deep Learning Containers. 

The full end-to-end example is available in [amazon-sagemaker-examples](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/autogluon-tabular-containers) repository.

## Pre-requisites
Before starting ensure that the latest version of sagemaker python API is installed via (`pip install --upgrade sagemaker`). 
This is required to ensure the information about newly released containers is available.

## Endpoint Deployment - Inference Script

To start using the containers, an inference script and the [wrapper classes](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/autogluon-tabular-containers/ag_model.py) are required.
When authoring an inference [scripts](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/autogluon-tabular-containers/scripts/), 
please refer to SageMaker [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html).

Here is one of the possible inference scripts. 

- the `model_fn` function is responsible for loading your model. It takes a `model_dir` argument that specifies where the model is stored. 

- the `transform_fn` function is responsible for deserializing your input data so that it can be passed to your model. It takes input data and 
content type as parameters, and returns deserialized data. The SageMaker inference toolkit provides a default implementation that deserializes 
the following content types: JSON, CSV, numpy array, NPZ.

```{.python}
from autogluon.tabular import TabularPredictor
# or from autogluon.multimodal import MultiModalPredictor for example
import os
import json
from io import StringIO
import pandas as pd
import numpy as np


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir)  # or model = MultiModalPredictor.load(model_dir) for example 
    model.persist_models()  # This line only works for TabularPredictor
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):

    if input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf, header=None)
        num_cols = len(data.columns)
        
    else:
        raise Exception(f"{input_content_type} content type not supported")

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1)

    return prediction.to_json(), output_content_type
```
For inference with other types of AutoGluon Predictors, i.e. TextPredictor, the inference script you provided will be quite similar to the one above.
Mostly, you just need to replace `TabularPredictor` to be `TextPredictor` for example.

### Note on image modality
To do inference on image modality, you would need to embed the image info, as bytes for example, into a column of the test data.
Then in the inference container, you would need to store the image into the disk and update the test data with the image paths accordingly.

For example, to encode the image:
```{.python}
def read_image_bytes_and_encode(image_path):
    image_obj = open(image_path, 'rb')
    image_bytes = image_obj.read()
    image_obj.close()
    b85_image = base64.b85encode(image_bytes).decode("utf-8")

    return b85_image


def convert_image_path_to_encoded_bytes_in_dataframe(dataframe, image_column):
    assert image_column in dataframe, 'Please specify a valid image column name'
    dataframe[image_column] = [read_image_bytes_and_encode(path) for path in dataframe[image_column]]

    return dataframe

test_data_image_column = "YOUR_COLUMN_CONTAINING_IMAGE_PATH"
test_data = convert_image_path_to_encoded_bytes_in_dataframe(test_data, test_data_image_column)
```

For example, to decode the image and save to disk in the inference container:
```{.python}
image_index = 0


def _save_image_and_update_dataframe_column(bytes):
    global image_index
    im = Image.open(BytesIO(base64.b85decode(bytes)))
    im_name = f'Image_{image_index}.png'
    im.save(im_name)
    image_index += 1

    return im_name


test_data[image_column] = [_save_image_and_update_dataframe_column(bytes) for bytes in test_data[image_column]]
```

## Deployment as an inference endpoint

To deploy AutoGluon model as a SageMaker inference endpoint, we configure SageMaker session first:

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

Upload the model archive trained earlier (if you trained AutoGluon model locally, it must be a zip archive of the model output directory).
Remember, you would need to save the model artifacts with `standalone=True` if the model artifact is of type `TextPredictor`/`MultiModalPredictor`.
Otherwise, you will have trouble loading the model in the container.

```{.python}
endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-autogluon-serving-trained-model")

model_data = sagemaker_session.upload_data(
    path=os.path.join(".", "model.tar.gz"), key_prefix=f"{endpoint_name}/models"
)
```

Deploy the model:

```{.python}
instance_type = "ml.m5.2xlarge"  # You might want to use GPU instances, i.e. ml.g4dn.2xlarge for Text/Image/MultiModal Predictors etc

model = AutoGluonInferenceModel(
    model_data=model_data,
    role=role,
    region=region,
    framework_version="0.5.2",  # Replace this with the AutoGluon DLC container version you want to use
    py_version="py38",
    instance_type=instance_type,
    entry_point="YOUR_SERVING_SCRIPT_PATH",  # example: "tabular_serve.py"
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

Upload the model archive trained earlier (if you trained AutoGluon model locally, it must be a zip archive of the model output directory).
Remember, you would need to save the model artifacts with `standalone=True` if the model artifact is of type `TextPredictor`/`MultiModalPredictor`.
Otherwise, you will have trouble loading the model in the container.

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
instance_type = "ml.m5.2xlarge"  # You might want to use GPU instances, i.e. ml.g4dn.2xlarge for Text/Image/MultiModal Predictors etc

model = AutoGluonInferenceModel(
    model_data=model_data,
    role=role,
    region=region,
    framework_version="0.5.2",  # Replace this with the AutoGluon DLC container version you want to use
    py_version="py38",
    instance_type=instance_type,
    entry_point="YOUR_BATCH_SERVE_SCRIPT",  # example: "tabular_serve.py"
    predictor_cls=AutoGluonTabularPredictor,
    # or AutoGluonMultiModalPredictor if model is trained by MultiModalPredictor.
    # Please refer to https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/autogluon-tabular-containers/ag_model.py#L60-L64 on how you would customize it.
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

Upload the test data.

```{.python}
test_input = transformer.sagemaker_session.upload_data(
    path=os.path.join("data", "test.csv"), key_prefix=s3_prefix
)
```

The inference script would be identical to the one used for deployment:

```{.python}
from autogluon.tabular import TabularPredictor
# or from autogluon.multimodal import MultiModalPredictor for example
import os
import json
from io import StringIO
import pandas as pd
import numpy as np


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir)  # or model = MultiModalPredictor.load(model_dir) for example 
    model.persist_models()  # This line only works for TabularPredictor
    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):

    if input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf, header=None)
        num_cols = len(data.columns)
        
    else:
        raise Exception(f"{input_content_type} content type not supported")

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1)

    return prediction.to_json(), output_content_type
```

Run the transform job.

When making predictions on a large dataset, you can exclude attributes that aren't needed for prediction. After the predictions have been made, you can 
associate some of the excluded attributes with those predictions or with other input data in your report. By using batch transform to perform these data 
processing steps, you can often eliminate additional preprocessing or postprocessing. You can use input files in JSON and CSV format only. 
More details on how to use filters are available here: [Associate Prediction Results with Input Records](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html).
In this specific case we will use `input_filter` argument to get first 14 columns, thus removing target variable from the test set and `output_filter` to
extract only the actual classes predictions without scores.

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

## Conclusion

In this tutorial we explored a few options how to deploy AutoGluon models using SageMaker. To explore more, refer to 
[SageMaker inference](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html) documentation.
