# Cloud Training with AWS SageMaker
:label:`cloud_aws_sagemaker_fit`

To help with AutoGluon models training, AWS developed a set of training and inference [deep learning containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers). 
The containers can be used to train models with CPU and GPU instances and deployed as a SageMaker endpoint or used as a batch transform job.

The full end-to-end example is available in [amazon-sagemaker-examples](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/autogluon-tabular-containers) repository.

## Pre-requisites
Before starting ensure that the latest version of sagemaker python API is installed via (`pip install --upgrade sagemaker`). 
This is required to ensure the information about newly released containers is available.

## Training Scripts

To start using the containers, a user training script and the [wrapper classes](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/autogluon-tabular-containers/ag_model.py) are required.
When authoring a training/inference [scripts](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/autogluon-tabular-containers/scripts/), 
please refer to SageMaker [documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script).

Here is one of the possible `TabularPredictor` training scripts, which takes AutoGluon parameters as a YAML config and outputs predictions, models leaderboard and feature importance:

```{.python}
import argparse
import os
import shutil
from pprint import pprint
import yaml
from autogluon.tabular import TabularDataset, TabularPredictor


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        print(f"WARN: more than one file is found in {channel} directory")
    print(f"Using {file}")
    filename = f"{path}/{file}"
    return filename


def get_env_if_present(name):
    result = None
    if name in os.environ:
        result = os.environ[name]
    return result


if __name__ == "__main__":
    # Disable Autotune
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    # ------------------------------------------------------------ Arguments parsing
    print("Starting AG")
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=get_env_if_present("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model-dir", type=str, default=get_env_if_present("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", type=str, default=get_env_if_present("SM_NUM_GPUS"))
    parser.add_argument("--train_dir", type=str, default=get_env_if_present("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test_dir", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TEST"))
    parser.add_argument("--ag_config", type=str, default=get_env_if_present("SM_CHANNEL_CONFIG"))
    parser.add_argument("--serving_script", type=str, default=get_env_if_present("SM_CHANNEL_SERVING"))

    args, _ = parser.parse_known_args()

    print(f"Args: {args}")

    # See SageMaker-specific environment variables: https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    os.makedirs(args.output_data_dir, mode=0o777, exist_ok=True)

    config_file = get_input_path(args.ag_config)
    with open(config_file) as f:
        config = yaml.safe_load(f)  # AutoGluon-specific config

    if args.n_gpus:
        config["num_gpus"] = int(args.n_gpus)

    print("Running training job with the config:")
    pprint(config)

    # ---------------------------------------------------------------- Training

    train_file = get_input_path(args.train_dir)
    train_data = TabularDataset(train_file)

    save_path = os.path.normpath(args.model_dir)

    ag_predictor_args = config["ag_predictor_args"]
    ag_predictor_args["path"] = save_path
    ag_fit_args = config["ag_fit_args"]

    predictor = TabularPredictor(**ag_predictor_args).fit(train_data, **ag_fit_args)

    # --------------------------------------------------------------- Inference

    if args.test_dir:
        test_file = get_input_path(args.test_dir)
        test_data = TabularDataset(test_file)

        # Predictions
        y_pred_proba = predictor.predict_proba(test_data)
        if config.get("output_prediction_format", "csv") == "parquet":
            y_pred_proba.to_parquet(f"{args.output_data_dir}/predictions.parquet")
        else:
            y_pred_proba.to_csv(f"{args.output_data_dir}/predictions.csv")

        # Leaderboard
        if config.get("leaderboard", False):
            lb = predictor.leaderboard(test_data, silent=False)
            lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

        # Feature importance
        if config.get("feature_importance", False):
            feature_importance = predictor.feature_importance(test_data)
            feature_importance.to_csv(f"{args.output_data_dir}/feature_importance.csv")
    else:
        if config.get("leaderboard", False):
            lb = predictor.leaderboard(silent=False)
            lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

    if args.serving_script:
        print("Saving serving script")
        serving_script_saving_path = os.path.join(save_path, "code")
        os.mkdir(serving_script_saving_path)
        serving_script_path = get_input_path(args.serving_script)
        shutil.move(
            serving_script_path,
            os.path.join(
                serving_script_saving_path, os.path.basename(serving_script_path)
            ),
        )
```
For training other types of AutoGluon Predictors, i.e. MultiModalPredictor, the training script you provided will be quite similar to the one above.
Mostly, you just need to replace `TabularPredictor` to be `MultiModalPredictor` for example.
Keep in mind that the specific Predictor type you want to train might not support the same feature sets as `TabularPredictor`.
For example, `leaderboard` does not exist for all Predictors.


### Notes for Training
1. If your use case involves image modality, you will need to pass the images as a compressed file to the training container (similarly to how you would pass in train data), decompress the file in the training container, and update the training data columns with the updated image path in the container.

2. If you wish to deploy or do batch inference on the trained TextPredictor/MultiModalPredictor on sagemaker later, you will need to save the model with `standalone` flag, which avoids internet access to load the model later.
For example, `predictor.save(path='MY_PATH', standalone=True)`.
SageMaker container is known to have issue connecting to HuggingFace. That's why we need to save the artifacts in offline mode.

Tabular example YAML config:

```yaml
# AutoGluon Predictor constructor arguments
# - see https://github.com/autogluon/autogluon/blob/v0.5.2/tabular/src/autogluon/tabular/predictor/predictor.py#L56-L181
ag_predictor_args:
  eval_metric: roc_auc
  label: class

# AutoGluon Predictor.fit arguments
# - see https://github.com/autogluon/autogluon/blob/v0.5.2/tabular/src/autogluon/tabular/predictor/predictor.py#L286-L711
ag_fit_args:
  presets: "medium_quality_faster_train"
  num_bag_folds: 2
  num_bag_sets: 1
  num_stack_levels: 0

output_prediction_format: csv  # predictions output format: csv or parquet
feature_importance: true       # calculate and save feature importance if true
leaderboard: true              # save leaderboard output if true
```

Another example, MultiModal example YAML config:

```yaml
# AutoGluon Predictor constructor arguments
# - see https://github.com/autogluon/autogluon/blob/v0.5.2/multimodal/src/autogluon/multimodal/predictor.py#L123-L180
ag_predictor_args:
  eval_metric: acc
  label: label

# AutoGluon Predictor.fit arguments
# - see https://github.com/autogluon/autogluon/blob/v0.5.2/multimodal/src/autogluon/multimodal/predictor.py#L246-L363
ag_fit_args:
  presets: "high_quality"
  time_limit: 120

output_prediction_format: csv  # predictions output format: csv or parquet
```

Other predictors would follow similar format as the previous two examples.

## Training

Note the `ag_model` imports are sourced from [this helper package](https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/autogluon-tabular-containers/ag_model.py).

To train AutoGluon model, set up a SageMaker session:

```{.python}
import sagemaker
import pandas as pd

# Helper wrappers referred earlier
from ag_model import (
    AutoGluonSagemakerEstimator,
    AutoGluonNonRepackInferenceModel,
    AutoGluonSagemakerInferenceModel,
    AutoGluonRealtimePredictor,
    AutoGluonBatchPredictor,
)
from sagemaker import utils
from sagemaker.serializers import CSVSerializer
import os
import boto3

role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.session.Session()
region = sagemaker_session._region_name

bucket = sagemaker_session.default_bucket()
s3_prefix = f"autogluon_sm/{utils.sagemaker_timestamp()}"
output_path = f"s3://{bucket}/{s3_prefix}/output/"
```

Create a training task:

```{.python}
ag = AutoGluonSagemakerEstimator(
    role=role,
    entry_point="YOUR_TRAINING_SCRIPT_PATH",
    region=region,
    instance_count=1,
    instance_type="ml.m5.2xlarge",  # You might want to use GPU instances for Text/Image/MultiModal Predictors etc
    framework_version="0.6",  # Replace this with the AutoGLuon DLC container version you want to use
    py_version="py38",
    base_job_name="YOUR_JOB_NAME",
    # Disable torch profiler instrumentation to avoid deserialization issues during deployment
    disable_profiler=True,
    debugger_hook_config=False,
)
```

Upload the required inputs, via SageMaker session (in this case it is a training set, test set and training YAML config) and start the training job.
Please read more on "Why do I see a repack step in my SageMaker pipeline?" [here](https://docs.aws.amazon.com/sagemaker/latest/dg/mlopsfaq.html).
Example of inference script can be found here: [tabular_serve.py](https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/autogluon-tabular-containers/scripts/tabular_serve.py)

```{.python}
s3_prefix = f"autogluon_sm/{utils.sagemaker_timestamp()}"
train_input = ag.sagemaker_session.upload_data(
    path=os.path.join("data", "train.csv"), key_prefix=s3_prefix
)
eval_input = ag.sagemaker_session.upload_data(
    path=os.path.join("data", "test.csv"), key_prefix=s3_prefix
)
config_input = ag.sagemaker_session.upload_data(
    path=os.path.join("config", "config-med.yaml"), key_prefix=s3_prefix
)
inference_script = ag.sagemaker_session.upload_data(
    path=os.path.join("scripts", "INFERENCE_SCRIPT_LOCATION"), key_prefix=s3_prefix
)

job_name = utils.unique_name_from_base("test-autogluon-image")
ag.fit(
    {
        "config": config_input,
        "train": train_input,
        "test": eval_input,
        "serving": inference_script,
    },
    job_name=job_name,
)
```

Once the models are trained, they will be available in S3 location specified in `ag.model_data` field. The model is fully portable and can be downloaded locally
if needed.

## Conclusion

In this tutorial we explored how to train AutoGluon models using SageMaker. Learn how to deploy the trained models using 
AWS SageMaker - :ref:`cloud_aws_sagemaker_deploy` or AWS Lambda - :ref:`cloud_aws_lambda_deploy`.
