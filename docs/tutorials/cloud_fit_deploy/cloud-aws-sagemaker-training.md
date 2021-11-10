# Cloud Training with AWS SageMaker
:label:`cloud_aws_sagemaker`

To help with AutoGluon models training, AWS developed a set of training and inference [deep learning containers](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#autogluon-training-containers). 
The containers can be used to train models with CPU and GPU instances and deployed as a SageMaker endpoint or used as a batch transform job.

The full end-to-end example is available in [amazon-sagemaker-examples](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/autogluon-tabular-containers) repository.

## Pre-requisites
Before starting ensure that the latest version of sagemaker python API is installed via `pip install --upgrade sagemaker`. 
This is required to ensure the information about newly released containers is available.

## Training Scripts

To start using the containers, a user training script and the [wrapper classes](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/autogluon-tabular-containers/ag_model.py) are required.
When authoring a training/inference [scripts](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/autogluon-tabular-containers/scripts/), 
please refer to SageMaker [documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script).

Here is one of the possible training scripts, which takes AutoGluon parameters as a YAML config and outputs predictions, models leaderboard and feature importance:

```{.python .input}
import argparse
import os
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
    parser.add_argument("--training_dir", type=str, default=get_env_if_present("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--test_dir", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TEST")
    )
    parser.add_argument("--ag_config", type=str, default=get_env_if_present("SM_CHANNEL_CONFIG"))

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

    train_file = get_input_path(args.training_dir)
    train_data = TabularDataset(train_file)

    ag_predictor_args = config["ag_predictor_args"]
    ag_predictor_args["path"] = args.model_dir
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
```

YAML config:

```yaml
# AutoGluon Predictor constructor arguments
# - see https://github.com/awslabs/autogluon/blob/ef3a5312dc2eaa0c6afde042d671860ac42cbafb/tabular/src/autogluon/tabular/predictor/predictor.py#L51-L159
ag_predictor_args:
  eval_metric: roc_auc
  label: class

# AutoGluon Predictor.fit arguments
# - see https://github.com/awslabs/autogluon/blob/ef3a5312dc2eaa0c6afde042d671860ac42cbafb/tabular/src/autogluon/tabular/predictor/predictor.py#L280-L651
ag_fit_args:
  presets: "medium_quality_faster_train"
  num_bag_folds: 2
  num_bag_sets: 1
  num_stack_levels: 0

output_prediction_format: csv  # predictions output format: csv or parquet
feature_importance: true       # calculate and save feature importance if true
leaderboard: true              # save leaderboard output if true
```

## Training

To train AutoGluon model, set up a SageMaker session:

```python
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

Create a training task:

```python
ag = AutoGluonTraining(
    role=role,
    entry_point="scripts/tabular_train.py",
    region=region,
    instance_count=1,
    instance_type="ml.m5.2xlarge",
    framework_version="0.3.1",
    base_job_name="autogluon-tabular-train",
)
```

Upload the required inputs, via SageMaker session (in this case it is a training set, test set and training YAML config) and start the training job:

```python
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

job_name = utils.unique_name_from_base("test-autogluon-image")
ag.fit(
    {"config": config_input, "train": train_input, "test": eval_input},
    job_name=job_name,
)
```

Once the models are trained, they will be available in S3 location specified in `ag.model_data` field. The model is fully portable and can be downloaded locally
if needed.
