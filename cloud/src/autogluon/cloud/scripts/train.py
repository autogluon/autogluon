import argparse
import os
from pprint import pprint

import yaml
from autogluon.tabular import TabularPredictor
from autogluon.text import TextPredictor


def get_input_path(path):
    file = os.listdir(path)[0]
    if len(os.listdir(path)) > 1:
        raise ValueError(f"WARN: more than one file is found in {channel} directory")
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

    # ------------------------------------------------------------ Args parsing
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
        "--tune_dir", type=str, required=False, default=get_env_if_present("SM_CHANNEL_TUNE")
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

    predictor_type = config["predictor_type"]
    predictor_init_args = config["predictor_init_args"]
    predictor_init_args["path"] = args.model_dir
    predictor_fit_args = config["predictor_fit_args"]
    assert predictor_type in ['tabular', 'text']
    if predictor_type == 'tabular':
        predictor_cls = TabularPredictor
    elif predictor_type == 'text':
        predictor_cls = TextPredictor

    train_file = get_input_path(args.training_dir)

    tune_file = None
    if args.tune_dir:
        tune_file = get_input_path(args.tune_dir)
    predictor = predictor_cls(**predictor_init_args).fit(train_file, tuning_data=tune_file, **predictor_fit_args)

    if predictor_cls == TabularPredictor:
        if config.get("leaderboard", False):
            lb = predictor.leaderboard(silent=False)
            lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")
