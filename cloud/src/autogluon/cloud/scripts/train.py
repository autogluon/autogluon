import argparse
import os
import shutil
import yaml

from autogluon.tabular import TabularPredictor, TabularDataset, FeatureMetadata
from autogluon.vision import ImagePredictor
from autogluon.text import TextPredictor
from autogluon.multimodal import MultiModalPredictor

from pprint import pprint


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
    parser.add_argument(
        "--images_dir", type=str, required=False, default=get_env_if_present("SM_CHANNEL_IMAGES")
    )
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
    save_path = os.path.normpath(args.model_dir)
    predictor_type = config["predictor_type"]
    predictor_init_args = config["predictor_init_args"]
    predictor_init_args["path"] = save_path
    predictor_fit_args = config["predictor_fit_args"]
    valid_predictor_types = ['tabular', 'text', 'image', 'multimodal']
    assert predictor_type in valid_predictor_types, f'predictor_type {predictor_type} not supported. Valid options are {valid_predictor_types}'
    if predictor_type == 'tabular':
        predictor_cls = TabularPredictor
        if 'feature_meatadata' in predictor_fit_args:
            predictor_fit_args['feature_meatadata'] = FeatureMetadata(**predictor_fit_args['feature_meatadata'])
    elif predictor_type == 'text':
        predictor_cls = TextPredictor
    elif predictor_type == 'image':
        predictor_cls = ImagePredictor
    elif predictor_type == 'multimodal':
        predictor_cls = MultiModalPredictor

    train_file = get_input_path(args.training_dir)
    training_data = TabularDataset(train_file)
    if predictor_type == 'tabular' and 'image_column' in config:
        feature_metadata = predictor_fit_args.get('feature_metadata', None)
        if feature_metadata is None:
            feature_metadata = FeatureMetadata.from_df(training_data)
        feature_metadata = feature_metadata.add_special_types({config['image_column']: ['image_path']})
        predictor_fit_args['feature_metadata'] = feature_metadata

    tuning_data = None
    if args.tune_dir:
        tune_file = get_input_path(args.tune_dir)
        tuning_data = TabularDataset(tune_file)

    if args.images_dir:
        image_compressed_file = get_input_path(args.images_dir)
        shutil.unpack_archive(image_compressed_file)

    predictor = predictor_cls(**predictor_init_args).fit(training_data, tuning_data=tuning_data, **predictor_fit_args)

    # When use automm backend, predictor needs to be saved with standalone flag to avoid need of internet access when loading
    # This is required because of https://discuss.huggingface.co/t/error-403-when-downloading-model-for-sagemaker-batch-inference/12571/6
    if predictor_type in ('text', 'multimodal'):
        try:
            # Need os.path.sep because text/multimodal predictor has a bug where the old path has separator in the end, and the comparison doesn't use normpath
            # TODO: remove this try except after 0.6 release
            predictor.save(path=save_path+os.path.sep, standalone=True)
        except:
            predictor.save(path=save_path+os.path.sep)
    elif predictor_type == 'image':
        predictor.save()

    if predictor_cls == TabularPredictor:
        if config.get("leaderboard", False):
            lb = predictor.leaderboard(silent=False)
            lb.to_csv(f"{args.output_data_dir}/leaderboard.csv")

    print('Saving serving script')
    serving_script_saving_path = os.path.join(save_path, 'code')
    os.mkdir(serving_script_saving_path)
    serving_script_path = get_input_path(args.serving_script)
    shutil.move(serving_script_path, os.path.join(serving_script_saving_path, os.path.basename(serving_script_path)))
