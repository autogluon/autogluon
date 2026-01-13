import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append(
    "../input/autogluon-standalone-install/autogluon_standalone/antlr4-python3-runtime-4.8/antlr4-python3-runtime-4.8/src/"
)
# !pip install --no-deps --no-index --quiet ../input/autogluon-standalone-install/autogluon_standalone/*.whl --find-links autogluon_standalone

import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor

data_path = "../input/petfinder-pawpularity-score/"

config_6 = {
    "save_path": "../input/pawpularity-automm-result/result6/result",
    "per_gpu_batch_size_evaluation": 32,
    "N_fold": 5,
}
config_7 = {
    "save_path": "../input/pawpularity-automm-result/result7/result",
    "per_gpu_batch_size_evaluation": 3,
    "N_fold": 5,
}
config_13 = {
    "save_path": "../input/pawpularity-automm-result/result13/result",
    "per_gpu_batch_size_evaluation": 32,
    "N_fold": 5,
}
config_26 = {
    "save_path": "../input/pawpularity-automm-result26/result",
    "per_gpu_batch_size_evaluation": 3,
    "N_fold": 5,
}
config_30 = {
    "save_path": "../input/pawpularity-automm-result30/result",
    "per_gpu_batch_size_evaluation": 32,
    "N_fold": 5,
}


def load_data(data_path):
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    train_df.rename(columns={"Id": "Image Path"}, inplace=True)
    train_df["Image Path"] = train_df["Image Path"].apply(lambda s: os.path.join(data_path, "train", s + ".jpg"))

    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    test_df.rename(columns={"Id": "Image Path"}, inplace=True)
    test_df["Image Path"] = test_df["Image Path"].apply(lambda s: os.path.join(data_path, "test", s + ".jpg"))
    return train_df, test_df


train_df, test_df = load_data(data_path)

if __name__ == "__main__":
    submission = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")

    configs = [config_6, config_7, config_26]
    model_preds = np.empty(shape=[0, submission.shape[0]])
    for perconfig in configs:
        print(perconfig)
        save_standalone_path = perconfig["save_path"] + "_standalone"
        all_preds = []
        for fold in range(perconfig["N_fold"]):
            predictor = MultiModalPredictor(
                label="Pawpularity",
                problem_type="regression",
                eval_metric="rmse",
                path=perconfig["save_path"],
                verbosity=4,
            )
            pretrained_model = predictor.load(path=save_standalone_path + f"_fold{fold}/")
            pretrained_model._config.env.per_gpu_batch_size_evaluation = perconfig["per_gpu_batch_size_evaluation"]
            df_test = pretrained_model.predict(test_df)
            all_preds.append(df_test)
            del predictor
            torch.cuda.empty_cache()
        model_preds = np.append(model_preds, [np.mean(np.stack(all_preds), axis=0)], axis=0)

    submission["Pawpularity"] = model_preds[0] * 0.25 + model_preds[1] * 0.5 + model_preds[2] * 0.25  # Model ensemble.
    submission.to_csv("submission.csv", index=False)

    print(submission)
