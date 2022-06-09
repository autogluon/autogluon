import sys
sys.path.append("../input/autogluon/")
sys.path.append("../input/nptyping/")
sys.path.append("../input/typish/")
sys.path.append("../input/timm-pytorch-image-models/pytorch-image-models-master/")
sys.path.append("../input/omegaconf/")
sys.path.append("../input/antlr4/")

from autogluon.text.automm import AutoMMPredictor
import pandas as pd
import numpy as np
import torch
import os


def load_data(data_path):
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    train_df.rename(columns={"Id": "Image Path"}, inplace=True)
    train_df["Image Path"] = train_df["Image Path"].apply(lambda s: os.path.join(data_path, "train", s + ".jpg"))

    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    test_df.rename(columns={"Id": "Image Path"}, inplace=True)
    test_df["Image Path"] = test_df["Image Path"].apply(lambda s: os.path.join(data_path, "test", s + ".jpg"))
    return train_df, test_df


if __name__ == "__main__":
    data_path = "../input/petfinder-pawpularity-score/"
    train_df, test_df = load_data(data_path)

    submission = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")
    N_fold = 5

    save_path = "../input/pawpularity-result/result1/result"
    save_standalone_path = save_path + "_standalone"
    all_preds_1 = []
    for fold in range(N_fold):
        predictor = AutoMMPredictor(
            label="Pawpularity",
            problem_type="regression",
            eval_metric="rmse",
            path=save_path,
            verbosity=4,
        )
        pretrained_model = predictor.load(path=save_standalone_path + f"_fold{fold}/")  # Load the predictor.
        test_pred = pretrained_model.predict(test_df)  # Predict the test dataset.
        all_preds_1.append(test_pred)
        del predictor
        torch.cuda.empty_cache()
    preds_1 = np.mean(np.stack(all_preds_1), axis=0)  # Fold ensemble.

    save_path = "../input/pawpularity-result/result2/result"
    save_standalone_path = save_path + "_standalone"
    all_preds_2 = []
    for fold in range(N_fold):
        predictor = AutoMMPredictor(
            label="Pawpularity",
            problem_type="regression",
            eval_metric="rmse",
            path=save_path,
            verbosity=4,
        )
        pretrained_model = predictor.load(path=save_standalone_path + f"_fold{fold}/")
        test_pred = pretrained_model.predict(test_df)
        all_preds_2.append(test_pred)
        del predictor
        torch.cuda.empty_cache()
    preds_2 = np.mean(np.stack(all_preds_2), axis=0)

    save_path = "../input/pawpularity-result/result3/result"
    save_standalone_path = save_path + "_standalone"
    all_preds_3 = []
    for fold in range(N_fold):
        predictor = AutoMMPredictor(
            label="Pawpularity",
            problem_type="regression",
            eval_metric="rmse",
            path=save_path,
            verbosity=4,
        )
        pretrained_model = predictor.load(path=save_standalone_path + f'_fold{fold}/')
        test_pred = pretrained_model.predict(test_df)
        all_preds_3.append(test_pred)
        del predictor
        torch.cuda.empty_cache()
    preds_3 = np.mean(np.stack(all_preds_3), axis=0)

    save_path = "../input/pawpularity-result/result4/result"
    save_standalone_path = save_path + "_standalone"
    all_preds_4 = []
    for fold in range(N_fold):
        predictor = AutoMMPredictor(
            label="Pawpularity",
            problem_type="regression",
            eval_metric="rmse",
            path=save_path,
            verbosity=4,
        )
        pretrained_model = predictor.load(path=save_standalone_path + f"_fold{fold}/")
        test_pred = pretrained_model.predict(test_df)
        all_preds_4.append(test_pred)
        del predictor
        torch.cuda.empty_cache()
    preds_4 = np.mean(np.stack(all_preds_4), axis=0)

    submission["Pawpularity"] = (preds_1 + preds_2 + preds_3 + preds_4) / 4  # Model ensemble.
    submission.to_csv("submission.csv", index=False)

    print(submission)
