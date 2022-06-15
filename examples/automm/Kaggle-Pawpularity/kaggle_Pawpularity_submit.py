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
    save_paths = ["../input/pawpularity-automm-result/result1/result",
                  "../input/pawpularity-automm-result/result2/result",
                  "../input/pawpularity-automm-result/result3/result", ] #Model lists.

    model_preds = np.array([])
    for perpath in save_paths:
        N_fold = 5
        save_standalone_path = perpath + '_standalone'
        all_preds = []
        for fold in range(N_fold):
            predictor = AutoMMPredictor(
                label='Pawpularity',
                problem_type='regression',
                eval_metric='rmse',
                path=perpath,
                verbosity=4,
            )
            pretrained_model = predictor.load(path=save_standalone_path + f'_fold{fold}/')
            df_test = pretrained_model.predict(test_df)
            all_preds.append(df_test)
            del predictor
            torch.cuda.empty_cache()
        model_preds = np.append(model_preds, np.mean(np.stack(all_preds), axis=0))

    submission["Pawpularity"] = np.mean(np.stack(model_preds), axis=0) # Model ensemble.
    submission.to_csv("submission.csv", index=False)

    print(submission)
