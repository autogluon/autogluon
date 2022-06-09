from autogluon.text.automm import AutoMMPredictor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import torch
import os


def load_data(data_path: str):
    """
    Load training and testing datasets and split folds.
    Parameters
    ----------
    data_path
        The path of training and testing files.

    Return
    ----------
    The pure training dataset and the folds split information and the testing dataset.
    """
    # Load training dataset.
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    train_df.rename(columns={"Id": "Image Path"}, inplace=True)
    train_df["Image Path"] = train_df["Image Path"].apply(lambda s: os.path.join(data_path, "train", s + ".jpg"))

    # Creat the split information of folds.
    num_bins = int(np.ceil(2 * ((len(train_df)) ** (1. / 3))))
    train_df_bins = pd.cut(train_df["Pawpularity"], bins=num_bins, labels=False)
    train_df_fold = pd.Series([-1] * len(train_df))
    strat_kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
    for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df_bins)):
        train_df_fold[train_index] = i
    train_df_fold = train_df_fold.astype("int")

    # Load testing dataset.
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    test_df.rename(columns={"Id": "Image Path"}, inplace=True)
    test_df["Image Path"] = test_df["Image Path"].apply(lambda s: os.path.join(data_path, "test", s + ".jpg"))

    return train_df, train_df_fold, test_df


if __name__ == "__main__":
    data_path = "./data/"  # The path of the training and testing data.
    save_path = "./pawpularity-result/result1/result"  # The path of saving the model.
    save_standalone_path = save_path + "_standalone" # The path of saving the standalone model which includes downloaded model.

    N_FOLDS = 5  # The number of folds.

    all_score = []  # The result of folds.
    train_df, train_df_fold, _ = load_data(data_path)

    for i in range(N_FOLDS):
        # The predictor in use.
        predictor = AutoMMPredictor(
            label="Pawpularity",  # label indicates the target value
            problem_type="regression",  # problem_type indicates the type of the problem. It can choose "multiclass", # "binary" or "regression"
            eval_metric="rmse",  # eval_metric indicates the evaluation index of the model
            path=save_path,
            verbosity=4,  # verbosity controls how much information is printed.
        )

        # Training process.
        training_df = train_df[train_df_fold != i]
        valid_df = train_df[train_df_fold == i]
        predictor.fit(
            train_data=training_df,
            tuning_data=valid_df,
            save_path=save_path + f"_fold{i}",
            hyperparameters={
                # "model.names": "['timm_image']",
                "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224",
                "model.timm_image.train_transform_types": "['resize_shorter_side','center_crop','randaug']",
                "data.categorical.convert_to_text": "False",
                "env.per_gpu_batch_size": "8",
                "env.per_gpu_batch_size_evaluation": "32",
                "env.precision": "32",
                "optimization.learning_rate": "2e-5",
                "optimization.weight_decay": "0",
                "optimization.lr_decay": "1",
                "optimization.max_epochs": "5",
                "optimization.warmup_steps": "0",
                "optimization.loss_function": "bcewithlogitsloss",
            },
            seed=1,
        )

        # Manual Validating process.
        valid_pred = predictor.predict(data=valid_df)
        score = mean_squared_error(valid_df["Pawpularity"].values, valid_pred, squared=False)
        print(f"Fold {i} | Score: {score}")
        predictor.save(
            path=save_standalone_path + f"_fold{i}",
            standalone=True,
        )
        all_score.append(score)

        del predictor
        torch.cuda.empty_cache()

    print(f"all-scores: {all_score}")
    print(f"mean_rmse: {np.mean(all_score)}")
