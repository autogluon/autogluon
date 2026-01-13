import os
import time

import numpy as np
import openml
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from autogluon.core.data.label_cleaner import LabelCleaner
from autogluon.core.models import BaggedEnsembleModel
from autogluon.features import AutoMLPipelineFeatureGenerator
from autogluon.tabular.models.mitra.mitra_model import MitraModel
from autogluon.tabular.testing import FitHelper


def test_mitra():
    model_hyperparameters = {"n_estimators": 1}

    try:
        model_cls = MitraModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )


def run_bagging(task_id, fold, bagging=True, target_dataset="tabrepo10fold", file_name=None, t="classification"):
    print("Task id", task_id, "Fold", fold)

    task = openml.tasks.get_task(task_id, download_splits=False)
    X, y, _, _ = task.get_dataset().get_data(task.target_name)
    train_indices, test_indices = task.get_train_test_split_indices(fold=fold)
    x_train, x_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    n_class = len(np.unique(y_train.values))
    if t == "classification":
        problem_type = "multiclass" if n_class > 2 else "binary"
    elif t == "regression":
        problem_type = "regression"
    else:
        raise ValueError(f"Unsupported task type: {t}")

    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    feature_generator = AutoMLPipelineFeatureGenerator()
    x_train = feature_generator.fit_transform(X=x_train, y=y_train)
    y_train = label_cleaner.transform(y_train)
    x_test = feature_generator.transform(X=x_test)
    y_test = label_cleaner.transform(y_test).values

    bagged_custom_model = BaggedEnsembleModel(MitraModel(problem_type=problem_type))
    custom_model = MitraModel(problem_type=problem_type)
    bagged_custom_model.params["fold_fitting_strategy"] = "sequential_local"

    time1 = time.time()

    try:
        if bagging:
            bagged_custom_model.fit(X=x_train, y=y_train, k_fold=8, save_space=True)  # Perform 8-fold bagging
        else:
            custom_model.fit(X=x_train, y=y_train, time_limit=3600)
    except ValueError:
        return

    time2 = time.time()

    if bagging:
        out = bagged_custom_model.predict_proba(x_test)
    else:
        out = custom_model.predict_proba(x_test)

    if n_class == 2 and (out.ndim == 1 or out.shape[1] == 1):
        out = np.vstack([1 - out[:, 0], out[:, 0]]).T if out.ndim > 1 else np.vstack([1 - out, out]).T

    time3 = time.time()

    train_time = time2 - time1
    infer_time = time3 - time2

    if t == "classification":
        accuracy = accuracy_score(y_test, out[:, :n_class].argmax(axis=-1))
        ce = log_loss(y_test, out[:, :n_class], labels=list(range(n_class)))
        if n_class == 2:
            roc = roc_auc_score(y_test, out[:, :2][:, 1])
        else:
            roc = roc_auc_score(y_test, out[:, :n_class], multi_class="ovo", labels=list(range(n_class)))

        print(f"accuracy: {accuracy}, ce: {ce}, roc: {roc}")

        file_path = f"/fsx/results/{target_dataset}/{file_name}.csv"

        file_exists = os.path.isfile(file_path)
        df = pd.DataFrame(
            {
                "roc": roc,
                "ce": ce,
                "accuracy": accuracy,
                "time_train_s": train_time,
                "time_infer_s": infer_time,
            },
            index=[f"tabrepo_{task_id}" + f"_fold_{fold}"],
        )

    elif t == "regression":
        mse = mean_squared_error(y_test, out)
        mae = mean_absolute_error(y_test, out)
        rmse = root_mean_squared_error(y_test, out)
        r2 = r2_score(y_test, out)

        print(f"mse: {mse}, mae: {mae}, rmse: {rmse}, r2: {r2}")

        file_path = f"/fsx/results/{target_dataset}/{file_name}.csv"
        file_exists = os.path.isfile(file_path)
        df = pd.DataFrame(
            {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "time_train_s": train_time,
                "time_infer_s": infer_time,
            },
            index=[f"tabrepo_{task_id}" + f"_fold_{fold}"],
        )

    df.to_csv(file_path, mode="a", index=True, header=not file_exists, float_format="%.4f")

    bagged_custom_model.delete_from_disk(silent=True)


if __name__ == "__main__":
    # test_mitra()

    # 8 * 6 + 9 * 2 = 66
    # 0-8, 8-16, 16-24, 24-32, 32-40, 40-48, 48-57, 57-66
    tabrepo = [
        2,
        11,
        37,
        2073,
        2077,
        3512,
        3549,
        3560,
        3581,
        3583,
        3606,
        3608,
        3616,
        3623,
        3664,
        3667,
        3690,
        3702,
        3704,
        3747,
        3749,
        3766,
        3783,
        3793,
        3799,
        3800,
        3812,
        3903,
        3913,
        3918,
        9904,
        9905,
        9906,
        9909,
        9915,
        9924,
        9925,
        9926,
        9970,
        9971,
        9979,
        14954,
        125920,
        125921,
        146800,
        146818,
        146819,
        168757,
        168784,
        190137,
        190146,
        359954,
        359955,
        359956,
        359958,
        359959,
        359960,
        359962,
        359963,
        361333,
        361335,
        361336,
        361339,
        361340,
        361341,
        361345,
    ]

    # 3 * 3 + 4 * 5 = 29
    # 0-3, 3-6, 6-9, 9-13, 13-17, 17-21, 21-25, 25-29
    amlb = [
        2073,
        146818,
        146820,
        168350,
        168757,
        168784,
        168911,
        190137,
        190146,
        190392,
        190410,
        190411,
        359954,
        359955,
        359956,
        359958,
        359959,
        359960,
        359961,
        359962,
        359963,
        359964,
        359965,
        359968,
        359969,
        359970,
        359972,
        359974,
        359975,
    ]

    # 9 * 5 + 10 * 3 = 75
    # 0-9, 9-18, 18-27, 27-36, 36-45, 45-55, 55-65, 65-75
    tabzilla = [
        4,
        9,
        10,
        11,
        14,
        15,
        16,
        18,
        22,
        23,
        25,
        27,
        29,
        31,
        35,
        37,
        39,
        40,
        42,
        47,
        48,
        50,
        53,
        54,
        59,
        2079,
        2867,
        3512,
        3540,
        3543,
        3549,
        3560,
        3561,
        3602,
        3620,
        3647,
        3731,
        3739,
        3748,
        3779,
        3797,
        3902,
        3903,
        3913,
        3917,
        3918,
        9946,
        9957,
        9971,
        9978,
        9979,
        9984,
        10089,
        10093,
        10101,
        14954,
        14967,
        125920,
        125921,
        145793,
        145799,
        145847,
        145977,
        145984,
        146024,
        146063,
        146065,
        146192,
        146210,
        146800,
        146817,
        146818,
        146819,
        146821,
        146822,
    ]

    tabrepo_reg = [167210, 359930, 359931, 359932, 359933, 359935, 359942, 359944, 359950, 359951]

    nature_reg = [
        167210,
        359940,
        359948,
        359939,
        359951,
        233215,
        359944,
        359942,
        359945,
        360945,
        361235,
        361236,
        361237,
        361617,
        361243,
        361619,
        361621,
        361251,
        361256,
        361258,
        361259,
        361622,
        359934,
        359933,
        359950,
        359932,
        359931,
        359930,
    ]

    test_reg = [363612]

    dataset_name, target_dataset, start, end = tabrepo, "tabrepo10fold", 0, 66

    for did in dataset_name[start:end]:
        for fold in range(10):
            begin_time = time.time()

            run_bagging(
                task_id=did,
                fold=fold,
                bagging=True,
                target_dataset=target_dataset,
                file_name=f"mitra_bagging_ft_save_ckpt",
                t="classification",
            )

            end_time = time.time()

            print(end_time - begin_time)
