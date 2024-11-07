import copy
import time

import numpy as np
import pandas as pd


def get_model_true_infer_speed_per_row_batch(data: pd.DataFrame, *, predictor, batch_size: int = 100000, repeats=1, persist=True, silent=False):
    """
    Get per-model true inference speed per row for a given batch size of data.

    Parameters
    ----------
    data : :class:`pd.DataFrame`
        Table of the data, which is similar to a pandas DataFrame.
        Must contain the label column to be compatible with leaderboard call.
    predictor : TabularPredictor
        Fitted predictor to get inference speeds for.
    batch_size : int, default = 100000
        Batch size to use when calculating speed. `data` will be modified to have this many rows.
        If simulating large-scale batch inference, values of 100000+ are recommended to get genuine throughput estimates.
    repeats : int, default = 1
        Repeats of calling leaderboard. Repeat times are averaged to get more stable inference speed estimates.
    persist : bool, default = True
        If True, attempts to persist models into memory for more genuine throughput calculation.
    silent : False
        If False, logs information regarding the speed of each model + feature preprocessing.

    Returns
    -------
    time_per_row_df : pd.DataFrame, time_per_row_transform : float
        time_per_row_df contains each model as index.
            'pred_time_test_with_transform' is the end-to-end prediction time per row in seconds if calling `predictor.predict(data, model=model)`
            'pred_time_test' is the end-to-end prediction time per row in seconds minus the global feature preprocessing time.
            'pred_time_test_marginal' is the prediction time needed to predict for this particular model minus dependent model inference times and global preprocessing time.
        time_per_row_transform is the time in seconds per row to do the feature preprocessing.
    """
    data_batch = copy.deepcopy(data)
    len_data = len(data_batch)
    if len_data == batch_size:
        pass
    elif len_data < batch_size:
        # add more rows
        duplicate_count = int(np.ceil(batch_size / len_data))
        data_batch = pd.concat([data_batch for _ in range(duplicate_count)])
        len_data = len(data_batch)
    if len_data > batch_size:
        # sample rows
        data_batch = data_batch.sample(n=batch_size, random_state=0)
        len_data = len(data_batch)

    if len_data != batch_size:
        raise AssertionError(f"len(data_batch) must equal batch_size! ({len_data} != {batch_size})")

    if persist:
        predictor.persist(models="all")

    ts = time.time()
    for i in range(repeats):
        predictor.transform_features(data_batch)
    time_transform = (time.time() - ts) / repeats

    leaderboards = []
    for i in range(repeats):
        leaderboard = predictor.leaderboard(data_batch, skip_score=True)
        leaderboard = leaderboard[leaderboard["can_infer"]][["model", "pred_time_test", "pred_time_test_marginal"]]
        leaderboard = leaderboard.set_index("model")
        leaderboards.append(leaderboard)
    leaderboard = pd.concat(leaderboards)
    time_per_batch_df = leaderboard.groupby(level=0).mean()
    time_per_batch_df["pred_time_test_with_transform"] = time_per_batch_df["pred_time_test"] + time_transform
    time_per_row_df = time_per_batch_df / batch_size
    time_per_row_transform = time_transform / batch_size

    if not silent:
        print(f"Throughput for batch_size={batch_size}:")
        for index, row in time_per_row_df.iterrows():
            time_per_row = row["pred_time_test_with_transform"]
            time_per_row_print = time_per_row
            unit = "s"
            if time_per_row_print < 1e-2:
                time_per_row_print *= 1000
                unit = "ms"
                if time_per_row_print < 1e-2:
                    time_per_row_print *= 1000
                    unit = "μs"
            print(f"\t{round(time_per_row_print, 3)}{unit} per row | {index}")
        time_per_row_transform_print = time_per_row_transform
        unit = "s"
        if time_per_row_transform_print < 1e-2:
            time_per_row_transform_print *= 1000
            unit = "ms"
            if time_per_row_transform_print < 1e-2:
                time_per_row_transform_print *= 1000
                unit = "μs"
        print(f"\t{round(time_per_row_transform_print, 3)}{unit} per row | transform_features")

    return time_per_row_df, time_per_row_transform


def get_model_true_infer_speed_per_row_batch_bulk(
    data: pd.DataFrame, *, predictor, batch_sizes: list = None, repeats=1, persist=True, include_transform_features=False, silent=False
) -> (pd.DataFrame, pd.DataFrame):
    """
    Get per-model true inference speed per row for a list of batch sizes of data.

    Parameters
    ----------
    data : :class:`pd.DataFrame`
        Table of the data, which is similar to a pandas DataFrame.
        Must contain the label column to be compatible with leaderboard call.
    predictor : TabularPredictor
        Fitted predictor to get inference speeds for.
    batch_sizes : List[int], default = [1, 10, 100, 1000, 10000]
        Batch sizes to use when calculating speed. `data` will be modified to have this many rows.
        If simulating large-scale batch inference, values of 100000+ are recommended to get genuine throughput estimates.
    repeats : int, default = 1
        Repeats of calling leaderboard. Repeat times are averaged to get more stable inference speed estimates.
    persist : bool, default = True
        If True, attempts to persist models into memory for more genuine throughput calculation.
    include_transform_features : bool, default = False
        If True, adds transform_features (data preprocessing) speeds into the first DataFrame output as if it was a model with the name "transform_features".
        Useful when plotting model throughput to see if data preprocessing is a bottleneck.
    silent : False
        If False, logs information regarding the speed of each model + feature preprocessing.

    Returns
    -------
    time_per_row_df : pd.DataFrame, time_per_row_df_transform : pd.DataFrame
        time_per_row_df contains the following columns.
            'model' is the model name.
            'pred_time_test_with_transform' is the end-to-end prediction time per row in seconds if calling `predictor.predict(data, model=model)`
            'pred_time_test' is the end-to-end prediction time per row in seconds minus the global feature preprocessing time.
            'pred_time_test_marginal' is the prediction time needed to predict for this particular model minus dependent model inference times and global preprocessing time.
            'batch_size' is the inference batch size used to calculate the pred_time columns.
        time_per_row_df_transform is the time in seconds per row to do the feature preprocessing.
            It contains the same columns as time_per_row_df but without the model column.
    """
    if batch_sizes is None:
        batch_sizes = [
            1,
            10,
            100,
            1000,
            10000,
        ]
    infer_dfs = dict()
    infer_transform_dfs = dict()

    if persist:
        predictor.persist(models="all")

    for batch_size in batch_sizes:
        infer_df, time_per_row_transform = get_model_true_infer_speed_per_row_batch(
            data=data, predictor=predictor, batch_size=batch_size, repeats=repeats, persist=False, silent=silent
        )
        infer_dfs[batch_size] = infer_df
        infer_transform_dfs[batch_size] = time_per_row_transform
    for key in infer_dfs.keys():
        infer_dfs[key] = infer_dfs[key].reset_index()
        infer_dfs[key]["batch_size"] = key

    infer_df_full_transform = pd.Series(infer_transform_dfs, name="pred_time_test").to_frame().rename_axis("batch_size")
    infer_df_full_transform["pred_time_test_marginal"] = infer_df_full_transform["pred_time_test"]
    infer_df_full_transform["pred_time_test_with_transform"] = infer_df_full_transform["pred_time_test"]
    infer_df_full_transform = infer_df_full_transform.reset_index()

    infer_df_full = pd.concat([infer_dfs[key] for key in infer_dfs.keys()])

    if include_transform_features:
        infer_df_full_transform_include = infer_df_full_transform.copy()
        infer_df_full_transform_include["model"] = "transform_features"
        infer_df_full = pd.concat([infer_df_full, infer_df_full_transform_include])

    infer_df_full = infer_df_full.sort_values(by=["batch_size"])
    infer_df_full = infer_df_full.reset_index(drop=True)

    return infer_df_full, infer_df_full_transform
