from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from autogluon.tabular import TabularPredictor


def _cumulative_min_idx(series: pd.Series) -> pd.Series:
    """
    
    Parameters
    ----------
    series: pd.Series

    Returns
    -------
    pd.Series
        The index of the cumulative min of the series values.

    """
    min_val = float('inf')
    min_index = -1
    result = []
    for i, val in enumerate(series):
        if pd.isna(val):
            result.append(min_index)
        elif val < min_val:
            min_val = val
            min_index = i
            result.append(min_index)
        else:
            result.append(min_index)
    return pd.Series(series.index[result], index=series.index)


def compute_cumulative_leaderboard_stats(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    leaderboard: pd.DataFrame

    Returns
    -------
    leaderboard_stats: pd.DataFrame

    """
    leaderboard = leaderboard.copy(deep=True)
    leaderboard = leaderboard.sort_values(by=["fit_order"]).set_index("model")
    leaderboard["best_model_so_far"] = _cumulative_min_idx(leaderboard["metric_error_val"])
    leaderboard["best_idx_so_far"] = leaderboard["best_model_so_far"].map(leaderboard["fit_order"])
    leaderboard["time_so_far"] = leaderboard["fit_time_marginal"].cumsum()
    leaderboard["metric_error_val_so_far"] = leaderboard["best_model_so_far"].map(leaderboard["metric_error_val"])
    if "metric_error_test" in leaderboard:
        leaderboard["metric_error_test_so_far"] = leaderboard["best_model_so_far"].map(leaderboard["metric_error_test"])
    leaderboard = leaderboard.reset_index(drop=False).set_index("fit_order")
    return leaderboard


# TODO: Include constraints as options:
#  infer_limit
#  disk_usage
# TODO: Avoid calling leaderboard on the original models again
# TODO: Calibration?
def compute_cumulative_leaderboard_stats_ensemble(
    leaderboard: pd.DataFrame,
    predictor: TabularPredictor,
    test_data: pd.DataFrame | None = None,
    cleanup_ensembles: bool = True,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    leaderboard: pd.DataFrame
    predictor: TabularPredictor
    test_data: pd.DataFrame | None, default None
    cleanup_ensembles: bool, default True

    Returns
    -------
    leaderboard_stats: pd.DataFrame

    """
    leaderboard_stats = compute_cumulative_leaderboard_stats(leaderboard)
    model_fit_order = list(leaderboard_stats["model"])
    ens_names = []
    for i in range(len(model_fit_order)):
        models_to_ens = model_fit_order[:i + 1]
        ens_name = predictor.fit_weighted_ensemble(base_models=models_to_ens, name_suffix=f"_fit_{i + 1}")[0]
        ens_names.append(ens_name)

    leaderboard_stats_ens = predictor.leaderboard(test_data, score_format="error", display=False)
    leaderboard_stats_ens = leaderboard_stats_ens[leaderboard_stats_ens["model"].isin(ens_names)]
    leaderboard_stats_ens = leaderboard_stats_ens.set_index("model").reindex(ens_names).reset_index()
    leaderboard_stats_ens["fit_order"] = leaderboard_stats.index
    leaderboard_stats_ens["model"] = leaderboard_stats["model"].values
    leaderboard_stats_ens = compute_cumulative_leaderboard_stats(leaderboard_stats_ens)

    leaderboard_stats["metric_error_val_so_far_ens"] = leaderboard_stats_ens["metric_error_val_so_far"]
    if test_data is not None:
        leaderboard_stats["metric_error_test_so_far_ens"] = leaderboard_stats_ens["metric_error_test_so_far"]
    leaderboard_stats["best_idx_so_far_ens"] = leaderboard_stats_ens["best_idx_so_far"]
    leaderboard_stats["best_model_so_far_ens"] = leaderboard_stats_ens["best_model_so_far"]
    if cleanup_ensembles:
        predictor.delete_models(models_to_delete=ens_names, dry_run=False)

    return leaderboard_stats


def plot_leaderboard_from_predictor(
    predictor: TabularPredictor,
    test_data: pd.DataFrame | None = None,
    ensemble: bool = False,
    include_val: bool = True,
) -> tuple[Figure, pd.DataFrame]:
    """

    Parameters
    ----------
    predictor: TabularPredictor
    test_data: pd.DataFrame | None, default None
        If specified, plots the test error.
    ensemble: bool, default False
        If True, additionally plots the results of cumulatively ensembling models at each step.
    include_val: bool, default True
        If True, plots the validation error.

    Returns
    -------
    fig: Figure
    leaderboard_stats: pd.DataFrame

    Examples
    --------
    >>> data_root = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
    >>> predictor_example = TabularPredictor(label="class").fit(train_data=data_root + "train.csv", time_limit=60)
    >>> figure, lb = plot_leaderboard_from_predictor(predictor=predictor_example, test_data=data_root + "test.csv", ensemble=True)
    >>> with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
    >>>     print(lb)
    >>> figure.savefig("example_leaderboard_plot.png")
    """
    leaderboard = predictor.leaderboard(test_data, score_format="error", display=False)
    if ensemble:
        leaderboard_order_sorted = compute_cumulative_leaderboard_stats_ensemble(leaderboard=leaderboard, test_data=test_data, predictor=predictor)
    else:
        leaderboard_order_sorted = compute_cumulative_leaderboard_stats(leaderboard=leaderboard)
    return plot_leaderboard(leaderboard=leaderboard_order_sorted, preprocess=False, ensemble=ensemble, include_val=include_val)


def plot_leaderboard(
    leaderboard: pd.DataFrame,
    preprocess: bool = True,
    ensemble: bool = False,
    include_val: bool = True,
    include_test: bool | None = None,
) -> tuple[Figure, pd.DataFrame]:
    """

    Parameters
    ----------
    leaderboard: pd.DataFrame
        Either the raw leaderboard output of `predictor.leaderboard(..., score_format="error")` or the output of `compute_cumulative_leaderboard_stats`.
    preprocess: bool, default True
        Whether to preprocess the leaderboard to obtain leaderboard_stats.
        Set to False if `leaderboard` has already been transformed
        via `compute_cumulative_leaderboard_stats` or `compute_cumulative_leaderboard_stats_ensemble`.
    ensemble: bool, default False
        If True, additionally plots the results of cumulatively ensembling models at each step.
        Can only be set to True if ensemble columns are present in the leaderboard,
        which are generated by first calling `compute_cumulative_leaderboard_stats_ensemble`.
    include_val: bool, default True
        If True, plots the validation error.
    include_test: bool | None, default None
        If True, plots the test error.
        If None, infers based on the existence of the test error column in `leaderboard`.

    Returns
    -------
    fig: Figure
    leaderboard_stats: pd.DataFrame

    """
    leaderboard_order_sorted = leaderboard
    if preprocess:
        if ensemble:
            raise AssertionError(
                f"Cannot have both `preprocess=True` and `ensemble=True`."
                f"Instead call `plot_leaderboard_from_predictor(..., ensemble=True)`"
            )
        leaderboard_order_sorted = compute_cumulative_leaderboard_stats(leaderboard=leaderboard_order_sorted)

    eval_metric = leaderboard_order_sorted["eval_metric"].iloc[0]
    if include_test is None:
        include_test = "metric_error_test_so_far" in leaderboard_order_sorted

    # TODO: View on inference time, can take from ensemble model, 3rd dimension, color?
    fig, axes = plt.subplots(1, 2, sharey=True)
    fig.suptitle('AutoGluon Metric Error Over Time')

    ax = axes[0]

    if include_test:
        ax.plot(leaderboard_order_sorted.index, leaderboard_order_sorted["metric_error_test_so_far"].values, '-', color="b", label="test")
    if include_val:
        ax.plot(leaderboard_order_sorted.index, leaderboard_order_sorted["metric_error_val_so_far"].values, '-', color="orange", label="val")
    if ensemble:
        if include_test:
            ax.plot(leaderboard_order_sorted.index, leaderboard_order_sorted["metric_error_test_so_far_ens"].values, '--', color="b", label="test (ens)")
        if include_val:
            ax.plot(leaderboard_order_sorted.index, leaderboard_order_sorted["metric_error_val_so_far_ens"].values, '--', color="orange", label="val (ens)")
    ax.set_xlim(left=1, right=leaderboard_order_sorted.index.max())
    ax.set_xlabel('# Models Fit')
    ax.set_ylabel(f'Metric Error ({eval_metric})')
    ax.grid()

    ax = axes[1]

    if include_test:
        ax.plot(leaderboard_order_sorted["time_so_far"].values, leaderboard_order_sorted["metric_error_test_so_far"].values, '-', color="b", label="test")
    if include_val:
        ax.plot(leaderboard_order_sorted["time_so_far"].values, leaderboard_order_sorted["metric_error_val_so_far"].values, '-', color="orange", label="val")
    if ensemble:
        if include_test:
            ax.plot(leaderboard_order_sorted["time_so_far"].values, leaderboard_order_sorted["metric_error_test_so_far_ens"].values, '--', color="b", label="test (ens)")
        if include_val:
            ax.plot(leaderboard_order_sorted["time_so_far"].values, leaderboard_order_sorted["metric_error_val_so_far_ens"].values, '--', color="orange", label="val (ens)")
    ax.set_xlabel('Time Elapsed (s)')
    ax.grid()
    ax.legend()

    return fig, leaderboard_order_sorted
