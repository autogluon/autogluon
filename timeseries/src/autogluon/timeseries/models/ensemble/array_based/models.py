from abc import ABC
from typing import Any, Type

from autogluon.timeseries.dataset import TimeSeriesDataFrame

from .abstract import ArrayBasedTimeSeriesEnsembleModel
from .regressor import (
    EnsembleRegressor,
    LinearStackerEnsembleRegressor,
    MedianEnsembleRegressor,
    PerQuantileTabularEnsembleRegressor,
    TabularEnsembleRegressor,
)


class MedianEnsemble(ArrayBasedTimeSeriesEnsembleModel):
    """Robust ensemble that computes predictions as the element-wise median of base model mean
    and quantile forecasts, providing robustness to outlier predictions.

    Other Parameters
    ----------------
    isotonization : str, default = "sort"
        The isotonization method to use (i.e. the algorithm to prevent quantile non-crossing).
        Currently only "sort" is supported.
    detect_and_ignore_failures : bool, default = True
        Whether to detect and ignore "failed models", defined as models which have a loss that is larger
        than 10x the median loss of all the models. This can be very important for the regression-based
        ensembles, as moving the weight from such a "failed model" to zero can require a long training
        time.
    """

    def _get_ensemble_regressor(self) -> MedianEnsembleRegressor:
        return MedianEnsembleRegressor()


class BaseTabularEnsemble(ArrayBasedTimeSeriesEnsembleModel, ABC):
    ensemble_regressor_type: Type[EnsembleRegressor]

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        default_hps = super()._get_default_hyperparameters()
        default_hps.update({"model_name": "GBM", "model_hyperparameters": {}})
        return default_hps

    def _get_ensemble_regressor(self):
        hyperparameters = self.get_hyperparameters()
        return self.ensemble_regressor_type(
            quantile_levels=list(self.quantile_levels),
            model_name=hyperparameters["model_name"],
            model_hyperparameters=hyperparameters["model_hyperparameters"],
        )


class TabularEnsemble(BaseTabularEnsemble):
    """Tabular ensemble that uses a single AutoGluon-Tabular model to learn ensemble combinations.

    This ensemble trains a single tabular model (such as gradient boosting machines) to predict all
    quantiles simultaneously from base model predictions. The tabular model learns complex non-linear
    patterns in how base models should be combined, potentially capturing interactions and conditional
    dependencies that simple weighted averages cannot represent.

    Other Parameters
    ----------------
    model_name : str, default = "GBM"
        Name of the AutoGluon-Tabular model to use for ensemble learning. Model name should be registered
        in AutoGluon-Tabular model registry.
    model_hyperparameters : dict, default = {}
        Hyperparameters to pass to the underlying AutoGluon-Tabular model.
    isotonization : str, default = "sort"
        The isotonization method to use (i.e. the algorithm to prevent quantile non-crossing).
        Currently only "sort" is supported.
    detect_and_ignore_failures : bool, default = True
        Whether to detect and ignore "failed models", defined as models which have a loss that is larger
        than 10x the median loss of all the models. This can be very important for the regression-based
        ensembles, as moving the weight from such a "failed model" to zero can require a long training
        time.
    """

    ensemble_regressor_type = TabularEnsembleRegressor


class PerQuantileTabularEnsemble(BaseTabularEnsemble):
    """Tabular ensemble using separate AutoGluon-Tabular models for each quantile and mean forecast.

    This ensemble trains dedicated tabular models for each quantile level plus a separate model
    for the mean prediction. Each model specializes in learning optimal combinations for its
    specific target, allowing for quantile-specific ensemble strategies that can capture different
    model behaviors across the prediction distribution.

    Other Parameters
    ----------------
    model_name : str, default = "GBM"
        Name of the AutoGluon-Tabular model to use for ensemble learning. Model name should be registered
        in AutoGluon-Tabular model registry.
    model_hyperparameters : dict, default = {}
        Hyperparameters to pass to the underlying AutoGluon-Tabular model.
    isotonization : str, default = "sort"
        The isotonization method to use (i.e. the algorithm to prevent quantile non-crossing).
        Currently only "sort" is supported.
    detect_and_ignore_failures : bool, default = True
        Whether to detect and ignore "failed models", defined as models which have a loss that is larger
        than 10x the median loss of all the models. This can be very important for the regression-based
        ensembles, as moving the weight from such a "failed model" to zero can require a long training
        time.
    """

    ensemble_regressor_type = PerQuantileTabularEnsembleRegressor


class LinearStackerEnsemble(ArrayBasedTimeSeriesEnsembleModel):
    """Linear stacking ensemble that learns optimal linear combination weights through gradient-based
    optimization.

    Weighted combinations can be per model or per model-quantile, model-horizon, model-quantile-horizon
    combinations. These choices are controlled by the ``weights_per`` hyperparameter.

    The optimization process uses gradient descent with configurable learning rates and convergence
    criteria, allowing for flexible training dynamics. Weight pruning can be applied to remove
    models with negligible contributions, resulting in sparse and interpretable ensembles.

    Other Parameters
    ----------------
    weights_per : str, default = "m"
        Granularity of weight learning.

        - "m": single weight per model
        - "mq": single weight for each model-quantile combination
        - "mt": single weight for each model-time step where time steps run across the prediction horizon
        - "mtq": single weight for each model-quantile-time step combination
    lr : float, default = 0.1
        Learning rate for PyTorch optimizer during weight training.
    max_epochs : int, default = 10000
        Maximum number of training epochs for weight optimization.
    relative_tolerance : float, default = 1e-7
        Relative tolerance for convergence detection during training.
    prune_below : float, default = 0.0
        Threshold below which weights are pruned to zero for sparsity. The weights are redistributed across
        remaining models after pruning.
    isotonization : str, default = "sort"
        The isotonization method to use (i.e. the algorithm to prevent quantile non-crossing).
        Currently only "sort" is supported.
    detect_and_ignore_failures : bool, default = True
        Whether to detect and ignore "failed models", defined as models which have a loss that is larger
        than 10x the median loss of all the models. This can be very important for the regression-based
        ensembles, as moving the weight from such a "failed model" to zero can require a long training
        time.
    """

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        default_hps = super()._get_default_hyperparameters()
        default_hps.update(
            {
                "weights_per": "m",
                "lr": 0.1,
                "max_epochs": 10000,
                "relative_tolerance": 1e-7,
                "prune_below": 0.0,
            }
        )
        return default_hps

    def _get_ensemble_regressor(self) -> LinearStackerEnsembleRegressor:
        hps = self.get_hyperparameters()
        return LinearStackerEnsembleRegressor(
            quantile_levels=list(self.quantile_levels),
            weights_per=hps["weights_per"],
            lr=hps["lr"],
            max_epochs=hps["max_epochs"],
            relative_tolerance=hps["relative_tolerance"],
            prune_below=hps["prune_below"],
        )

    def _fit(
        self,
        predictions_per_window: dict[str, list[TimeSeriesDataFrame]],
        data_per_window: list[TimeSeriesDataFrame],
        model_scores: dict[str, float] | None = None,
        time_limit: float | None = None,
    ) -> None:
        super()._fit(predictions_per_window, data_per_window, model_scores, time_limit)

        assert isinstance(self.ensemble_regressor, LinearStackerEnsembleRegressor)

        if self.ensemble_regressor.kept_indices is not None:
            original_names = self._model_names
            self._model_names = [original_names[i] for i in self.ensemble_regressor.kept_indices]
