import logging
import os
import time
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Self

from autogluon.common.loaders import load_pkl
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.utils.features import CovariateMetadata

if TYPE_CHECKING:
    from tirex2 import ForecastModel

logger = logging.getLogger(__name__)


class TiRex2Model(AbstractTimeSeriesModel):
    """TiRex-2 [Auer2026]_ pretrained time series forecasting model.

    TiRex-2 is a pretrained time series foundation model built by NXAI, generalizing the univariate
    `TiRex <https://github.com/NX-AI/tirex>`_ model to multivariate forecasting. It is built on a recurrent
    (xLSTM-based) architecture designed for efficient streaming inference and produces quantile forecasts zero-shot.
    The model is available on `Hugging Face <https://huggingface.co/NX-AI/TiRex-2>`_.

    AutoGluon supports TiRex-2 for **inference only**, i.e., the model will not be trained or fine-tuned on the
    provided training data. TiRex-2 natively conditions on past and future-known covariates, which are passed to the
    model when available. Real-valued covariates are passed through as-is and categorical covariates are target-encoded.

    TiRex-2 requires the ``tirex-2`` package. As the PyPI release has very restrictive dependencies, we recommend
    installing it from source::

        pip install "tirex-2 @ git+https://github.com/NX-AI/tirex-2.git@6b232232e3150a7897f6d43f640d5a3928e2868e"

    References
    ----------
    .. [Auer2026] Auer, Andreas et al.
        "TiRex-2: Generalizing TiRex to Multivariate Data and Streaming." (2026).
        https://arxiv.org/abs/2607.01204


    Other Parameters
    ----------------
    model_path : str, default = "NX-AI/TiRex-2"
        Model path used for the model, i.e., a HuggingFace ``name_or_path``. Can be a compatible model name on
        HuggingFace Hub or a local path to a model directory.
    batch_size : int, default = 256
        Size of batches used during inference. The batch size is automatically halved and retried on out-of-memory
        errors, so this is an upper bound.
    device : str, default = None
        Device to use for inference. If None, model will use the GPU if available, and the CPU otherwise.
    context_length : int, default = 2048
        The maximum context length (number of past observations) to use in the model. Longer series are truncated to
        the most recent ``context_length`` observations. Shorter context lengths result in faster inference.
    use_target_encoding : bool, default = True
        How categorical covariates are encoded into the real-valued features passed to the model. If True, categorical
        covariates are target-encoded (each category is replaced by a smoothed per-item mean of the target); if False,
        they are ordinal-encoded to their integer category codes. Only affects categorical covariates.
    """

    ag_priority = 50

    # TiRex-2 natively conditions on past and future-known covariates. Real-valued covariates are passed through and
    # categorical covariates are target-encoded (see `_to_timeseries`).
    _supports_known_covariates = True
    _supports_past_covariates = True

    default_model_path: str = "NX-AI/TiRex-2"

    def __init__(
        self,
        path: str | None = None,
        name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        freq: str | None = None,
        prediction_length: int = 1,
        covariate_metadata: CovariateMetadata | None = None,
        target: str = "target",
        quantile_levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        eval_metric: Any = None,
    ):
        hyperparameters = hyperparameters if hyperparameters is not None else {}

        self.model_path = hyperparameters.get("model_path", self.default_model_path)

        super().__init__(
            path=path,
            name=name,
            hyperparameters=hyperparameters,
            freq=freq,
            prediction_length=prediction_length,
            covariate_metadata=covariate_metadata,
            target=target,
            quantile_levels=quantile_levels,
            eval_metric=eval_metric,
        )

        self._model: "ForecastModel | None" = None

    def save(self, path: str | None = None, verbose: bool = True) -> str:
        model = self._model
        self._model = None
        path = super().save(path=path, verbose=verbose)
        self._model = model

        return str(path)

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, load_oof: bool = False, verbose: bool = True) -> Self:
        model = load_pkl.load(path=os.path.join(path, cls.model_file_name), verbose=verbose)
        if reset_paths:
            model.set_contexts(path)

        return model

    def _is_gpu_available(self) -> bool:
        import torch.cuda

        return torch.cuda.is_available()

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        minimum_resources: dict[str, int | float] = {"num_cpus": 1}
        if is_gpu_available:
            minimum_resources["num_gpus"] = 1
        return minimum_resources

    def _get_device(self) -> str:
        device = self.get_hyperparameter("device")
        if device is None:
            device = "cuda" if self._is_gpu_available() else "cpu"
        return device

    def load_model(self):
        try:
            from tirex2 import load_model
        except ImportError as err:
            raise ImportError(
                f"{self.name} requires the `tirex-2` package to be installed. Because the PyPI release has very "
                "restrictive dependencies, we recommend installing it from source with "
                "`pip install 'tirex-2 @ git+https://github.com/NX-AI/tirex-2.git"
                "@6b232232e3150a7897f6d43f640d5a3928e2868e'`."
            ) from err

        self._model = load_model(self.model_path, device=self._get_device())

    def persist(self) -> Self:
        if self._model is None:
            self.load_model()
        return self

    def _get_default_hyperparameters(self) -> dict:
        return {
            "batch_size": 256,
            "device": None,
            "context_length": 2048,
            "use_target_encoding": True,
        }

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return super().allowed_hyperparameters + [
            "model_path",
            "batch_size",
            "device",
            "context_length",
            "use_target_encoding",
        ]

    def _more_tags(self) -> dict:
        return {
            "allow_nan": True,
            "can_use_train_data": False,
            "can_use_val_data": False,
        }

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None = None,
        time_limit: float | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        self._check_fit_params()
        self._log_unused_hyperparameters()
        self.load_model()

    @staticmethod
    def _interpolate_quantiles(
        quantile_forecast: np.ndarray, knots: np.ndarray, quantile_levels: np.ndarray
    ) -> np.ndarray:
        """Linearly interpolate the quantiles at the requested levels from the quantiles at the model's native knots.

        Levels outside the range spanned by ``knots`` are clipped to the extreme knots.

        Parameters
        ----------
        quantile_forecast
            Shape ``(num_rows, len(knots))`` array with the quantiles predicted at the native ``knots``.
        knots
            Native quantile levels produced by the model, sorted in ascending order.
        quantile_levels
            Quantile levels to interpolate.

        Returns
        -------
        Shape ``(num_rows, len(quantile_levels))`` array with the quantiles at the requested ``quantile_levels``.
        """
        idx_above = np.clip(np.searchsorted(knots, quantile_levels), 1, len(knots) - 1)
        knot_below, knot_above = knots[idx_above - 1], knots[idx_above]
        weight = np.clip((quantile_levels - knot_below) / (knot_above - knot_below), 0.0, 1.0)
        return quantile_forecast[:, idx_above - 1] * (1 - weight) + quantile_forecast[:, idx_above] * weight

    def _to_timeseries(
        self,
        data: TimeSeriesDataFrame,
        context_length: int,
        known_covariates: TimeSeriesDataFrame | None = None,
        use_target_encoding: bool = True,
    ) -> list:
        """Build a list of univariate ``TimeseriesType``, one per item, truncated to ``context_length``.

        Past covariates are passed as ``past_covariates`` of shape ``(n_past, context_length)``, and known covariates
        as ``future_covariates`` of shape ``(n_known, context_length + prediction_length)`` (the historical values
        from ``data`` concatenated with the future values from ``known_covariates``). Real-valued covariates are passed
        through and categorical covariates are encoded (target encoding if ``use_target_encoding`` else ordinal),
        reusing Chronos-2's ``preprocess.from_data_frame`` which produces one ``PreparedInput`` per item with the
        covariate rows ordered past-only first, known-future last. Encoding uses the full history; the context is
        truncated to ``context_length`` afterwards.
        """
        import torch
        from chronos.chronos2 import preprocess
        from tirex2 import TimeseriesType

        past_df = data.reset_index().to_data_frame()
        future_df = known_covariates.reset_index().to_data_frame() if known_covariates is not None else None

        # `from_data_frame` returns one PreparedInput per item, in sorted (item_id, timestamp) order, matching
        # `get_forecast_horizon_index(data)`. `validate_inputs=False` skips re-sorting: AG data is already sorted.
        prepared = preprocess.from_data_frame(
            past_df,
            target_columns=[self.target],
            prediction_length=self.prediction_length,
            future_df=future_df,
            use_target_encoding=use_target_encoding,
            validate_inputs=False,
        )

        timeseries = []
        for item in prepared:
            n_targets = item["n_targets"]
            n_covariates = item["n_covariates"]
            n_known = item["n_future_covariates"]
            n_past_only = n_covariates - n_known

            context = item["context"]  # (n_targets + n_covariates, series_length)
            future = item["future_covariates"]  # (n_targets + n_covariates, prediction_length)

            # Truncate the context (target and the historical part of every covariate) to the most recent steps.
            if context.shape[-1] > context_length:
                context = context[:, -context_length:]

            target = context[:n_targets].contiguous()

            past_covariates = None
            if n_past_only > 0:
                # (n_past, context_length)
                past_covariates = context[n_targets : n_targets + n_past_only].contiguous()

            future_covariates = None
            if n_known > 0:
                known_rows = slice(n_targets + n_past_only, n_targets + n_covariates)
                # Known covariates span context and horizon: historical values then future values.
                # (n_known, context_length + prediction_length)
                future_covariates = torch.cat([context[known_rows], future[known_rows]], dim=-1)

            timeseries.append(
                TimeseriesType(
                    target=target,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                )
            )
        return timeseries

    def _predict(
        self, data: TimeSeriesDataFrame, known_covariates: TimeSeriesDataFrame | None = None, **kwargs
    ) -> TimeSeriesDataFrame:
        if self._model is None:
            self.load_model()
        assert self._model is not None, "TiRex-2 model failed to load"

        hyperparameters = self.get_hyperparameters()
        timeseries = self._to_timeseries(
            data,
            context_length=hyperparameters["context_length"],
            known_covariates=known_covariates,
            use_target_encoding=hyperparameters["use_target_encoding"],
        )

        # Native quantile levels produced by TiRex-2, sorted in ascending order.
        knots = np.array([round(float(q), 6) for q in self._model.quantiles], dtype=np.float64)

        time_limit = kwargs.get("time_limit")
        start_time = time.monotonic()

        # `forecast` batches internally and halves the batch size on device OOM. Each yielded batch is a list of
        # per-series `(n_variates=1, len(knots), horizon)` arrays; we check the time limit between batches.
        forecast_per_item = []
        for batch in self._model.forecast(
            timeseries,
            prediction_length=self.prediction_length,
            output_type="numpy",
            batch_size=hyperparameters["batch_size"],
            yield_per_batch=True,
        ):
            for forecast in batch:
                # (n_variates=1, len(knots), horizon) -> (horizon, len(knots))
                forecast_per_item.append(forecast.squeeze(0).T.astype(np.float64))
            if time_limit is not None and time.monotonic() - start_time > time_limit:
                raise TimeLimitExceeded

        quantile_forecast = np.concatenate(forecast_per_item, axis=0)  # (num_items * horizon, len(knots))
        # The median is requested first and used as the mean forecast.
        interpolated = self._interpolate_quantiles(
            quantile_forecast=quantile_forecast,
            knots=knots,
            quantile_levels=np.array([0.5, *self.quantile_levels], dtype=np.float64),
        )

        predictions = {"mean": interpolated[:, 0]}
        predictions |= {str(q): interpolated[:, i + 1] for i, q in enumerate(self.quantile_levels)}

        df = pd.DataFrame(predictions, index=self.get_forecast_horizon_index(data))
        return TimeSeriesDataFrame(df)
