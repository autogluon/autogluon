import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.registry import ag_model_registry
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.models.autogluon_tabular.transforms import MLForecastScaler, apply_inverse_transform
from autogluon.timeseries.utils.datetime import get_lags_for_frequency, get_time_features_for_frequency
from autogluon.timeseries.utils.forecast import make_future_data_frame

NIXTLA_TO_AG = {"unique_id": "item_id", "ds": "timestamp"}


class RegressionModel:
    def __init__(self, model_cls, quantile_levels, model_hyperparameters):
        self.model_cls = model_cls
        self.model_hyperparameters = model_hyperparameters
        self.quantile_levels = quantile_levels

    def fit(
        self,
        X,
        y,
        step,
        time_limit,
        num_cpus: int,
        val_frac: float | None = None,
    ):
        model = self.model_cls(path="", problem_type="regression", hyperparameters=self.model_hyperparameters)
        # model = self.model_cls(
        #     path="", problem_type="quantile", hyperparameters={"ag.quantile_levels": self.quantile_levels}
        # )
        y_is_valid = np.isfinite(y)
        X, y = X[y_is_valid], y[y_is_valid]
        if val_frac is None:
            X_val = None
            y_val = None
        else:
            num_val = int(len(X) * val_frac)
            X_val, y_val = X.iloc[-num_val:], y.iloc[-num_val:]
            X, y = X.iloc[:-num_val], y.iloc[:-num_val]
        if len(y) == 0:
            raise ValueError("Not enough valid target values to fit model")
        self.model = model.fit(
            X=X, y=y, X_val=X_val, y_val=y_val, time_limit=time_limit, num_cpus=num_cpus, num_gpus=0
        )
        # self.model.model.coef_ = np.zeros_like(self.model.model.coef_)
        # self.model.model.coef_[self.model.features.index(f"lag{24 - step}")] = 1.0
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        # return preds
        # Repeat model prediction for each quantile
        predictions = np.stack([preds for _ in self.quantile_levels], axis=1)
        return predictions


class PerStepTabular(AbstractTimeSeriesModel):
    @property
    def _ag_to_nixtla(self) -> dict:
        return {self.target: "y", "item_id": "unique_id", "timestamp": "ds"}

    def _initialize_transforms_and_regressor(self):
        super()._initialize_transforms_and_regressor()
        # Do not create a scaler in the model, scaler will be passed to MLForecast
        self.target_scaler = None

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None = None,
        time_limit: float | None = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        model_params = self.get_hyperparameters()

        offset = pd.tseries.frequencies.to_offset(self.freq)
        target_transforms = []
        differences = model_params.get("differences", [])
        if differences is not None and len(differences) > 0:
            target_transforms.append(Differences(differences))
        scaler_type = model_params.get("target_scaler", "mean_abs")
        if scaler_type is not None:
            self.scaler = MLForecastScaler(scaler_type=scaler_type)
            target_transforms.append(self.scaler)

        default_lags = get_lags_for_frequency(self.freq, lag_ub=int(train_data.num_timesteps_per_item().median()))
        if model_params.get("lags") is None:
            lags = default_lags
        else:
            lags = [int(lag) for lag in model_params.get("lags")]
        self._mlf = MLForecast(
            models=[],
            freq=self.freq,
            lags=lags,
            target_transforms=target_transforms,
        )
        self.time_features = get_time_features_for_frequency(self.freq)

        train_df = train_data.to_data_frame().reset_index().rename(columns=self._ag_to_nixtla)
        target_df = train_df[["unique_id", "ds", "y"]].assign(y=train_df["y"].fillna(float("inf")))
        covariates_df = train_df.drop(columns=["y"])
        timestamps = pd.DatetimeIndex(covariates_df["ds"])

        # Includes known covariates and time features
        covariates_df = covariates_df.assign(**{feat.__name__: feat(timestamps) for feat in self.time_features})
        if train_data.static_features is not None:
            covariates_df = pd.merge(
                left=covariates_df, right=train_data.static_features, left_on="unique_id", right_index=True
            )
        # Includes lag features
        features_df = self._mlf.preprocess(
            target_df,
            max_horizon=self.prediction_length,
            static_features=[],
            dropna=True,  # return_X_y=False
        )
        target_cols = [f"y{i}" for i in range(self.prediction_length)]
        lag_features_df = features_df.drop(columns=target_cols)
        stacked_targets = features_df[target_cols].to_numpy()
        lag_features_df = lag_features_df.replace(float("inf"), float("nan"))
        stacked_targets[np.isinf(stacked_targets)] = float("nan")

        X_y_per_step: list[tuple[pd.DataFrame, pd.Series]] = []
        for i in range(self.prediction_length):
            X_for_step = lag_features_df.copy()
            X_for_step["ds"] = pd.DatetimeIndex(X_for_step["ds"]) + i * offset
            X_for_step = X_for_step.merge(covariates_df, on=["unique_id", "ds"], how="left")
            y_for_step = pd.Series(stacked_targets[:, i])
            # Sort chronologically for efficient train/val split
            order = X_for_step["ds"].argsort().to_numpy()
            X_y_per_step.append((X_for_step.drop(columns=["unique_id", "ds"]).iloc[order], y_for_step.iloc[order]))

        model_cls = ag_model_registry.key_to_cls(model_params.get("model_name", "CAT"))
        model_hyperparameters = model_params.get("model_hyperparameters", {})
        # Estimate memory usage
        X, y = X_y_per_step[0]
        num_cpus = cpu_count(only_physical_cores=True)
        n_parallel_jobs = min(num_cpus, self.prediction_length)

        try:
            mem_usage = model_cls.estimate_memory_usage_static(X=X, y=y, problem_type="regression")
            n_parallel_jobs = min(n_parallel_jobs, ResourceManager.get_available_virtual_mem() // mem_usage)
        except:
            pass
        n_parallel_jobs = max(n_parallel_jobs, 1)

        if time_limit is not None:
            time_limit_per_model = time_limit * n_parallel_jobs / self.prediction_length
        else:
            time_limit_per_model = None
        fit_kwargs = dict(
            model_cls=model_cls,
            quantile_levels=self.quantile_levels,
            time_limit=time_limit_per_model,
            val_frac=model_params.get("val_frac", 0.1),
            num_cpus=max(num_cpus // n_parallel_jobs, 1),
            model_hyperparameters=model_hyperparameters.copy(),
        )
        print(f"Fitting with {n_parallel_jobs=}, {time_limit_per_model=}, num_cpus_per_model={fit_kwargs['num_cpus']}")
        self.model_per_horizon = Parallel(n_jobs=n_parallel_jobs)(
            delayed(self._fit_single_model)(X=X, y=y, step=h, **fit_kwargs) for h, (X, y) in enumerate(X_y_per_step)
        )

    @staticmethod
    def _fit_single_model(
        model_cls,
        model_hyperparameters,
        X,
        y,
        step,
        time_limit,
        val_frac,
        quantile_levels,
        num_cpus,
    ) -> RegressionModel:
        return RegressionModel(
            model_cls,
            model_hyperparameters=model_hyperparameters,
            quantile_levels=quantile_levels,
        ).fit(
            X=X,
            y=y,
            step=step,
            val_frac=val_frac,
            time_limit=time_limit,
            num_cpus=num_cpus,
        )

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        df = data.to_data_frame().reset_index().rename(columns=self._ag_to_nixtla)
        if known_covariates is not None:
            X_df = known_covariates.to_data_frame().reset_index()
        else:
            X_df = make_future_data_frame(data, prediction_length=self.prediction_length, freq=self.freq)
        X_df = X_df.rename(columns=self._ag_to_nixtla)
        timestamps = pd.DatetimeIndex(X_df["ds"])
        X_df = X_df.assign(**{feat.__name__: feat(timestamps) for feat in self.time_features})

        # df = df.assign(**{feat.__name__: feat(pd.DatetimeIndex(df["ds"])) for feat in self.time_features})
        # self._mlf.models_ = {"LR": self.model_per_horizon}
        # self._mlf.predict(h=self.prediction_length, new_df=df, X_df=X_df)

        if data.static_features is not None:
            X_df = pd.merge(left=X_df, right=data.static_features, left_on="unique_id", right_index=True)

        X_df_grouped = X_df.groupby("unique_id", sort=False, as_index=False)
        df = pd.concat([df[["unique_id", "ds", "y"]], X_df_grouped[["unique_id", "ds"]].head(1)], ignore_index=True)
        df = df.sort_values(by=["unique_id", "ds"])
        df = df.assign(y=df["y"].fillna(float("inf")))

        lags_for_prediction = (
            self._mlf.preprocess(df, static_features=[], dropna=True, max_horizon=self.prediction_length)
            .groupby("unique_id", sort=False, as_index=False)
            .tail(1)
        )
        lags_for_prediction = (
            lags_for_prediction.replace(float("inf"), float("nan"))
            .drop(columns=["unique_id", "ds"] + [c for c in lags_for_prediction.columns if c.startswith("y")])
            .reset_index(drop=True)
        )
        predictions_per_step = []
        for h in range(self.prediction_length):
            X = pd.concat([lags_for_prediction, X_df_grouped.nth(h).reset_index(drop=True)], axis=1)
            predictions_per_step.append(self.model_per_horizon[h].predict(X))
            # predictions_per_step.append(np.zeros([len(X), len(self.quantile_levels)]))
        predictions = X_df[["unique_id", "ds"]]
        pred_array = (
            np.stack(predictions_per_step, axis=0).transpose(1, 0, 2).reshape(-1, predictions_per_step[0].shape[1])
        )
        predictions = predictions.assign(**{str(q): pred_array[:, i] for i, q in enumerate(self.quantile_levels)})
        predictions["mean"] = predictions["0.5"]
        # predictions = self.scaler.inverse_transform(predictions)
        if hasattr(self._mlf.ts, "target_transforms"):
            for tfm in self._mlf.ts.target_transforms[::-1]:
                predictions = apply_inverse_transform(predictions, transform=tfm)
        col_order = ["mean"] + [str(q) for q in self.quantile_levels]
        return TimeSeriesDataFrame(predictions.rename(columns=NIXTLA_TO_AG))[col_order]


if __name__ == "__main__":
    full_data = TimeSeriesDataFrame("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/test.csv")

    full_data = full_data.loc[full_data.item_ids[[0]]]
    full_data["target"] = np.sin(np.arange(len(full_data)) * np.pi / 12)
    prediction_length = 24
    target = "target"
    # prediction_length = 8
    # target = "unit_sales"
    known_covariates_names = list(full_data.columns.drop(target))
    train_data, test_data = full_data.train_test_split(prediction_length)
    # lags = np.array(get_lags_for_frequency(full_data.freq, num_default_lags=24))
    lags = list(range(1, 25))

    predictor = TimeSeriesPredictor(
        target=target,
        prediction_length=prediction_length,
        known_covariates_names=known_covariates_names,
        eval_metric="SQL",
        quantile_levels=[0.501, 0.5],  # 0.9],
    ).fit(
        train_data,
        hyperparameters={
            # "DirectTabular": {},
            # "RecursiveTabular": {},
            PerStepTabular: {
                "model_name": "LR",
                # "val_frac": 0.1,
                "target_scaler": None,
                "lags": lags,
                # "differences": [24],
            },
            # "SeasonalNaive": {},
        },
        time_limit=40,
        num_val_windows=3,
        refit_every_n_windows=None,
        enable_ensemble=False,
    )
    predictor.leaderboard(test_data, display=True)
