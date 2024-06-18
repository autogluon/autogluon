import logging
import os
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
from autogluon.timeseries.models.chronos.model import MODEL_ALIASES
from autogluon.timeseries.models.chronos.pipeline import ChronosPipeline
from autogluon.timeseries.models.chronos.utils import (
    ChronosInferenceDataLoader,
    ChronosInferenceDataset,
    timeout_callback,
)

logger = logging.getLogger(__name__)


class ChronosAugmentedGeneration(AbstractTimeSeriesModel):
    # default number of samples for prediction
    default_num_samples: int = 20
    default_model_path = "autogluon/chronos-t5-mini"
    default_torch_dtype = "bfloat16"
    default_device = "cpu"

    maximum_context_length = 512
    default_batch_size = 8
    default_use_raw_features = True

    lgbm_default_params = {
        "n_estimators": 300,
        "n_jobs": -1,
        "random_state": 0,
    }

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: str = None,
        hyperparameters: Dict[str, Any] = None,
        **kwargs,  # noqa
    ):

        self._feature_generator = None
        self.embedding_model = None
        self.head_model = None
        model_path_input = hyperparameters.get("model_path", self.default_model_path)
        self.model_path = MODEL_ALIASES.get(model_path_input, model_path_input)
        self.batch_size = hyperparameters.get("batch_size", self.default_batch_size)
        self.num_samples = hyperparameters.get("num_samples", self.default_num_samples)
        self.device = hyperparameters.get("device", self.default_device)
        self.use_raw_features = hyperparameters.get("use_raw_features", self.default_use_raw_features)

        # if the model requires a GPU, set the torch dtype to bfloat16
        self.torch_dtype = hyperparameters.get("torch_dtype", self.default_torch_dtype)

        self.data_loader_num_workers = hyperparameters.get("data_loader_num_workers", 0)
        self.optimization_strategy: Optional[Literal["onnx", "openvino"]] = hyperparameters.get(
            "optimization_strategy"
        )
        self.context_length = hyperparameters.get("context_length", self.maximum_context_length)
        if self.context_length is not None and self.context_length > self.maximum_context_length:
            logger.warning(
                f"\tContext length {self.context_length} exceeds maximum context length {self.maximum_context_length}."
                f"Context length will be set to {self.maximum_context_length}."
            )
            self.context_length = self.maximum_context_length

        # we truncate the name to avoid long path errors on Windows
        model_path_safe = str(model_path_input).replace("/", "__").replace(os.path.sep, "__")[-50:]
        name = (name if name is not None else "Chronos") + f"[{model_path_safe}]"
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self.time_limit: Optional[float] = None

        self.embedding_model = ChronosPipeline.from_pretrained(
            self.model_path, device_map=hyperparameters["device_map"]
        )

        lgbm_params = kwargs.get("lgbm_params", self.lgbm_default_params)

        lgbm = LGBModel()  # TODO: this is the correct lgbm to use in autogluon, get train_data as input
        lgbm = LGBMRegressor(
            **lgbm_params
        )  # TODO: this is the incorrect lgbm to use in autogluon, but get X and y inputs
        self.head_model = MultiOutputRegressor(lgbm)

    def _preprocess(
        self,
        train_data: TimeSeriesDataFrame,
        is_train,
    ) -> np.ndarray:

        # TODO: it only embedds the target variable - need to loop over target_column
        chronos_dataset = ChronosInferenceDataset(
            train_data, context_length=self.context_length, target_column="target"
        )

        inference_data_loader = ChronosInferenceDataLoader(
            chronos_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.data_loader_num_workers,
            on_batch=timeout_callback(seconds=self.time_limit),
        )
        scaled_embedding_list = []
        raw_features_list = []
        for batch in inference_data_loader:
            # TODO: this section only emebbeed the univariate target variable
            ts_raw_embedding, ts_scale = self.embedding_model.embed(context=batch)
            # TODO: this section only emebbeed the univariate target variable
            mask = ~torch.isnan(batch)
            max_seq_lens = mask.sum(dim=1)
            ts_raw_embedding_no_eot = ts_raw_embedding[:, :-1, :]
            unpacked_embeddings = ts_raw_embedding_no_eot[
                mask.unsqueeze(-1).expand_as(ts_raw_embedding_no_eot)
            ].reshape(mask.shape[0], -1, ts_raw_embedding_no_eot.shape[-1])
            ts_nopadding_embedding = unpacked_embeddings[:, : max_seq_lens.max()]
            ts_clean_embedding = (
                ts_nopadding_embedding.detach().cpu().numpy()
            )  # need to remove the last as it is EOT symbol
            ts_scale = ts_scale.detach().cpu().numpy()
            embedding_mean = ts_clean_embedding.mean(axis=1)
            scaled_embedding = np.concatenate((ts_scale.reshape(-1, 1), embedding_mean), axis=1)
            scaled_embedding_list.extend(scaled_embedding)

            # Select non-NaN values from the batch
            batch_without_padding = torch.masked_select(batch, mask).view(-1, mask.sum(dim=1).max())
            raw_features_list.extend(batch_without_padding.detach().cpu().numpy())

        scaled_embedding_np = np.array(scaled_embedding_list)
        raw_features_np = np.array(raw_features_list)

        if self.use_raw_features:
            X_concat = np.concatenate([raw_features_np, scaled_embedding_np], axis=1)
        else:
            X_concat = scaled_embedding_np

        X = np.nan_to_num(X_concat).astype(np.float32)

        return X

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[int] = None,
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        print("Entering the `_fit` method")
        self.time_limit = time_limit

        # train_data - long dataframe
        # X_train = train_data.slice_by_timestep(None, -self.prediction_length)
        X = self._preprocess(train_data, is_train=True)
        y_train = train_data.slice_by_timestep(-self.prediction_length, None)["target"].values
        y = y_train.reshape(X.shape[0], self.prediction_length)

        self.head_model.fit(X, y)
        print("Exiting the `_fit` method")

    def _predict(
        self,
        data: Union[TimeSeriesDataFrame, Dict[str, TimeSeriesDataFrame]],
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:

        pass


def main():
    # Load the multivariate time series dataset
    prediction_length = 24
    id_column = "item_id"
    timestamp_column = "timestamp"
    target_column = "target"
    parquet_file_path = "./wallmart_data.parquet"
    cag_hyperparameters_dict = dict(data_loader_num_workers=2, device_map="cuda")

    df_wallmart = pd.read_parquet(parquet_file_path)
    tsdata_wallmart = TimeSeriesDataFrame.from_data_frame(
        df_wallmart,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    # Split the data into train and validation sets
    train_data, val_data = tsdata_wallmart.train_test_split(prediction_length=prediction_length)
    # make the validation set the same size as the train set on the time split to fit a tabular classifier
    y_val = val_data.slice_by_timestep(-prediction_length, None)[target_column]

    cag_model = ChronosAugmentedGeneration(
        target="target",
        path="autogluon/chronos-t5-base",  # Use the base model
        prediction_length=prediction_length,
        freq="W",
        eval_metric="RMSE",
        hyperparameters=cag_hyperparameters_dict,
    )

    cag_model.fit(train_data)
    val_pred = cag_model.predict(val_data)
    evaluation_scores = cag_model.evaluate(y_val, val_pred)
    print(f"Evaluation scores: {evaluation_scores}")


if __name__ == "__main__":
    main()
