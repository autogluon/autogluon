import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.t0 import T0Model
from autogluon.timeseries.models.t0.dataloader import T0DataLoader, T0InferenceDataset
from autogluon.timeseries.utils.features import CovariateMetadata

from ..common import get_data_frame_with_item_index, get_data_frame_with_variable_lengths


def noop():
    # Pickleable no-op used by the shared model tests.
    pass


class MockT0Forecaster:
    def __init__(self):
        self.last_quantiles = None
        self.last_context = None
        self.last_future_covariates = None
        self.last_model_input = None
        self.last_inference_kwargs = None
        self.last_past_covariates = None

    def predict(self, model_input, horizon, quantile_levels, *, context_length, **inference_kwargs):
        from t0.data import MaskType, VariateType

        self.last_quantiles = quantile_levels
        self.last_model_input = model_input
        self.last_inference_kwargs = inference_kwargs
        row_types = model_input.variate_type[:, -1]
        target_rows = row_types == VariateType.TARGET
        future_rows = row_types == VariateType.FUTURE
        historical_rows = row_types == VariateType.HISTORICAL
        target = model_input.variates[target_rows, :context_length]
        target_mask = model_input.mask[target_rows, :context_length]
        self.last_group_ids = model_input.group_ids[:, -1]
        # Every item has exactly one target row; cross learning may put several items in one group.
        num_items = int(target_rows.sum())
        self.last_context = target.unflatten(0, (num_items, -1))
        self.last_future_covariates = (
            model_input.variates[future_rows].unflatten(0, (num_items, -1)) if future_rows.any() else None
        )
        self.last_past_covariates = (
            model_input.variates[historical_rows, :context_length].unflatten(0, (num_items, -1))
            if historical_rows.any()
            else None
        )
        observed = target.masked_fill(target_mask != MaskType.VALID, torch.nan)
        mean = torch.nan_to_num(torch.nanmean(observed, dim=-1), nan=0.0)
        quantile_forecast = mean[:, None, None].repeat(1, horizon, len(quantile_levels))
        return Mock(quantiles=quantile_forecast)


class OutOfMemoryT0Forecaster(MockT0Forecaster):
    """Raises ``error`` for any batch holding more than ``max_items`` items."""

    def __init__(self, max_items: int, error: RuntimeError):
        super().__init__()
        self.max_items = max_items
        self.error = error
        self.batch_sizes = []

    def predict(self, model_input, *args, **kwargs):
        from t0.data import VariateType

        num_items = int((model_input.variate_type[:, -1] == VariateType.TARGET).sum())
        self.batch_sizes.append(num_items)
        if num_items > self.max_items:
            raise self.error
        return super().predict(model_input, *args, **kwargs)


def make_future_covariates(data: TimeSeriesDataFrame, prediction_length: int, values: list[float]):
    index = []
    for item_id in data.item_ids:
        last_timestamp = data.loc[item_id].index[-1]
        future_timestamps = pd.date_range(last_timestamp, periods=prediction_length + 1, freq=data.freq)[1:]
        index.extend((item_id, timestamp) for timestamp in future_timestamps)
    return TimeSeriesDataFrame(
        pd.DataFrame(
            {"known": values},
            index=pd.MultiIndex.from_tuples(index, names=["item_id", "timestamp"]),
        )
    )


class TestT0Dataset:
    @pytest.mark.parametrize(
        "input_data_length, context_length",
        [
            (100, 10),
            (100, 100),
            (5, 100),
        ],
    )
    def test_when_dataset_iterated_then_context_has_correct_length(self, input_data_length, context_length):
        df = get_data_frame_with_item_index(["A", "B", "C", "D"], data_length=input_data_length)
        dataset = T0InferenceDataset(df, max_context_length=context_length)

        for i in range(len(dataset)):
            model_input, item_context_length = dataset[i]
            assert model_input.variates.shape == (1, min(context_length, input_data_length))
            assert item_context_length == min(context_length, input_data_length)

    @pytest.mark.parametrize("max_data_length", [10, 100])
    def test_when_dataset_with_uneven_lengths_iterated_then_items_have_correct_length(self, max_data_length):
        item_id_to_length = {"A": 1, "B": max_data_length // 2, "C": max_data_length}
        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
        dataset = T0InferenceDataset(df, max_context_length=max_data_length)

        for i, item_length in zip(range(len(dataset)), item_id_to_length.values()):
            model_input, context_length = dataset[i]
            assert model_input.variates.shape == (1, item_length)
            assert context_length == item_length

    def test_dataset_keeps_known_covariates_split_until_t0_input_is_created(self):
        df = get_data_frame_with_item_index(["A"], data_length=4)
        df["past"] = [10.0, 11.0, float("nan"), 13.0]
        df["known"] = [20.0, 21.0, 22.0, 23.0]
        future = make_future_covariates(df, prediction_length=2, values=[24.0, 25.0])

        dataset = T0InferenceDataset(
            df,
            max_context_length=3,
            past_covariate_columns=["past"],
            known_covariate_columns=["known"],
            known_covariates=future,
            prediction_length=2,
        )

        from t0.data import MaskType, VariateType

        model_input, context_length = dataset[0]
        assert context_length == 3
        # Rows: target, known covariate, past covariate.
        assert np.allclose(model_input.variates[0, :3], df["target"].to_numpy()[-3:])
        assert np.allclose(dataset.past_known_covariate_array.T, [[20.0, 21.0, 22.0, 23.0]])
        assert np.allclose(dataset.future_known_covariate_array.T, [[24.0, 25.0]])
        assert np.allclose(model_input.variates[1], [21.0, 22.0, 23.0, 24.0, 25.0])
        assert np.allclose(model_input.variates[2, :3], [11.0, 0.0, 13.0])
        assert model_input.mask[0].tolist() == [
            MaskType.VALID,
            MaskType.VALID,
            MaskType.VALID,
            MaskType.WITHHELD,
            MaskType.WITHHELD,
        ]
        assert model_input.mask[2, :3].tolist() == [MaskType.VALID, MaskType.MISSING, MaskType.VALID]
        assert model_input.variate_type[:, -1].tolist() == [
            VariateType.TARGET,
            VariateType.FUTURE,
            VariateType.HISTORICAL,
        ]
        assert model_input.group_ids[:, -1].tolist() == [0, 0, 0]


class TestT0DataLoader:
    @pytest.mark.parametrize("num_past, num_known", [(0, 0), (2, 0), (0, 2), (2, 2)])
    @pytest.mark.parametrize("context_length", [3, 100])
    @pytest.mark.parametrize("cross_learning", [False, True])
    @pytest.mark.parametrize("start, stop", [(0, 3), (1, 3)])
    def test_direct_batch_matches_native_t0_batch(
        self, num_past, num_known, context_length, cross_learning, start, stop
    ):
        from t0 import TimeSeries

        df = get_data_frame_with_variable_lengths({"A": 12, "B": 1, "C": 7})
        df.iloc[0, 0] = np.nan
        past_columns = [f"past_{i}" for i in range(num_past)]
        known_columns = [f"known_{i}" for i in range(num_known)]
        for i, column in enumerate(past_columns + known_columns):
            df[column] = np.arange(len(df), dtype=np.float32) + i
            df.loc[("B", df.loc["B"].index[0]), column] = np.nan
        future = make_future_covariates(df, prediction_length=2, values=[np.nan, 2, 3, 4, 5, 6])
        for i, column in enumerate(known_columns):
            future[column] = future["known"] + i
        dataset = T0InferenceDataset(
            df,
            max_context_length=context_length,
            past_covariate_columns=past_columns,
            known_covariate_columns=known_columns,
            known_covariates=future if num_known else None,
            prediction_length=2,
        )
        expected = TimeSeries.batch([dataset[i][0] for i in range(start, stop)])
        if cross_learning:
            expected.group_ids = torch.where(expected.group_ids >= 0, 0, -1)

        indices, actual, actual_context_length = T0DataLoader(
            dataset, device="cpu", cross_learning=cross_learning
        ).make_batch(start, stop)

        assert indices == list(range(start, stop))
        assert actual_context_length == min(context_length, max([12, 1, 7][start:stop]))
        for field in ("variates", "mask", "group_ids", "variate_type"):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_batch_made_then_it_is_on_correct_device(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        df = get_data_frame_with_item_index([f"item{x:03d}" for x in range(50)], data_length=100)
        dataset = T0InferenceDataset(df, max_context_length=100)
        loader = T0DataLoader(dataset, device=device)

        indices, model_input, context_length = loader.make_batch(0, 32)
        assert indices == list(range(32))
        assert model_input.device == torch.device(device)
        assert context_length == 100

    def test_when_lengths_are_uneven_then_t0_batches_with_padding(self):
        from t0 import MaskType
        from t0.data import VariateType

        item_id_to_length = {"A": 20, "B": 5}
        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
        dataset = T0InferenceDataset(df, max_context_length=100)
        loader = T0DataLoader(dataset, device="cpu")

        indices, model_input, context_length = loader.make_batch(0, 2)
        assert indices == [0, 1]
        assert context_length == 20
        assert model_input.variates.shape == (2, 20)
        assert torch.all(model_input.mask[0] == MaskType.VALID)
        assert model_input.mask[1].tolist() == [MaskType.PAD] * 15 + [MaskType.VALID] * 5
        assert model_input.group_ids[1].tolist() == [-1] * 15 + [1] * 5
        assert model_input.variate_type[1].tolist() == [-1] * 15 + [VariateType.TARGET] * 5

    def test_when_cross_learning_set_then_batch_items_share_a_group(self):
        from t0 import MaskType
        from t0.data import VariateType

        item_id_to_length = {"A": 20, "B": 5, "C": 8, "D": 6}
        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
        df["past"] = 1.0
        dataset = T0InferenceDataset(df, max_context_length=100, past_covariate_columns=["past"])
        loader = T0DataLoader(dataset, device="cpu", cross_learning=True)

        indices, model_input, context_length = loader.make_batch(0, 3)
        assert indices == [0, 1, 2]
        assert context_length == 20
        # Rows: A target, A past, B target, B past, C target, C past.
        assert model_input.variate_type[:, -1].tolist() == [VariateType.TARGET, VariateType.HISTORICAL] * 3
        assert model_input.group_ids[:, -1].tolist() == [0] * 6
        assert model_input.mask[2].tolist() == [MaskType.PAD] * 15 + [MaskType.VALID] * 5
        assert model_input.group_ids[2].tolist() == [-1] * 15 + [0] * 5
        # A group never spans batches: D alone in the next batch is its own group.
        indices, model_input, _ = loader.make_batch(3, 4)
        assert indices == [3]
        assert model_input.group_ids[:, -1].tolist() == [0, 0]

    def test_when_cross_learning_disabled_then_each_item_is_its_own_group(self):
        df = get_data_frame_with_item_index(["A", "B", "C"], data_length=10)
        dataset = T0InferenceDataset(df, max_context_length=100)
        loader = T0DataLoader(dataset, device="cpu", cross_learning=False)

        _, model_input, _ = loader.make_batch(0, 3)
        assert model_input.group_ids[:, -1].tolist() == [0, 1, 2]

    def test_when_target_has_missing_values_then_they_are_marked_as_missing(self):
        from t0 import MaskType

        df = get_data_frame_with_item_index(["A"], data_length=4)
        df["target"] = [1.0, float("nan"), 3.0, float("nan")]
        dataset = T0InferenceDataset(df, max_context_length=100)

        _, model_input, _ = T0DataLoader(dataset, device="cpu").make_batch(0, 1)
        assert torch.isfinite(model_input.variates).all()
        assert model_input.mask[0].tolist() == [
            MaskType.VALID,
            MaskType.MISSING,
            MaskType.VALID,
            MaskType.MISSING,
        ]


class TestT0Model:
    def test_autogluon_imports_without_optional_t0_runtime(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                'import sys; sys.modules["t0"] = None\n'
                "from autogluon.timeseries import TimeSeriesPredictor\n"
                "from autogluon.timeseries.models import T0Model\n"
                "try:\n"
                "    T0Model().load_model()\n"
                "except ImportError as error:\n"
                '    assert "autogluon.timeseries[t0]" in str(error)\n'
                "else:\n"
                '    raise AssertionError("Missing runtime must raise ImportError")\n',
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_load_model_uses_checkpoint_and_device(self):
        loaded_model = Mock()
        loaded_model.eval.return_value = loaded_model
        loaded_model.to.return_value = loaded_model
        model = T0Model(hyperparameters={"model_path": "t0-gamma", "device": "cpu"})

        with patch("t0.T0Forecaster.from_pretrained", return_value=loaded_model) as from_pretrained:
            model.load_model()

        from_pretrained.assert_called_once_with("theforecastingcompany/t0-gamma")
        loaded_model.eval.assert_called_once_with()
        loaded_model.to.assert_called_once_with("cpu")
        assert model._model is loaded_model

    @pytest.mark.parametrize(
        "model_path, expected_path",
        [
            (None, "theforecastingcompany/t0-gamma"),
            ("t0-gamma", "theforecastingcompany/t0-gamma"),
            ("theforecastingcompany/t0-gamma", "theforecastingcompany/t0-gamma"),
            ("/local/checkpoint", "/local/checkpoint"),
        ],
    )
    def test_model_path_resolves_aliases(self, model_path, expected_path):
        hyperparameters = {} if model_path is None else {"model_path": model_path}
        assert T0Model(hyperparameters=hyperparameters).model_path == expected_path

    def test_recommended_inference_settings_are_passed_to_predict(self):
        df = get_data_frame_with_item_index(["A"], data_length=10)
        mock_model = MockT0Forecaster()
        model = T0Model(prediction_length=2, hyperparameters={"device": "cpu", "snap_integers": False})

        with patch.object(model, "load_model"), patch.object(model, "_model", mock_model):
            model._predict(df)

        assert mock_model.last_inference_kwargs == {
            "trim_leading_missing": True,
            "symmetric_averaging": True,
            "clip_nonnegative": True,
            "snap_integers": False,
        }

    @pytest.mark.parametrize(
        "hyperparameters, expected_group_ids",
        [
            ({}, [0, 1, 2, 3, 4]),
            ({"cross_learning": True}, [0, 0, 0, 0, 0]),
            # The mock keeps the last batch only: item E, alone in the third batch of two.
            ({"cross_learning": True, "batch_size": 2}, [0]),
        ],
    )
    def test_cross_learning_groups_items_and_preserves_order(self, hyperparameters, expected_group_ids):
        item_id_to_length = {"A": 10, "B": 5, "C": 7, "D": 9, "E": 6}
        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
        for value, item_id in enumerate(item_id_to_length, start=1):
            df.loc[item_id, "target"] = value
        mock_model = MockT0Forecaster()
        model = T0Model(prediction_length=2, hyperparameters={"device": "cpu", **hyperparameters})

        with patch.object(model, "load_model"), patch.object(model, "_model", mock_model):
            predictions = model._predict(df)

        assert mock_model.last_group_ids.tolist() == expected_group_ids
        assert np.allclose(predictions["mean"], np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 2))

    def test_cross_learning_changes_real_t0_forecasts(self):
        from t0 import T0Config, T0Forecaster

        torch.manual_seed(0)
        tiny_model = T0Forecaster.from_config(
            T0Config(
                embed_dim=32,
                num_layers=2,
                num_heads=4,
                mlp_hidden_dim=64,
                patch_size=4,
                group_every_n=1,
                dropout=0.0,
                quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
            )
        ).eval()
        df = get_data_frame_with_variable_lengths({"A": 12, "B": 7, "C": 9})

        predictions = {}
        for cross_learning in (False, True):
            model = T0Model(
                prediction_length=3,
                hyperparameters={"device": "cpu", "cross_learning": cross_learning},
            )
            with patch.object(model, "load_model"), patch.object(model, "_model", tiny_model):
                predictions[cross_learning] = model._predict(df).to_numpy()

        assert np.isfinite(predictions[True]).all()
        assert not np.allclose(predictions[False], predictions[True])

    def test_predict_with_tiny_real_t0_model(self):
        from t0 import T0Config, T0Forecaster

        torch.manual_seed(0)
        tiny_model = T0Forecaster.from_config(
            T0Config(
                embed_dim=32,
                num_layers=2,
                num_heads=4,
                mlp_hidden_dim=64,
                patch_size=4,
                group_every_n=1,
                dropout=0.0,
                quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
                scaler_use_arcsinh=True,
            )
        ).eval()
        df = get_data_frame_with_variable_lengths({"A": 12, "B": 7})
        df.loc[("A", df.loc["A"].index[3]), "target"] = float("nan")
        df["past"] = np.arange(len(df), dtype=np.float32)
        df["known"] = np.arange(len(df), dtype=np.float32) + 100
        future = make_future_covariates(
            df,
            prediction_length=3,
            values=[200.0, 201.0, 202.0, 300.0, 301.0, 302.0],
        )
        model = T0Model(
            prediction_length=3,
            quantile_levels=[0.1, 0.5, 0.9],
            hyperparameters={"batch_size": 2, "device": "cpu"},
            covariate_metadata=CovariateMetadata(past_covariates_real=["past"], known_covariates_real=["known"]),
        )

        with patch.object(model, "load_model"), patch.object(model, "_model", tiny_model):
            predictions = model._predict(df, known_covariates=future)

        assert predictions.shape == (6, 4)
        assert np.isfinite(predictions.to_numpy()).all()

    @pytest.mark.parametrize("num_items", [5, 23])
    @pytest.mark.parametrize("batch_size", [4, 32])
    def test_predict_returns_correct_format(self, num_items, batch_size):
        item_index = [f"item{x:03d}" for x in range(num_items)]
        df = get_data_frame_with_item_index(item_index, data_length=50)
        for i, item_id in enumerate(item_index):
            df.loc[item_id, "target"] = i + 1

        model = T0Model(
            prediction_length=10,
            quantile_levels=[0.1, 0.5, 0.9],
            hyperparameters={"batch_size": batch_size, "device": "cpu"},
        )

        with patch.object(model, "load_model"), patch.object(model, "_model", MockT0Forecaster()):
            predictions = model._predict(df)

        assert len(predictions) == num_items * 10
        assert list(predictions.columns) == ["mean", "0.1", "0.5", "0.9"]
        assert predictions.index.names == ["item_id", "timestamp"]
        assert np.allclose(np.repeat(np.arange(1, num_items + 1), 10), predictions["mean"])

    @pytest.mark.parametrize(
        "error",
        [
            torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
            RuntimeError("MPS backend out of memory (MPS allocated: 9.00 GB)"),
        ],
    )
    def test_when_batch_runs_out_of_memory_then_batch_size_is_halved(self, error):
        item_index = [f"item{x:02d}" for x in range(10)]
        df = get_data_frame_with_item_index(item_index, data_length=20)
        for i, item_id in enumerate(item_index):
            df.loc[item_id, "target"] = i + 1
        mock_model = OutOfMemoryT0Forecaster(max_items=3, error=error)
        model = T0Model(prediction_length=2, hyperparameters={"batch_size": 8, "device": "cpu"})

        with patch.object(model, "load_model"), patch.object(model, "_model", mock_model):
            predictions = model._predict(df)

        # 8 fails, 4 fails, then batches of 2 cover the 10 items without retrying 8 again.
        assert mock_model.batch_sizes == [8, 4, 2, 2, 2, 2, 2]
        assert np.allclose(predictions["mean"], np.repeat(np.arange(1, 11), 2))

    @pytest.mark.parametrize(
        "error, batch_size",
        [
            (RuntimeError("shape mismatch"), 8),
            (torch.cuda.OutOfMemoryError("CUDA out of memory"), 1),
        ],
    )
    def test_when_error_is_not_recoverable_then_it_is_raised(self, error, batch_size):
        df = get_data_frame_with_item_index(["A", "B"], data_length=20)
        mock_model = OutOfMemoryT0Forecaster(max_items=0, error=error)
        model = T0Model(prediction_length=2, hyperparameters={"batch_size": batch_size, "device": "cpu"})

        with patch.object(model, "load_model"), patch.object(model, "_model", mock_model):
            with pytest.raises(type(error), match=str(error)):
                model._predict(df)
        assert mock_model.batch_sizes == [min(batch_size, 2)]

    def test_when_cross_learning_batch_runs_out_of_memory_then_it_is_not_retried(self):
        df = get_data_frame_with_item_index([f"item{x:02d}" for x in range(10)], data_length=20)
        mock_model = OutOfMemoryT0Forecaster(max_items=3, error=torch.cuda.OutOfMemoryError("CUDA out of memory"))
        model = T0Model(
            prediction_length=2, hyperparameters={"batch_size": 8, "device": "cpu", "cross_learning": True}
        )

        with patch.object(model, "load_model"), patch.object(model, "_model", mock_model):
            with pytest.raises(RuntimeError, match="Lower batch_size"):
                model._predict(df)
        assert mock_model.batch_sizes == [8]

    def test_when_batch_runs_out_of_memory_after_time_limit_then_it_is_not_retried(self):
        df = get_data_frame_with_item_index([f"item{x:02d}" for x in range(10)], data_length=20)
        mock_model = OutOfMemoryT0Forecaster(max_items=3, error=torch.cuda.OutOfMemoryError("CUDA out of memory"))
        model = T0Model(prediction_length=2, hyperparameters={"batch_size": 8, "device": "cpu"})

        with patch.object(model, "load_model"), patch.object(model, "_model", mock_model):
            with pytest.raises(TimeLimitExceeded):
                # A negative limit has always passed, so the deadline check before the retry fires.
                model._predict(df, time_limit=-1)
        assert mock_model.batch_sizes == [8]

    def test_median_is_requested_for_mean_when_not_in_quantile_levels(self):
        df = get_data_frame_with_item_index(["A"], data_length=10)
        mock_model = MockT0Forecaster()
        model = T0Model(
            prediction_length=2,
            quantile_levels=[0.1, 0.9],
            hyperparameters={"device": "cpu"},
        )

        with patch.object(model, "load_model"), patch.object(model, "_model", mock_model):
            predictions = model._predict(df)

        assert list(predictions.columns) == ["mean", "0.1", "0.5", "0.9"]
        assert mock_model.last_quantiles == [0.1, 0.5, 0.9]

    def test_predictions_preserve_item_order_when_context_lengths_differ(self):
        item_id_to_length = {"A": 10, "B": 5, "C": 7}
        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
        for value, item_id in enumerate(item_id_to_length, start=1):
            df.loc[item_id, "target"] = value

        model = T0Model(prediction_length=2, hyperparameters={"batch_size": 3, "device": "cpu"})
        with patch.object(model, "load_model"), patch.object(model, "_model", MockT0Forecaster()):
            predictions = model._predict(df)

        assert np.allclose(predictions["mean"], np.repeat([1.0, 2.0, 3.0], 2))

    def test_predict_uses_numeric_past_and_known_covariates(self):
        df = get_data_frame_with_item_index(["A", "B"], data_length=4)
        df.loc["A", "target"] = 1.0
        df.loc["B", "target"] = 2.0
        df["past"] = np.arange(len(df), dtype=np.float32) + 10
        df["known"] = np.arange(len(df), dtype=np.float32) + 20
        future = make_future_covariates(df, prediction_length=2, values=[30.0, 31.0, 40.0, 41.0])
        mock_model = MockT0Forecaster()
        model = T0Model(
            prediction_length=2,
            hyperparameters={"batch_size": 2, "device": "cpu"},
            covariate_metadata=CovariateMetadata(past_covariates_real=["past"], known_covariates_real=["known"]),
        )

        with patch.object(model, "load_model"), patch.object(model, "_model", mock_model):
            model.fit(df)
            predictions = model.predict(df, known_covariates=future)

        assert model.supports_past_covariates
        assert model.supports_known_covariates
        assert mock_model.last_context.shape == (2, 1, 4)
        assert mock_model.last_past_covariates.shape == (2, 1, 4)
        assert mock_model.last_future_covariates.shape == (2, 1, 6)
        assert torch.allclose(mock_model.last_context[:, 0].mean(dim=-1), torch.tensor([1.0, 2.0]))
        assert torch.allclose(mock_model.last_past_covariates[1, 0], torch.tensor([14.0, 15.0, 16.0, 17.0]))
        assert torch.allclose(mock_model.last_future_covariates[0, 0, -2:], torch.tensor([30.0, 31.0]))
        assert np.allclose(predictions["mean"], np.repeat([1.0, 2.0], 2))

    def test_predict_does_not_use_disabled_covariates(self):
        df = get_data_frame_with_item_index(["A"], data_length=4)
        df["past"] = 10.0
        df["known"] = 20.0
        mock_model = MockT0Forecaster()
        model = T0Model(
            prediction_length=2,
            hyperparameters={
                "device": "cpu",
                "disable_past_covariates": True,
                "disable_known_covariates": True,
            },
            covariate_metadata=CovariateMetadata(past_covariates_real=["past"], known_covariates_real=["known"]),
        )

        with patch.object(model, "load_model"), patch.object(model, "_model", mock_model):
            model._predict(df)

        assert mock_model.last_context.shape == (1, 1, 4)
        assert mock_model.last_past_covariates is None
        assert mock_model.last_future_covariates is None
