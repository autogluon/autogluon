from unittest.mock import Mock

import numpy as np
import pytest
import torch

from autogluon.timeseries.models.toto._internal.forecaster import Forecast
from autogluon.timeseries.models.toto.dataloader import (
    TotoDataLoader,
    TotoInferenceDataset,
    freq_to_seconds,
)

from ..common import get_data_frame_with_item_index


def noop():
    # pickleable no-op function
    pass


class MockTotoForecaster:
    def __init__(self):
        self.model = Mock()
        self.model.device = torch.device("cpu")

    def forecast(self, inputs, prediction_length, num_samples=None, samples_per_batch=32):
        inputs.series[~inputs.padding_mask] = torch.nan
        input_mean = torch.nanmean(inputs.series, dim=-1, keepdim=True)
        mean = torch.nan_to_num(input_mean.repeat(1, 1, prediction_length), nan=0.0)

        if num_samples is not None:
            samples = mean.unsqueeze(-1).repeat(1, 1, 1, num_samples)
        else:
            samples = None

        return Forecast(mean=mean, samples=samples)


class TestTotoDataset:
    @pytest.mark.parametrize(
        "input_freq, expected_freq, expected_num_seconds",
        [
            ("15min", "15min", 15 * 60),
            ("h", "h", 60 * 60),
            ("D", "D", 24 * 60 * 60),
        ],
    )
    def test_when_dataset_created_then_frequency_set_correctly(self, input_freq, expected_freq, expected_num_seconds):
        df = get_data_frame_with_item_index(["A", "B", "C", "D"], freq=input_freq, data_length=100)

        dset = TotoInferenceDataset(df, context_length=10)
        assert dset.freq == expected_freq
        assert freq_to_seconds(dset.freq) == expected_num_seconds  # type: ignore

    @pytest.mark.parametrize(
        "input_data_length, context_length",
        [
            (100, 10),
            (5, 10),
            (100, 100),
            (5, 100),
        ],
    )
    def test_when_dataset_iterated_then_context_has_correct_length(self, input_data_length, context_length):
        df = get_data_frame_with_item_index(["A", "B", "C", "D"], data_length=input_data_length)

        dset = TotoInferenceDataset(df, context_length=context_length)

        for i in range(len(dset)):
            assert len(dset[i]) == context_length

    @pytest.mark.parametrize(
        "input_data_length, context_length",
        [
            (2, 10),
            (5, 10),
            (20, 100),
            (5, 100),
        ],
    )
    def test_when_dataset_iterated_then_context_is_left_padded(self, input_data_length, context_length):
        df = get_data_frame_with_item_index(["A", "B", "C", "D"], data_length=input_data_length)

        dset = TotoInferenceDataset(df, context_length=context_length)

        for i in range(len(dset)):
            assert np.all(np.isnan(dset[i][: (context_length - input_data_length)]))


class TestTotoDataloader:
    @pytest.fixture(scope="class")
    def dataset(self):
        df = get_data_frame_with_item_index([f"item{x:03d}" for x in range(100)], data_length=100)
        return TotoInferenceDataset(df, context_length=32)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_dataloader_iterated_then_batches_are_on_correct_device(self, device, dataset):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip()

        loader = TotoDataLoader(dataset, batch_size=32, device=device)

        for masked_timeseries in loader:
            for tensor in [
                masked_timeseries.series,
                masked_timeseries.padding_mask,
                masked_timeseries.id_mask,
                masked_timeseries.timestamp_seconds,
                masked_timeseries.time_interval_seconds,
            ]:
                assert tensor.device == torch.device(device)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("batch_size", [4, 8, 16, 32])
    def test_when_dataloader_iterated_then_batches_have_correct_shape(self, device, batch_size, dataset):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip()

        loader = TotoDataLoader(dataset, batch_size=batch_size, device=device)

        context_length = 32
        nr_full_batches, remainder_batch_size = divmod(len(dataset), batch_size)
        expected_sizes = [batch_size] * nr_full_batches + ([remainder_batch_size] if remainder_batch_size > 0 else [])

        for masked_timeseries, expected_size in zip(loader, expected_sizes):
            assert masked_timeseries.series.shape == (expected_size, 1, context_length)
            assert masked_timeseries.padding_mask.shape == (
                expected_size,
                1,
                context_length,
            )
            assert masked_timeseries.id_mask.shape == (expected_size, 1, context_length)
            assert masked_timeseries.timestamp_seconds.shape == (
                expected_size,
                1,
                context_length,
            )
            assert masked_timeseries.time_interval_seconds.shape == (expected_size, 1)


class TestTotoModel:
    @pytest.mark.parametrize("num_items", [5, 100])
    @pytest.mark.parametrize("batch_size", [4, 32])
    def test_predict_returns_correct_format(self, num_items, batch_size):
        from unittest.mock import patch

        from autogluon.timeseries.models.toto import TotoModel

        item_index = [f"item{x:03d}" for x in range(num_items)]
        df = get_data_frame_with_item_index(item_index, data_length=50)  # type: ignore
        for i, item_id in enumerate(item_index):
            df.loc[item_id, "target"] = i + 1

        model = TotoModel(
            prediction_length=10,
            quantile_levels=[0.1, 0.5, 0.9],
            hyperparameters={"batch_size": batch_size},
        )

        with patch.object(model, "load_forecaster"), patch.object(model, "_forecaster", MockTotoForecaster()):
            predictions = model._predict(df)

            assert len(predictions) == num_items * 10
            assert list(predictions.columns) == ["mean", "0.1", "0.5", "0.9"]
            assert predictions.index.names == ["item_id", "timestamp"]

            assert np.allclose(
                np.repeat(np.arange(1, num_items + 1), 10),
                predictions["mean"],
            )
