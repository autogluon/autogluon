import numpy as np
import pytest
import torch

from autogluon.timeseries.models.toto.dataloader import Toto2DataLoader, TotoInferenceDataset

from ..common import get_data_frame_with_item_index, get_data_frame_with_variable_lengths

PATCH_SIZE = 32
MODEL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def noop(*args, **kwargs):
    # pickleable no-op function
    pass


class MockToto2Model:
    """Lightweight stand-in for ``toto2.Toto2Model`` used in tests.

    Produces a forecast where every quantile equals the per-item mean of the observed context,
    repeated across the horizon.
    """

    class _Config:
        patch_size = PATCH_SIZE

    class _OutputHead:
        knots = MODEL_QUANTILES

    def __init__(self):
        self.config = self._Config()
        self.output_head = self._OutputHead()

    def forecast(self, inputs, horizon, **kwargs):
        target = inputs["target"]  # (batch, n_var, time)
        mask = inputs["target_mask"]
        masked = torch.where(mask, target, torch.nan)
        means = torch.nan_to_num(torch.nanmean(masked, dim=-1), nan=0.0)  # (batch, n_var)
        # (num_quantiles, batch, n_var, horizon)
        return means[None, :, :, None].repeat(len(MODEL_QUANTILES), 1, 1, horizon)


class TestToto2Dataset:
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

        dset = TotoInferenceDataset(df, max_context_length=context_length)

        for i in range(len(dset)):
            assert len(dset[i]) == min(context_length, input_data_length)

    @pytest.mark.parametrize("max_data_length", [10, 100])
    def test_when_dataset_with_uneven_lengths_iterated_then_items_have_correct_length(self, max_data_length):
        item_id_to_length = {"A": 1, "B": max_data_length // 2, "C": max_data_length // 2, "D": max_data_length}

        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)

        dset = TotoInferenceDataset(df, max_context_length=max_data_length)

        for i, item_length in zip(range(len(dset)), item_id_to_length.values()):
            assert len(dset[i]) == item_length


class TestToto2Dataloader:
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_dataloader_iterated_then_batches_are_on_correct_device(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        df = get_data_frame_with_item_index([f"item{x:03d}" for x in range(50)], data_length=100)
        dataset = TotoInferenceDataset(df, max_context_length=100)
        loader = Toto2DataLoader(dataset, batch_size=32, device=device)

        for batch in loader:
            for tensor in [batch["target"], batch["target_mask"], batch["series_ids"]]:
                assert tensor.device == torch.device(device)

    @pytest.mark.parametrize("pad_to_multiple", [1, 16, 32])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_pad_to_multiple_set_then_context_is_trimmed_to_floor_multiple(self, pad_to_multiple, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        # 50 is not a multiple of 16 or 32
        data_length = 50
        df = get_data_frame_with_item_index(["A", "B", "C", "D"], data_length=data_length)
        dataset = TotoInferenceDataset(df, max_context_length=1000)
        loader = Toto2DataLoader(dataset, batch_size=4, pad_to_multiple=pad_to_multiple, device=device)

        for batch in loader:
            expected_length = (data_length // pad_to_multiple) * pad_to_multiple
            assert batch["target"].shape[-1] == expected_length
            assert batch["target"].shape[-1] % pad_to_multiple == 0
            # Series longer than a patch are trimmed (not padded), so every position stays observed.
            assert torch.all(batch["target_mask"])

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_series_shorter_than_patch_then_left_padded_up(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        pad_to_multiple = 32
        data_length = 10  # shorter than a single patch
        df = get_data_frame_with_item_index(["A", "B"], data_length=data_length)
        dataset = TotoInferenceDataset(df, max_context_length=1000)
        loader = Toto2DataLoader(dataset, batch_size=2, pad_to_multiple=pad_to_multiple, device=device)

        for batch in loader:
            assert batch["target"].shape[-1] == pad_to_multiple
            for item_mask in batch["target_mask"]:
                # Leading positions are padded (masked out), trailing positions are observed.
                assert not torch.any(item_mask[0, : pad_to_multiple - data_length])
                assert torch.all(item_mask[0, pad_to_multiple - data_length :])

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_dataset_with_uneven_lengths_iterated_then_context_is_correctly_masked(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        max_input_length = 20
        item_id_to_length = {"A": 1, "B": 10, "C": 10, "D": max_input_length}
        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)

        dataset = TotoInferenceDataset(df, max_context_length=1000)
        loader = Toto2DataLoader(dataset, batch_size=4, device=device)

        for batch in loader:
            assert batch["target"].shape[-1] == max_input_length
            assert not torch.any(torch.isnan(batch["target"]))  # NaNs replaced with zeros
            for item_mask, true_length in zip(batch["target_mask"], item_id_to_length.values()):
                # left-padded positions are masked out, observed positions are kept
                assert not torch.any(item_mask[0, : max_input_length - true_length])
                assert torch.all(item_mask[0, max_input_length - true_length :])

    @pytest.mark.parametrize("input_length", [100, 500])
    @pytest.mark.parametrize("max_context_length", [20, 50])
    def test_when_long_data_loaded_then_max_context_is_enforced(self, input_length, max_context_length):
        df = get_data_frame_with_item_index(["A", "B", "C", "D"], data_length=input_length)

        dataset = TotoInferenceDataset(df, max_context_length=max_context_length)
        loader = Toto2DataLoader(dataset, batch_size=4, device="cpu")

        for batch in loader:
            assert batch["target"].shape[-1] == max_context_length


class TestToto2Model:
    @pytest.mark.parametrize("num_items", [5, 100])
    @pytest.mark.parametrize("batch_size", [4, 32])
    def test_predict_returns_correct_format(self, num_items, batch_size):
        from unittest.mock import patch

        from autogluon.timeseries.models.toto import Toto2Model

        item_index = [f"item{x:03d}" for x in range(num_items)]
        df = get_data_frame_with_item_index(item_index, data_length=50)
        for i, item_id in enumerate(item_index):
            df.loc[item_id, "target"] = i + 1

        model = Toto2Model(
            prediction_length=10,
            quantile_levels=[0.1, 0.5, 0.9],
            hyperparameters={"batch_size": batch_size, "device": "cpu"},
        )

        with patch.object(model, "load_model", noop), patch.object(model, "_model", MockToto2Model()):
            predictions = model._predict(df)

            assert len(predictions) == num_items * 10
            assert list(predictions.columns) == ["mean", "0.1", "0.5", "0.9"]
            assert predictions.index.names == ["item_id", "timestamp"]

            assert np.allclose(
                np.repeat(np.arange(1, num_items + 1), 10),
                predictions["mean"],
            )

    def test_predict_interpolates_requested_quantile_levels(self):
        from unittest.mock import patch

        from autogluon.timeseries.models.toto import Toto2Model

        df = get_data_frame_with_item_index(["A", "B"], data_length=50)

        # request levels that are not among the model's native quantiles
        model = Toto2Model(
            prediction_length=5,
            quantile_levels=[0.05, 0.15, 0.5, 0.95],
            hyperparameters={"device": "cpu"},
        )

        with patch.object(model, "load_model", noop), patch.object(model, "_model", MockToto2Model()):
            predictions = model._predict(df)
            assert list(predictions.columns) == ["mean", "0.05", "0.15", "0.5", "0.95"]
            assert not predictions.isna().any().any()
