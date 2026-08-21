import pytest
import torch

from autogluon.timeseries.models.toto2.dataloader import Toto2DataLoader, Toto2InferenceDataset

from ..common import get_data_frame_with_item_index, get_data_frame_with_variable_lengths


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

        dset = Toto2InferenceDataset(df, max_context_length=context_length)

        for i in range(len(dset)):
            assert len(dset[i]) == min(context_length, input_data_length)

    @pytest.mark.parametrize("max_data_length", [10, 100])
    def test_when_dataset_with_uneven_lengths_iterated_then_items_have_correct_length(self, max_data_length):
        item_id_to_length = {"A": 1, "B": max_data_length // 2, "C": max_data_length // 2, "D": max_data_length}

        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)

        dset = Toto2InferenceDataset(df, max_context_length=max_data_length)

        for i, item_length in zip(range(len(dset)), item_id_to_length.values()):
            assert len(dset[i]) == item_length


class TestToto2Dataloader:
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_dataloader_iterated_then_batches_are_on_correct_device(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        df = get_data_frame_with_item_index([f"item{x:03d}" for x in range(50)], data_length=100)
        dataset = Toto2InferenceDataset(df, max_context_length=100)
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
        dataset = Toto2InferenceDataset(df, max_context_length=1000)
        loader = Toto2DataLoader(dataset, batch_size=4, pad_to_multiple=pad_to_multiple, device=device)

        for batch in loader:
            expected_length = (data_length // pad_to_multiple) * pad_to_multiple
            assert batch["target"].shape[-1] == expected_length
            assert batch["target"].shape[-1] % pad_to_multiple == 0
            # Series longer than a patch are trimmed (not padded), so every position stays observed.
            assert torch.all(batch["target_mask"])

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_lengths_are_uneven_then_each_series_is_trimmed_to_patch_boundary(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        pad_to_multiple = 32
        # None of these lengths is a multiple of 32, and only "D" is trimmed by the batch length alone
        item_id_to_length = {"A": 100, "B": 70, "C": 33, "D": 20}
        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
        dataset = Toto2InferenceDataset(df, max_context_length=1000)
        loader = Toto2DataLoader(dataset, batch_size=4, pad_to_multiple=pad_to_multiple, device=device)

        for batch in loader:
            batch_length = batch["target"].shape[-1]
            assert batch_length == 96
            expected_num_observed = [96, 64, 32, 20]
            for item_mask, num_observed in zip(batch["target_mask"], expected_num_observed):
                assert item_mask[0].sum() == num_observed
                # The oldest observation of each trimmed series falls on a patch boundary
                if num_observed >= pad_to_multiple:
                    assert (batch_length - num_observed) % pad_to_multiple == 0

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_all_series_shorter_than_patch_then_batch_is_left_padded_up(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        pad_to_multiple = 32
        data_length = 10  # shorter than a single patch
        df = get_data_frame_with_item_index(["A", "B"], data_length=data_length)
        dataset = Toto2InferenceDataset(df, max_context_length=1000)
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

        dataset = Toto2InferenceDataset(df, max_context_length=1000)
        loader = Toto2DataLoader(dataset, batch_size=4, device=device)

        for batch in loader:
            assert batch["target"].shape[-1] == max_input_length
            assert not torch.any(torch.isnan(batch["target"]))
            for item_mask, true_length in zip(batch["target_mask"], item_id_to_length.values()):
                # left-padded positions are masked out, observed positions are kept
                assert not torch.any(item_mask[0, : max_input_length - true_length])
                assert torch.all(item_mask[0, max_input_length - true_length :])

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_series_left_padded_then_pad_is_filled_with_first_observed_value(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        item_id_to_length = {"A": 20, "B": 5}
        df = get_data_frame_with_variable_lengths(item_id_to_length=item_id_to_length)
        df["target"] += 1000.0  # move away from zero to distinguish backfill from zero-filling
        dataset = Toto2InferenceDataset(df, max_context_length=1000)
        loader = Toto2DataLoader(dataset, batch_size=2, device=device)

        for batch in loader:
            target, mask = batch["target"][1, 0], batch["target_mask"][1, 0]
            first_observed_value = df.loc["B"]["target"].iloc[0]
            assert torch.allclose(target[~mask], torch.full_like(target[~mask], first_observed_value))

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_target_has_missing_values_then_they_are_forward_filled(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        df = get_data_frame_with_item_index(["A"], data_length=8)
        df["target"] = [10.0, float("nan"), float("nan"), 40.0, 50.0, float("nan"), 70.0, float("nan")]
        dataset = Toto2InferenceDataset(df, max_context_length=1000)
        loader = Toto2DataLoader(dataset, batch_size=1, device=device)

        expected = [10.0, 10.0, 10.0, 40.0, 50.0, 50.0, 70.0, 70.0]
        for batch in loader:
            assert batch["target"][0, 0].tolist() == expected
            assert batch["target_mask"][0, 0].tolist() == [True, False, False, True, True, False, True, False]

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_when_series_is_entirely_missing_then_target_is_filled_with_zeros(self, device):
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip(reason="No GPU available")

        df = get_data_frame_with_item_index(["A"], data_length=8)
        df["target"] = float("nan")
        dataset = Toto2InferenceDataset(df, max_context_length=1000)
        loader = Toto2DataLoader(dataset, batch_size=1, device=device)

        for batch in loader:
            assert not torch.any(batch["target_mask"])
            assert torch.all(batch["target"] == 0.0)

    @pytest.mark.parametrize("input_length", [100, 500])
    @pytest.mark.parametrize("max_context_length", [20, 50])
    def test_when_long_data_loaded_then_max_context_is_enforced(self, input_length, max_context_length):
        df = get_data_frame_with_item_index(["A", "B", "C", "D"], data_length=input_length)

        dataset = Toto2InferenceDataset(df, max_context_length=max_context_length)
        loader = Toto2DataLoader(dataset, batch_size=4, device="cpu")

        for batch in loader:
            assert batch["target"].shape[-1] == max_context_length
