import numpy as np
import pytest
from chronos import ChronosConfig
from flaky import flaky

from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.timeseries.models.chronos.utils import (
    ChronosFineTuningDataset,
    ChronosInferenceDataLoader,
    ChronosInferenceDataset,
    PseudoShuffledIterableDataset,
    timeout_callback,
)

from ...common import (
    DATAFRAME_WITH_COVARIATES,
    DATAFRAME_WITH_STATIC,
    DUMMY_TS_DATAFRAME,
    get_data_frame_with_item_index,
    get_data_frame_with_variable_lengths,
)

DATASETS = [DUMMY_TS_DATAFRAME, DATAFRAME_WITH_STATIC, DATAFRAME_WITH_COVARIATES]


# PseudoShuffledIterableDataset tests


@flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize("iterable_size,shuffle_buffer_size", [(10, 10), (100, 10), (1000, 100), (1000, 1000)])
def test_pseudo_shuffled_iterable_dataset_shuffles_the_iterable(iterable_size, shuffle_buffer_size):
    iterable = list(range(iterable_size))
    shuffled_dataset = PseudoShuffledIterableDataset(iterable, shuffle_buffer_size=shuffle_buffer_size)
    result = list(shuffled_dataset)

    # result contains all elements
    assert sorted(result) == iterable

    # result in not the same order as the original iterable
    assert result != iterable


# ChronosFineTuningDataset tests


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("context_length", [5, 10, 20])
@pytest.mark.parametrize("prediction_length", [4, 8, 10])
@pytest.mark.parametrize("mode", ["training", "validation"])
def test_chronos_fine_tuning_dataset_returns_data_in_chronos_format_when_tokenizer_is_given(
    data, context_length, prediction_length, mode
):
    tokenizer = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15, "high_limit": 15},
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    ).create_tokenizer()
    fine_tuning_dataset = ChronosFineTuningDataset(
        data,
        context_length=context_length,
        prediction_length=prediction_length,
        tokenizer=tokenizer,
        mode=mode,
    )

    entry = next(iter(fine_tuning_dataset))
    expected_keys = ["input_ids", "attention_mask", "labels"]
    assert all([exp_key in entry for exp_key in expected_keys])

    # +1 for the EOS token
    assert entry["input_ids"].shape[-1] == context_length + 1
    assert entry["attention_mask"].shape[-1] == context_length + 1
    assert entry["labels"].shape[-1] == prediction_length + 1


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("context_length", [5, 10, 20])
@pytest.mark.parametrize("prediction_length", [4, 8, 10])
@pytest.mark.parametrize("mode", ["training", "validation"])
def test_chronos_fine_tuning_dataset_returns_data_in_chronos_bolt_format_when_tokenizer_is_not_given(
    data, context_length, prediction_length, mode
):
    fine_tuning_dataset = ChronosFineTuningDataset(
        data,
        context_length=context_length,
        prediction_length=prediction_length,
        mode=mode,
    )

    entry = next(iter(fine_tuning_dataset))
    expected_keys = ["context", "target"]
    assert all([exp_key in entry for exp_key in expected_keys])
    assert entry["context"].shape[-1] == context_length
    assert entry["target"].shape[-1] == prediction_length


@pytest.mark.parametrize(
    "shuffle_buffer_size, expected_type",
    [(100, PseudoShuffledIterableDataset), (0, ChronosFineTuningDataset), (None, ChronosFineTuningDataset)],
)
def test_chronos_fine_tuning_dataset_shuffle_returns_shuffled_dataset(shuffle_buffer_size, expected_type):
    shuffled_dataset = ChronosFineTuningDataset(DUMMY_TS_DATAFRAME).shuffle(shuffle_buffer_size)

    assert isinstance(shuffled_dataset, expected_type)


# ChronosInferenceDataset tests


@pytest.mark.parametrize("data", DATASETS)
@pytest.mark.parametrize("context_length", [5, 10, 20])
def test_when_context_length_provided_then_inference_dataset_context_length_used(data, context_length):
    inference_dataset = ChronosInferenceDataset(data, context_length=context_length)
    item = inference_dataset[0]

    assert item.shape[-1] == context_length


@pytest.mark.parametrize("context_length", [5, 10, 20])
def test_when_context_length_provided_then_padding_correct(context_length):
    data = get_data_frame_with_item_index(list(range(20)), data_length=5)
    inference_dataset = ChronosInferenceDataset(data, context_length=context_length)
    item = inference_dataset[0]

    assert np.sum(np.isnan(item)) == context_length - 5
    assert not np.isnan(item[-1])  # padding left


@pytest.mark.parametrize(
    "item_id_to_length, expected_indptr",
    [
        ({"A": 20, "B": 12}, [0, 20, 32]),
        ({"A": 20, "B": 12, "C": 1}, [0, 20, 32, 33]),
        ({"A": 20}, [0, 20]),
        ({"A": 1}, [0, 1]),
        ({"B": 10, "A": 10}, [0, 10, 20]),
    ],
)
def test_when_inference_dataset_initialized_then_indptr_set_correctly(item_id_to_length, expected_indptr):
    dataset = get_data_frame_with_variable_lengths(item_id_to_length)
    inference_dataset = ChronosInferenceDataset(dataset, context_length=5)

    assert inference_dataset.indptr.tolist() == expected_indptr


# ChronosInferenceDataLoader tests


@pytest.mark.parametrize("data_loader_num_workers", [0, 1, 2])
def test_when_chronos_inference_dataloader_used_and_time_limit_exceeded_then_exception_is_raised(
    data_loader_num_workers,
):
    data_loader = ChronosInferenceDataLoader(
        range(100_000_000),
        batch_size=2,
        num_workers=data_loader_num_workers,
        on_batch=timeout_callback(seconds=0.5),
    )

    with pytest.raises(TimeLimitExceeded):
        for _ in data_loader:
            pass
