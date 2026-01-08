from unittest import mock

import numpy as np
import pytest
from fastai.callback.core import CancelFitException

from autogluon.tabular.models.fastainn.callbacks import BatchTimeTracker
from autogluon.tabular.models.fastainn.tabular_nn_fastai import NNFastAiTabularModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"epochs": 3}


def test_tabular_nn_fastai():
    model_cls = NNFastAiTabularModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
    )


__GET_EPOCHS_NUMBER_CASES = {
    "happy_path": [dict(time_left=45, batch_size=256, epochs="auto"), 2],
    "given negative time return 0 epochs": [dict(time_left=-45, batch_size=256, epochs="auto"), 0],
    "given time for more than default_epochs epochs, return default_epochs": [
        dict(time_left=21 * 31, epochs="auto", batch_size=256, default_epochs=12),
        12,
    ],
    "given time for less than 1 epoch, return 0 epoch": [dict(time_left=10, batch_size=256, epochs="auto"), 0],
    "given no time_left, return default_epochs": [dict(epochs="auto", batch_size=256, default_epochs=14), 14],
    "given there is not enough batches to get min_batches_count, return default_epochs": [
        dict(epochs="auto", time_left=45, batch_size=4096, default_epochs=14),
        14,
    ],
    "given explicit epoch count return specified value": [dict(time_left=45, batch_size=256, epochs=100), 100],
}


@pytest.mark.parametrize("test_input", __GET_EPOCHS_NUMBER_CASES.values(), ids=__GET_EPOCHS_NUMBER_CASES.keys())
def test_get_epochs_number(test_input):
    args, epochs_expected = test_input
    with mock.patch.object(NNFastAiTabularModel, "_measure_batch_times", return_value=1.2732) as mock_method:
        # batches = (4000/256) + 1 = 16
        # est_epoch_time = 16 * 1.2732 = 20.371
        # time_left:45/est_epoch_time:20.371 = 2
        model = NNFastAiTabularModel(path="", name="")
        assert epochs_expected == model._get_epochs_number(4000, min_batches_count=4, **args)
        if "time_left" in args:
            if args.get("epochs", None) == "auto" and args["batch_size"] * 4 <= 4000:
                mock_method.assert_called_with(4)


__GET_BATCH_SIZE_CASES = {
    "given batch size provided return specified value": [dict(bs=111, input_size=400), 111],
    "given batch size larger than dataset use default_batch_size_for_small_inputs": [dict(bs=111, input_size=100), 32],
    "given batch size auto return default_batch_size_for_small_inputs for small datasets": [
        dict(bs="auto", input_size=100),
        32,
    ],
    "given batch size auto return value for small datasets": [dict(bs="auto", input_size=2048), 256],
    "given batch size auto return value for large datasets": [dict(bs="auto", input_size=200000), 512],
}


@pytest.mark.parametrize("test_input", __GET_BATCH_SIZE_CASES.values(), ids=__GET_BATCH_SIZE_CASES.keys())
def test_get_batch_size_with_bs_provided(test_input):
    args, expected_bs = test_input
    bs, input_size = args["bs"], args["input_size"]
    model = NNFastAiTabularModel(path="", name="")
    model.params["bs"] = bs
    x = np.arange(input_size)
    assert expected_bs == model._get_batch_size(x)


def test_BatchTimeTracker():
    with mock.patch.object(BatchTimeTracker, "_time_now") as mock_method:
        mock_method.side_effect = [10, 40]
        iterations_to_complete = 3
        tracker = BatchTimeTracker(batches_to_measure=iterations_to_complete)
        stopped = False
        for i in range(iterations_to_complete + 1):
            try:
                tracker.after_batch()
                if i == iterations_to_complete:
                    raise ValueError("CancelFitException should be raised, but was not")
            except CancelFitException:
                stopped = True
        assert stopped
        assert tracker.batch_measured_time == 10.0
