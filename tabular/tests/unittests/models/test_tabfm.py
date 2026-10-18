import logging

import pytest

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.tabfm.tabfm_model import TabFMModel, _root_logging_left_unconfigured
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 1}


@pytest.mark.skip(
    reason="TabFM is a very large model; run manually on a machine with a GPU and the checkpoint cached."
)
def test_tabfm():
    pytest.importorskip("tabfm")
    if ResourceManager.get_gpu_count_torch() == 0:
        pytest.skip("Skip, no GPU available.")

    FitHelper.verify_model(
        model_cls=TabFMModel,
        model_hyperparameters=toy_model_params,
        verify_load_wo_cuda=True,
        # TabFM returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


def test_tabfm_weight_fetch_policy_resolves_from_the_local_cache_only():
    pytest.importorskip("tabfm")
    from huggingface_hub import constants
    from huggingface_hub.errors import LocalEntryNotFoundError

    from autogluon.common.utils.pretrained_weights import PretrainedWeightsUnavailableError
    from autogluon.tabular.models.tabfm._weight_fetch import weight_fetch_policy

    before = constants.HF_HUB_OFFLINE
    with pytest.raises(PretrainedWeightsUnavailableError):
        with weight_fetch_policy(False, stage="load", model_name="TabFM"):
            assert constants.HF_HUB_OFFLINE is True
            raise LocalEntryNotFoundError("not cached")
    assert constants.HF_HUB_OFFLINE == before, "the guard restores the offline setting"

    with weight_fetch_policy(True, stage="fit", model_name="TabFM"):
        assert constants.HF_HUB_OFFLINE == before, "allowed fetches are left untouched"


def test_tabfm_leaves_the_root_logger_unconfigured():
    root = logging.getLogger()
    handlers = list(root.handlers)
    with _root_logging_left_unconfigured():
        root.addHandler(logging.StreamHandler())
    assert root.handlers == handlers
