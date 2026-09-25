import pytest

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.causilo.causilo_model import CausiloModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_estimators": 1}


@pytest.mark.skip(
    reason="Fitting downloads the Causilo checkpoint; run manually on a machine with a GPU and the checkpoint cached."
)
def test_causilo():
    pytest.importorskip("causilo")
    if ResourceManager.get_gpu_count_torch() == 0:
        pytest.skip("Skip, no GPU available.")

    FitHelper.verify_model(
        model_cls=CausiloModel,
        model_hyperparameters=toy_model_params,
        verify_load_wo_cuda=True,
        # Causilo returns different predictions when predicting on an individual sample
        verify_single_prediction_equivalent_to_multi=False,
    )


def test_causilo_weight_fetch_policy_refuses_a_cache_miss(monkeypatch):
    checkpoints = pytest.importorskip("causilo.checkpoints")
    from huggingface_hub.errors import LocalEntryNotFoundError

    from autogluon.common.utils.pretrained_weights import PretrainedWeightsUnavailableError
    from autogluon.tabular.models.causilo._weight_fetch import weight_fetch_policy

    calls = []

    def snapshot_download(*args, **kwargs):
        calls.append(kwargs.get("local_files_only"))
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(checkpoints, "snapshot_download", snapshot_download)
    with weight_fetch_policy(False, stage="fit", model_name="Causilo"):
        with pytest.raises(PretrainedWeightsUnavailableError):
            checkpoints.snapshot_download("nums-ai/causilo")
    assert calls == [True], "the fetch is restricted to the local cache"
    assert checkpoints.snapshot_download is snapshot_download, "the guard restores the library's function"

    with weight_fetch_policy(True, stage="fit", model_name="Causilo"):
        assert checkpoints.snapshot_download is snapshot_download, "allowed fetches are left untouched"
