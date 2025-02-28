import pytest

from autogluon.common.utils.resource_utils import ResourceManager


@pytest.mark.gpu
def test_automm_sts(fit_helper):
    if ResourceManager.get_gpu_count_torch() == 0:
        # Skip test if no GPU available
        pytest.skip("Skip, no GPU available.")
    fit_args = dict(
        hyperparameters={"AG_AUTOMM": {"env.num_workers": 0, "env.num_workers_inference": 0}},
    )
    dataset_name = "sts"
    fit_helper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        sample_size=100,
        refit_full=False,
    )


def test_handle_text_automm():
    hyperparameters = {"AG_AUTOMM": {}}
    from autogluon.tabular import TabularPredictor

    assert TabularPredictor._check_if_hyperparameters_handle_text(hyperparameters)
