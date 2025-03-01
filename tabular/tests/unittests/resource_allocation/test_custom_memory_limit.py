import pytest

from autogluon.tabular.testing import FitHelper


@pytest.fixture()
def get_and_assert_max_memory():
    import os

    import psutil

    # Mock patch to guarantee that test does not fail
    # due to memory changing between calls to psutil.
    p = psutil.Process()
    allocated_memory = p.memory_info()
    psutil.Process.memory_info = lambda _: allocated_memory

    # Import after mock patch above.
    from autogluon.common.utils.resource_utils import ResourceManager

    max_memory = 48.0
    yield max_memory  # Wait for test to set up custom memory limit

    # Check custom memory limit
    try:
        assert ResourceManager.get_memory_size(format="GB") == max_memory
        assert (max_memory * (1024.0**3)) - allocated_memory.rss == ResourceManager.get_available_virtual_mem(
            format="B",
        )
    finally:
        del os.environ["AG_MEMORY_LIMIT_IN_GB"]


def test_custom_memory_soft_limit_tabular_fit(get_and_assert_max_memory):
    fit_args = dict(
        hyperparameters={"DUMMY": {}},
        memory_limit=get_and_assert_max_memory,
        dynamic_stacking=False,
        fit_weighted_ensemble=False,
        num_bag_folds=2,
        num_bag_sets=1,
        num_stack_levels=0,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    dataset_name = "adult"

    FitHelper.fit_and_validate_dataset(
        dataset_name=dataset_name,
        fit_args=fit_args,
        expected_model_count=1,
        refit_full=False,
    )
