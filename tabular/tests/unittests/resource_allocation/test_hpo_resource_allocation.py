import pandas as pd
import pytest

from autogluon.core.hpo.executors import CustomHpoExecutor, RayHpoExecutor
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.tabular.models import AbstractModel


class DummyBaseModel(AbstractModel):
    def __init__(self, minimum_resources=None, default_resources=None, **kwargs):
        self._minimum_resources = minimum_resources
        self._default_resources = default_resources
        super().__init__(**kwargs)

    def get_minimum_resources(self, **kwargs):
        return self._minimum_resources

    def _get_default_resources(self):
        num_cpus = self._default_resources.get("num_cpus")
        num_gpus = self._default_resources.get("num_gpus")
        return num_cpus, num_gpus


class DummyModel(DummyBaseModel):
    pass


class DummyBaggedModel(BaggedEnsembleModel):
    pass


dummy_x = pd.DataFrame([1, 2, 3], columns=["Dummy"])


def _initialize_executor(executor_cls, hyperparameter_tune_kwargs):
    executor = executor_cls()
    executor.initialize(hyperparameter_tune_kwargs)
    return executor


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_and_bagging_invalid_resources_per_fold_more_than_total(mock_system_resources_ctx_mgr, executor_cls):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 2}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        # Test invalid resources per fold larger than total resources
        base_model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
            hyperparameters={"ag_args_fit": {"num_cpus": 10, "num_gpus": 1}},
        )
        base_model.initialize()
        bagged_model = DummyBaggedModel(
            model_base=base_model, hyperparameters={"ag_args_fit": {"num_cpus": 2, "num_gpus": 2}}
        )
        bagged_model.initialize()
        with pytest.raises(AssertionError) as e:
            executor.register_resources(bagged_model, k_fold=8, X=dummy_x, **total_resources)


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_and_bagging_invalid_resources_per_fold_less_than_minimum(mock_system_resources_ctx_mgr, executor_cls):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 2}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        # Test invalid resources per fold less than minimum resources required
        base_model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
            hyperparameters={"ag_args_fit": {"num_cpus": 1, "num_gpus": 0}},
        )
        base_model.initialize()
        bagged_model = DummyBaggedModel(
            model_base=base_model, hyperparameters={"ag_args_fit": {"num_cpus": 2, "num_gpus": 2}}
        )
        bagged_model.initialize()
        with pytest.raises(AssertionError) as e:
            executor.register_resources(bagged_model, k_fold=8, X=dummy_x, **total_resources)


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_and_bagging_invalid_resources_per_trial_more_than_total(mock_system_resources_ctx_mgr, executor_cls):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 2}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        # Test invalid resources per trial larger than total resources
        base_model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
            hyperparameters={"ag_args_fit": {"num_cpus": 1, "num_gpus": 1}},
        )
        base_model.initialize()
        bagged_model = DummyBaggedModel(
            model_base=base_model, hyperparameters={"ag_args_fit": {"num_cpus": 10, "num_gpus": 1}}
        )
        bagged_model.initialize()
        with pytest.raises(AssertionError) as e:
            executor.register_resources(bagged_model, k_fold=8, X=dummy_x, **total_resources)


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_and_bagging_invalid_resources_per_trial_less_than_minimum(mock_system_resources_ctx_mgr, executor_cls):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 2}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        # Test invalid resources per trial less than minimum resources required
        base_model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
            hyperparameters={"ag_args_fit": {"num_cpus": 1, "num_gpus": 0.1}},
        )
        base_model.initialize()
        bagged_model = DummyBaggedModel(
            model_base=base_model, hyperparameters={"ag_args_fit": {"num_cpus": 1, "num_gpus": 0}}
        )
        bagged_model.initialize()
        with pytest.raises(AssertionError) as e:
            executor.register_resources(bagged_model, k_fold=8, X=dummy_x, **total_resources)


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_and_bagging_valid_resources_per_fold_and_valid_resources_per_trial(
    mock_system_resources_ctx_mgr, executor_cls
):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 2}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        # Test valid resources per fold and resources per trial
        base_model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
            hyperparameters={"ag_args_fit": {"num_cpus": 1, "num_gpus": 0.1}},
        )
        base_model.initialize()
        bagged_model = DummyBaggedModel(
            model_base=base_model, hyperparameters={"ag_args_fit": {"num_cpus": 4, "num_gpus": 0.5}}
        )
        bagged_model.initialize()
        executor.register_resources(bagged_model, k_fold=8, X=dummy_x, **total_resources)
        # 1 bag in parallel, 4 folds in parallel per bagged ensemble, each using 1 cpu and 0.1 gpus
        assert executor.hyperparameter_tune_kwargs["resources_per_trial"] == {"num_cpus": 4, "num_gpus": 0.4}


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_and_bagging_valid_resources_per_fold_and_no_resources_per_trial(
    mock_system_resources_ctx_mgr, executor_cls
):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 2}
    # Test valid resources per fold and no resources per trial
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        base_model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.01},
            hyperparameters={"ag_args_fit": {"num_cpus": 1, "num_gpus": 0.1}},
        )
        base_model.initialize()
        bagged_model = DummyBaggedModel(
            model_base=base_model,
        )
        bagged_model.initialize()
        executor.register_resources(bagged_model, k_fold=8, X=dummy_x, **total_resources)
        # 1 bag in parallel, 8 folds in parallel per bagged ensemble, each using 1 cpu and 0.1 gpus
        assert executor.hyperparameter_tune_kwargs["resources_per_trial"] == {"num_cpus": 8, "num_gpus": 0.8}


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_and_bagging_valid_resources_per_trial_and_no_resources_per_fold(
    mock_system_resources_ctx_mgr, executor_cls
):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 2}
    # Test valid resources per trial and no resources per fold
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        base_model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
        )
        base_model.initialize()
        bagged_model = DummyBaggedModel(
            model_base=base_model, hyperparameters={"ag_args_fit": {"num_cpus": 1, "num_gpus": 0.1}}
        )
        bagged_model.initialize()
        executor.register_resources(bagged_model, k_fold=8, X=dummy_x, **total_resources)
        # 1 bag in parallel, 1 fold in parallel per bagged ensemble, using 1 cpu and 0.1 gpus
        assert executor.hyperparameter_tune_kwargs["resources_per_trial"] == {"num_cpus": 1, "num_gpus": 0.1}


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_and_bagging_no_resources_per_trial_and_no_resources_per_fold(mock_system_resources_ctx_mgr, executor_cls):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 2}
    # Test valid resources per trial and no resources per fold
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        base_model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.5},
        )
        base_model.initialize()
        bagged_model = DummyBaggedModel(model_base=base_model, hyperparameters={})
        bagged_model.initialize()
        executor.register_resources(bagged_model, k_fold=8, X=dummy_x, **total_resources)
        # Only 1 trial can run at a time. Give full resources to it
        assert executor.hyperparameter_tune_kwargs["resources_per_trial"] == {"num_cpus": 8, "num_gpus": 1}


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_without_bagging_invalid_resources_per_trial_more_than_total_resources(
    mock_system_resources_ctx_mgr, executor_cls
):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 4}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        # Test valid resources per fold and resources per trial
        model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
            hyperparameters={"ag_args_fit": {"num_cpus": 10, "num_gpus": 0.2}},
        )
        model.initialize()
        with pytest.raises(AssertionError) as e:
            executor.register_resources(model, X=dummy_x, **total_resources)


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_without_bagging_invalid_resources_per_trial_less_than_minimum_resources(
    mock_system_resources_ctx_mgr, executor_cls
):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 4}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        # Test valid resources per fold and resources per trial
        model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
            hyperparameters={"ag_args_fit": {"num_cpus": 1, "num_gpus": 0.01}},
        )
        model.initialize()
        with pytest.raises(AssertionError) as e:
            executor.register_resources(model, X=dummy_x, **total_resources)


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_without_bagging_valid_resources_per_trial(mock_system_resources_ctx_mgr, executor_cls):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 4}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        # Test valid resources per fold and resources per trial
        model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
            hyperparameters={"ag_args_fit": {"num_cpus": 1, "num_gpus": 0.2}},
        )
        model.initialize()
        executor.register_resources(model, X=dummy_x, **total_resources)
        # 4 trials in parallel, each using 1 cpu and 0.2 gpus
        assert executor.hyperparameter_tune_kwargs["resources_per_trial"] == {"num_cpus": 1, "num_gpus": 0.2}


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_without_bagging_zero_gpus_per_trial_no_cpus_specified(mock_system_resources_ctx_mgr, executor_cls):
    """Regression test for a ZeroDivisionError: specifying only `num_gpus: 0` in
    `ag_args_fit` (a common way to pin a model to CPU, e.g. to keep it off the GPU
    while other models use it) used to crash `register_resources`, which
    unconditionally computed `num_gpus // user_specified_trial_num_gpus` when cpus
    per trial weren't specified. A gpu-per-trial request of 0 places no constraint
    on parallelism, so the fix gives the trial all available cpus instead.
    """
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 4}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0},
            hyperparameters={"ag_args_fit": {"num_gpus": 0}},
        )
        model.initialize()
        executor.register_resources(model, X=dummy_x, **total_resources)
        assert executor.hyperparameter_tune_kwargs["resources_per_trial"] == {"num_cpus": 8, "num_gpus": 0}


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_without_bagging_zero_cpus_per_trial_no_gpus_specified(mock_system_resources_ctx_mgr, executor_cls):
    """Symmetric regression test for the `num_cpus: 0`-only case, which hit the same
    ZeroDivisionError pattern on the other branch of `register_resources`.
    """
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 4}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 2}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        model = DummyModel(
            minimum_resources={"num_cpus": 0, "num_gpus": 0.1},
            hyperparameters={"ag_args_fit": {"num_cpus": 0}},
        )
        model.initialize()
        executor.register_resources(model, X=dummy_x, **total_resources)
        assert executor.hyperparameter_tune_kwargs["resources_per_trial"] == {"num_cpus": 0, "num_gpus": 2}


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_without_bagging_more_cpus_per_trial_than_total(mock_system_resources_ctx_mgr, executor_cls):
    """Requesting more cpus per trial than the predictor was granted reports what is wrong.

    The requested value is validated before the unspecified gpu count is derived from it,
    which would otherwise divide by a trial count of `num_cpus // 100 == 0`.
    """
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 4}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 2}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0},
            hyperparameters={"ag_args_fit": {"num_cpus": 100}},
        )
        model.initialize()
        with pytest.raises(AssertionError, match="Detected trial level cpu requirement = 100"):
            executor.register_resources(model, X=dummy_x, **total_resources)


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_without_bagging_more_gpus_per_trial_than_total(mock_system_resources_ctx_mgr, executor_cls):
    """Requesting gpus per trial on a predictor granted no gpus reports what is wrong."""
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 4}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 0}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0},
            hyperparameters={"ag_args_fit": {"num_gpus": 1}},
        )
        model.initialize()
        with pytest.raises(AssertionError, match="Detected trial level gpu requirement = 1"):
            executor.register_resources(model, X=dummy_x, **total_resources)


@pytest.mark.parametrize("executor_cls", [RayHpoExecutor, CustomHpoExecutor])
def test_hpo_without_bagging_no_resources_per_trial(mock_system_resources_ctx_mgr, executor_cls):
    hyperparameter_tune_kwargs = {"scheduler": "local", "searcher": "random", "num_trials": 4}
    executor = _initialize_executor(executor_cls, hyperparameter_tune_kwargs)
    total_resources = {"num_cpus": 8, "num_gpus": 1}
    with mock_system_resources_ctx_mgr(num_cpus=total_resources["num_cpus"], num_gpus=total_resources["num_gpus"]):
        # Test valid resources per fold and resources per trial
        model = DummyModel(
            minimum_resources={"num_cpus": 1, "num_gpus": 0.1},
        )
        model.initialize()
        executor.register_resources(model, X=dummy_x, **total_resources)
        if executor_cls == RayHpoExecutor:
            # 4 trials in parallel, each using 1 cpu and 0.25 gpus(the maximum possible)
            assert executor.hyperparameter_tune_kwargs["resources_per_trial"] == {"num_cpus": 2, "num_gpus": 0.25}
        elif executor_cls == CustomHpoExecutor:
            # custom backend use all resources for one trial
            assert executor.hyperparameter_tune_kwargs["resources_per_trial"] == {"num_cpus": 8, "num_gpus": 1}
