import pytest

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.tabular.models import AbstractModel


class DummyBaseModel(AbstractModel):
    def __init__(self, minimum_resources={}, **kwargs):
        self._minimum_resources = minimum_resources
        super().__init__(**kwargs)

    def get_minimum_resources(self, **kwargs):
        return self._minimum_resources

    def _get_default_resources(self):
        num_cpus = 1
        num_gpus = 1
        return num_cpus, num_gpus


class DummyModel(DummyBaseModel):
    pass


class DummyBaggedModel(BaggedEnsembleModel):
    pass


def test_bagged_model_with_total_resources(mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus, k_fold):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        model_base = DummyModel()
        model_base.initialize()
        bagged_model = DummyBaggedModel(model_base)
        total_resources = {
            "num_cpus": 1,
            "num_gpus": 0,
        }
        bagged_model.initialize()
        resources = bagged_model._preprocess_fit_resources(total_resources=total_resources, k_fold=k_fold)
        resources.pop("k_fold")
        assert resources == total_resources

        # Given total resources more than what the system has
        total_resources = {
            "num_cpus": 99999,
            "num_gpus": 99999,
        }
        resources = bagged_model._preprocess_fit_resources(total_resources=total_resources, k_fold=k_fold)
        resources.pop("k_fold")
        assert resources == {"num_cpus": ResourceManager.get_cpu_count(), "num_gpus": ResourceManager.get_gpu_count()}


def test_bagged_model_with_total_resources_and_ensemble_resources(
    mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus, k_fold
):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        total_resources = {
            "num_cpus": 8,
            "num_gpus": 1,
        }
        model_base = DummyModel()
        model_base.initialize()
        bagged_model = DummyBaggedModel(
            model_base,
            hyperparameters={
                "ag_args_fit": {
                    "num_cpus": 10,
                    "num_gpus": 1,
                }
            },
        )
        bagged_model.initialize()
        with pytest.raises(AssertionError) as e:
            bagged_model._preprocess_fit_resources(total_resources=total_resources, k_fold=k_fold)

        total_resources = {
            "num_cpus": 8,
            "num_gpus": 1,
        }
        model_base = DummyModel()
        ensemble_ag_args_fit = {
            "num_cpus": 4,
            "num_gpus": 1,
        }
        model_base.initialize()
        bagged_model = DummyBaggedModel(model_base, hyperparameters={"ag_args_fit": ensemble_ag_args_fit})
        bagged_model.initialize()
        resources = bagged_model._preprocess_fit_resources(total_resources=total_resources, k_fold=k_fold)
        resources.pop("k_fold")
        assert resources == ensemble_ag_args_fit


def test_bagged_model_with_total_resources_but_no_gpu_specified(
    mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus, k_fold
):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        model_base = DummyModel()
        model_base.initialize()
        total_resources = {
            "num_cpus": 2,
        }
        bagged_model = DummyBaggedModel(model_base)
        bagged_model.initialize()
        resources = bagged_model._preprocess_fit_resources(total_resources=total_resources, k_fold=k_fold)
        resources.pop("k_fold")
        default_model_resources = {
            "num_cpus": 2,
            "num_gpus": ResourceManager.get_gpu_count(),
        }  # return all gpu resources as default needs gpu
        assert resources == default_model_resources


def test_bagged_model_without_total_resources_but_with_ensemble_resources(
    mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus, k_fold
):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        model_base = DummyModel()
        model_base.initialize()
        bagged_model = DummyBaggedModel(
            model_base,
            hyperparameters={
                "ag_args_fit": {
                    "num_cpus": 99999,
                    "num_gpus": 99999,
                }
            },
        )
        bagged_model.initialize()
        with pytest.raises(AssertionError) as e:
            bagged_model._preprocess_fit_resources(k_fold=k_fold)

        model_base = DummyModel()
        model_base.initialize()
        ensemble_ag_args_fit = {
            "num_cpus": 1,
            "num_gpus": 0,
        }
        bagged_model = DummyBaggedModel(model_base, hyperparameters={"ag_args_fit": ensemble_ag_args_fit})
        bagged_model.initialize()
        resources = bagged_model._preprocess_fit_resources(k_fold=k_fold)
        resources.pop("k_fold")
        assert resources == ensemble_ag_args_fit


def test_bagged_model_without_total_resources_and_without_model_resources(
    mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus, k_fold
):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        model_base = DummyModel()
        model_base.initialize()
        bagged_model = DummyBaggedModel(model_base)
        bagged_model.initialize()
        resources = bagged_model._preprocess_fit_resources(k_fold=k_fold)
        resources.pop("k_fold")
        # Bagged model should take all resources and internally calculate correct resources given ag_args_ensemble and ag_args_fit
        expected_model_resources = {
            "num_cpus": ResourceManager.get_cpu_count(),
            "num_gpus": ResourceManager.get_gpu_count(),
        }
        assert resources == expected_model_resources


def test_nonbagged_model_with_total_resources(mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        model_base = DummyModel()
        total_resources = {
            "num_cpus": 1,
            "num_gpus": 0,
        }
        resources = model_base._preprocess_fit_resources(total_resources=total_resources)
        assert resources == total_resources


def test_nonbagged_model_with_total_resources_but_no_gpu_specified(
    mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus
):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        # If model by default needs gpu, we use it even if user didn't specify it
        model_base = DummyModel()
        total_resources = {
            "num_cpus": 2,
        }
        resources = model_base._preprocess_fit_resources(total_resources=total_resources)
        _, default_model_num_gpus = model_base._get_default_resources()
        default_model_resources = {"num_cpus": 2, "num_gpus": default_model_num_gpus}
        assert resources == default_model_resources


def test_nonbagged_model_with_total_resources_and_model_resources(
    mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus
):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        model_base = DummyModel(hyperparameters={"ag_args_fit": {"num_cpus": 2, "num_gpus": 1}})
        total_resources = {
            "num_cpus": 1,
            "num_gpus": 1,
        }
        with pytest.raises(AssertionError) as e:
            model_base._preprocess_fit_resources(total_resources=total_resources)

        ag_args_fit = {"num_cpus": 1, "num_gpus": 0}
        model_base = DummyModel(hyperparameters={"ag_args_fit": ag_args_fit})
        total_resources = {
            "num_cpus": 8,
            "num_gpus": 1,
        }
        resources = model_base._preprocess_fit_resources(total_resources=total_resources)
        # Here both total_resources and ag_args_fit are specified, respect ag_args_fit
        assert resources == ag_args_fit


def test_nonbagged_model_without_total_resources(mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        model_base = DummyModel()
        resources = model_base._preprocess_fit_resources()
        default_model_num_cpus, default_model_num_gpus = model_base._get_default_resources()
        default_model_resources = {"num_cpus": default_model_num_cpus, "num_gpus": default_model_num_gpus}
        assert resources == default_model_resources


def test_nonbagged_model_without_total_resources_but_with_model_resources(
    mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus
):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        model_base = DummyModel(hyperparameters={"ag_args_fit": {"num_cpus": 99999, "num_gpus": 99999}})
        with pytest.raises(AssertionError) as e:
            model_base._preprocess_fit_resources()

        ag_args_fit = {"num_cpus": 2, "num_gpus": 2}
        model_base = DummyModel(hyperparameters={"ag_args_fit": ag_args_fit})
        resources = model_base._preprocess_fit_resources()
        assert resources == ag_args_fit


def test_nonbagged_model_without_total_resources_and_without_model_resources(
    mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus
):
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        model_base = DummyModel(hyperparameters={})
        resources = model_base._preprocess_fit_resources()
        default_model_num_cpus, default_model_num_gpus = model_base._get_default_resources()
        default_model_resources = {
            "num_cpus": min(default_model_num_cpus, ResourceManager.get_cpu_count()),
            "num_gpus": min(default_model_num_gpus, ResourceManager.get_gpu_count()),
        }
        assert resources == default_model_resources
