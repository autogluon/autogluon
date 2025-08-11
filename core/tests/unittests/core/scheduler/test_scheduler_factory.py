import pytest

from autogluon.core.scheduler.scheduler_factory import get_hyperparameter_tune_kwargs_preset, scheduler_factory
from autogluon.core.scheduler.seq_scheduler import LocalSequentialScheduler


def test_scheduler_factory__can_construct_valid_config_with_str_scheduler():
    scheduler_cls, scheduler_params = scheduler_factory(
        hyperparameter_tune_kwargs={"scheduler": "local", "searcher": "grid", "custom_option": "value"}
    )

    assert scheduler_cls == LocalSequentialScheduler, "scheduler_cls must be correct"
    assert scheduler_params["resource"]["num_cpus"] is not None, "resources/num_cpus must be present"
    assert scheduler_params["resource"]["num_gpus"] is not None, "resources/num_cpus must be present"

    expected_values = {
        "searcher": "grid",
        "search_options": {},
        "checkpoint": None,
        "resume": False,
        "num_trials": None,
        "reward_attr": "validation_performance",
        "time_attr": "epoch",
        "visualizer": "none",
        "dist_ip_addrs": [],
        "scheduler": "local",
        "custom_option": "value",
    }
    for k, v in expected_values.items():
        assert scheduler_params[k] == v, f"{k} must be {v}"


def test_scheduler_factory__can_construct_valid_config_with_class_scheduler():
    scheduler_cls, scheduler_params = scheduler_factory(
        hyperparameter_tune_kwargs={"scheduler": LocalSequentialScheduler, "searcher": "local_random"}
    )
    assert scheduler_cls == LocalSequentialScheduler, "scheduler_cls must be correct"


def test_scheduler_factory__reaises_exception_on_missing_scheduler():
    with pytest.raises(ValueError, match="Required key 'scheduler' is not present in hyperparameter_tune_kwargs"):
        scheduler_factory(hyperparameter_tune_kwargs={"searcher": "local_random"})


def test_scheduler_factory__reaises_exception_on_unknown_str_scheduler():
    with pytest.raises(
        ValueError, match="Required key 'scheduler' in hyperparameter_tune_kwargs must be one of the values dict_keys"
    ):
        scheduler_factory(hyperparameter_tune_kwargs={"scheduler": "_some_value_", "searcher": "local_random"})


def test_scheduler_factory__reaises_exception_on_missing_searcher():
    with pytest.raises(ValueError, match="Required key 'searcher' is not present in hyperparameter_tune_kwargs"):
        scheduler_factory(hyperparameter_tune_kwargs={"scheduler": "local"})


def test_get_hyperparameter_tune_kwargs_preset__preset_exists():
    assert get_hyperparameter_tune_kwargs_preset(preset="auto") == {"scheduler": "local", "searcher": "local_random"}


def test_get_hyperparameter_tune_kwargs_preset__preset_missing():
    with pytest.raises(ValueError, match='Invalid hyperparameter_tune_kwargs preset value "unknown"'):
        get_hyperparameter_tune_kwargs_preset(preset="unknown")
