import tempfile

import pytest
from ray import tune

from autogluon.core.hpo.ray_hpo import AutommRayTuneAdapter, RayTuneAdapter, TabularRayTuneAdapter, run
from autogluon.core.hpo.ray_tune_constants import SCHEDULER_PRESETS, SEARCHER_PRESETS


class DummyAdapter(RayTuneAdapter):
    supported_searchers = list(SEARCHER_PRESETS.keys())
    supported_schedulers = list(SCHEDULER_PRESETS.keys())

    @property
    def adapter_type(self):
        return "dummy"

    def get_resource_calculator(self, **kwargs):
        pass

    def get_resources_per_trial(self, total_resources, num_samples, **kwargs):
        return {"cpu": 1}

    def trainable_args_update_method(self, trainable_args):
        return {}


DUMMY_SEARCH_SPACE = {"a": tune.uniform(0, 1), "b": tune.uniform(0, 20)}


def _dummy_objective(x, a, b):
    return a * (x**0.5) + b


def _dummy_trainable(config):
    for x in range(20):
        score = _dummy_objective(x, config["a"], config["b"])

        tune.report({"score": score})


def test_invalid_searcher():
    hyperparameter_tune_kwargs = dict(
        searcher="abc",
        scheduler="FIFO",
        num_trials=1,
    )
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(Exception) as e_info:
            run(
                trainable=_dummy_trainable,
                trainable_args=dict(),
                search_space=DUMMY_SEARCH_SPACE,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                metric="score",
                mode="min",
                save_dir=root,
                ray_tune_adapter=DummyAdapter(),
            )


def test_invalid_scheduler():
    hyperparameter_tune_kwargs = dict(
        searcher="random",
        scheduler="abc",
        num_trials=1,
    )
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(Exception) as e_info:
            run(
                trainable=_dummy_trainable,
                trainable_args=dict(),
                search_space=DUMMY_SEARCH_SPACE,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                metric="score",
                mode="min",
                save_dir=root,
                ray_tune_adapter=DummyAdapter(),
            )


def test_invalid_preset():
    hyperparameter_tune_kwargs = "abc"
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(Exception) as e_info:
            run(
                trainable=_dummy_trainable,
                trainable_args=dict(),
                search_space=DUMMY_SEARCH_SPACE,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                metric="score",
                mode="min",
                save_dir=root,
                ray_tune_adapter=DummyAdapter(),
            )


def test_empty_search_space():
    hyperparameter_tune_kwargs = dict(
        searcher="random",
        scheduler="FIFO",
        num_trials=1,
    )
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(Exception) as e_info:
            run(
                trainable=_dummy_trainable,
                trainable_args=dict(),
                search_space=dict(),
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                metric="score",
                mode="min",
                save_dir=root,
                ray_tune_adapter=DummyAdapter(),
            )


@pytest.mark.platform
@pytest.mark.parametrize("searcher", list(SEARCHER_PRESETS.keys()))
@pytest.mark.parametrize("scheduler", list(SCHEDULER_PRESETS.keys()))
def test_run(searcher, scheduler):
    hyperparameter_tune_kwargs = dict(
        searcher=searcher,
        scheduler=scheduler,
        num_trials=2,
    )
    with tempfile.TemporaryDirectory() as root:
        analysis = run(
            trainable=_dummy_trainable,
            trainable_args=dict(),
            search_space=DUMMY_SEARCH_SPACE,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            metric="score",
            mode="min",
            save_dir=root,
            ray_tune_adapter=DummyAdapter(),
        )
        assert analysis is not None
        assert analysis.best_result is not None
        assert analysis.best_result["score"] is not None
