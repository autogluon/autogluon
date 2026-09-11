"""`_ray_fit` with `keep_fold=False`: the worker saves nothing and returns the fold's trained parameters."""

import os

import numpy as np
import pandas as pd
import pytest
from autogluon.common.utils.cv_splitter import CVSplitter
from autogluon.core.models import BaggedEnsembleModel
from autogluon.core.models.dummy.dummy_model import DummyModel
from autogluon.core.models.ensemble.fold_fitting_strategy import _ray_fit


@pytest.mark.parametrize("keep_fold", [True, False])
def test_ray_fit_keep_fold(tmp_path, keep_fold):
    ray = pytest.importorskip("ray")
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, size=40))
    fold_fit_args_list, _, _ = BaggedEnsembleModel._generate_fold_configs(
        X=X,
        y=y,
        cv_splitter=CVSplitter(n_splits=2, n_repeats=1, stratify=True, random_state=0),
        k_fold_start=0,
        k_fold_end=2,
        n_repeat_start=0,
        n_repeat_end=1,
        vary_seed_across_folds=False,
        random_seed_offset=0,
    )
    bag_path = str(tmp_path / "bag")
    model_base = DummyModel(path=bag_path, name="Dummy", problem_type="binary", eval_metric="log_loss")
    model_base.initialize(X=X, y=y)
    out = _ray_fit(
        model_base=model_base,
        bagged_ensemble_model_path=bag_path,
        X=X,
        y=y,
        X_pseudo=None,
        y_pseudo=None,
        task_id=0,
        fold_ctx=fold_fit_args_list[0],
        task_gpu_ids=[],
        time_limit_fold=None,
        save_bag_folds=False,
        resources={"num_cpus": 1, "num_gpus": 0},
        kwargs_fold={},
        head_node_id=ray.get_runtime_context().get_node_id(),
        keep_fold=keep_fold,
    )
    assert isinstance(out, tuple) and len(out) == 10
    name, params_trained = out[0], out[-1]
    assert name == "DummyS1F1"
    assert os.path.isdir(os.path.join(bag_path, name)) == keep_fold
    if keep_fold:
        assert params_trained is None
    else:
        assert isinstance(params_trained, dict)
