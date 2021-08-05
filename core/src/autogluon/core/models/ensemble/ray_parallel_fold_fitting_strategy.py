import ray
from numpy import ndarray
from pandas import DataFrame, Series
from ray.util import placement_group, placement_group_table
import time
from time import sleep

from autogluon.core.models.ensemble.fold_fitting_strategy import SequentialLocalFoldFittingStrategy


@ray.remote
def model_fit_task_ray(X_fold, X_val_fold, fold_model, kwargs_fold, time_limit_fold, y_fold, y_val_fold):
    fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **kwargs_fold)
    time_train_end_fold = time.time()
    return fold_model, time_train_end_fold

class RayParallelFitter(SequentialLocalFoldFittingStrategy):

    def __init__(self, bagged_ensemble_model, X: DataFrame, y: Series, sample_weight, time_limit: float, time_start: float, models: list, oof_pred_proba: ndarray, oof_pred_model_repeats: ndarray, save_folds: bool):
        super().__init__(bagged_ensemble_model, X, y, sample_weight, time_limit, time_start, models, oof_pred_proba, oof_pred_model_repeats, save_folds)
        # ray.util.connect("localhost:10001")
        # ray.init(address='auto')
        print('init')
        ray.init()

    def schedule_fold_model_fit(self, model_base, fold_ctx, kwargs):
        args = [model_base, fold_ctx, kwargs]
        args_refs = [ray.put(arg) for arg in args]
        print('...model_fit')

        pg = placement_group([{"CPU": 2}], strategy="STRICT_SPREAD")
        ray.get(pg.ready())
        print(placement_group_table(pg))
        results_ref = model_fit_task_ray.options(placement_group=pg).remote(*args_refs)
        self.jobs.append((results_ref, time_start_fold, on_fit_end_fn))

    def wait_for_completion(self):
        print('...wait')
        for (results_ref, time_start_fold, on_fit_end_fn) in self.jobs:
            fold_model, time_train_end_fold = ray.get(results_ref)
            print(f'{fold_model} | {time_train_end_fold}')
            on_fit_end_fn(fold_model, time_train_end_fold, time_start_fold)

        ray.shutdown()
