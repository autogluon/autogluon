from ..abstract.abstract_model import AbstractModel
from .default_parameters import get_default_parameters
from gluonts.model.seq2seq._mq_dnn_estimator import MQCNNEstimator
import os
import time
from ......core import Int, Space
from ..abstract.model_trial import model_trial
from tqdm import tqdm
from ......scheduler.fifo import FIFOScheduler


class MQCNNModel(AbstractModel):

    def __init__(self, hyperparameters=None, model=None):
        super().__init__(hyperparameters, model)
        self.set_default_parameters()
        if hyperparameters is not None:
            self.params.update(hyperparameters)

    def set_default_parameters(self):
        self.params = get_default_parameters()

    def create_model(self):
        self.model = MQCNNEstimator.from_hyperparameters(**self.params)

    def fit(self, train_ds):
        # print(self.params)
        self.create_model()
        self.model = self.model.train(train_ds)

    def predict(self, test_ds, num_samples=100):
        # self.model = self.model.predict(test_ds)
        from gluonts.evaluation.backtest import make_evaluation_predictions
        forecast_it, ts_it = make_evaluation_predictions(dataset=test_ds,
                                                   predictor=self.model,
                                                   num_samples=num_samples)
        return list(tqdm(forecast_it, total=len(test_ds))), list(tqdm(ts_it, total=len(test_ds)))

    def score(self, y, quantiles=[0.9]):
        from gluonts.evaluation import Evaluator
        evaluator = Evaluator(quantiles=quantiles)
        forecasts, tss = self.predict(y)
        num_series = len(tss)
        # print(num_series, forecasts, tss)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)

        # print(json.dumps(agg_metrics, indent=4))
        return agg_metrics["mean_wQuantileLoss"]
    #
    # def hyperparameter_tune(self, train_data, test_data, scheduler_options, **kwargs):
    #     # verbosity = kwargs.get('verbosity', 2)
    #     time_start = time.time()
    #     # logger.log(15, "Starting generic AbstractModel hyperparameter tuning for %s model..." % self.name)
    #     # self._set_default_searchspace()
    #     params_copy = self.params.copy()
    #     # directory = self.path  # also create model directory if it doesn't exist
    #     # TODO: This will break on S3. Use tabular/utils/savers for datasets, add new function
    #     scheduler_func, scheduler_options = scheduler_options  # Unpack tuple
    #     if scheduler_func is None or scheduler_options is None:
    #         raise ValueError("scheduler_func and scheduler_options cannot be None for hyperparameter tuning")
    #     params_copy['num_threads'] = scheduler_options['resource'].get('num_cpus', None)
    #     params_copy['num_gpus'] = scheduler_options['resource'].get('num_gpus', None)
    #     # dataset_train_filename = 'dataset_train.p'
    #     # train_path = directory + dataset_train_filename
    #     # save_pkl.save(path=train_path, object=(X_train, y_train))
    #     #
    #     # dataset_val_filename = 'dataset_val.p'
    #     # val_path = directory + dataset_val_filename
    #     # save_pkl.save(path=val_path, object=(X_val, y_val))
    #
    #     # if not any(isinstance(params_copy[hyperparam], Space) for hyperparam in params_copy):
    #     #     logger.warning(
    #     #         "Attempting to do hyperparameter optimization without any search space (all hyperparameters are already fixed values)")
    #     # else:
    #     #     logger.log(15, "Hyperparameter search space for %s model: " % self.name)
    #     #     for hyperparam in params_copy:
    #     #         if isinstance(params_copy[hyperparam], Space):
    #     #             logger.log(15, f"{hyperparam}:   {params_copy[hyperparam]}")
    #
    #     util_args = dict(
    #         dataset_train_filename=train_data,
    #         dataset_val_filename=test_data,
    #         model=self,
    #         time_start=time_start,
    #         time_limit=scheduler_options['time_out']
    #     )
    #
    #     model_trial.register_args(util_args=util_args, **params_copy)
    #     scheduler: FIFOScheduler = scheduler_func(model_trial, **scheduler_options)
    #
    #     scheduler.run()
    #     scheduler.join_jobs()
    #
    #     return self._get_hpo_results(scheduler=scheduler, scheduler_options=scheduler_options,
    #                                  time_start=time_start)
