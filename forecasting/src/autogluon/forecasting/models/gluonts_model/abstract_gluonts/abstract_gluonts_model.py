from pathlib import Path
import pandas as pd
import numpy as np
import copy
import os
import time
from tqdm import tqdm
import logging

from gluonts.model.predictor import Predictor
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.forecast import SampleForecast, QuantileForecast
from gluonts.evaluation import Evaluator

import autogluon.core.utils.savers.save_pkl as save_pkl
import autogluon.core.utils.loaders.load_pkl as load_pkl
from autogluon.core.scheduler.fifo import FIFOScheduler
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.task.base.base_predictor import BasePredictor
from autogluon.core.constants import AG_ARGS_FIT, BINARY, REGRESSION, REFIT_FULL_SUFFIX, OBJECTIVES_TO_NORMALIZE
from ...abstract.abstract_model import AbstractModel
from ..abstract_gluonts.model_trial import model_trial

logger = logging.getLogger(__name__)


class AbstractGluonTSModel(AbstractModel):

    model_file_name = "model.pkl"
    gluonts_model_path = "gluon_ts"
    prev_fitting_time = []

    def __init__(self, path: str, freq: str, prediction_length: int, name: str, eval_metric: str = None,
                 hyperparameters=None, model=None, **kwargs):
        """
        Create a new model
        Args:
            path(str): directory where to store all the model
            freq(str): frequency
            name(str): name of subdirectory inside path where model will be saved.
            eval_metric(str): objective function the model intends to optimize, will use mean_wQuantileLoss by default
            hyperparameters: various hyperparameters that will be used by model (can be search spaces instead of fixed values).
        """
        super().__init__()
        self.name = name
        self.path_root = path
        self.path_suffix = self.name + os.path.sep
        self.path = self.path_root + self.path_suffix

        if eval_metric is None:
            eval_metric = "mean_wQuantileLoss"
        if eval_metric is not None and eval_metric not in ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]:
            raise ValueError(f"metric {eval_metric} is not available yet.")
        self.eval_metric = eval_metric

        self.params = {}
        self.set_default_parameters()
        self.params["freq"] = freq
        self.params["prediction_length"] = prediction_length

        self.best_configs = self.params.copy()

        self.nondefault_parameters = []
        if hyperparameters is not None:
            self.params.update(hyperparameters)
            self.nondefault_parameters = list(hyperparameters.keys())[:]

        self.model = model
        self.val_score = None
        self.fit_time = None
        self.predict_time = None
        self.quantiles = kwargs.get("quantiles", ["0.5"])
        self.params["quantiles"] = self.quantiles

    def set_contexts(self, path_context):
        self.path = self.create_contexts(path_context)
        self.path_suffix = self.name + os.path.sep
        self.path_root = self.path.rsplit(self.path_suffix, 1)[0]

    def set_default_parameters(self):
        self.params = {}

    @staticmethod
    def create_contexts(path_context):
        path = path_context
        return path

    def save(self, path: str = None,):
        if path is None:
            path = self.path
        # save gluonts model
        weight_path = path + self.gluonts_model_path
        os.makedirs(weight_path, exist_ok=True)
        self.model.serialize(Path(weight_path))
        self.model = None
        # save self
        file_path = path + self.model_file_name
        save_pkl.save(path=file_path, object=self)

        return path

    @classmethod
    def load(cls, path: str, reset_path=True):
        file_path = path + cls.model_file_name
        model = load_pkl.load(path=file_path,)
        # if reset_path:
        #     model.set_context(path)
        weight_path = path + cls.gluonts_model_path
        model.model = Predictor.deserialize(Path(weight_path))
        return model

    def create_model(self):
        """
        Create the model using gluonts.
        """
        pass

    def fit(self, train_data, val_data=None, time_limit=None):
        """
        Fitting the model.
        """
        if time_limit is None or time_limit > 0:
            start_time = time.time()
            self.create_model()
            self.model = self.model.train(train_data, validation_data=val_data)
            end_time = time.time()
            AbstractGluonTSModel.prev_fitting_time.append(end_time - start_time)
        else:
            raise TimeLimitExceeded

    def predict(self, data, quantiles=None):
        """
        Return forecasts for given dataset and quantiles.

        Parameters
        __________
        data: dataset in the same format as train data

        quantiles: list of ints, default=None
              if quantiles=None, it will by default give all the quantiles that the model is trained for.
        """
        logger.log(30, f"Predicting with model {self.name}")
        if quantiles is None:
            quantiles = [str(q) for q in self.quantiles]
        else:
            quantiles = [str(q) for q in quantiles]
        result_dict = {}
        predicted_targets = list(self.model.predict(data))
        if isinstance(predicted_targets[0], QuantileForecast):
            status = [0 < float(quantiles[i]) < 1 and str(quantiles[i]) in predicted_targets[0].forecast_keys for i in range(len(quantiles))]
        elif isinstance(predicted_targets[0], SampleForecast):
            transformed_targets = []
            for forecast in predicted_targets:
                tmp = []
                for quantile in quantiles:
                    tmp.append(forecast.quantile(quantile))
                transformed_targets.append(QuantileForecast(forecast_arrays=np.array(tmp),
                                                            start_date=forecast.start_date,
                                                            freq=forecast.freq,
                                                            forecast_keys=quantiles,
                                                            item_id=forecast.item_id))
            predicted_targets = copy.deepcopy(transformed_targets)
            status = [0 < float(quantiles[i]) < 1 and str(quantiles[i]) in predicted_targets[0].forecast_keys for i in range(len(quantiles))]
        else:
            raise TypeError("DistributionForecast is not yet supported.")

        if not all(status):
            raise ValueError("Invalid quantile value.")
        if isinstance(data, pd.DataFrame):
            index = data.get_index()
        else:
            index = [i["item_id"] for i in data]

        index_count = {}
        for idx in index:
            index_count[idx] = index_count.get(idx, 0) + 1

        for i in range(len(index)):
            tmp_dict = {}
            for quantile in quantiles:
                tmp_dict[quantile] = predicted_targets[i].quantile(str(quantile))
            df = pd.DataFrame(tmp_dict)
            df.index = pd.date_range(start=predicted_targets[i].start_date,
                                     periods=self.params["prediction_length"],
                                     freq=self.params["freq"])
            if index_count[index[i]] > 1:
                result_dict[f"{index[i]}_{predicted_targets[i].start_date}"] = df
            else:
                result_dict[index[i]] = df
        return result_dict

    def predict_for_scoring(self, data, num_samples=100):
        forecast_it, ts_it = make_evaluation_predictions(dataset=data,
                                                         predictor=self.model,
                                                         num_samples=num_samples)
        return list(tqdm(forecast_it, total=len(data))), list(tqdm(ts_it, total=len(data)))

    def score(self, data, metric=None, num_samples=100):
        """
        Return the evaluation scores for given metric and dataset.

        Parameters
        __________
        data: dataset for evaluation in the same format as train dataset

        metric: str, default=None
                if metric is None, we will by default use mean_wQuantileLoss for scoring.
                should be one of "MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"

        num_samples: int, default=100
                number of samples selected for evaluation if the output of the model is DistributionForecast in gluonts
        """
        if metric is None:
            metric = self.eval_metric
        if metric is not None and metric not in ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]:
            raise ValueError(f"metric {metric} is not available yet.")

        # if quantiles are given, use the given one, otherwise use the default
        if "quantiles" in self.params:
            evaluator = Evaluator(quantiles=self.params["quantiles"])
        else:
            evaluator = Evaluator()

        forecasts, tss = self.predict_for_scoring(data, num_samples=num_samples)
        num_series = len(tss)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)
        return agg_metrics[metric]

    def hyperparameter_tune(self, train_data, val_data, scheduler_options, time_limit=None, **kwargs):
        """
        Do hyperparamter tuning, return hyperparamemter tuning results.

        Parameters
        __________
        train_data: data for training

        val_data: data for validation, used for evaluting parameter settings

        scheduler_options: tuple in the form (scheduler_cls, scheduler_params)

        time_limit: roughly how long will the hyperparameter tuning process last.
        """
        time_start = time.time()
        logger.log(30, f"Start hyperparameter tuning for {self.name}")
        params_copy = self.params.copy()

        directory = self.path

        dataset_train_filename = 'dataset_train.p'
        train_path = directory + dataset_train_filename
        save_pkl.save(path=train_path, object=train_data)

        dataset_val_filename = 'dataset_val.p'
        val_path = directory + dataset_val_filename
        save_pkl.save(path=val_path, object=val_data)
        scheduler_func, scheduler_params = scheduler_options
        util_args = dict(
            train_data_path=dataset_train_filename,
            val_data_path=dataset_val_filename,
            directory=directory,
            model=self,
            time_start=time_start,
            time_limit=scheduler_params.get("time_out", None)
        )

        model_trial.register_args(util_args=util_args, **params_copy)
        scheduler = scheduler_func(model_trial, **scheduler_params)
        scheduler.run()
        scheduler.join_jobs()
        self.best_configs.update(scheduler.get_best_config())
        return self._get_hpo_results(scheduler, scheduler_params, time_start)

    def _get_hpo_results(self, scheduler, scheduler_options, time_start):
        # Store results / models from this HPO run:
        best_hp = scheduler.get_best_config()  # best_hp only contains searchable stuff
        hpo_results = {
            'best_reward': scheduler.get_best_reward(),
            'best_config': best_hp,
            'total_time': time.time() - time_start,
            'metadata': scheduler.metadata,
            'training_history': scheduler.training_history,
            'config_history': scheduler.config_history,
            'reward_attr': scheduler._reward_attr,
            'args': model_trial.args
        }

        hpo_results = BasePredictor._format_results(hpo_results)  # results summarizing HPO for this model

        hpo_models = {}  # stores all the model names and file paths to model objects created during this HPO run.
        hpo_model_performances = {}
        for trial in sorted(hpo_results['trial_info'].keys()):
            file_id = "trial_" + str(trial)  # unique identifier to files from this trial
            trial_model_name = self.name + os.path.sep + file_id
            trial_model_path = self.path_root + trial_model_name + os.path.sep
            hpo_models[trial_model_name] = trial_model_path
            hpo_model_performances[trial_model_name] = hpo_results['trial_info'][trial][scheduler._reward_attr]

        logger.log(15, "Time for %s model HPO: %s" % (self.name, str(hpo_results['total_time'])))
        logger.log(15, "Best hyperparameter configuration for %s model: " % self.name)
        logger.log(15, str(best_hp))
        return hpo_models, hpo_model_performances, hpo_results

    def get_info(self):
        info = {
            'name': self.name,
            'model_type': type(self).__name__,
            'eval_metric': self.eval_metric,
            'fit_time': self.fit_time,
            'predict_time': self.predict_time,
            'val_score': self.val_score,
            'hyperparameters': self.params,
        }
        return info

    def __repr__(self):
        return self.name

    # After calling this function, returned model should be able to be fit as if it was new, as well as deep-copied.
    def convert_to_template(self):
        model = self.model
        self.model = None
        template = copy.deepcopy(self)
        template.reset_metrics()
        self.model = model
        return template

    # After calling this function, model should be able to be fit without test data using the iterations trained by the original model
    def convert_to_refit_full_template(self):
        template = self.convert_to_template()
        template.name = template.name + REFIT_FULL_SUFFIX
        template.set_contexts(self.path_root + template.name + os.path.sep)
        return template

    def reset_metrics(self):
        """
        Reset metrics to be None, usually used for refitting.
        """
        self.fit_time = None
        self.predict_time = None
        self.val_score = None
