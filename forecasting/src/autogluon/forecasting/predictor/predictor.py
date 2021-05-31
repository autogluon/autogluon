import copy
import logging
import pprint
import pandas as pd
import numpy as np

from autogluon.core.dataset import TabularDataset
from autogluon.core.task.base.base_task import schedulers
from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils.utils import setup_outputdir
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.forecasting.utils.dataset_utils import rebuild_tabular, train_test_split_gluonts, \
    train_test_split_dataframe
from autogluon.forecasting.utils.dataset_utils import TimeSeriesDataset

from gluonts.dataset.common import FileDataset, ListDataset
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import SampleForecast, QuantileForecast

from ..learner import AbstractLearner, DefaultLearner
from ..trainer import AbstractTrainer
from ..utils.dataset_utils import time_series_dataset

logger = logging.getLogger()  # return root logger


class ForecastingPredictor:
    Dataset = TabularDataset
    predictor_file_name = 'predictor.pkl'

    def __init__(
            self,
            eval_metric=None,
            path=None,
            verbosity=2,
            **kwargs
    ):
        self.verbosity = verbosity
        set_logger_verbosity(self.verbosity, logger=logger)
        # self._validate_init_kwargs(kwargs)
        path = setup_outputdir(path)

        learner_type = kwargs.pop('learner_type', DefaultLearner)
        learner_kwargs = kwargs.pop('learner_kwargs', dict())

        self.eval_metric = eval_metric
        self._learner: AbstractLearner = learner_type(path_context=path, eval_metric=eval_metric, **learner_kwargs)
        self._learner_type = type(self._learner)
        self._trainer = None

        self.index_column = None
        self.time_column = None
        self.target_column = None

        self.static_cat_columns = None
        self.static_real_columns = None
        self.use_feat_static_cat = False
        self.use_feat_static_real = False
        self.cardinality = None
        self.prev_inferred_static_features = {}

    def fit(self,
            train_data,
            prediction_length,
            index_column="index_column",
            time_column="time_column",
            target_column="target_column",
            val_data=None,
            hyperparameters=None,
            hyperparameter_tune_kwargs=None,
            time_limits=None,
            static_features=None,
            **kwargs):
        if self._learner.is_fit:
            raise AssertionError(
                'Predictor is already fit! To fit additional models, refer to `predictor.fit_extra`, or create a new '
                '`Predictor`.')

        self.index_column = index_column
        self.time_column = time_column
        self.target_column = target_column
        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_kwargs(kwargs)
        if not self._validate_hyperparameter_tune_kwargs(hyperparameter_tune_kwargs, time_limits=time_limits):
            hyperparameter_tune_kwargs = None
            logger.warning(30, "Invalid hyperparameter_tune_kwarg, disabling hyperparameter tuning.")

        verbosity = kwargs.get('verbosity', self.verbosity)
        set_logger_verbosity(verbosity, logger=logger)
        if hyperparameter_tune_kwargs is None:
            hyperparameter_tune = False
        else:
            hyperparameter_tune = True

        if verbosity >= 3:
            logger.log(20, '============ fit kwarg info ============')
            logger.log(20, 'User Specified kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs_orig)}')
            logger.log(20, 'Full kwargs:')
            logger.log(20, f'{pprint.pformat(kwargs)}')
            logger.log(20, '========================================')

        freq = kwargs.get("freq", None)
        set_logger_verbosity(verbosity, logger)
        if isinstance(train_data, pd.DataFrame):
            logger.log(30, "Training with dataset in tabular format...")
            train_data = rebuild_tabular(train_data,
                                         index_column=index_column,
                                         target_column=target_column,
                                         time_column=time_column)
            logger.log(30, "Finish rebuilding the data, showing the top five rows.")
            logger.log(30, train_data.head())
            if val_data is None:
                logger.log(30, "Validation data is None, will do auto splitting...")
                train_data, val_data = train_test_split_dataframe(train_data, prediction_length)
            else:
                val_data = rebuild_tabular(val_data,
                                           index_column=index_column,
                                           target_column=target_column,
                                           time_column=time_column)
            train_data = TimeSeriesDataset(train_data, index_column=index_column, static_features=static_features)

            self.static_cat_columns = train_data.static_cat_columns()
            self.static_real_columns = train_data.static_real_columns()
            self.use_feat_static_cat = train_data.use_feat_static_cat()
            self.use_feat_static_real = train_data.use_feat_static_real()
            self.cardinality = train_data.get_static_cat_cardinality()
            self.prev_inferred_static_features = {"static_cat_columns": self.static_cat_columns,
                                                  "static_real_columns": self.static_real_columns,
                                                  "cardinality": self.cardinality}
            freq = train_data.get_freq()

            val_data = TimeSeriesDataset(val_data, index_column=index_column, static_features=static_features, prev_inferred=self.prev_inferred_static_features)
        elif isinstance(train_data, FileDataset) or isinstance(train_data, ListDataset):
            logger.log(30, "Training with dataset in gluon-ts format...")
            if val_data is None:
                logger.log(30, "Validation data is not specified, will do auto splitting...")
                train_data, val_data = train_test_split_gluonts(train_data, prediction_length, freq)
        else:
            raise TypeError("Does not support dataset type:", type(train_data))

        refit_full = kwargs["refit_full"]
        save_data = kwargs["save_data"]
        if not save_data:
            logger.log(30,
                       'Warning: `save_data=False` will disable or limit advanced functionality after training such '
                       'as feature importance calculations. It is recommended to set `save_data=True` unless you '
                       'explicitly wish to not have the data saved to disk.')
            if refit_full:
                raise ValueError(
                    '`refit_full=True` is only available when `cache_data=True`. Set `cache_data=True` to utilize '
                    '`refit_full`.')

        set_best_to_refit_full = kwargs['set_best_to_refit_full']
        if set_best_to_refit_full and not refit_full:
            raise ValueError(
                '`set_best_to_refit_full=True` is only available when `refit_full=True`. Set `refit_full=True` to '
                'utilize `set_best_to_refit_full`.')

        random_seed = kwargs.get('random_seed', 0)
        logger.log(30, f"Random seed set to {random_seed}")
        quantiles = kwargs.get("quantiles", ["0.5"])
        logger.log(30, f"All models will be trained for quantiles {quantiles}.")

        scheduler_cls, scheduler_params = scheduler_factory(hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                                                            time_out=time_limits,
                                                            nthreads_per_trial='auto', ngpus_per_trial='auto')
        if time_limits is None and scheduler_params["num_trials"] is None:
            logger.log(30, "None of time_limits and num_tirals are set, by default setting num_tirals=2")
            scheduler_params["num_trials"] = 2

        scheduler_options = (scheduler_cls, scheduler_params)

        self._learner.fit(train_data=train_data,
                          freq=freq,
                          prediction_length=prediction_length,
                          use_feat_static_cat=self.use_feat_static_cat,
                          use_feat_static_real=self.use_feat_static_real,
                          cardinality=self.cardinality,
                          val_data=val_data,
                          scheduler_options=scheduler_options,
                          hyperparameters=hyperparameters,
                          hyperparameter_tune=hyperparameter_tune,
                          quantiles=quantiles,
                          time_limits=time_limits,)

        self._set_post_fit_vars()
        self._post_fit(
            keep_only_best=kwargs["keep_only_best"],
            refit_full=kwargs['refit_full'],
            set_best_to_refit_full=kwargs["set_best_to_refit_full"],
        )
        self.save()
        return self

    def _validate_fit_kwargs(self, kwargs):
        kwargs_default = {
            "set_best_to_refit_full": False,
            "keep_only_best": False,
            "refit_full": False,
            "save_data": True,
            "freq": None,
            "quantiles": ["0.5"]
        }
        copied_kwargs = copy.deepcopy(kwargs_default)
        copied_kwargs.update(kwargs)
        return copied_kwargs

    def _set_post_fit_vars(self, learner: AbstractLearner = None):
        if learner is not None:
            self._learner: AbstractLearner = learner
        self._learner_type = type(self._learner)
        if self._learner.trainer_path is not None:
            self._trainer: AbstractTrainer = self._learner.load_trainer()

    def get_model_names(self):
        """Returns the list of model names trained in this `predictor` object."""
        return self._trainer.get_model_names_all()

    def preprocessing(self, data, time_series_to_predict=None, static_features=None):
        if (self.use_feat_static_cat or self.use_feat_static_real) and static_features is None:
            raise ValueError("Static features are used for training, cannot predict without static features.")
        if isinstance(data, pd.DataFrame):
            data = time_series_dataset(data,
                                       index_column=self.index_column,
                                       target_column=self.target_column,
                                       time_column=self.time_column,
                                       chosen_ts=time_series_to_predict,
                                       static_features=static_features,
                                       prev_inferred=self.prev_inferred_static_features)
        return data

    def predict(self, data, time_series_to_predict=None, model=None, for_score=False, static_features=None, **kwargs):
        processed_data = self.preprocessing(data, time_series_to_predict=time_series_to_predict, static_features=static_features)
        predict_targets = self._learner.predict(processed_data, model=model, for_score=for_score, **kwargs)
        return predict_targets

    def evaluate(self, data, static_features=None, **kwargs):
        processed_data = self.preprocessing(data, static_features=static_features)
        perf = self._learner.score(processed_data, **kwargs)
        return perf

    # def evaluate_predictions(self, forecasts, tss, **kwargs):
    #
    #     return self._learner.evaluate(forecasts, tss, **kwargs)

    @classmethod
    def load(cls, output_directory, verbosity=2):
        if output_directory is None:
            raise ValueError("output_directory cannot be None in load()")
        output_directory = setup_outputdir(output_directory)  # replace ~ with absolute path if it exists
        logger.log(30, f"Loading predictor from path {output_directory}")
        learner = AbstractLearner.load(output_directory)
        predictor = load_pkl.load(path=learner.path + cls.predictor_file_name)
        predictor._learner = learner
        predictor._trainer = learner.load_trainer()
        return predictor

    def save(self):
        """ Save this predictor to file in directory specified by this Predictor's `output_directory`.
            Note that `fit()` already saves the predictor object automatically
            (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        tmp_learner = self._learner
        tmp_trainer = self._trainer
        self._learner = None
        self._trainer = None
        save_pkl.save(path=tmp_learner.path + self.predictor_file_name, object=self)
        self._learner = tmp_learner
        self._trainer = tmp_trainer

    def info(self):
        return self._learner.get_info(include_model_info=True)

    def get_model_best(self):
        return self._trainer.get_model_best()

    def leaderboard(self, data=None, static_features=None):
        if data is not None:
            data = self.preprocessing(data, static_features=static_features)
        return self._learner.leaderboard(data)

    def fit_summary(self, verbosity=3):
        """
            Output summary of information about models produced during `fit()`.
            May create various generated summary plots and store them in folder: `Predictor.output_directory`.

            Parameters
            ----------
            verbosity : int, default = 3
                Controls how detailed of a summary to ouput.
                Set <= 0 for no output printing, 1 to print just high-level summary,
                2 to print summary and create plots, >= 3 to print all information produced during fit().

            Returns
            -------
            Dict containing various detailed information. We do not recommend directly printing this dict as it may be very large.
        """
        hpo_used = len(self._trainer.hpo_results) > 0
        model_types = self._trainer.get_models_attribute_dict(attribute='type')
        model_typenames = {key: model_types[key].__name__ for key in model_types}

        unique_model_types = set(model_typenames.values())  # no more class info
        # all fit() information that is returned:
        results = {
            'model_types': model_typenames,  # dict with key = model-name, value = type of model (class-name)
            'model_performance': self._trainer.get_models_attribute_dict('score'),
            # dict with key = model-name, value = validation performance
            'model_best': self._trainer.model_best,  # the name of the best model (on validation data)
            'model_paths': self._trainer.get_models_attribute_dict('path'),
            # dict with key = model-name, value = path to model file
            'model_fit_times': self._trainer.get_models_attribute_dict('fit_time'),
            'hyperparameter_tune': hpo_used,
            'hyperparameters_userspecified': self._trainer.hyperparameters,
        }
        if hpo_used:
            results['hpo_results'] = self._trainer.hpo_results
        # get dict mapping model name to final hyperparameter values for each model:
        model_hyperparams = {}
        for model_name in self._trainer.get_model_names_all():
            model_obj = self._trainer.load_model(model_name)
            model_hyperparams[model_name] = model_obj.params
        results['model_hyperparams'] = model_hyperparams

        if verbosity > 0:  # print stuff
            print("*** Summary of fit() ***")
            print("Estimated performance of each model:")
            results['leaderboard'] = self._learner.leaderboard()
            # self._summarize('model_performance', 'Validation performance of individual models', results)
            #  self._summarize('model_best', 'Best model (based on validation performance)', results)
            # self._summarize('hyperparameter_tune', 'Hyperparameter-tuning used', results)
            print("Number of models trained: %s" % len(results['model_performance']))
            print("Types of models trained:")
            print(unique_model_types)
            hpo_str = ""
            if hpo_used and verbosity <= 2:
                hpo_str = " (call fit_summary() with verbosity >= 3 to see detailed HPO info)"
            print("Hyperparameter-tuning used: %s %s" % (hpo_used, hpo_str))
            # TODO: uncomment once feature_prune is functional:  self._summarize('feature_prune', 'feature-selection used', results)
            print("User-specified hyperparameters:")
            print(results['hyperparameters_userspecified'])
            print("Feature Metadata (Processed):")
            print("(raw dtype, special dtypes):")
        if verbosity > 2:  # print detailed information
            if hpo_used:
                hpo_results = results['hpo_results']
                print("*** Details of Hyperparameter optimization ***")
                for model_type in hpo_results:
                    hpo_model = hpo_results[model_type]
                    if 'trial_info' in hpo_model:
                        print(
                            f"HPO for {model_type} model:  Num. configurations tried = {len(hpo_model['trial_info'])}, Time spent = {hpo_model['total_time']}s, Search strategy = {hpo_model['search_strategy']}")
                        print(
                            f"Best hyperparameter-configuration (validation-performance: {self.eval_metric} = {hpo_model['validation_performance']}):")
                        print(hpo_model['best_config'])
        if verbosity > 0:
            print("*** End of fit() summary ***")
        return results

    def _post_fit(self, keep_only_best=False, refit_full=False, set_best_to_refit_full=False):
        if refit_full is True:
            if set_best_to_refit_full is True:
                refit_full = 'best'
            else:
                refit_full = 'all'

        if refit_full is not False:
            trainer_model_best = self._trainer.get_model_best()

            self.refit_full(models=refit_full)
            self._trainer = self._learner.load_trainer()
            if set_best_to_refit_full:
                if trainer_model_best in self._trainer.model_full_dict.keys():
                    self._trainer.model_best = self._trainer.model_full_dict[trainer_model_best]
                    # Note: model_best will be overwritten if additional training is done with new models, since model_best will have validation score of None and any new model will have a better validation score.
                    # This has the side-effect of having the possibility of model_best being overwritten by a worse model than the original model_best.
                    self._trainer.save()
                else:
                    logger.warning(f'Best model ({trainer_model_best}) is not present in refit_full dictionary. Training may have failed on the refit model. AutoGluon will default to using {trainer_model_best} for predictions.')

    def refit_full(self, models='all'):
        return self._learner.refit_full(models=models)

    def _validate_hyperparameter_tune_kwargs(self, hyperparameter_tune_kwargs, time_limits=None):
        """
        Returns True if hyperparameter_tune_kwargs is None or can construct a valid scheduler.
        Returns False if hyperparameter_tune_kwargs results in an invalid scheduler.
        """
        if hyperparameter_tune_kwargs is None:
            return True

        scheduler_cls, scheduler_params = scheduler_factory(hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, time_out=time_limits,
                                                            nthreads_per_trial='auto', ngpus_per_trial='auto')
        assert scheduler_params['searcher'] != 'bayesopt_hyperband', "searcher == 'bayesopt_hyperband' not yet supported"
        if scheduler_params.get('dist_ip_addrs', None):
            logger.warning('Warning: dist_ip_addrs does not currently work for Tabular. Distributed instances will not be utilized.')

        if scheduler_params['num_trials'] == 1:
            logger.warning('Warning: Specified num_trials == 1 for hyperparameter tuning, disabling HPO. This can occur if time_limit was not specified in `fit()`.')
            return False

        scheduler_ngpus = scheduler_params['resource'].get('num_gpus', 0)
        if scheduler_ngpus is not None and isinstance(scheduler_ngpus, int) and scheduler_ngpus > 1:
            logger.warning(f"Warning: TabularPredictor currently doesn't use >1 GPU per training run. Detected {scheduler_ngpus} GPUs.")
        return True

    @classmethod
    def evaluate_predictions(cls, forecasts, targets, eval_metric=None):
        quantile_forecasts = []
        for ts_id, forecast in forecasts.items():
            tmp = []
            for quantile in forecast.columns:
                tmp.append(forecast[quantile])
            quantile_forecasts.append(QuantileForecast(
                forecast_arrays=np.array(tmp),
                start_date=forecast.index[0],
                freq=pd.infer_freq(forecast.index),
                forecast_keys=forecast.columns,
                item_id=ts_id,
            ))
        evaluator = Evaluator()
        num_series = len(targets)
        agg_metrics, item_metrics = evaluator(iter(targets), iter(quantile_forecasts), num_series=num_series)
        if eval_metric is None:
            return agg_metrics
        else:
            return agg_metrics[eval_metric]