import copy
import logging
import pprint
import time

import pandas as pd
import numpy as np

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.dataset import TabularDataset
from autogluon.core.scheduler.scheduler_factory import scheduler_factory
from autogluon.core.utils.decorators import apply_presets
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl

from gluonts.dataset.common import FileDataset, ListDataset
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import QuantileForecast

from ..configs.presets_configs import forecasting_presets_configs
from ..learner import AbstractLearner, DefaultLearner
from ..trainer import AbstractTrainer
from ..utils.dataset_utils import (
    time_series_dataset,
    rebuild_tabular,
    train_test_split_gluonts,
    train_test_split_dataframe,
    TimeSeriesDataset,
)
from ..utils.warning_filters import evaluator_warning_filter

logger = logging.getLogger()  # return root logger


class ForecastingPredictor:
    """
    AutoGluon ForecastingPredictor predicts future (numeric) values of multiple related time-series.

    Parameters
    ----------
    eval_metric : str, default = None
        Metric by which predictions will be ultimately evaluated on future test data.
        AutoGluon tunes factors such as hyperparameters, early-stopping, etc. in order to improve this metric on validation data.

        Available options include:
           ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]

        If `eval_metric = None`, it is set by default as "mean_wQuantileLoss".
        For more information about these options, please see the GluonTS documentation: https://ts.gluon.ai/api/gluonts/gluonts.evaluation.metrics.html
    path : str, default = None
       Path to directory where models and intermediate outputs should be saved.
       If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
       Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
       Otherwise files from first `fit()` will be overwritten by second `fit()`.
    verbosity : int, default = 2
       Verbosity levels range from 0 to 4 and control how much information is printed.
       Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
       If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
       where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
    **kwargs :
       learner_type : AbstractLearner, default = DefaultLearner
           A class which inherits from `AbstractLearner`. This dictates the inner logic of `predictor`.
           If you don't know what this is, keep it as the default.
       learner_kwargs : dict, default = None
           Kwargs to send to the learner (for advanced users only). Options include:

           trainer_type : AbstractTrainer, default = AutoTrainer
               A class inheriting from `AbstractTrainer` that controls training of many models.
               If you don't know what this is, keep it as the default.

    Attributes
    ----------
    path : str
    Path to directory where all models used by this Predictor are stored.
    eval_metric: function or str
       What metric is used to evaluate predictive performance.
    index_column: str or None
       Name of column in training/validation data that contains an index ID specifying which time series is being observed at each time-point (for datasets containing multiple time-series).
       If None, we will assume that there is only one time series in the dataset.
    time_column: str
       Name of column in training/validation data that lists the time of each observation.
    target_column: str
       Name of column in training/validation data that contains the target time-series value to be predicted.
    static_cat_columns: str, default = None
       Names of columns that contain static (non time-varying) categorical features.
       These are automatically inferred if a static feature dataframe is provided in `fit()`.
    static_real_columns: str, default = None
        Names of columns that contain static (non time-varying) numeric features.
        These are automatically inferred if a static feature dataframe is provided in `fit()`.
    use_feat_static_cat: bool
       Whether to use categorical static features when training models, assuming static features are provided in `fit()`.
       If any static feature is inferred to be categorical, `use_feat_static_cat` will be set to True by default.
       You can override this default via the `hyperparameters` argument of `fit()`.
    use_feat_static_real: bool
       Whether to use numeric static features when training models, assuming static features are provided in `fit()`.
       If any static feature is inferred to be real-valued, `use_feat_static_real` will be set to True by default.
       You can override this default via the `hyperparameters` argument of `fit()`.
    cardinality: List of ints, default = None
       The cardinality (number of categories) of each categorical static feature.
       This is automatically inferred from the set of observed categories in any static feature found to be categorical.
    prev_inferred_static_features:
       Static features type will be inferred when processing training data.
       This dictionary contains the inferred type of each static feature, and is utilized during processing of static features in extra test data.
    """

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

    @apply_presets(forecasting_presets_configs)
    def fit(self,
            train_data,
            prediction_length,
            index_column,
            time_column,
            target_column,
            val_data=None,
            presets=None,
            hyperparameters=None,
            hyperparameter_tune_kwargs=None,
            time_limit=None,
            static_features=None,
            **kwargs):
        """
        Fit models to predict (distributional) forecasts of future values of multiple related time series based on their historical observations.

        Parameters
        ----------
        train_data: pd.DataFrame or GluonTS FileDataset/ListDataset
            More info about GluonTS data formats and popular time-series datasets provided in GluonTS is available here: https://ts.gluon.ai/api/gluonts/gluonts.dataset.common.html
            A full description of the data formatting requirements for pd.DataFrame is provided in the Forecasting Time-Series - Quick Start Tutorial.
            When it is pd.DataFrame, each row in the table `train_data` corresponds to one observation of one time-series at a particular time.
            If it is pd.DataFrame, `train_data` should look roughly like this (see description of index/time/target columns below):

            ================  ===============  =================
              index_column      time_column      target_column
            ================  ===============  =================
              A                 2020-01-22       1
              A                 2020-01-23       2
              A                 2020-01-24       3
              B                 2020-01-22       1
              B                 2020-01-23       2
              B                 2020-01-24       3
              C                 2020-01-22       1
              C                 2020-01-23       2
              C                 2020-01-24       3
            ================  ===============  =================

        prediction_length: int
            How many time points into the future should our forecasters be trained to predict.
            For example, if our time-series contains daily observations, setting `prediction_length=3` will train models that predict up to 3 days in the future from the most recent observation.
        index_column: str
            For pd.DataFrame datasets containing multiple individual time-series, this is the name of column containing index ID indicating which of these series is being observed.
            `index_column` is not needed if the data only contain observations of a single time-series.

            By default, `index_column="index_column"`.
        time_column: str
            For pd.DataFrame datasets, this is the name of column that indicates the time of each observation.
            The time values in `time_column` must be in a valid pandas time-series format: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

            Time values must be uniformly spaced (eg. every hour, every day, etc.) such that the `pandas.infer_freq` function can be successfully applied to these time values: https://pandas.pydata.org/docs/reference/api/pandas.infer_freq.html

            By default, `time_column="time_column"`.
        target_column: str
            For pd.DataFrame datasets, name of column that contains the target values to forecast (ie. the numeric observations of the time-series).
            This column must contain numeric values, and missing target values should be in a pandas compatible format: https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html

            Ideally each index will be observed over the same time points, but if some observations are missing for a particular (index, time) combination, then you can add these to your dataset with target value NA.
            Each (index, time) combination must only appear in your dataset once with a single target value.

            By default, `target_column="target_column"`.
        val_data: pd.DataFrame or GluonTS FileDataset/ListDataset, default = None
            Validation data reserved for model selection or hyperparameter tuning, rather than training individual models.
            If provided, it should have the same format as `train_data`.
            If None, AutoGluon will reserve the most recent portion of `train_data` for validation.
            Validation scores will by default be computed over the last `prediction_length` time points in the validation data.
        presets: str, default = None
                Optional preset configurations for various arguments in `fit()`.
                Can significantly impact predictive accuracy, memory-footprint, and inference latency of trained models, and various other properties of the returned predictor.
                It is recommended to specify presets and avoid specifying most other `fit()` arguments or model hyperparameters prior to becoming familiar with AutoGluon.
                For example, set `presets="best_quality"` to get a high-accuracy predictor, or set `presets="low_quality"` to get a toy predictor that trains very quick but lacks accuracy.
                Available presets: ["best_quality", "high_quality", "good_quality", "medium_quality", "low_quality", "low_quality_hpo"]
                Details for these presets can be found in the file: forecasting/src/autogluon/forecasting/configs/presets_configs.py
                Any user-specified arguments in `fit()` will override the values used by presets.
        hyperparameter_tune_kwargs: Optional, None by default, can be str or dict
            Valid str values:
                'auto': Uses the 'bayesopt' preset.
                'random': Performs HPO via random search using local scheduler.
                'bayesopt': Performs HPO via bayesian optimization using local scheduler.
            For valid dictionary keys, refer to :class:`autogluon.core.scheduler.FIFOScheduler` documentation.
                The 'searcher' key is required when providing a dict.
                The following is an example dictionary for hyperparameter_tune_kwargs:
                hyperparameter_tune_kwargs={
                                            'scheduler': 'local',
                                            'searcher': 'random',
                                            'num_trials': 2
                                            }
        hyperparameters: str or dict, default = None
            Determines the hyperparameters used by each model.
            If str is passed, will use a preset hyperparameter configuration, can be one of ["default", "default_hpo", "toy", "toy_hpo"], where "toy" settings correspond to tiny models only intended for prototyping.
            If dict is provided, the keys are strings that indicate which model types to train.
            Stable model options include:
               'DeepAR',
               'MQCNN',
               'SFF' (SimpleFeedForward),
            Experimental model options include:
               'AutoTabular' (which adapts Autogluon's TabularPredictor for forecasting)
            If a certain key is missing from hyperparameters, then `fit()` will not train any models of that type.
            For example, set `hyperparameters = { 'SFF':{...} }` means you only want to train SimpleFeedForward models and not any other model.

            Values in the `hyperparameters` dict are themselves dictionaries of hyperparameter settings for each model type.
            Each hyperparameter can either be a single fixed value or a search space containing many possible values.
            A search space should only be provided when `hyperparameter_tune_kwargs` is specified (ie. hyperparameter-tuning is utilized).
            Any omitted hyperparameters not specified here will be set to default values.
            Default `hyperparameters` settings are listed in file: `autogluon/forecasting/trainer/model_presets/presets.py`
            and details regarding the hyperparameters you can specify for each model are provided in the following links:
            DeepAR: https://ts.gluon.ai/api/gluonts/gluonts.model.deepar.html

            MQCNN: https://ts.gluon.ai/api/gluonts/gluonts.model.seq2seq.html

            SFF: https://ts.gluon.ai/api/gluonts/gluonts.model.simple_feedforward.html

            AutoTabular: https://ts.gluon.ai/api/gluonts/gluonts.nursery.autogluon_tabular.html

        time_limit: int, default=None
            Approximately how long hyperparameter tuning will run for (wallclock time in seconds).
            Only considered when `hyperparameter_tune_kwargs` is not None.

        static_features: pd.Dataframe, default=None
            static features used for training.

    **kwargs:

        refit_full: bool, default = False
            Whether to retrain models on all of the data (training + validation) after validation scores have been computed.
            If `refit_full=True`, it will be treated as `refit_full='all'`.
            If `refit_full=False`, refitting will not occur and models will never been trained on the validation data.
            Valid str values:
                `all`: refits all models.
                `best`: refits only the best model (and its ancestors if it is a stacker model).

        set_best_to_refit_full: bool, default=False
            If True, will change the default model used for prediction (when model is not specified) to the refit_full version of the model that exhibited the highest validation score.
            Only valid if `refit_full` is set. If `refit_full` is set = True and `set_best_to_refit_full` is not specified, the latter will be by default set = True.

        quantiles: List[float], default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            List of increasing floats in [0.1, 0.2, ..., 0.9] that specifies which quantiles should be estimated when making distributional forecasts.

        freq: str, default = None
            Only need to provide this when using a built-in GluonTS dataset.
            The frequency of your time-series which corresponds to the spacing between adjacent time-points (e.g. daily vs weekly vs monthly, etc.).
            Models will be trained to forecast at this granularity.
            An example of a valid frequency would be: "1D".
        """
        start_time = time.time()
        if self._learner.is_fit:
            raise AssertionError(
                'Predictor is already fit! To fit additional models, refer to `predictor.fit_extra`, or create a new '
                '`Predictor`.')
        if presets is not None:
            logger.log(30, f"presets is set to be {presets}")
        if index_column is None:
            logger.log(30, "index_column=None, assuming there is only one time series in the dataset.")
        self.index_column = index_column
        self.time_column = time_column
        self.target_column = target_column
        kwargs_orig = kwargs.copy()
        kwargs = self._validate_fit_kwargs(kwargs)
        if not self._validate_hyperparameter_tune_kwargs(hyperparameter_tune_kwargs, time_limit=time_limit):
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
            # Inform the user extra columns in dataset will not be used.
            extra_columns = [c for c in train_data.columns.copy() if c not in [index_column, time_column, target_column]]
            if len(extra_columns) > 0:
                logger.log(30, f"Find more than 3 columns, columns {extra_columns} will not be used.")
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
            train_data = TimeSeriesDataset(train_data,
                                           index_column=index_column,
                                           static_features=static_features)

            self.static_cat_columns = train_data.static_cat_columns()
            self.static_real_columns = train_data.static_real_columns()
            self.use_feat_static_cat = train_data.use_feat_static_cat()
            self.use_feat_static_real = train_data.use_feat_static_real()
            self.cardinality = train_data.get_static_cat_cardinality()
            self.prev_inferred_static_features = {"static_cat_columns": self.static_cat_columns,
                                                  "static_real_columns": self.static_real_columns,
                                                  "cardinality": self.cardinality}
            freq = train_data.get_freq()

            val_data = TimeSeriesDataset(val_data,
                                         index_column=index_column,
                                         static_features=static_features,
                                         prev_inferred=self.prev_inferred_static_features)
        elif isinstance(train_data, FileDataset) or isinstance(train_data, ListDataset):
            logger.log(30, "Training with dataset in gluon-ts format...")
            if val_data is None:
                logger.log(30, "Validation data is not specified, will do auto splitting...")
                train_data, val_data = train_test_split_gluonts(train_data, prediction_length, freq)
        else:
            raise TypeError("Does not support dataset type:", type(train_data))
        time_preprocessing_end = time.time()
        processing_time = time_preprocessing_end - start_time
        logger.log(30, f"Finished processing data, using {processing_time}s.")
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
        quantiles = kwargs.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        logger.log(30, f"All models will be trained for quantiles {quantiles}.")
        if hyperparameter_tune_kwargs is not None:
            if time_limit is None and hyperparameter_tune_kwargs.get("num_trials", None) is None:
                logger.log(30, "None of time_limit and num_tirals are set, by default setting num_tirals=2")
                num_trials = 2
            else:
                if isinstance(hyperparameter_tune_kwargs, str):
                    num_trials = 9999
                elif isinstance(hyperparameter_tune_kwargs, dict):
                    num_trials = hyperparameter_tune_kwargs.get("num_trials", 9999)
            scheduler_cls, scheduler_params = scheduler_factory(hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                                                                nthreads_per_trial='auto', ngpus_per_trial='auto', num_trials=num_trials)
            scheduler_options = (scheduler_cls, scheduler_params)
        else:
            scheduler_options = (None, None)
        if time_limit is not None:
            time_left = time_limit - processing_time
        else:
            time_left = time_limit
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
                          time_limit=time_left,)

        self._set_post_fit_vars()
        self._post_fit(
            keep_only_best=kwargs["keep_only_best"],
            refit_full=kwargs['refit_full'],
            set_best_to_refit_full=kwargs["set_best_to_refit_full"],
        )
        self.save()
        return self

    def _validate_fit_kwargs(self, kwargs):
        """
        Validate kwargs given in .fit()
        """
        kwargs_default = {
            "set_best_to_refit_full": False,
            "keep_only_best": False,
            "refit_full": False,
            "save_data": True,
            "freq": None,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        if kwargs.get("refit_full", False):
            if "set_best_to_refit_full" not in kwargs:
                kwargs["set_best_to_refit_full"] = True
                logger.log(30, "refit_full is set while set_best_to_refit_full is not set, automatically setting set_best_to_refit_full=True"
                               "to make sure that the model will predict with refit full model by default.")
        copied_kwargs = copy.deepcopy(kwargs_default)
        copied_kwargs.update(kwargs)
        return copied_kwargs

    def _set_post_fit_vars(self, learner: AbstractLearner = None):
        """
        Variable settings after fitting.
        """
        if learner is not None:
            self._learner: AbstractLearner = learner
        self._learner_type = type(self._learner)
        if self._learner.trainer_path is not None:
            self._trainer: AbstractTrainer = self._learner.load_trainer()

    def get_model_names(self):
        """Returns the list of model names trained in this `predictor` object."""
        return self._trainer.get_model_names_all()

    def preprocessing(self, data, time_series_to_predict=None, static_features=None):
        """
        Preprocessing your dataset. It will transform your tabular dataset to gluonts ListDataset.
        """
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
        """
        Return forecasts given a dataset

        Parameters
        ----------
        data: dataset to forecast,
              should be in the same format as train_data when you call .fit()
        time_series_to_predict: List, default=None,
              Time series index for which you want to forecast, if None, predict() will return the forecasts for every time series presented in data
        model: str, default=None
              Name of the model that you would like to use for forecasting. If None, it will by default use the best model from trainer.
        for_score: bool, default=False
              Whether you are using this predict() method for evaluation.
              We do not recommend you setting this to be True directly. If you want to evaluate your model on your dataset, you should directly using .evaluate()
        """
        processed_data = self.preprocessing(data, time_series_to_predict=time_series_to_predict, static_features=static_features)
        predict_targets = self._learner.predict(processed_data, model=model, for_score=for_score, **kwargs)
        return predict_targets

    def evaluate(self, data, static_features=None, **kwargs):
        """
        Evaluate the performace for given dataset.
        """
        processed_data = self.preprocessing(data, static_features=static_features)
        perf = self._learner.score(processed_data, **kwargs)
        return perf

    @classmethod
    def load(cls, output_directory, verbosity=2):
        """
        Load an existing ForecastingPredictor from output_directory
        """
        if output_directory is None:
            raise ValueError("output_directory cannot be None in load()")
        output_directory = setup_outputdir(output_directory, warn_if_exist=False)  # replace ~ with absolute path if it exists
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
        """
        Get information from learner.
        """
        return self._learner.get_info(include_model_info=True)

    def get_model_best(self):
        """
        Get the best model from trainer.
        """
        return self._trainer.get_model_best()

    def leaderboard(self, data=None, static_features=None):
        """
        Return a leaderboard showing the performance of every trained model

        Parameters
        ----------
        data: a dataset in the same format of the train_data input input ForecastingPredictor().fit()
              used for additional evaluation aside from the validation set.
        static_features: a Dataframe containing static_features,
              must be provided if static_features is provided when calling ForecastingPredictor().fit()
        """
        if data is not None:
            data = self.preprocessing(data, static_features=static_features)
        return self._learner.leaderboard(data)

    def fit_summary(self, verbosity=3):
        """
            Output summary of information about models produced during `fit()`.
            May create various generated summary plots and store them in folder: `Predictor.output_directory`.

            Parameters
            ----------
            verbosity : int, default = 3
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
            'model_best': self._trainer.model_best,  # the name of the best model (on validation data)
            'model_paths': self._trainer.get_models_attribute_dict('path'),
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
        results['leaderboard'] = self._learner.leaderboard()
        if verbosity > 0:  # print stuff
            print("*** Summary of fit() ***")
            print("Estimated performance of each model:")
            print(results["leaderboard"])
            print("Number of models trained: %s" % len(results['model_performance']))
            print("Types of models trained:")
            print(unique_model_types)
            hpo_str = ""
            if hpo_used and verbosity <= 2:
                hpo_str = " (call fit_summary() with verbosity >= 3 to see detailed HPO info)"
            print("Hyperparameter-tuning used: %s %s" % (hpo_used, hpo_str))
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
                            f"HPO for {model_type} model:  Num. configurations tried = {len(hpo_model['trial_info'])}, Time spent = {hpo_model['total_time']}s")
                        print(
                            f"Best hyperparameter-configuration (validation-performance: {self.eval_metric} = {hpo_model['validation_performance']}):")
                        print(hpo_model['best_config'])
        if verbosity > 0:
            print("*** End of fit() summary ***")
        return results

    def _post_fit(self, keep_only_best=False, refit_full=False, set_best_to_refit_full=False):
        """
        Post fit operations.
        """
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
                    # Note: model_best will be overwritten if additional training is done with new models,
                    # since model_best will have validation score of None and any new model will have a better validation score.
                    # This has the side-effect of having the possibility of model_best being overwritten by a worse model than the original model_best.
                    self._trainer.save()
                else:
                    logger.warning(f'Best model ({trainer_model_best}) is not present in refit_full dictionary. '
                                   f'Training may have failed on the refit model. AutoGluon will default to using {trainer_model_best} for predictions.')

    def refit_full(self, models='all'):
        """
        Rifit models on the whole dataset(train + validation)
        """
        return self._learner.refit_full(models=models)

    def _validate_hyperparameter_tune_kwargs(self, hyperparameter_tune_kwargs, time_limit=None):
        """
        Returns True if hyperparameter_tune_kwargs is None or can construct a valid scheduler.
        Returns False if hyperparameter_tune_kwargs results in an invalid scheduler.
        """
        if hyperparameter_tune_kwargs is None:
            return True

        scheduler_cls, scheduler_params = scheduler_factory(hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, time_out=time_limit,
                                                            nthreads_per_trial='auto', ngpus_per_trial='auto')
        assert scheduler_params['searcher'] != 'bayesopt_hyperband', "searcher == 'bayesopt_hyperband' not yet supported"
        if scheduler_params.get('dist_ip_addrs', None):
            logger.warning('Warning: dist_ip_addrs does not currently work for Tabular. Distributed instances will not be utilized.')

        if scheduler_params['num_trials'] == 1:
            logger.warning('Warning: Specified num_trials == 1 for hyperparameter tuning, disabling HPO. '
                           'This can occur if time_limit was not specified in `fit()`.')
            return False

        scheduler_ngpus = scheduler_params['resource'].get('num_gpus', 0)
        if scheduler_ngpus is not None and isinstance(scheduler_ngpus, int) and scheduler_ngpus > 1:
            logger.warning(f"Warning: ForecastingPredictor currently doesn't use >1 GPU per training run. Detected {scheduler_ngpus} GPUs.")
        return True

    @classmethod
    def evaluate_predictions(cls,
                             forecasts,
                             targets,
                             index_column,
                             time_column,
                             target_column,
                             eval_metric=None,
                             quantiles=None):
        """
        Evaluate predictions once future targets are received.

        Parameters
        ----------
        forecasts: dict, produced by ForecastingPredictor().predict()
            a dictionary containing predictions for different targets.
            Keys are time series index
            Values are pandas Dataframe containing predictions for different quantiles.

        targets: a Dataframe which has the same format as what you have for train_data/test_data,
            must contain targets for all time presented in forecasts.

        index_column: str or None
            Name of column in targets that contains an index ID specifying which time series is being observed at each time-point (for datasets containing multiple time-series).
            If None, we will assume that there is only one time series in the dataset.

        time_column: str
            Name of column in targets that lists the time of each observation.

        target_column: str
            Name of column in targets that contains the target time-series value to be predicted.
        """
        with evaluator_warning_filter():
            targets = rebuild_tabular(targets,
                                      index_column=index_column,
                                      target_column=target_column,
                                      time_column=time_column).set_index(index_column).transpose()

            required_time = list(forecasts.values())[0].index
            targets_time = pd.DatetimeIndex(targets.index, freq=pd.infer_freq(targets.index))
            targets.index = targets_time

            for time in required_time:
                if time not in targets_time:
                    raise ValueError(f"Time {time} is presented in predictions but not given in targets. Please check your targets.")

            formated_targets = []
            quantile_forecasts = []
            for ts_id, forecast in forecasts.items():
                tmp_targets = targets.loc[required_time,ts_id]
                formated_targets.append(tmp_targets)

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
            evaluator = Evaluator(quantile_forecasts[0].forecast_keys)
            num_series = len(formated_targets)
            agg_metrics, item_metrics = evaluator(iter(formated_targets), iter(quantile_forecasts), num_series=num_series)
            if eval_metric is None:
                return agg_metrics
            else:
                return agg_metrics[eval_metric]
