import copy
import logging

import pandas as pd

from autogluon.core.task.base import BasePredictor
from autogluon.core.utils.savers import save_pkl
from autogluon.core.utils.loaders import load_pkl
from ...learner import AbstractLearner as Learner  # TODO: Keep track of true type of learner for loading
from ...trainer import AbstractTrainer  # TODO: Keep track of true type of trainer for loading
from autogluon.core.utils.utils import setup_outputdir
from ...utils.dataset_utils import time_series_dataset

__all__ = ['ForecastingPredictor']

logger = logging.getLogger()  # return root logger


class ForecastingPredictor:

    predictor_file_name = "predictor.pkl"

    def __init__(self, learner, index_column="index", target_column="target", time_column="date"):
        """ Creates TabularPredictor object.
            You should not construct a TabularPredictor yourself, it is only intended to be produced during fit().

            Parameters
            ----------
            learner : `AbstractLearner` object
                Object that implements the `AbstractLearner` APIs.

            To access any learner method `func()` from this Predictor, use: `predictor._learner.func()`.
            To access any trainer method `func()` from this `Predictor`, use: `predictor._trainer.func()`.
        """
        self.index_column = index_column
        self.target_column = target_column
        self.time_column = time_column
        self._learner: Learner = learner  # Learner object
        self._trainer: AbstractTrainer = self._learner.load_trainer()  # Trainer object
        self.output_directory = self._learner.path
        self.eval_metric = self._learner.eval_metric

    def get_model_names(self):
        """Returns the list of model names trained in this `predictor` object."""
        return self._trainer.get_model_names_all()

    def preprocessing(self, data):
        if isinstance(data, pd.DataFrame):
            data = time_series_dataset(data,
                                       index_column=self.index_column,
                                       target_column=self.target_column,
                                       time_column=self.time_column)
        return data

    def predict(self, data, model=None, for_score=False, **kwargs):
        processed_data = self.preprocessing(data)
        predict_targets = self._learner.predict(processed_data, model=model, for_score=for_score, **kwargs)
        return predict_targets

    def evaluate(self, data, **kwargs):
        processed_data = self.preprocessing(data)
        perf = self._learner.score(processed_data, **kwargs)
        return perf

    def evaluate_predictions(self, forecasts, tss, **kwargs):

        return self._learner.evaluate(forecasts, tss, **kwargs)

    @classmethod
    def load(cls, output_directory, verbosity=2):
        if output_directory is None:
            raise ValueError("output_directory cannot be None in load()")
        output_directory = setup_outputdir(output_directory)  # replace ~ with absolute path if it exists
        logger.log(30, f"Loading predictor from path {output_directory}")
        learner = Learner.load(output_directory)
        predictor = load_pkl.load(path=learner.path + cls.predictor_file_name)
        predictor._learner = learner
        predictor._trainer = learner.load_trainer()
        return predictor

    def save(self, output_directory):
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

    def leaderboard(self, data=None):
        # TODO: allow a dataset as input
        if data is not None:
            data = self.preprocessing(data)
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

    def _post_fit(self, keep_only_best=False, refit_full=False, set_best_to_refit_full=False, save_space=False):
        if refit_full is True:
            if keep_only_best is True:
                if set_best_to_refit_full is True:
                    refit_full = 'best'
                else:
                    logger.warning(f'refit_full was set to {refit_full}, but keep_only_best=True and set_best_to_refit_full=False. Disabling refit_full to avoid training models which would be automatically deleted.')
                    refit_full = False
            else:
                refit_full = 'all'

        if refit_full is not False:
            trainer_model_best = self._trainer.get_model_best()
            self.refit_full(models=refit_full)
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