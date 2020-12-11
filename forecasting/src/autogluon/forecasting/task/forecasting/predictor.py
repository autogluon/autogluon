import copy
import logging

import pandas as pd

from .dataset import TimeSeriesDataset
from autogluon.core.task.base import BasePredictor
from autogluon.core.utils import plot_performance_vs_trials, plot_summary_of_models, plot_tabular_models, verbosity2loglevel
from ...learner import AbstractLearner as Learner  # TODO: Keep track of true type of learner for loading
from ...trainer import AbstractTrainer  # TODO: Keep track of true type of trainer for loading
from autogluon.core.utils.utils import setup_outputdir

__all__ = ['ForecastingPredictor']

logger = logging.getLogger()  # return root logger


class ForecastingPredictor(BasePredictor):

    def __init__(self, learner):
        """ Creates TabularPredictor object.
            You should not construct a TabularPredictor yourself, it is only intended to be produced during fit().

            Parameters
            ----------
            learner : `AbstractLearner` object
                Object that implements the `AbstractLearner` APIs.

            To access any learner method `func()` from this Predictor, use: `predictor._learner.func()`.
            To access any trainer method `func()` from this `Predictor`, use: `predictor._trainer.func()`.
        """
        self._learner: Learner = learner  # Learner object
        self._trainer: AbstractTrainer = self._learner.load_trainer()  # Trainer object
        self.output_directory = self._learner.path
        self.eval_metric = self._learner.eval_metric

    def get_model_names(self):
        """Returns the list of model names trained in this `predictor` object."""
        return self._trainer.get_model_names_all()

    def predict(self, data, model=None, for_score=False):
        predict_targets = self._learner.predict(data, model=model, for_score=for_score)
        return predict_targets

    def evaluate(self, data, **kwargs):
        perf = self._learner.score(data, **kwargs)
        return perf

    def evaluate_predictions(self, forecasts, tss, **kwargs):

        return self._learner.evaluate(forecasts, tss, **kwargs)

    @classmethod
    def load(cls, output_directory, verbosity=2):
        if output_directory is None:
            raise ValueError("output_directory cannot be None in load()")

        output_directory = setup_outputdir(output_directory)  # replace ~ with absolute path if it exists
        learner = Learner.load(output_directory)

        return cls(learner=learner)

    def save(self, output_directory):
        """ Save this predictor to file in directory specified by this Predictor's `output_directory`.
            Note that `fit()` already saves the predictor object automatically
            (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        self._learner.save()

    def predict_proba(self, X):
        pass

    def info(self):
        return self._learner.get_info(include_model_info=True)

    def get_model_best(self):
        return self._trainer.get_model_best()

    def leaderboard(self):
        # TODO: allow a dataset as input
        return self._learner.leaderboard()

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
            'model_performance': self._trainer.get_models_attribute_dict('score'),  # dict with key = model-name, value = validation performance
            'model_best': self._trainer.model_best,  # the name of the best model (on validation data)
            'model_paths': self._trainer.get_models_attribute_dict('path'),  # dict with key = model-name, value = path to model file
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
            num_fold_str = ""
            bagging_used = results['num_bagging_folds'] > 0
            if bagging_used:
                num_fold_str = f" (with {results['num_bagging_folds']} folds)"
            print("Bagging used: %s %s" % (bagging_used, num_fold_str))
            num_stack_str = ""
            stacking_used = results['stack_ensemble_levels'] > 0
            if stacking_used:
                num_stack_str = f" (with {results['stack_ensemble_levels']} levels)"
            print("Stack-ensembling used: %s %s" % (stacking_used, num_stack_str))
            hpo_str = ""
            if hpo_used and verbosity <= 2:
                hpo_str = " (call fit_summary() with verbosity >= 3 to see detailed HPO info)"
            print("Hyperparameter-tuning used: %s %s" % (hpo_used, hpo_str))
            # TODO: uncomment once feature_prune is functional:  self._summarize('feature_prune', 'feature-selection used', results)
            print("User-specified hyperparameters:")
            print(results['hyperparameters_userspecified'])
            print("Feature Metadata (Processed):")
            print("(raw dtype, special dtypes):")
        if verbosity > 1:  # create plots
            plot_tabular_models(results, output_directory=self.output_directory,
                                save_file="SummaryOfModels.html",
                                plot_title="Models produced during fit()")
            if hpo_used:
                for model_type in results['hpo_results']:
                    if 'trial_info' in results['hpo_results'][model_type]:
                        plot_summary_of_models(
                            results['hpo_results'][model_type],
                            output_directory=self.output_directory, save_file=model_type + "_HPOmodelsummary.html",
                            plot_title=f"Models produced during {model_type} HPO")
                        plot_performance_vs_trials(
                            results['hpo_results'][model_type],
                            output_directory=self.output_directory, save_file=model_type + "_HPOperformanceVStrials.png",
                            plot_title=f"HPO trials for {model_type} models")
        if verbosity > 2:  # print detailed information
            if hpo_used:
                hpo_results = results['hpo_results']
                print("*** Details of Hyperparameter optimization ***")
                for model_type in hpo_results:
                    hpo_model = hpo_results[model_type]
                    if 'trial_info' in hpo_model:
                        print(f"HPO for {model_type} model:  Num. configurations tried = {len(hpo_model['trial_info'])}, Time spent = {hpo_model['total_time']}s, Search strategy = {hpo_model['search_strategy']}")
                        print(f"Best hyperparameter-configuration (validation-performance: {self.eval_metric} = {hpo_model['validation_performance']}):")
                        print(hpo_model['best_config'])
        if verbosity > 0:
            print("*** End of fit() summary ***")
        return results



