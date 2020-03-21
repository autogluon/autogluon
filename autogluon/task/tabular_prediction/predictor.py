import logging

import pandas as pd

from .dataset import TabularDataset
from ..base.base_predictor import BasePredictor
from ...utils import plot_performance_vs_trials, plot_summary_of_models, plot_tabular_models, verbosity2loglevel
from ...utils.tabular.ml.constants import REGRESSION
from ...utils.tabular.ml.learner.abstract_learner import AbstractLearner as Learner  # TODO: Keep track of true type of learner for loading
from ...utils.tabular.ml.trainer.abstract_trainer import AbstractTrainer  # TODO: Keep track of true type of trainer for loading
from ...utils.tabular.ml.utils import setup_outputdir

__all__ = ['TabularPredictor']

logger = logging.getLogger()  # return root logger


class TabularPredictor(BasePredictor):
    """ Object returned by `fit()` in Tabular Prediction tasks.
        Use for making predictions on new data and viewing information about models trained during `fit()`.

        Attributes
        ----------
        output_directory : str
            Path to directory where all models used by this Predictor are stored.
        problem_type : str
            What type of prediction problem this Predictor has been trained for.
        eval_metric : function or str
            What metric is used to evaluate predictive performance.
        label_column : str
            Name of table column that contains data from the variable to predict (often referred to as: labels, response variable, target variable, dependent variable, Y, etc).
        feature_types : dict
            Inferred data type of each predictive variable (i.e. column of training data table used to predict `label_column`).
        model_names : list
            List of model names trained during `fit()`.
        model_performance : dict
            Maps names of trained models to their predictive performance values attained on the validation dataset during `fit()`.
        class_labels : list
            For multiclass problems, this list contains the class labels in sorted order of `predict_proba()` output. Is = None for problems that are not multiclass.
            For example if `pred = predict_proba(x)`, then ith index of `pred` provides predicted probability that `x` belongs to class given by `class_labels[i]`.

        Examples
        --------
        >>> from autogluon import TabularPrediction as task
        >>> train_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/train.csv')
        >>> predictor = task.fit(train_data=train_data, label='class')
        >>> results = predictor.fit_summary()
        >>> test_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/Inc/test.csv')
        >>> perf = predictor.evaluate(test_data)

    """

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
        self.problem_type = self._learner.problem_type
        self.eval_metric = self._learner.objective_func
        self.label_column = self._learner.label
        self.feature_types = self._trainer.feature_types_metadata
        self.model_names = self._trainer.get_model_names_all()
        self.model_performance = self._trainer.model_performance
        self.class_labels = self._learner.class_labels

    def predict(self, dataset, model=None, as_pandas=False, use_pred_cache=False, add_to_pred_cache=False):
        """ Use trained models to produce predicted labels (in classification) or response values (in regression).

            Parameters
            ----------
            dataset : str or :class:`TabularDataset` or `pandas.DataFrame`
                The dataset to make predictions for. Should contain same column names as training Dataset and follow same format
                (may contain extra columns that won't be used by Predictor, including the label-column itself).
                If str is passed, `dataset` will be loaded using the str value as the file path.
            model : str (optional)
                The name of the model to get predictions from. Defaults to None, which uses the highest scoring model on the validation set.
                Valid models are listed in this `predictor` by calling `predictor.model_names`
            as_pandas : bool (optional)
                Whether to return the output as a pandas Series (True) or numpy array (False)
            use_pred_cache : bool (optional)
                Whether to used previously-cached predictions for table rows we have already predicted on before
                (can speedup repeated runs of `predict()` on multiple datasets with overlapping rows between them).
            add_to_pred_cache : bool (optional)
                Whether these predictions should be cached for reuse in future `predict()` calls on the same table rows
                (can speedup repeated runs of `predict()` on multiple datasets with overlapping rows between them).

            Returns
            -------
            Array of predictions, one corresponding to each row in given dataset. Either numpy Ndarray or pandas Series depending on `as_pandas` argument.

        """
        dataset = self.__get_dataset(dataset)
        return self._learner.predict(X_test=dataset, model=model, as_pandas=as_pandas, use_pred_cache=use_pred_cache, add_to_pred_cache=add_to_pred_cache)

    def predict_proba(self, dataset, model=None, as_pandas=False):
        """ Use trained models to produce predicted class probabilities rather than class-labels (if task is classification).

            Parameters
            ----------
            dataset : str or :class:`TabularDataset` or `pandas.DataFrame`
                The dataset to make predictions for. Should contain same column names as training Dataset and follow same format
                (may contain extra columns that won't be used by Predictor, including the label-column itself).
                If str is passed, `dataset` will be loaded using the str value as the file path.
            model : str (optional)
                The name of the model to get prediction probabilities from. Defaults to None, which uses the highest scoring model on the validation set.
                Valid models are listed in this `predictor` by calling `predictor.model_names`
            as_pandas : bool (optional)
                Whether to return the output as a pandas object (True) or numpy array (False).
                Pandas object is a DataFrame if this is a multiclass problem, otherwise it is a Series.

            Returns
            -------
            Array of predicted class-probabilities, corresponding to each row in the given dataset.
            May be a numpy Ndarray or pandas Series/Dataframe depending on `as_pandas` argument and the type of prediction problem.
        """
        dataset = self.__get_dataset(dataset)
        return self._learner.predict_proba(X_test=dataset, model=model, as_pandas=as_pandas)

    def evaluate(self, dataset, silent=False):
        """ Report the predictive performance evaluated for a given Dataset.
            This is basically a shortcut for: `pred = predict(dataset); evaluate_predictions(dataset[label_column], preds, auxiliary_metrics=False)`
            that automatically uses `predict_proba()` instead of `predict()` when appropriate.

            Parameters
            ----------
            dataset : str or :class:`TabularDataset` or `pandas.DataFrame`
                This Dataset must also contain the label-column with the same column-name as specified during `fit()`.
                If str is passed, `dataset` will be loaded using the str value as the file path.

            silent : bool (optional)
                Should performance results be printed?

            Returns
            -------
            Predictive performance value on the given dataset, based on the `eval_metric` used by this Predictor.
        """
        dataset = self.__get_dataset(dataset)
        perf = self._learner.score(dataset)
        sign = self._learner.objective_func._sign
        perf = perf * sign  # flip negative once again back to positive (so higher is no longer necessarily better)
        if not silent:
            print("Predictive performance on given dataset: %s = %s" % (self.eval_metric, perf))
        return perf

    def evaluate_predictions(self, y_true, y_pred, silent=False, auxiliary_metrics=False, detailed_report=True):
        """ Evaluate the provided predictions against ground truth labels.

            Parameters
            ----------
            y_true : list or `numpy.array`
                The ordered collection of ground-truth labels.
            y_pred : list or `numpy.array`
                The ordered collection of predictions.
                For certain types of `eval_metric` (such as AUC), `y_pred` must be predicted-probabilities rather than predicted labels.
            silent : bool (optional)
                Should performance results be printed?
            auxiliary_metrics: bool (optional)
                Should we compute other (`problem_type` specific) metrics in addition to the default metric?
            detailed_report : bool (optional)
                Should we computed more detailed versions of the `auxiliary_metrics`? (requires `auxiliary_metrics = True`)

            Returns
            -------
            Scalar performance value if `auxiliary_metrics = False`.
            If `auxiliary_metrics = True`, returns dict where keys = metrics, values = performance along each metric.
        """
        return self._learner.evaluate(y_true=y_true, y_pred=y_pred, silent=silent,
                                      auxiliary_metrics=auxiliary_metrics, detailed_report=detailed_report)

    def leaderboard(self, dataset=None, silent=False):
        """
            Output summary of information about models produced during fit() as a pandas DataFrame.
            Includes information on test and validation scores for all models, model training times and stack levels.

            Parameters
            ----------
            dataset : str or :class:`TabularDataset` or `pandas.DataFrame` (optional)
                This Dataset must also contain the label-column with the same column-name as specified during fit().
                If specified, then the leaderboard returned will contain an additional column 'score_test'
                'score_test' is the score of the model on the validation_metric for the dataset provided
                If str is passed, `dataset` will be loaded using the str value as the file path.
            silent: bool (optional)
                Should leaderboard DataFrame be printed?

            Returns
            -------
            Pandas `pandas.DataFrame` of model performance summary information.
        """
        dataset = self.__get_dataset(dataset) if dataset is not None else dataset
        return self._learner.leaderboard(X=dataset, silent=silent)

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
        model_typenames = {key: self._trainer.model_types[key].__name__ for key in self._trainer.model_types}
        model_innertypenames = {key: self._trainer.model_types_inner[key].__name__ for key in self._trainer.model_types if key in self._trainer.model_types_inner}
        MODEL_STR = 'Model'
        ENSEMBLE_STR = 'Ensemble'
        for model in model_typenames:
            if (model in model_innertypenames) and (ENSEMBLE_STR not in model_innertypenames[model]) and (ENSEMBLE_STR in model_typenames[model]):
                new_model_typename = model_typenames[model] + "_" + model_innertypenames[model]
                if new_model_typename.endswith(MODEL_STR):
                    new_model_typename = new_model_typename[:-len(MODEL_STR)]
                model_typenames[model] = new_model_typename

        unique_model_types = set(model_typenames.values())  # no more class info
        # all fit() information that is returned:
        results = {
            'model_types': model_typenames,  # dict with key = model-name, value = type of model (class-name)
            'model_performance': self.model_performance,  # dict with key = model-name, value = validation performance
            'model_best': self._trainer.model_best,  # the name of the best model (on validation data)
            'model_paths': self._trainer.model_paths,  # dict with key = model-name, value = path to model file
            'model_fit_times': self._trainer.model_fit_times,
            'model_pred_times': self._trainer.model_pred_times,
            'num_bagging_folds': self._trainer.kfolds,
            'stack_ensemble_levels': self._trainer.stack_ensemble_levels,
            'feature_prune': self._trainer.feature_prune,
            'hyperparameter_tune': hpo_used,
            'hyperparameters_userspecified': self._trainer.hyperparameters,
        }
        if self.problem_type != REGRESSION:
            results['num_classes'] = self._trainer.num_classes
        if hpo_used:
            results['hpo_results'] = self._trainer.hpo_results
        # get dict mapping model name to final hyperparameter values for each model:
        model_hyperparams = {}
        for model_name in results['model_performance']:
            model_obj = self._trainer.load_model(model_name)
            model_hyperparams[model_name] = model_obj.params
        results['model_hyperparams'] = model_hyperparams

        if verbosity > 0:  # print stuff
            print("*** Summary of fit() ***")
            print("Estimated performance of each model:")
            results['leaderboard'] = self._learner.leaderboard(silent=False)
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
        if verbosity > 1:  # create plots
            plot_tabular_models(results, output_directory=self.output_directory,
                                save_file="SummaryOfModels.html",
                                plot_title="Models produced during fit()")
            if hpo_used:
                for model_type in results['hpo_results']:
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
                    print("HPO for %s model:  Num. configurations tried = %s, Time spent = %s, Search strategy = %s"
                          % (model_type, len(hpo_model['trial_info']), hpo_model['total_time'], hpo_model['search_strategy']))
                    print("Best hyperparameter-configuration (validation-performance: %s = %s):"
                          % (self.eval_metric, hpo_model['validation_performance']))
                    print(hpo_model['best_config'])
            """
            if bagging_used:
                pass # TODO: print detailed bagging info
            if stacking_used:
                pass # TODO: print detailed stacking info, like how much it improves validation performance
            if results['feature_prune']:
                pass # TODO: print detailed feature-selection info once feature-selection is functional.
            """
        if verbosity > 0:
            print("*** End of fit() summary ***")
        return results

    # TODO: Consider adding time_limit option to early stop the feature importance process
    def feature_importance(self, model=None, dataset=None, features=None, raw=True, subsample_size=10000, silent=False):
        """
        Calculates feature importance scores for the given model.
        A feature's importance score represents the performance drop that results when the model makes predictions on a perturbed copy of the dataset where this feature's values have been randomly shuffled across rows.
        A feature score of 0.01 would indicate that the predictive performance dropped by 0.01 when the feature was randomly shuffled.
        The higher the score a feature has, the more important it is to the model's performance.
        If a feature has a negative score, this means that the feature is likely harmful to the final model, and a model trained with the feature removed would be expected to achieve a better predictive performance.
        Note that calculating feature importance can be a very computationally expensive process, particularly if the model uses hundreds or thousands of features. In many cases, this can take longer than the original model training.
        To estimate how long `feature_importance(model, dataset, features)` will take, it is roughly the time taken by `predict_proba(dataset, model)` multiplied by the number of features.

        Parameters
        ----------
        model : str, default = None
            Model to get feature importances for, if None the best model is chosen.
            Valid models are listed in this `predictor` by calling `predictor.model_names`
        dataset : str or :class:`TabularDataset` or `pandas.DataFrame` (optional)
            This Dataset must also contain the label-column with the same column-name as specified during fit().
            If specified, then the dataset is used to calculate the feature importance scores.
            If str is passed, `dataset` will be loaded using the str value as the file path.
            If not specified, the original dataset used during fit() will be used if `cache_data=True`. Otherwise, an exception will be raised.
            Do not pass the training data through this argument, as the feature importance scores calculated will be inaccurate.
        features : list, default = None
            List of str feature names that feature importances are calculated for and returned, specify None to get all feature importances.
            If you only want to compute feature importances for some of the features, you can pass their names in as a list of str.
        raw : bool, default = True
            Whether to compute feature importance on raw features in the original data (after automated feature engineering) or on the features used by the particular model.
            For example, a stacker model uses both the original features and the predictions of the lower-level models.
            Note that for bagged models, feature importance calculation is not yet supported when both `raw=True` and `dataset=None`. Doing so will raise an exception.
        subsample_size : int, default = 10000
            The number of rows to sample from `dataset` when computing feature importance.
            If `subsample_size=None` or `dataset` contains fewer than `subsample_size` rows, all rows will be used during computation.
            Larger values increase the accuracy of the feature importance scores.
            Runtime linearly scales with `subsample_size`.
        silent : bool, default = False
            Whether to suppress logging output

        Returns
        -------
        Pandas `pandas.Series` of feature importance scores.

        """
        if (dataset is None) and (not self._trainer.is_data_saved):
            raise AssertionError('No dataset was provided and there is no cached data to load for feature importance calculation. `cache_data=True` must be set in the `TabularPrediction.fit()` call to enable this functionality when dataset is not specified.')

        return self._learner.get_feature_importance(model=model, X=dataset, features=features, raw=raw, subsample_size=subsample_size, silent=silent)

    @classmethod
    def load(cls, output_directory, verbosity=2):
        """
        Load a predictor object previously produced by `fit()` from file and returns this object.
        Is functionally equivalent to :meth:`autogluon.task.tabular_prediction.TabularPrediction.load`.

        Parameters
        ----------
        output_directory : str
            Path to directory where trained models are stored (i.e. the `output_directory` specified in previous call to `fit()`).
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is generally printed by this Predictor.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50 (Note: higher values `L` correspond to fewer print statements, opposite of verbosity levels)

        Returns
        -------
        :class:`TabularPredictor` object
        """
        logger.setLevel(verbosity2loglevel(verbosity))  # Reset logging after load (may be in new Python session)
        if output_directory is None:
            raise ValueError("output_directory cannot be None in load()")

        output_directory = setup_outputdir(output_directory)  # replace ~ with absolute path if it exists
        learner = Learner.load(output_directory)
        return cls(learner=learner)

    def save(self):
        """ Save this predictor to file in directory specified by this Predictor's `output_directory`.
            Note that `fit()` already saves the predictor object automatically
            (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        self._learner.save()
        logger.log(20, "TabularPredictor saved. To load, use: TabularPredictor.load(\"%s\")" % self.output_directory)

    @staticmethod
    def _summarize(key, msg, results):
        if key in results:
            print(msg + ": " + str(results[key]))

    @staticmethod
    def __get_dataset(dataset):
        if isinstance(dataset, TabularDataset):
            return dataset
        if isinstance(dataset, str):
            return TabularDataset(file_path=dataset)
        if isinstance(dataset, pd.DataFrame):
            return TabularDataset(df=dataset)
        if isinstance(dataset, pd.Series):
            raise TypeError("dataset must be TabularDataset or pandas.DataFrame, not pandas.Series. \
                   To predict on just single example (ith row of table), use dataset.iloc[[i]] rather than dataset.iloc[i]")
        else:
            raise TypeError("dataset must be TabularDataset or pandas.DataFrame or str file path to dataset")
