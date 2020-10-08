import copy
import logging

import pandas as pd

from .dataset import TabularDataset
from .hyperparameter_configs import get_hyperparameter_config
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
        feature_metadata : :class:`autogluon.utils.tabular.features.feature_metadata.FeatureMetadata`
            Inferred data type of each predictive variable after preprocessing transformation (i.e. column of training data table used to predict `label_column`).
            Contains both raw dtype and special dtype information. Each feature has exactly 1 raw dtype (such as 'int', 'float', 'category') and zero to many special dtypes (such as 'datetime_as_int', 'text', 'text_ngram').
            Special dtypes are AutoGluon specific feature types that are used to identify features with meaning beyond what the raw dtype can convey.
                `feature_metadata.type_map_raw`: Dictionary of feature name -> raw dtype mappings.
                `feature_metadata.type_group_map_special`: Dictionary of lists of special feature names, grouped by special feature dtype.
        class_labels : list
            For multiclass problems, this list contains the class labels in sorted order of `predict_proba()` output.
            For binary problems, this list contains the class labels in sorted order of `predict_proba(as_multiclass=True)` output.
                `class_labels[0]` corresponds to internal label = 0 (negative class), `class_labels[1]` corresponds to internal label = 1 (positive class).
                This is relevant for certain metrics such as F1 where True and False labels impact the metric score differently.
            For other problem types, will equal None.
            For example if `pred = predict_proba(x, as_multiclass=True)`, then ith index of `pred` provides predicted probability that `x` belongs to class given by `class_labels[i]`.
        class_labels_internal : list
            For multiclass problems, this list contains the internal class labels in sorted order of internal `predict_proba()` output.
            For binary problems, this list contains the internal class labels in sorted order of internal `predict_proba(as_multiclass=True)` output.
                The value will always be `class_labels_internal=[0, 1]` for binary problems, with 0 as the negative class, and 1 as the positive class.
            For other problem types, will equal None.
        class_labels_internal_map : dict
            For binary and multiclass classification problems, this dictionary contains the mapping of the original labels to the internal labels.
            For example, in binary classification, label values of 'True' and 'False' will be mapped to the internal representation `1` and `0`.
                Therefore, class_labels_internal_map would equal {'True': 1, 'False': 0}
            For other problem types, will equal None.
            For multiclass, it is possible for not all of the label values to have a mapping.
                This indicates that the internal models will never predict those missing labels, and training rows associated with the missing labels were dropped.

        Examples
        --------
        >>> from autogluon import TabularPrediction as task
        >>> train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
        >>> predictor = task.fit(train_data=train_data, label='class')
        >>> results = predictor.fit_summary()
        >>> test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
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
        self._learner.persist_trainer(low_memory=True)
        self._trainer: AbstractTrainer = self._learner.load_trainer()  # Trainer object
        self.output_directory = self._learner.path
        self.problem_type = self._learner.problem_type
        self.eval_metric = self._learner.eval_metric
        self.label_column = self._learner.label
        self.feature_metadata = self._trainer.feature_metadata
        self.class_labels = self._learner.class_labels
        self.class_labels_internal = self._learner.label_cleaner.ordered_class_labels_transformed
        self.class_labels_internal_map = self._learner.label_cleaner.inv_map

    @property
    def model_names(self):
        logger.warning('WARNING: `predictor.model_names` is a deprecated `predictor` variable. Use `predictor.get_model_names()` instead. Use of `predictor.model_names` will result in an exception starting in autogluon==0.1')
        return self.get_model_names()

    @property
    def model_performance(self):
        logger.warning('WARNING: `predictor.model_performance` is a deprecated `predictor` variable. Use `predictor.leaderboard()` instead. Use of `predictor.model_performance` will result in an exception starting in autogluon==0.1')
        return self._trainer.get_models_attribute_dict(attribute='val_score')

    def predict(self, dataset, model=None, as_pandas=False):
        """ Use trained models to produce predicted labels (in classification) or response values (in regression).

            Parameters
            ----------
            dataset : str or :class:`TabularDataset` or `pandas.DataFrame`
                The dataset to make predictions for. Should contain same column names as training Dataset and follow same format
                (may contain extra columns that won't be used by Predictor, including the label-column itself).
                If str is passed, `dataset` will be loaded using the str value as the file path.
            model : str (optional)
                The name of the model to get predictions from. Defaults to None, which uses the highest scoring model on the validation set.
                Valid models are listed in this `predictor` by calling `predictor.get_model_names()`
            as_pandas : bool (optional)
                Whether to return the output as a pandas Series (True) or numpy array (False)

            Returns
            -------
            Array of predictions, one corresponding to each row in given dataset. Either numpy Ndarray or pandas Series depending on `as_pandas` argument.

        """
        dataset = self.__get_dataset(dataset)
        return self._learner.predict(X=dataset, model=model, as_pandas=as_pandas)

    def predict_proba(self, dataset, model=None, as_pandas=False, as_multiclass=False):
        """ Use trained models to produce predicted class probabilities rather than class-labels (if task is classification).
            If `predictor.problem_type` is regression, this functions identically to `predict`, returning the same output.

            Parameters
            ----------
            dataset : str or :class:`TabularDataset` or `pandas.DataFrame`
                The dataset to make predictions for. Should contain same column names as training Dataset and follow same format
                (may contain extra columns that won't be used by Predictor, including the label-column itself).
                If str is passed, `dataset` will be loaded using the str value as the file path.
            model : str (optional)
                The name of the model to get prediction probabilities from. Defaults to None, which uses the highest scoring model on the validation set.
                Valid models are listed in this `predictor` by calling `predictor.get_model_names()`
            as_pandas : bool (optional)
                Whether to return the output as a pandas object (True) or numpy array (False).
                Pandas object is a DataFrame if this is a multiclass problem or `as_multiclass=True`, otherwise it is a Series.
                If the output is a DataFrame, the column order will be equivalent to `predictor.class_labels`.
            as_multiclass : bool (optional)
                Whether to return binary classification probabilities as if they were for multiclass classification.
                    Output will contain two columns, and if `as_pandas=True`, the column names will correspond to the binary class labels.
                    The columns will be the same order as `predictor.class_labels`.
                Only impacts output for binary classification problems.

            Returns
            -------
            Array of predicted class-probabilities, corresponding to each row in the given dataset.
            May be a numpy ndarray or pandas Series/DataFrame depending on `as_pandas` and `as_multiclass` arguments and the type of prediction problem.
            For binary classification problems, the output contains for each datapoint only the predicted probability of the positive class, unless you specify `as_multiclass=True`.
        """
        dataset = self.__get_dataset(dataset)
        return self._learner.predict_proba(X=dataset, model=model, as_pandas=as_pandas, as_multiclass=as_multiclass)

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
        sign = self._learner.eval_metric._sign
        perf = perf * sign  # flip negative once again back to positive (so higher is no longer necessarily better)
        if not silent:
            print("Predictive performance on given dataset: %s = %s" % (self.eval_metric, perf))
        return perf

    def evaluate_predictions(self, y_true, y_pred, silent=False, auxiliary_metrics=False, detailed_report=True):
        """ Evaluate the provided predictions against ground truth labels.
            Evaluation is based on the `eval_metric` previously specifed to `fit()`, or default metrics if none was specified.

            Parameters
            ----------
            y_true : list or `numpy.array`
                The ordered collection of ground-truth labels.
            y_pred : list or `numpy.array`
                The ordered collection of predictions.
                Caution: For certain types of `eval_metric` (such as 'roc_auc'), `y_pred` must be predicted-probabilities rather than predicted labels.
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

    def leaderboard(self, dataset=None, extra_info=False, only_pareto_frontier=False, silent=False):
        """
            Output summary of information about models produced during fit() as a pandas DataFrame.
            Includes information on test and validation scores for all models, model training times, inference times, and stack levels.
            Output DataFrame columns include:
                'model': The name of the model.

                'score_val': The validation score of the model on the 'eval_metric'.

                'pred_time_val': The inference time required to compute predictions on the validation data end-to-end.
                    Equivalent to the sum of all 'pred_time_val_marginal' values for the model and all of its base models.
                'fit_time': The fit time required to train the model end-to-end (Including base models if the model is a stack ensemble).
                    Equivalent to the sum of all 'fit_time_marginal' values for the model and all of its base models.
                'pred_time_val_marginal': The inference time required to compute predictions on the validation data (Ignoring inference times for base models).
                    Note that this ignores the time required to load the model into memory when bagging is disabled.
                'fit_time_marginal': The fit time required to train the model (Ignoring base models).
                'stack_level': The stack level of the model.
                    A model with stack level N can take any set of models with stack level less than N as input, with stack level 0 models having no model inputs.
                'can_infer': If model is able to perform inference on new data. If False, then the model either was not saved, was deleted, or an ancestor of the model cannot infer.
                    `can_infer` is often False when `save_bagged_folds=False` was specified in initial `task.fit`.
                'fit_order': The order in which models were fit. The first model fit has `fit_order=1`, and the Nth model fit has `fit_order=N`. The order corresponds to the first child model fit in the case of bagged ensembles.

            Parameters
            ----------
            dataset : str or :class:`TabularDataset` or `pandas.DataFrame` (optional)
                This Dataset must also contain the label-column with the same column-name as specified during fit().
                If specified, then the leaderboard returned will contain additional columns 'score_test', 'pred_time_test', and 'pred_time_test_marginal'.
                    'score_test': The score of the model on the 'eval_metric' for the dataset provided.
                    'pred_time_test': The true end-to-end wall-clock inference time of the model for the dataset provided.
                        Equivalent to the sum of all 'pred_time_test_marginal' values for the model and all of its base models.
                    'pred_time_test_marginal': The inference time of the model for the dataset provided, minus the inference time for the model's base models, if it has any.
                        Note that this ignores the time required to load the model into memory when bagging is disabled.
                If str is passed, `dataset` will be loaded using the str value as the file path.
            extra_info : bool, default = False
                If `True`, will return extra columns with advanced info.
                This requires additional computation as advanced info data is calculated on demand.
                Additional output columns when `extra_info=True` include:
                    'num_features': Number of input features used by the model.
                        Some models may ignore certain features in the preprocessed data.
                    'num_models': Number of models that actually make up this "model" object.
                        For non-bagged models, this is 1. For bagged models, this is equal to the number of child models (models trained on bagged folds) the bagged ensemble contains.
                    'num_models_w_ancestors': Equivalent to the sum of 'num_models' values for the model and its' ancestors (see below).
                    'memory_size': The amount of memory in bytes the model requires when persisted in memory. This is not equivalent to the amount of memory the model may use during inference.
                        For bagged models, this is the sum of the 'memory_size' of all child models.
                    'memory_size_w_ancestors': Equivalent to the sum of 'memory_size' values for the model and its' ancestors.
                        This is the amount of memory required to avoid loading any models in-between inference calls to get predictions from this model.
                        For online-inference, this is critical. It is important that the machine performing online inference has memory more than twice this value to avoid loading models for every call to inference by persisting models in memory.
                    'memory_size_min': The amount of memory in bytes the model minimally requires to perform inference.
                        For non-bagged models, this is equivalent to 'memory_size'.
                        For bagged models, this is equivalent to the largest child model's 'memory_size_min'.
                        To minimize memory usage, child models can be loaded and un-persisted one by one to infer. This is the default behavior if a bagged model was not already persisted in memory prior to inference.
                    'memory_size_min_w_ancestors': Equivalent to the max of the 'memory_size_min' values for the model and its' ancestors.
                        This is the minimum required memory to infer with the model by only loading one model at a time, as each of its ancestors will also have to be loaded into memory.
                        For offline-inference where latency is not a concern, this should be used to determine the required memory for a machine if 'memory_size_w_ancestors' is too large.
                    'num_ancestors': Number of ancestor models for the given model.

                    'num_descendants': Number of descendant models for the given model.

                    'model_type': The type of the given model.
                        If the model is an ensemble type, 'child_model_type' will indicate the inner model type. A stack ensemble of bagged LightGBM models would have 'StackerEnsembleModel' as its model type.
                    'child_model_type': The child model type. None if the model is not an ensemble. A stack ensemble of bagged LightGBM models would have 'LGBModel' as its child type.
                        child models are models which are used as a group to generate a given bagged ensemble model's predictions. These are the models trained on each fold of a bagged ensemble.
                        For 10-fold bagging, the bagged ensemble model would have 10 child models.
                        For 10-fold bagging with 3 repeats, the bagged ensemble model would have 30 child models.
                        Note that child models are distinct from ancestors and descendants.
                    'hyperparameters': The hyperparameter values specified for the model.
                        All hyperparameters that do not appear in this dict remained at their default values.
                    'hyperparameters_fit': The hyperparameters set by the model during fit.
                        This overrides the 'hyperparameters' value for a particular key if present in 'hyperparameters_fit' to determine the fit model's final hyperparameters.
                        This is most commonly set for hyperparameters that indicate model training iterations or epochs, as early stopping can find a different value from what 'hyperparameters' indicated.
                        In these cases, the provided hyperparameter in 'hyperparameters' is used as a maximum for the model, but the model is still able to early stop at a smaller value during training to achieve a better validation score or to satisfy time constraints.
                        For example, if a NN model was given `epochs=500` as a hyperparameter, but found during training that `epochs=60` resulted in optimal validation score, it would use `epoch=60` and `hyperparameters_fit={'epoch': 60}` would be set.
                    'AG_args_fit': Special AutoGluon arguments that influence model fit.
                        See the documentation of the `hyperparameters` argument in `TabularPrediction.fit()` for more information.
                    'features': List of feature names used by the model.

                    'child_hyperparameters': Equivalent to 'hyperparameters', but for the model's children.

                    'child_hyperparameters_fit': Equivalent to 'hyperparameters_fit', but for the model's children.

                    'child_AG_args_fit': Equivalent to 'AG_args_fit', but for the model's children.

                    'ancestors': The model's ancestors. Ancestor models are the models which are required to make predictions during the construction of the model's input features.
                        If A is an ancestor of B, then B is a descendant of A.
                        If a model's ancestor is deleted, the model is no longer able to infer on new data, and its 'can_infer' value will be False.
                        A model can only have ancestor models whose 'stack_level' are lower than itself.
                        'stack_level'=0 models have no ancestors.
                    'descendants': The model's descendants. Descendant models are the models which require this model to make predictions during the construction of their input features.
                        If A is a descendant of B, then B is an ancestor of A.
                        If this model is deleted, then all descendant models will no longer be able to infer on new data, and their 'can_infer' values will be False.
                        A model can only have descendant models whose 'stack_level' are higher than itself.

            only_pareto_frontier : bool, default = False
                If `True`, only return model information of models in the Pareto frontier of the accuracy/latency trade-off (models which achieve the highest score within their end-to-end inference time).
                At minimum this will include the model with the highest score and the model with the lowest inference time.
                This is useful when deciding which model to use during inference if inference time is a consideration.
                Models filtered out by this process would never be optimal choices for a user that only cares about model inference time and score.
            silent : bool, default = False
                Should leaderboard DataFrame be printed?

            Returns
            -------
            Pandas `pandas.DataFrame` of model performance summary information.
        """
        dataset = self.__get_dataset(dataset) if dataset is not None else dataset
        return self._learner.leaderboard(X=dataset, extra_info=extra_info, only_pareto_frontier=only_pareto_frontier, silent=silent)

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
        model_inner_types = self._trainer.get_models_attribute_dict(attribute='type_inner')
        model_typenames = {key: model_types[key].__name__ for key in model_types}
        model_innertypenames = {key: model_inner_types[key].__name__ for key in model_types if key in model_inner_types}
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
            'model_performance': self._trainer.get_models_attribute_dict('val_score'),  # dict with key = model-name, value = validation performance
            'model_best': self._trainer.model_best,  # the name of the best model (on validation data)
            'model_paths': self._trainer.get_models_attribute_dict('path'),  # dict with key = model-name, value = path to model file
            'model_fit_times': self._trainer.get_models_attribute_dict('fit_time'),
            'model_pred_times': self._trainer.get_models_attribute_dict('predict_time'),
            'num_bagging_folds': self._trainer.k_fold,
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
        for model_name in self._trainer.get_model_names_all():
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
            print("Feature Metadata (Processed):")
            print("(raw dtype, special dtypes):")
            print(self.feature_metadata)
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

    def transform_features(self, dataset=None, model=None, base_models=None, return_original_features=True):
        """
        Transforms dataset features through the AutoGluon feature generator.
        This is useful to gain an understanding of how AutoGluon interprets the dataset features.
        The output of this function can be used to train further models, even outside of AutoGluon.
        This can be useful for training your own models on the same data representation as AutoGluon.
        Individual AutoGluon models like the neural network may apply additional feature transformations that are not reflected in this method.
        This method only applies universal transforms employed by all AutoGluon models.
        When `dataset=None`, `base_models=[{best_model}], and bagging was enabled during fit():
            This returns the out-of-fold predictions of the best model, which can be used as training input to a custom user stacker model.

        Parameters
        ----------
        dataset : str or :class:`TabularDataset` or `pandas.DataFrame` (optional)
            The dataset to apply feature transformation to.
            This dataset does not require the label column.
            If str is passed, `dataset` will be loaded using the str value as the file path.
            If not specified, the original dataset used during fit() will be used if fit() was previously called with `cache_data=True`. Otherwise, an exception will be raised.
                For non-bagged mode predictors:
                    The dataset used when not specified is the validation set.
                    This can either be an automatically generated validation set or the user-defined `tuning_data` if passed during fit().
                    If all parameters are unspecified, then the output is equivalent to `predictor.load_data_internal(dataset='val', return_X=True, return_y=False)[0]`.
                    To get the label values of the output, call `predictor.load_data_internal(dataset='val', return_X=False, return_y=True)[1]`.
                    If the original training set is desired, it can be passed in through `dataset`.
                        Warning: Do not pass the original training set if `model` or `base_models` are set. This will result in overfit feature transformation.
                For bagged mode predictors:
                    The dataset used when not specified is the full training set.
                    If all parameters are unspecified, then the output is equivalent to `predictor.load_data_internal(dataset='train', return_X=True, return_y=False)[0]`.
                    To get the label values of the output, call `predictor.load_data_internal(dataset='train', return_X=False, return_y=True)[1]`.
                    `base_model` features generated in this instance will be from out-of-fold predictions.
                    Note that the training set may differ from the training set originally passed during fit(), as AutoGluon may choose to drop or duplicate rows during training.
                    Warning: Do not pass the original training set through `dataset` if `model` or `base_models` are set. This will result in overfit feature transformation. Instead set `dataset=None`.
        model : str, default = None
            Model to generate input features for.
            The output data will be equivalent to the input data that would be sent into `model.predict_proba(data)`.
                Note: This only applies to cases where `dataset` is not the training data.
            If `None`, then only return generically preprocessed features prior to any model fitting.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.
            Specifying a `refit_full` model will cause an exception if `dataset=None`.
            `base_models=None` is a requirement when specifying `model`.
        base_models : list, default = None
            List of model names to use as base_models for a hypothetical stacker model when generating input features.
            If `None`, then only return generically preprocessed features prior to any model fitting.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.
            If a stacker model S exists with `base_models=M`, then setting `base_models=M` is equivalent to setting `model=S`.
            `model=None` is a requirement when specifying `base_models`.
        return_original_features : bool, default = True
            Whether to return the original features.
            If False, only returns the additional output columns from specifying `model` or `base_models`.
                This is useful to set to False if the intent is to use the output as input to further stacker models without the original features.

        Returns
        -------
        Pandas `pandas.DataFrame` of the provided `dataset` after feature transformation has been applied.
        This output does not include the label column, and will remove it if present in the supplied `dataset`.
        If a transformed label column is desired, use `predictor.transform_labels`.

        Examples
        --------
        >>> from autogluon import TabularPrediction as task
        >>> train_data = task.Dataset('train.csv')
        >>> predictor = task.fit(train_data=train_data, label='class', auto_stack=True, cache_data=True)  # predictor is in bagged mode and `cache_data=True`.
        >>> model = 'weighted_ensemble_k0_l1'
        >>> test_data = task.Dataset('test.csv')
        >>> train_data_transformed = predictor.transform_features(model=model)  # Internal training DataFrame used as input to `model.fit()` during `predictor = task.fit(train_data=train_data, ...)`
        >>> test_data_transformed = predictor.transform_features(dataset=test_data, model=model)  # Internal test DataFrame used as input to `model.predict_proba()` during `predictor.predict_proba(test_data, model=model)`

        """
        dataset = self.__get_dataset(dataset) if dataset is not None else dataset
        return self._learner.get_inputs_to_stacker(dataset=dataset, model=model, base_models=base_models, use_orig_features=return_original_features)

    # TODO: Add support for DataFrame input and support for DataFrame output
    #  Add support for retaining DataFrame/Series indices in output
    #  Add as_pandas parameter
    def transform_labels(self, labels, inverse=False, proba=False):
        """
        Transforms dataset labels to the internal label representation.
        This can be useful for training your own models on the same data label representation as AutoGluon.
        Regression problems do not differ between original and internal representation, and thus this method will return the provided labels.
        Warning: When `inverse=False`, it is possible for the output to contain NaN label values in multiclass problems if the provided label was dropped during training.

        Parameters
        ----------
        labels : `numpy.ndarray` or `pandas.Series`
            Labels to transform.
            If `proba=False`, an example input would be the output of `predictor.predict(test_data)`.
            If `proba=True`, an example input would be the output of `predictor.predict_proba(test_data)`.
        inverse : boolean, default = False
            When `True`, the input labels are treated as being in the internal representation and the original representation is outputted.
        proba : boolean, default = False
            When `True`, the input labels are treated as probabilities and the output will be the internal representation of probabilities.
                In this case, it is expected that `labels` be a `numpy.ndarray`.
                If the `problem_type` is multiclass:
                    The input column order must be equal to `predictor.class_labels`.
                    The output column order will be equal to `predictor.class_labels_internal`.
                    if `inverse=True`, the same logic applies, but with input and output columns interchanged.
            When `False`, the input labels are treated as actual labels and the output will be the internal representation of the labels.
                In this case, it is expected that `labels` be a `pandas.Series` or `numpy.ndarray`.

        Returns
        -------
        Pandas `pandas.Series` of labels if `proba=False` or Numpy `numpy.ndarray` of label probabilities if `proba=True`

        """
        if inverse:
            if proba:
                labels_transformed = self._learner.label_cleaner.inverse_transform_proba(y=labels)
            else:
                labels_transformed = self._learner.label_cleaner.inverse_transform(y=labels)
        else:
            if proba:
                labels_transformed = self._learner.label_cleaner.transform_proba(y=labels)
            else:
                labels_transformed = self._learner.label_cleaner.transform(y=labels)
        return labels_transformed

    # TODO: Consider adding time_limit option to early stop the feature importance process
    # TODO: Add option to specify list of features within features list, to check importances of groups of features. Make tuple to specify new feature name associated with group.
    def feature_importance(self, dataset=None, model=None, features=None, feature_stage='original', subsample_size=1000, silent=False, **kwargs):
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
        dataset : str or :class:`TabularDataset` or `pandas.DataFrame` (optional)
            This dataset must also contain the label-column with the same column-name as specified during `fit()`.
            If specified, then the dataset is used to calculate the feature importance scores.
            If str is passed, `dataset` will be loaded using the str value as the file path.
            If not specified, the original dataset used during `fit()` will be used if `cache_data=True`. Otherwise, an exception will be raised.
            Do not pass the training data through this argument, as the feature importance scores calculated will be biased due to overfitting.
                More accurate feature importances will be obtained from new data that was held-out during `fit()`.
        model : str, default = None
            Model to get feature importances for, if None the best model is chosen.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`
        features : list, default = None
            List of str feature names that feature importances are calculated for and returned, specify None to get all feature importances.
            If you only want to compute feature importances for some of the features, you can pass their names in as a list of str.
            Valid feature names change depending on the `feature_stage`.
                To get the list of feature names for `feature_stage='transformed'`, call `list(predictor.transform_features().columns)`.
                To get the list of feature names for `feature_stage=`transformed_model`, call `list(predictor.transform_features(model={model_name}).columns)`.
        feature_stage : str, default = 'original'
            What stage of feature-processing should importances be computed for.
            Options:
                'original':
                    Compute importances of the original features.
                    Warning: `dataset` must be specified with this option, otherwise an exception will be raised.
                'transformed':
                    Compute importances of the post-internal-transformation features (after automated feature engineering). These features may be missing some original features, or add new features entirely.
                    An example of new features would be ngram features generated from a text column.
                    Warning: For bagged models, feature importance calculation is not yet supported with this option when `dataset=None`. Doing so will raise an exception.
                'transformed_model':
                    Compute importances of the post-model-transformation features. These features are the internal features used by the requested model. They may differ greatly from the original features.
                    If the model is a stack ensemble, this will include stack ensemble features such as the prediction probability features of the stack ensemble's base (ancestor) models.
        subsample_size : int, default = 1000
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
        allowed_kwarg_names = {'raw'}
        for kwarg_name in kwargs.keys():
            if kwarg_name not in allowed_kwarg_names:
                raise ValueError("Unknown keyword argument specified: %s" % kwarg_name)
        if 'raw' in kwargs.keys():
            logger.log(30, 'Warning: `raw` is a deprecated parameter. Use `feature_stage` instead. Starting from AutoGluon 0.1.0, specifying `raw` as a parameter will cause an exception.')
            logger.log(30, 'Overriding `feature_stage` value with `raw` value.')
            raw = kwargs['raw']
            if raw is True:
                feature_stage = 'transformed'
            elif raw is False:
                feature_stage = 'transformed_model'
            else:
                raise ValueError(f'`raw` must be one of [True, False], but was {raw}.')

        dataset = self.__get_dataset(dataset) if dataset is not None else dataset
        if (dataset is None) and (not self._trainer.is_data_saved):
            raise AssertionError('No dataset was provided and there is no cached data to load for feature importance calculation. `cache_data=True` must be set in the `TabularPrediction.fit()` call to enable this functionality when dataset is not specified.')

        return self._learner.get_feature_importance(model=model, X=dataset, features=features, feature_stage=feature_stage, subsample_size=subsample_size, silent=silent)

    def persist_models(self, models='best', with_ancestors=True, max_memory=0.1) -> list:
        """
        Persist models in memory for reduced inference latency. This is particularly important if the models are being used for online-inference where low latency is critical.
        If models are not persisted in memory, they are loaded from disk every time they are asked to make predictions.

        Parameters
        ----------
        models : list of str or str, default = 'best'
            Model names of models to persist.
            If 'best' then the model with the highest validation score is persisted (this is the model used for prediction by default).
            If 'all' then all models are persisted.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.
        with_ancestors : bool, default = True
            If True, all ancestor models of the provided models will also be persisted.
            If False, stacker models will not have the models they depend on persisted unless those models were specified in `models`. This will slow down inference as the ancestor models will still need to be loaded from disk for each predict call.
            Only relevant for stacker models.
        max_memory : float, default = 0.1
            Proportion of total available memory to allow for the persisted models to use.
            If the models' summed memory usage requires a larger proportion of memory than max_memory, they are not persisted. In this case, the output will be an empty list.
            If None, then models are persisted regardless of estimated memory usage. This can cause out-of-memory errors.

        Returns
        -------
        List of persisted model names.
        """
        return self._learner.persist_trainer(low_memory=False, models=models, with_ancestors=with_ancestors, max_memory=max_memory)

    def unpersist_models(self, models='all') -> list:
        """
        Unpersist models in memory for reduced memory usage.
        If models are not persisted in memory, they are loaded from disk every time they are asked to make predictions.
        Note: Another way to reset the predictor and unpersist models is to reload the predictor from disk via `predictor = TabularPredictor.load(predictor.output_directory)`.

        Parameters
        ----------
        models : list of str or str, default = 'all'
            Model names of models to unpersist.
            If 'all' then all models are unpersisted.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names_persisted()`.

        Returns
        -------
        List of unpersisted model names.
        """
        return self._learner.load_trainer().unpersist_models(model_names=models)

    def refit_full(self, model='all'):
        """
        Retrain model on all of the data (training + validation).
        For bagged models:
            Optimizes a model's inference time by collapsing bagged ensembles into a single model fit on all of the training data.
            This process will typically result in a slight accuracy reduction and a large inference speedup.
            The inference speedup will generally be between 10-200x faster than the original bagged ensemble model.
                The inference speedup factor is equivalent to (k * n), where k is the number of folds (`num_bagging_folds`) and n is the number of finished repeats (`num_bagging_sets`) in the bagged ensemble.
            The runtime is generally 10% or less of the original fit runtime.
                The runtime can be roughly estimated as 1 / (k * n) of the original fit runtime, with k and n defined above.
        For non-bagged models:
            Optimizes a model's accuracy by retraining on 100% of the data without using a validation set.
            Will typically result in a slight accuracy increase and no change to inference time.
            The runtime will be approximately equal to the original fit runtime.
        This process does not alter the original models, but instead adds additional models.
        If stacker models are refit by this process, they will use the refit_full versions of the ancestor models during inference.
        Models produced by this process will not have validation scores, as they use all of the data for training.
            Therefore, it is up to the user to determine if the models are of sufficient quality by including test data in `predictor.leaderboard(dataset=test_data)`.
            If the user does not have additional test data, they should reference the original model's score for an estimate of the performance of the refit_full model.
                Warning: Be aware that utilizing refit_full models without separately verifying on test data means that the model is untested, and has no guarantee of being consistent with the original model.
        `cache_data` must have been set to `True` during the original training to enable this functionality.

        Parameters
        ----------
        model : str, default = 'all'
            Model name of model to refit.
                If 'all' then all models are refitted.
                If 'best' then the model with the highest validation score is refit.
            All ancestor models will also be refit in the case that the selected model is a weighted or stacker ensemble.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`.

        Returns
        -------
        Dictionary of original model names -> refit_full model names.
        """
        refit_full_dict = self._learner.refit_ensemble_full(model=model)
        return refit_full_dict

    def get_model_best(self):
        """
        Returns the string model name of the best model by validation score.
        This is typically the same model used during inference when `predictor.predict` is called without specifying a model.

        Returns
        -------
        String model name of the best model
        """
        return self._trainer.get_model_best(can_infer=True)

    def get_model_full_dict(self):
        """
        Returns a dictionary of original model name -> refit full model name.
        Empty unless `refit_full=True` was set during fit or `predictor.refit_full()` was called.
        This can be useful when determining the best model based off of `predictor.leaderboard()`, then getting the _FULL version of the model by passing its name as the key to this dictionary.

        Returns
        -------
        Dictionary of original model name -> refit full model name.
        """
        return copy.deepcopy(self._trainer.model_full_dict)

    def info(self):
        """
        [EXPERIMENTAL] Returns a dictionary of `predictor` metadata.
        Warning: This functionality is currently in preview mode.
            The metadata information returned may change in structure in future versions without warning.
            The definitions of various metadata values are not yet documented.
            The output of this function should not be used for programmatic decisions.
        Contains information such as row count, column count, model training time, validation scores, hyperparameters, and much more.

        Returns
        -------
        Dictionary of `predictor` metadata.
        """
        return self._learner.get_info(include_model_info=True)

    # TODO: Handle cases where name is same as a previously fit model, currently overwrites old model.
    # TODO: Add dataset argument
    # TODO: Add option to disable OOF generation of newly fitted models
    # TODO: Move code logic to learner/trainer
    # TODO: Add task.fit arg to perform this automatically at end of training
    # TODO: Consider adding cutoff arguments such as top-k models
    def fit_weighted_ensemble(self, base_models: list = None, name_suffix='_custom', expand_pareto_frontier=False, time_limits=None):
        """
        Fits new weighted ensemble models to combine predictions of previously-trained models.
        `cache_data` must have been set to `True` during the original training to enable this functionality.

        Parameters
        ----------
        base_models : list, default = None
            List of model names the weighted ensemble can consider as candidates.
            If None, all previously trained models are considered except for weighted ensemble models.
            As an example, to train a weighted ensemble that can only have weights assigned to the models 'model_a' and 'model_b', set `base_models=['model_a', 'model_b']`
        name_suffix : str, default = '_custom'
            Name suffix to add to the name of the newly fitted ensemble model.
            Warning: If the name of a trained model from this function is identical to an existing model, it will overwrite the existing model.
            Ensure that `name_suffix` is unique each time `fit_weighted_ensemble` is called to avoid this.
        expand_pareto_frontier : bool, default = False
            If True, will train N-1 weighted ensemble models instead of 1, where `N=len(base_models)`.
            The final model trained when True is equivalent to the model trained when False.
            These weighted ensemble models will attempt to expand the pareto frontier.
            This will create many different weighted ensembles which have different accuracy/memory/inference-speed trade-offs.
            This is particularly useful when inference speed is an important consideration.
        time_limits : int, default = None
            Time in seconds each weighted ensemble model is allowed to train for. If `expand_pareto_frontier=True`, the `time_limits` value is applied to each model.
            If None, the ensemble models train without time restriction.

        Returns
        -------
        List of newly trained weighted ensemble model names.
        If an exception is encountered while training an ensemble model, that model's name will be absent from the list.
        """
        trainer = self._learner.load_trainer()

        if trainer.bagged_mode:
            X = trainer.load_X_train()
            y = trainer.load_y_train()
            fit = True
        else:
            X = trainer.load_X_val()
            y = trainer.load_y_val()
            fit = False

        stack_name = 'aux1'
        if base_models is None:
            base_models = trainer.get_model_names(stack_name='core')

        X_train_stack_preds = trainer.get_inputs_to_stacker(X=X, base_models=base_models, fit=fit, use_orig_features=False)

        models = []

        if expand_pareto_frontier:
            leaderboard = self.leaderboard(silent=True)
            leaderboard = leaderboard[leaderboard['model'].isin(base_models)]
            leaderboard = leaderboard.sort_values(by='pred_time_val')
            models_to_check = leaderboard['model'].tolist()
            for i in range(1, len(models_to_check) - 1):
                models_to_check_now = models_to_check[:i+1]
                max_base_model_level = max([trainer.get_model_level(base_model) for base_model in models_to_check_now])
                weighted_ensemble_level = max_base_model_level + 1
                models += trainer.generate_weighted_ensemble(X=X_train_stack_preds, y=y, level=weighted_ensemble_level, stack_name=stack_name, base_model_names=models_to_check_now, name_suffix=name_suffix + '_pareto' + str(i), time_limit=time_limits)

        max_base_model_level = max([trainer.get_model_level(base_model) for base_model in base_models])
        weighted_ensemble_level = max_base_model_level + 1
        models += trainer.generate_weighted_ensemble(X=X_train_stack_preds, y=y, level=weighted_ensemble_level, stack_name=stack_name, base_model_names=base_models, name_suffix=name_suffix, time_limit=time_limits)

        return models

    @classmethod
    def load(cls, output_directory, verbosity=2):
        """
        Load a predictor object previously produced by `fit()` from file and returns this object.
        It is highly recommended the predictor be loaded with the exact AutoGluon version it was fit with.
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
        try:
            from ...version import __version__
            version_inference = __version__
        except:
            version_inference = None
        try:
            version_fit = learner.version
        except:
            version_fit = None
        if version_fit is None:
            version_fit = 'Unknown (Likely <=0.0.11)'
        if version_inference != version_fit:
            logger.warning('')
            logger.warning('############################## WARNING ##############################')
            logger.warning('WARNING: AutoGluon version differs from the version used during the original model fit! This may lead to instability and it is highly recommended the model be loaded with the exact AutoGluon version it was fit with.')
            logger.warning(f'\tFit Version:     {version_fit}')
            logger.warning(f'\tCurrent Version: {version_inference}')
            logger.warning('############################## WARNING ##############################')
            logger.warning('')

        return cls(learner=learner)

    def save(self):
        """ Save this predictor to file in directory specified by this Predictor's `output_directory`.
            Note that `fit()` already saves the predictor object automatically
            (we do not recommend modifying the Predictor object yourself as it tracks many trained models).
        """
        self._learner.save()
        logger.log(20, "TabularPredictor saved. To load, use: TabularPredictor.load(\"%s\")" % self.output_directory)

    def load_data_internal(self, dataset='train', return_X=True, return_y=True):
        """
        Loads the internal data representation used during model training.
        Individual AutoGluon models like the neural network may apply additional feature transformations that are not reflected in this method.
        This method only applies universal transforms employed by all AutoGluon models.
        This will raise an exception if `cache_data=False` was previously set in `task.fit()`.
        Warning, the internal representation may:
            Have different features compared to the original data.
            Have different row counts compared to the original data.
            Have indices which do not align with the original data.
            Have label values which differ from those in the original data.
        Internal data representations should NOT be combined with the original data, in most cases this is not possible.

        Parameters
        ----------
        dataset : str, default = 'train'
            The dataset to load.
            Valid values are:
                'train':
                    Load the training data used during model training.
                    This is a transformed and augmented version of the `train_data` passed in `task.fit()`.
                'val':
                    Load the validation data used during model training.
                    This is a transformed and augmented version of the `tuning_data` passed in `task.fit()`.
                    If `tuning_data=None` was set in `task.fit()`, then `tuning_data` is an automatically generated validation set created by splitting `train_data`.
                    Warning: Will raise an exception if called by a bagged predictor, as bagged predictors have no validation data.
        return_X : bool, default = True
            Whether to return the internal data features
            If set to `False`, then the first element in the returned tuple will be None.
        return_y : bool, default = True
            Whether to return the internal data labels
            If set to `False`, then the second element in the returned tuple will be None.

        Returns
        -------
        Tuple of (`pandas.DataFrame`, `pandas.Series`) corresponding to the internal data features and internal data labels, respectively.

        """
        if dataset == 'train':
            load_X = self._trainer.load_X_train
            load_y = self._trainer.load_y_train
        elif dataset == 'val':
            load_X = self._trainer.load_X_val
            load_y = self._trainer.load_y_val
        else:
            raise ValueError(f'dataset must be one of: [\'train\', \'val\'], but was \'{dataset}\'.')
        X = load_X() if return_X else None
        y = load_y() if return_y else None
        return X, y

    def save_space(self, remove_data=True, remove_fit_stack=True, requires_save=True, reduce_children=False):
        """
        Reduces the memory and disk size of predictor by deleting auxiliary model files that aren't needed for prediction on new data.
        This function has NO impact on inference accuracy.
        It is recommended to invoke this method if the only goal is to use the trained model for prediction.
        However, certain advanced functionality may no longer be available after `save_space()` has been called.

        Parameters
        ----------
        remove_data : bool, default = True
            Whether to remove cached files of the original training and validation data.
            Only reduces disk usage, it has no impact on memory usage.
            This is especially useful when the original data was large.
            This is equivalent to setting `cache_data=False` during the original `fit()`.
                Will disable all advanced functionality that requires `cache_data=True`.
        remove_fit_stack : bool, default = True
            Whether to remove information required to fit new stacking models and continue fitting bagged models with new folds.
            Only reduces disk usage, it has no impact on memory usage.
            This includes:
                out-of-fold (OOF) predictions
            This is useful for multiclass problems with many classes, as OOF predictions can become very large on disk. (1 GB per model in extreme cases)
            This disables `predictor.refit_full()` for stacker models.
        requires_save : bool, default = True
            Whether to remove information that requires the model to be saved again to disk.
            Typically this only includes flag variables that don't have significant impact on memory or disk usage, but should technically be updated due to the removal of more important information.
                An example is the `is_data_saved` boolean variable in `trainer`, which should be updated to `False` if `remove_data=True` was set.
        reduce_children : bool, default = False
            Whether to apply the reduction rules to bagged ensemble children models. These are the models trained for each fold of the bagged ensemble.
            This should generally be kept as `False` since the most important memory and disk reduction techniques are automatically applied to these models during the original `fit()` call.

        """
        self._trainer.reduce_memory_size(remove_data=remove_data, remove_fit_stack=remove_fit_stack, remove_fit=True, remove_info=False, requires_save=requires_save, reduce_children=reduce_children)

    def delete_models(self, models_to_keep=None, models_to_delete=None, allow_delete_cascade=False, delete_from_disk=True, dry_run=True):
        """
        Deletes models from `predictor`.
        This can be helpful to minimize memory usage and disk usage, particularly for model deployment.
        This will remove all references to the models in `predictor`.
            For example, removed models will not appear in `predictor.leaderboard()`.
        WARNING: If `delete_from_disk=True`, this will DELETE ALL FILES in the deleted model directories, regardless if they were created by AutoGluon or not.
            DO NOT STORE FILES INSIDE OF THE MODEL DIRECTORY THAT ARE UNRELATED TO AUTOGLUON.

        Parameters
        ----------
        models_to_keep : str or list, default = None
            Name of model or models to not delete.
            All models that are not specified and are also not required as a dependency of any model in `models_to_keep` will be deleted.
            Specify `models_to_keep='best'` to keep only the best model and its model dependencies.
            `models_to_delete` must be None if `models_to_keep` is set.
            To see the list of possible model names, use: `predictor.get_model_names()` or `predictor.leaderboard()`.
        models_to_delete : str or list, default = None
            Name of model or models to delete.
            All models that are not specified but depend on a model in `models_to_delete` will also be deleted.
            `models_to_keep` must be None if `models_to_delete` is set.
        allow_delete_cascade : bool, default = False
            If `False`, if unspecified dependent models of models in `models_to_delete` exist an exception will be raised instead of deletion occurring.
                An example of a dependent model is m1 if m2 is a stacker model and takes predictions from m1 as inputs. In this case, m1 would be a dependent model of m2.
            If `True`, all dependent models of models in `models_to_delete` will be deleted.
            Has no effect if `models_to_delete=None`.
        delete_from_disk : bool, default = True
            If `True`, deletes the models from disk if they were persisted.
            WARNING: This deletes the entire directory for the deleted models, and ALL FILES located there.
                It is highly recommended to first run with `dry_run=True` to understand which directories will be deleted.
        dry_run : bool, default = True
            If `True`, then deletions don't occur, and logging statements are printed describing what would have occurred.
            Set `dry_run=False` to perform the deletions.

        """
        if models_to_keep == 'best':
            models_to_keep = self._trainer.model_best
            if models_to_keep is None:
                models_to_keep = self._trainer.get_model_best()
        self._trainer.delete_models(models_to_keep=models_to_keep, models_to_delete=models_to_delete, allow_delete_cascade=allow_delete_cascade, delete_from_disk=delete_from_disk, dry_run=dry_run)

    def get_model_names(self):
        """Returns the list of model names trained in this `predictor` object."""
        return self._trainer.get_model_names_all()

    def get_model_names_persisted(self):
        """Returns the list of model names which are persisted in memory."""
        return list(self._learner.load_trainer().models.keys())

    def distill(self, train_data=None, tuning_data=None, augmentation_data=None, time_limits=None, hyperparameters=None, holdout_frac=None,
                teacher_preds='soft', augment_method='spunge', augment_args={'size_factor':5,'max_size':int(1e5)}, models_name_suffix=None, verbosity=None):
        """
        Distill AutoGluon's most accurate ensemble-predictor into single models which are simpler/faster and require less memory/compute.
        Distillation can produce a model that is more accurate than the same model fit directly on the original training data.
        After calling `distill()`, there will be more models available in this Predictor, which can be evaluated using `predictor.leaderboard(test_data)` and deployed with: `predictor.predict(test_data, model=MODEL_NAME)`.
        This will raise an exception if `cache_data=False` was previously set in `task.fit()`.

        NOTE: Until catboost v0.24 is released, `distill()` with CatBoost students in multiclass classification requires you to first install catboost-dev: `pip install catboost-dev`

        Parameters
        ----------
        train_data : str or :class:`autogluon.task.tabular_prediction.TabularDataset` or `pandas.DataFrame`, default = None
            Same as `train_data` argument of :meth:`autogluon.task.tabular_prediction.TabularPrediction.fit`.
            If None, the same training data will be loaded from `fit()` call used to produce this Predictor.
        tuning_data : str or :class:`autogluon.task.tabular_prediction.TabularDataset` or `pandas.DataFrame`, default = None
            Same as `tuning_data` argument of :meth:`autogluon.task.tabular_prediction.TabularPrediction.fit`.
            If `tuning_data = None` and `train_data = None`: the same training/validation splits will be loaded from `fit()` call used to produce this Predictor,
            unless bagging/stacking was previously used in which case a new training/validation split is performed.
        augmentation_data : :class:`autogluon.task.tabular_prediction.TabularDataset` or `pandas.DataFrame`, default = None
            An optional extra dataset of unlabeled rows that can be used for augmenting the dataset used to fit student models during distillation (ignored if None).
        time_limits : int, default = None
            Approximately how long (in seconds) the distillation process should run for.
            If None, no time-constraint will be enforced allowing the distilled models to fully train.
        hyperparameters : dict or str, default = None
            Specifies which models to use as students and what hyperparameter-values to use for them.
            Same as `hyperparameters` argument of :meth:`autogluon.task.tabular_prediction.TabularPrediction.fit`.
            If = None, then student models will use the same hyperparameters from `fit()` used to produce this Predictor.
            Note: distillation is currently only supported for ['GBM','NN','RF','CAT'] student models, other models and their hyperparameters are ignored here.
        holdout_frac : float
            Same as `holdout_frac` argument of :meth:`autogluon.task.tabular_prediction.TabularPrediction.fit`.
        teacher_preds : str, default = 'soft'
            What form of teacher predictions to distill from (teacher refers to the most accurate AutoGluon ensemble-predictor).
            If None, we only train with original labels (no data augmentation).
            If 'hard', labels are hard teacher predictions given by: `teacher.predict()`
            If 'soft', labels are soft teacher predictions given by: `teacher.predict_proba()`
            Note: 'hard' and 'soft' are equivalent for regression problems.
            If `augment_method` is not None, teacher predictions are only used to label augmented data (training data keeps original labels).
            To apply label-smoothing: `teacher_preds='onehot'` will use original training data labels converted to one-hot vectors for multiclass problems (no data augmentation).
        augment_method : str, default='spunge'
            Specifies method to use for generating augmented data for distilling student models.
            Options include:
                None : no data augmentation performed.
                'munge' : The MUNGE algorithm (https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf).
                'spunge' : A simpler, more efficient variant of the MUNGE algorithm.
        augment_args : dict, default = {'size_factor':5, 'max_size': int(1e5)}
            Contains the following kwargs that control the chosen `augment_method` (these are ignored if `augment_method=None`):
                'num_augmented_samples': int, number of augmented datapoints used during distillation. Overrides 'size_factor', 'max_size' if specified.
                'max_size': float, the maximum number of augmented datapoints to add (ignored if 'num_augmented_samples' specified).
                'size_factor': float, if n = training data sample-size, we add int(n * size_factor) augmented datapoints, up to 'max_size'.
                Larger values in `augment_args` will slow down the runtime of distill(), and may produce worse results if provided time_limits are too small.
                You can also pass in kwargs for the `spunge_augment`, `munge_augment` functions in `autogluon/utils/tabular/ml/augmentation/distill_utils.py`.
        models_name_suffix : str, default = None
            Optional suffix that can be appended at the end of all distilled student models' names.
            Note: all distilled models will contain '_DSTL' substring in their name by default.
        verbosity : int, default = None
            Controls amount of printed output during distillation (4 = highest, 0 = lowest).
            Same as `verbosity` argument of :meth:`autogluon.task.tabular_prediction.TabularPrediction.fit`.
            If None, the same `verbosity` used in previous fit is employed again.

        Returns
        -------
        List of names (str) corresponding to the distilled models.

        Examples
        --------
        >>> from autogluon import TabularPrediction as task
        >>> train_data = task.Dataset('train.csv')
        >>> predictor = task.fit(train_data=train_data, label='class', auto_stack=True, cache_data=True)  # predictor is in bagged mode and `cache_data=True`.
        >>> distilled_model_names = predictor.distill()
        >>> test_data = task.Dataset('test.csv')
        >>> ldr = predictor.leaderboard(test_data)
        >>> model_todeploy = distilled_model_names[0]
        >>> predictor.predict(test_data, model=model_todeploy)

        """
        if isinstance(hyperparameters, str):
            hyperparameters = get_hyperparameter_config(hyperparameters)
        return self._learner.distill(X=train_data, X_val=tuning_data, time_limits=time_limits, hyperparameters=hyperparameters, holdout_frac=holdout_frac,
                                     verbosity=verbosity, models_name_suffix=models_name_suffix, teacher_preds=teacher_preds,
                                     augmentation_data=augmentation_data, augment_method=augment_method, augment_args=augment_args)

    @staticmethod
    def _summarize(key, msg, results):
        if key in results:
            print(msg + ": " + str(results[key]))

    @staticmethod
    def __get_dataset(dataset):
        if isinstance(dataset, TabularDataset):
            return dataset
        elif isinstance(dataset, pd.DataFrame):
            return TabularDataset(df=dataset)
        elif isinstance(dataset, str):
            return TabularDataset(file_path=dataset)
        elif isinstance(dataset, pd.Series):
            raise TypeError("dataset must be TabularDataset or pandas.DataFrame, not pandas.Series. \
                   To predict on just single example (ith row of table), use dataset.iloc[[i]] rather than dataset.iloc[i]")
        else:
            raise TypeError("dataset must be TabularDataset or pandas.DataFrame or str file path to dataset")
