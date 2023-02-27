import logging

import numpy as np
import pandas as pd

from .predictor import TabularPredictor

logger = logging.getLogger(__name__)


# TODO: Forcibly disable bagging, stacking, calibration, and non-interpretable models
class InterpretableTabularPredictor(TabularPredictor):
    def fit(self,
            train_data,
            tuning_data=None,
            time_limit=None,
            *,
            presets='interpretable',
            fit_weighted_ensemble=False,
            calibrate=False,
            **kwargs):
        logger.log(30, f'EXPERIMENTAL WARNING: Fitting {self.__class__.__name__}\n'
                       f'\tThis class is experimental and could be removed without warning in a future release.\n'
                       f'\tTo avoid confusing results, please only provide numeric features.\n'
                       f'\tCategorical, text, and datetime features will result in confusing rules that are hard to interpret.')

        return super().fit(
            train_data=train_data,
            tuning_data=tuning_data,
            time_limit=time_limit,
            presets=presets,
            fit_weighted_ensemble=fit_weighted_ensemble,
            **kwargs,
        )

    def interpretable_models_summary(self, verbosity=0):
        '''Summary of fitted interpretable models along with their corresponding complexities
        '''
        d = self.fit_summary(verbosity=verbosity)
        summaries = pd.DataFrame.from_dict(d)

        complexities = []
        info = self.info()
        for i in range(summaries.shape[0]):
            model_name = summaries.index.values[i]
            complexities.append(info['model_info'][model_name].get('complexity', np.nan))
        summaries.insert(2, 'complexity', complexities)
        summaries = summaries[~pd.isna(summaries.complexity)]  # remove non-interpretable models
        return summaries.sort_values(by=['model_performance', 'complexity'], ascending=[False, True])

    def print_interpretable_rules(self, complexity_threshold: int = 10, model_name: str = None):
        """
        Print the rules of the highest performing model below the complexity threshold.

        Parameters
        ----------
        complexity_threshold : int, default=10
            Threshold for complexity (number of rules) of fitted models to show.
            If not model complexity is below this threshold, prints the model with the lowest complexity.
        model_name : str,  default=None
            Optionally print rules for a particular model, ignoring the complexity threshold.
        """
        if model_name is None:
            summaries = self.interpretable_models_summary()
            summaries_filtered = summaries[summaries.complexity <= complexity_threshold]
            if summaries_filtered.shape[0] == 0:
                summaries_filtered = summaries
            model_name = summaries_filtered.index.values[0]  # best model is at top
        agmodel = self._trainer.load_model(model_name)
        imodel = agmodel.model
        print(imodel)

    def explain_classification_errors(self, data, model = None, print_rules: bool = True):
        """Explain classification errors by fitting a rule-based model to them

        Parameters
        ----------
        data : str or :class:`TabularDataset` or :class:`pd.DataFrame`
            The data to make predictions for. Should contain same column names as training Dataset and follow same format
            (may contain extra columns that won't be used by Predictor, including the label-column itself).
            If str is passed, `data` will be loaded using the str value as the file path.
        model : str (optional)
            The name of the model to get predictions from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.get_model_names()`
        print_rules : bool, optional
            Whether to print the learned rules

        Returns
        -------
        cls : imodels.classifier
            Interpretable rule-based classifier with fit/predict methods
        """
        import imodels
        data = self.__get_dataset(data)
        predictions = self._learner.predict(X=data, model=model, as_pandas=True)
        labels = data[self.label]
        cls, columns = imodels.explain_classification_errors(data, predictions, labels, print_rules=print_rules)
        return cls
