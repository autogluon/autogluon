import logging

import numpy as np
import pandas as pd

from .predictor import TabularPredictor

logger = logging.getLogger(__name__)


# TODO: Consider removing this unless improved. I suspect it will simply confuse users more than help them.
class InterpretableTabularPredictor(TabularPredictor):
    """
    EXPERIMENTAL

    AutoGluon InterpretableTabularPredictor predicts values in a column of a tabular dataset (classification or regression).
    InterpretableTabularPredictor shares the same functionality as TabularPredictor, but is limited to simple models
    that are easier to interpret visually via simple rules.

    InterpretableTabularPredictor should be used when accuracy is not important,
    and instead interpretability is the key requirement.

    Categorical features are one-hot-encoded to preserve interpretability.

    Stacking and bagging are not available in this predictor to preserve interpretability.
    """

    def fit(self, train_data, tuning_data=None, time_limit=None, *, presets="interpretable", **kwargs):
        logger.log(
            30,
            f"EXPERIMENTAL WARNING: Fitting {self.__class__.__name__}\n"
            f"\tThis class is experimental and could be removed without warning in a future release.\n"
            f"\tTo avoid confusing results, please only provide categorical and numeric features.\n"
            f"\tText and datetime features will result in confusing rules that are hard to interpret.",
        )

        return super().fit(
            train_data=train_data,
            tuning_data=tuning_data,
            time_limit=time_limit,
            presets=presets,
            **kwargs,
        )

    def _validate_fit_extra_kwargs(self, kwargs, extra_valid_keys=None) -> dict:
        kwargs = super()._validate_fit_extra_kwargs(kwargs=kwargs, extra_valid_keys=extra_valid_keys)
        print(kwargs)
        if "num_bag_folds" in kwargs and kwargs["num_bag_folds"] is not None and kwargs["num_bag_folds"] > 1:
            raise ValueError(f"{self.__class__.__name__} does not support `num_bag_folds`.")
        if "num_bag_sets" in kwargs and kwargs["num_bag_sets"] is not None and kwargs["num_bag_sets"] > 1:
            raise ValueError(f"{self.__class__.__name__} does not support `num_bag_sets`.")
        if "num_stack_levels" in kwargs and kwargs["num_stack_levels"] is not None and kwargs["num_stack_levels"] >= 1:
            raise ValueError(f"{self.__class__.__name__} does not support `num_stack_levels`.")
        if "auto_stack" in kwargs and kwargs["auto_stack"]:
            raise ValueError(f"{self.__class__.__name__} does not support `auto_stack`.")
        return kwargs

    def leaderboard_interpretable(self, verbose: bool = False, **kwargs) -> pd.DataFrame:
        """
        Leaderboard of fitted interpretable models along with their corresponding complexities.
        Identical to `.leaderboard`, but with an additional 'complexity' column indicating
        the number of rules used in the model.

        Models which do not support calculating 'complexity' will be filtered from this result.
        """
        silent = kwargs.pop("silent", None)
        if silent is not None:
            verbose = not silent
        leaderboard = self.leaderboard(**kwargs)

        complexities = []
        info = self.info()
        for i in range(leaderboard.shape[0]):
            model_name = leaderboard.iloc[i]["model"]
            complexities.append(info["model_info"][model_name].get("complexity", np.nan))
        leaderboard.insert(2, "complexity", complexities)  # insert directly after score_test/score_val
        leaderboard = leaderboard[~pd.isna(leaderboard.complexity)]  # remove non-interpretable models
        score_col = "score_test" if "score_test" in leaderboard.columns else "score_val"
        leaderboard = leaderboard.sort_values(by=[score_col, "complexity"], ascending=[False, True], ignore_index=True)
        if verbose:
            with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
                print(leaderboard)
        return leaderboard

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
            summaries = self.leaderboard_interpretable()
            summaries_filtered = summaries[summaries.complexity <= complexity_threshold]
            if summaries_filtered.shape[0] == 0:
                summaries_filtered = summaries
            model_name = summaries_filtered.iloc[0]["model"]  # best model is at top
        agmodel = self._trainer.load_model(model_name)
        imodel = agmodel.model
        print(imodel)

    # TODO: I have not been able to extract any insight from the output of this method.
    #  I don't see how it is useful.
    def explain_classification_errors(self, data, model=None, print_rules: bool = True):
        """Explain classification errors by fitting a rule-based model to them

        Parameters
        ----------
        data : str or :class:`pd.DataFrame`
            The data to make predictions for. Should contain same column names as training Dataset and follow same format
            (may contain extra columns that won't be used by Predictor, including the label-column itself).
            If str is passed, `data` will be loaded using the str value as the file path.
        model : str (optional)
            The name of the model to get predictions from. Defaults to None, which uses the highest scoring model on the validation set.
            Valid models are listed in this `predictor` by calling `predictor.model_names()`
        print_rules : bool, optional
            Whether to print the learned rules

        Returns
        -------
        cls : imodels.classifier
            Interpretable rule-based classifier with fit/predict methods
        """
        import imodels

        if model is None:
            model = self.model_best
        data = self._get_dataset(data)
        predictions = self.predict(data=data, model=model, as_pandas=True)
        labels = data[self.label]
        data_transformed = self.transform_features(data=data, model=model)
        labels_transformed = self.transform_labels(labels=labels)
        cls, columns = imodels.explain_classification_errors(data_transformed, predictions, labels_transformed, print_rules=print_rules)
        return cls
