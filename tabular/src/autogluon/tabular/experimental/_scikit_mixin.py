from __future__ import annotations

import pandas as pd
from sklearn.utils.validation import check_array, check_is_fitted


class ScikitMixin:
    def _get_init_args(self, problem_type: str) -> dict:
        init_args = self.init_args
        if init_args is None:
            init_args = dict()
        init_args = init_args.copy()
        if "label" not in init_args:
            init_args["label"] = "_target_"
        if "problem_type" not in init_args:
            init_args["problem_type"] = problem_type
        if "eval_metric" not in init_args:
            init_args["eval_metric"] = self.eval_metric
        if "path" not in init_args:
            init_args["path"] = self.path
        if "verbosity" not in init_args:
            init_args["verbosity"] = self.verbosity
        return init_args

    def _get_fit_args(self) -> dict:
        fit_args = self.fit_args
        if fit_args is None:
            fit_args = dict()
        fit_args = fit_args.copy()

        if "time_limit" not in fit_args:
            fit_args["time_limit"] = self.time_limit
        if "presets" not in fit_args:
            fit_args["presets"] = self.presets
        if "hyperparameters" not in fit_args:
            fit_args["hyperparameters"] = self.hyperparameters
        if fit_args["time_limit"] is None:
            # TODO: This isn't technically right if the user specified `None`. Can fix in future by setting Predictor's default `time_limit="auto"`
            fit_args.pop("time_limit")
        return fit_args

    def _validate_input(self, X):
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Inconsistent number of features between fit and predict calls: ({self.n_features_in_}, {X.shape[1]})")
        return X

    def _combine_X_y(self, X, y) -> pd.DataFrame:
        label = self.predictor_.label
        X = pd.DataFrame(X)
        assert label not in list(X.columns), f"Cannot have column named {label}. Please rename the column to a different value."
        X[label] = y
        return X

    def leaderboard(self, X, y, **kwargs) -> pd.DataFrame:
        data = self._combine_X_y(X=X, y=y)
        return self.predictor_.leaderboard(data=data, **kwargs)
