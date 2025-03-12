from __future__ import annotations

import numpy as np
import pandas as pd

from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.models import AbstractModel
from autogluon.core.utils import generate_train_test_split
from autogluon.features.generators import LabelEncoderFeatureGenerator


class TabPFNModel(AbstractModel):
    """
    AutoGluon model wrapper to the TabPFN model: https://github.com/automl/TabPFN

    Paper: "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
    Authors: Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter

    TabPFN is a viable model option when inference speed is not a concern,
    and the number of rows of training data is less than 10,000.

    Additionally, TabPFN is only available for classification tasks with up to 10 classes and 100 features.

    To use this model, `tabpfn` must be installed.
    To install TabPFN, you can run `pip install autogluon.tabular[tabpfn]` or `pip install tabpfn`.
    """
    ag_key = "TABPFN"
    ag_name = "TabPFN"
    ag_priority = 110

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        from tabpfn import TabPFNClassifier

        ag_params = self._get_ag_params()
        sample_rows = ag_params.get("sample_rows")
        max_features = ag_params.get("max_features")
        max_classes = ag_params.get("max_classes")
        if max_classes is not None and self.num_classes > max_classes:
            # TODO: Move to earlier stage when problem_type is checked
            raise AssertionError(f"Max allowed classes for the model is {max_classes}, " f"but found {self.num_classes} classes.")

        # TODO: Make sample_rows generic
        if sample_rows is not None and len(X) > sample_rows:
            X, y = self._subsample_train(X=X, y=y, num_rows=sample_rows)
        X = self.preprocess(X)
        num_features = X.shape[1]
        # TODO: Make max_features generic
        if max_features is not None and num_features > max_features:
            raise AssertionError(f"Max allowed features for the model is {max_features}, " f"but found {num_features} features.")
        hyp = self._get_model_params()
        N_ensemble_configurations = hyp.get("N_ensemble_configurations")
        self.model = TabPFNClassifier(device="cpu", N_ensemble_configurations=N_ensemble_configurations).fit(  # TODO: Add GPU option
            X, y, overwrite_warning=True
        )

    # TODO: Make this generic by creating a generic `preprocess_train` and putting this logic prior to `_preprocess`.
    def _subsample_train(self, X: pd.DataFrame, y: pd.Series, num_rows: int, random_state=0) -> (pd.DataFrame, pd.Series):
        num_rows_to_drop = len(X) - num_rows
        X, _, y, _ = generate_train_test_split(
            X=X,
            y=y,
            problem_type=self.problem_type,
            test_size=num_rows_to_drop,
            random_state=random_state,
            min_cls_count_train=1,
        )
        return X, y

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Converts categorical to label encoded integers
        Keeps missing values, as TabPFN automatically handles missing values internally.
        """
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        X = X.to_numpy(dtype=np.float32)
        return X

    def _set_default_params(self):
        """
        By default, we only use 1 ensemble configurations to speed up inference times.
        Increase the value to improve model quality while linearly increasing inference time.

        Model quality improvement diminishes significantly beyond `N_ensemble_configurations=8`.
        """
        default_params = {
            "N_ensemble_configurations": 1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass"]

    def _get_default_auxiliary_params(self) -> dict:
        """
        TabPFN was originally learned on synthetic datasets with 1024 rows, and struggles to
        leverage additional rows effectively beyond a certain point.

        In the TabPFN paper, performance appeared to stagnate around 4000 rows of training data (Figure 10).
        Thus, we set `sample_rows=4096`, to only use that many rows of training data, even if more is available.

        TODO: TabPFN scales poorly on large datasets, so we set `max_rows=20000`.
         Not implemented yet, first move this logic to the trainer level to avoid `refit_full` edge-case crashes.
        TabPFN only works on datasets with at most 100 features, so we set `max_features=100`.
        TabPFN only works on datasets with at most 10 classes, so we set `max_classes=10`.
        """
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "sample_rows": 4096,
                # 'max_rows': 20000,
                "max_features": 100,
                "max_classes": 10,
            }
        )
        return default_auxiliary_params

    # FIXME: Enabling parallel bagging TabPFN creates a lot of warnings / potential failures from Ray
    # TODO: Consider not setting `max_sets=1`, and only setting it in the preset hyperparameter definition.
    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """
        Set max_sets to 1 when bagging, otherwise inference time could become extremely slow.
        Set fold_fitting_strategy to sequential_local, as parallel folding causing many warnings / potential errors from Ray.
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {
            "max_sets": 1,
            "fold_fitting_strategy": "sequential_local",
        }
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _ag_params(self) -> set:
        return {"sample_rows", "max_features", "max_classes"}

    def _more_tags(self) -> dict:
        """
        Because TabPFN doesn't use validation data for early stopping, it supports refit_full natively.
        """
        tags = {"can_refit_full": True}
        return tags
