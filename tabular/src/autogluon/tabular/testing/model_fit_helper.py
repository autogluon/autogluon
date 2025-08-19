from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from autogluon.core.data.label_cleaner import LabelCleaner
from autogluon.core.models import AbstractModel, BaggedEnsembleModel
from autogluon.core.utils import generate_train_test_split, infer_problem_type
from autogluon.features.generators import AbstractFeatureGenerator, AutoMLPipelineFeatureGenerator
from autogluon.tabular.testing.fit_helper import FitHelper


# Helper functions for training models outside of predictors
class ModelFitHelper:
    """
    Helper functions to test and verify models when fit outside TabularPredictor's API (aka as stand-alone models)
    """

    @staticmethod
    def fit_and_validate_dataset(
        dataset_name: str,
        model: AbstractModel,
        fit_args: dict,
        sample_size: int = 1000,
        check_predict_children: bool = False,
    ) -> AbstractModel:
        directory_prefix = "./datasets/"
        train_data, test_data, dataset_info = FitHelper.load_dataset(
            name=dataset_name, directory_prefix=directory_prefix
        )
        label = dataset_info["label"]
        model, label_cleaner, feature_generator = ModelFitHelper.fit_dataset(
            train_data=train_data, model=model, label=label, fit_args=fit_args, sample_size=sample_size
        )
        if sample_size is not None and sample_size < len(test_data):
            test_data = test_data.sample(n=sample_size, random_state=0)

        X_test = test_data.drop(columns=[label])
        X_test = feature_generator.transform(X_test)

        y_pred = model.predict(X_test)
        assert isinstance(y_pred, np.ndarray), (
            f"Expected np.ndarray as model.predict(X_test) output. Got: {y_pred.__class__}"
        )

        y_pred_proba = model.predict_proba(X_test)
        assert isinstance(y_pred_proba, np.ndarray), (
            f"Expected np.ndarray as model.predict_proba(X_test) output. Got: {y_pred.__class__}"
        )
        model.get_info()

        if check_predict_children:
            assert isinstance(model, BaggedEnsembleModel)
            y_pred_children = model.predict_children(X_test)
            assert len(y_pred_children) == model.n_children
            if model.can_predict_proba:
                y_pred_proba_children = model.predict_proba_children(X_test)
                assert len(y_pred_proba_children) == model.n_children
                y_pred_proba_from_children = np.mean(y_pred_proba_children, axis=0)
                assert np.isclose(y_pred_proba_from_children, y_pred_proba).all()

                for y_pred_proba_child, y_pred_child in zip(y_pred_proba_children, y_pred_children):
                    y_pred_child_from_proba = model.predict_from_proba(y_pred_proba=y_pred_proba_child)
                    assert np.isclose(y_pred_child_from_proba, y_pred_child).all()

        return model

    @staticmethod
    def fit_dataset(
        train_data: pd.DataFrame,
        model: AbstractModel,
        label: str,
        fit_args: dict,
        sample_size: int = None,
    ) -> Tuple[AbstractModel, LabelCleaner, AbstractFeatureGenerator]:
        if sample_size is not None and sample_size < len(train_data):
            train_data = train_data.sample(n=sample_size, random_state=0)
        X = train_data.drop(columns=[label])
        y = train_data[label]

        problem_type = infer_problem_type(y)
        label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
        y = label_cleaner.transform(y)
        feature_generator = AutoMLPipelineFeatureGenerator()
        X = feature_generator.fit_transform(X, y)

        X, X_val, y, y_val = generate_train_test_split(X, y, problem_type=problem_type, test_size=0.2, random_state=0)

        model.fit(X=X, y=y, X_val=X_val, y_val=y_val, **fit_args)
        return model, label_cleaner, feature_generator
