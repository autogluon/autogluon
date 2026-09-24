"""`self.features` must remain the model's input columns, feature-space transforms go through preprocessing.

A custom model that assigned `self.features` to the output columns of a transform inside `_fit` fit and scored
OOF fine, but bagged inference failed later with an opaque `KeyError`: the bag selects `X[self.features]` on
the raw input once for all children, without routing through the child's `_predict_proba` (GitHub #5898).
`AbstractModel.fit` now fails fast, and the supported alternatives are covered here with bagging.
"""

import numpy as np
import pandas as pd
import pytest

from autogluon.common.features.types import R_FLOAT
from autogluon.core.models import AbstractModel, BaggedEnsembleModel
from autogluon.features.generators import AbstractFeatureGenerator


def _every_other_column(X: pd.DataFrame) -> pd.DataFrame:
    """A width-changing transform whose output columns are not in the input columns."""
    X_out = X.iloc[:, ::2]
    X_out.columns = [f"component_{i}" for i in range(X_out.shape[1])]
    return X_out


class _LstsqModel(AbstractModel):
    """Least-squares regression that fits and predicts through `self.preprocess`."""

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X, is_train=True)
        self.model = np.linalg.lstsq(np.asarray(X, dtype=float), np.asarray(y, dtype=float), rcond=None)[0]

    def _predict_proba(self, X, **kwargs):
        return np.asarray(self.preprocess(X, **kwargs), dtype=float) @ self.model

    @classmethod
    def supported_problem_types(cls):
        return ["regression"]


class _ResyncFeaturesModel(_LstsqModel):
    """The pattern from GitHub #5898: transform in `_fit` and point `self.features` at the transformed columns."""

    def _fit(self, X, y, **kwargs):
        X_t = _every_other_column(X)
        self.features = list(X_t.columns)
        self.feature_metadata = None
        self._preprocess_set_features(X_t)
        self.model = np.linalg.lstsq(X_t.to_numpy(dtype=float), np.asarray(y, dtype=float), rcond=None)[0]

    def _predict_proba(self, X, **kwargs):
        return _every_other_column(X).to_numpy(dtype=float) @ self.model


class _TransformInPreprocessModel(_LstsqModel):
    """Recommended: the transform lives in `_preprocess`, `self.features` is left alone."""

    def _preprocess(self, X, **kwargs):
        return _every_other_column(super()._preprocess(X, **kwargs))


class _NarrowFeaturesModel(_LstsqModel):
    """Narrowing `self.features` to a subset of the input columns stays allowed (e.g. dropping unused inputs)."""

    def _fit(self, X, y, **kwargs):
        self.features = self.features[1:]
        self._features_internal = self._features_internal_to_align = self.features
        super()._fit(X, y, **kwargs)


class _EveryOtherColumnFeatureGenerator(AbstractFeatureGenerator):
    """Recommended alternative: the transform as a model-specific feature generator."""

    def _fit_transform(self, X: pd.DataFrame, **kwargs):
        return self._transform(X), dict()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return _every_other_column(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_FLOAT])


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=[f"ch_{i}" for i in range(6)])
    y = pd.Series(X.iloc[:, ::2].sum(axis=1) + rng.normal(scale=0.01, size=len(X)))
    return X, y


def _make_model(model_cls, tmp_path, hyperparameters=None, bag: bool = False):
    model = model_cls(
        path=str(tmp_path),
        name=model_cls.__name__,
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        hyperparameters=hyperparameters,
    )
    if bag:
        model = BaggedEnsembleModel(
            model_base=model,
            path=str(tmp_path),
            name=f"{model_cls.__name__}_BAG",
            hyperparameters={"fold_fitting_strategy": "sequential_local"},
        )
    return model


def _fit(model, X, y, bag: bool):
    if bag:
        return model.fit(X=X, y=y, k_fold=2)
    return model.fit(X=X, y=y)


@pytest.mark.parametrize("bag", [False, True])
def test_features_replaced_in_fit_fails_fast(data, tmp_path, bag):
    X, y = data
    model = _make_model(_ResyncFeaturesModel, tmp_path, bag=bag)
    with pytest.raises(AssertionError, match=r"changed `self.features` during `_fit`.*component_0"):
        _fit(model, X, y, bag=bag)


@pytest.mark.parametrize(
    "model_cls, hyperparameters",
    [
        (_TransformInPreprocessModel, None),
        (
            _LstsqModel,
            # A single generator is used as-is and replaces the input columns (no passthrough wrapper)
            {
                "ag.model_specific_feature_generator_kwargs": {
                    "feature_generators": [(_EveryOtherColumnFeatureGenerator, {})]
                }
            },
        ),
        (_NarrowFeaturesModel, None),
    ],
    ids=["transform_in_preprocess", "model_specific_feature_generator", "narrow_features"],
)
@pytest.mark.parametrize("bag", [False, True])
def test_supported_feature_space_changes_predict(data, tmp_path, model_cls, hyperparameters, bag):
    X, y = data
    model = _make_model(model_cls, tmp_path, hyperparameters=hyperparameters, bag=bag)
    _fit(model, X, y, bag=bag)
    child = model.load_child(model.models[0]) if bag else model

    if model_cls is _NarrowFeaturesModel:
        assert child.features == list(X.columns[1:])
    else:
        # `self.features` stays the input columns, the transform is applied during preprocessing
        assert child.features == list(X.columns)
        if hyperparameters is not None:
            assert child._features_internal == ["component_0", "component_1", "component_2"]
        np.testing.assert_allclose(model.predict(X), y, atol=0.1)
    assert model.predict(X).shape == (len(X),)


def test_missing_input_features_error_names_the_model(data, tmp_path):
    X, y = data
    model = _make_model(_LstsqModel, tmp_path)
    model.fit(X=X, y=y)
    with pytest.raises(KeyError, match=r"_LstsqModel.*missing 1 of its 6 input features.*ch_2"):
        model.predict(X.drop(columns=["ch_2"]))
