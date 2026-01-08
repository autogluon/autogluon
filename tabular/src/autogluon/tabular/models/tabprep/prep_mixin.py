from __future__ import annotations

import logging
from typing import Type

import numpy as np
import pandas as pd

from autogluon.features import (
    ArithmeticFeatureGenerator,
    BulkFeatureGenerator,
    CategoricalInteractionFeatureGenerator,
    OOFTargetEncodingFeatureGenerator,
)
from autogluon.features.generators.abstract import AbstractFeatureGenerator

logger = logging.getLogger(__name__)

# TODO: In future we can have a feature generator registry like what is done for models
_feature_generator_class_lst = [
    ArithmeticFeatureGenerator,
    CategoricalInteractionFeatureGenerator,
    OOFTargetEncodingFeatureGenerator,
]

_feature_generator_class_map = {
    feature_generator_cls.__name__: feature_generator_cls for feature_generator_cls in _feature_generator_class_lst
}


def _recursive_expand_prep_param(prep_param: tuple | list[list | tuple]) -> list[tuple]:
    if isinstance(prep_param, list):
        if len(prep_param) == 0:
            param_type = "list"
        elif len(prep_param) == 2:
            if isinstance(prep_param[0], (str, AbstractFeatureGenerator)):
                param_type = "generator"
            else:
                param_type = "list"
        else:
            param_type = "list"
    elif isinstance(prep_param, tuple):
        param_type = "generator"
    else:
        raise ValueError(f"Invalid value for prep_param: {prep_param}")
    if param_type == "list":
        out = []
        for p in prep_param:
            out += _recursive_expand_prep_param(p)
        return out
    elif param_type == "generator":
        return [prep_param]
    else:
        raise ValueError(f"Invalid value for prep_param: {prep_param}")


# FIXME: Why is preprocessing twice as slow per fold when bagging LightGBM??? Need to investigate. Try sequential fold fit
# TODO: Why is `prep_params` a dict instead of a list?
class ModelAgnosticPrepMixin:
    def _estimate_dtypes_after_preprocessing(self, X: pd.DataFrame, **kwargs) -> int:
        prep_params = self._get_ag_params().get("prep_params", None)
        if prep_params is None:
            prep_params = []

        # FIXME: Temporarily simplify for memory calculation
        prep_params = _recursive_expand_prep_param(prep_params)

        X_nunique = X.nunique().values
        n_categorical = X.select_dtypes(exclude=[np.number]).shape[1]
        n_numeric = X.loc[:, X_nunique > 2].select_dtypes(include=[np.number]).shape[1]
        n_binary = (
            X.loc[:, X_nunique <= 2].select_dtypes(include=[np.number]).shape[1]
        )  # NOTE: It can happen that features have less than two unique values if cleaning is applied before the bagging, i.e. Bioresponse

        assert n_numeric + n_categorical + n_binary == X.shape[1]  # NOTE: FOr debugging, to be removed later
        for preprocessor_cls_name, init_params in prep_params:
            if preprocessor_cls_name == "ArithmeticFeatureGenerator":
                prep_cls = ArithmeticFeatureGenerator(target_type=self.problem_type, **init_params)
            elif preprocessor_cls_name == "CategoricalInteractionFeatureGenerator":
                prep_cls = CategoricalInteractionFeatureGenerator(target_type=self.problem_type, **init_params)
            elif preprocessor_cls_name == "OOFTargetEncodingFeatureGenerator":
                prep_cls = OOFTargetEncodingFeatureGenerator(target_type=self.problem_type, **init_params)
            else:
                raise ValueError(f"Unknown preprocessor class name: {preprocessor_cls_name}")
            n_numeric, n_categorical, n_binary = prep_cls.estimate_new_dtypes(
                n_numeric, n_categorical, n_binary, num_classes=self.num_classes
            )

        return n_numeric, n_categorical, n_binary

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        n_numeric, n_categorical, n_binary = self._estimate_dtypes_after_preprocessing(X=X, **kwargs)

        if hasattr(self, "_estimate_memory_usage_static_lite"):
            return self._estimate_memory_usage_static_lite(
                num_samples=X.shape[0],
                num_features=n_numeric + n_categorical + n_binary,
                num_bytes_per_cell=4,
                hyperparameters=hyperparameters,
                problem_type=self.problem_type,
                num_classes=self.num_classes,
                **kwargs,
            )

        # TODO: Replace with memory estimation logic based on no. of features instead of dataframe generation
        shape = X.shape[0]
        df_lst = []
        if n_numeric > 0:
            X_estimate = np.random.random(size=[shape, n_numeric]).astype(np.float32)
            X_estimate_numeric = pd.DataFrame(X_estimate)
            df_lst.append(X_estimate_numeric)
        if n_categorical > 0:
            cardinality = int(X.select_dtypes(exclude=[np.number]).nunique().mean())
            X_estimate = np.random.randint(0, cardinality, [shape, n_categorical]).astype("str")
            X_estimate_cat = pd.DataFrame(X_estimate)
            df_lst.append(X_estimate_cat)
        if n_binary > 0:
            X_estimate = np.random.randint(0, 2, [shape, n_binary]).astype(np.int8)
            X_estimate_binary = pd.DataFrame(X_estimate)
            df_lst.append(X_estimate_binary)
        X = pd.concat(df_lst, ignore_index=True, axis=1)

        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

    def _init_preprocessor(
        self,
        preprocessor_cls: Type[AbstractFeatureGenerator] | str,
        init_params: dict | None,
    ) -> AbstractFeatureGenerator:
        if isinstance(preprocessor_cls, str):
            preprocessor_cls = _feature_generator_class_map[preprocessor_cls]
        if init_params is None:
            init_params = {}
        _init_params = dict(
            verbosity=0,
            random_state=self.random_seed,  # FIXME: Not a generic param
            target_type=self.problem_type,  # FIXME: Not a generic param
        )
        _init_params.update(**init_params)
        return preprocessor_cls(
            **_init_params,
        )

    def _recursive_init_preprocessors(self, prep_param: tuple | list[list | tuple]):
        if isinstance(prep_param, list):
            if len(prep_param) == 0:
                param_type = "list"
            elif len(prep_param) == 2:
                if isinstance(prep_param[0], (str, AbstractFeatureGenerator)):
                    param_type = "generator"
                else:
                    param_type = "list"
            else:
                param_type = "list"
        elif isinstance(prep_param, tuple):
            param_type = "generator"
        else:
            raise ValueError(f"Invalid value for prep_param: {prep_param}")

        if param_type == "list":
            out = []
            for i, p in enumerate(prep_param):
                out.append(self._recursive_init_preprocessors(p))
            return out
        elif param_type == "generator":
            assert len(prep_param) == 2
            preprocessor_cls = prep_param[0]
            init_params = prep_param[1]
            return self._init_preprocessor(preprocessor_cls=preprocessor_cls, init_params=init_params)
        else:
            raise ValueError(f"Invalid value for prep_param: {prep_param}")

    def get_preprocessors(self) -> list[AbstractFeatureGenerator]:
        ag_params = self._get_ag_params()
        prep_params = ag_params.get("prep_params", None)
        passthrough_types = ag_params.get("prep_params.passthrough_types", None)
        if prep_params is None:
            return []
        if not prep_params:
            return []

        preprocessors = self._recursive_init_preprocessors(prep_param=prep_params)
        if len(preprocessors) == 0:
            return []
        if len(preprocessors) == 1 and isinstance(preprocessors[0], AbstractFeatureGenerator):
            return preprocessors
        else:
            preprocessors = [
                BulkFeatureGenerator(
                    generators=preprocessors,
                    # TODO: "false_recursive" technically can slow down inference, but need to optimize `True` first
                    #  Refer to `Bioresponse` dataset where setting to `True` -> 200s fit time vs `false_recursive` -> 1s fit time
                    remove_unused_features="false_recursive",
                    post_drop_duplicates=True,
                    passthrough=True,
                    passthrough_types=passthrough_types,
                    verbosity=0,
                )
            ]
            return preprocessors

    def _preprocess(self, X: pd.DataFrame, y=None, is_train: bool = False, **kwargs):
        if is_train:
            self.preprocessors = self.get_preprocessors()
            if self.preprocessors:
                assert y is not None, (
                    f"y must be specified to fit preprocessors... Likely the inheriting class isn't passing `y` in its `preprocess` call."
                )
                # FIXME: add `post_drop_useless`, example: anneal has many useless features
                feature_metadata_in = self._feature_metadata
                for prep in self.preprocessors:
                    X = prep.fit_transform(X, y, feature_metadata_in=feature_metadata_in)
                    # FIXME: Nick: This is incorrect because it strips away special dtypes. Need to do this properly by fixing in the preprocessors
                    feature_metadata_in = prep.feature_metadata
                self._feature_metadata = feature_metadata_in
                self._features_internal = self._feature_metadata.get_features()
        else:
            for prep in self.preprocessors:
                X = prep.transform(X)

        return super()._preprocess(X, y=y, is_train=is_train, **kwargs)
