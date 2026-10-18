from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from typing import Any, Type

import numpy as np
import pandas as pd
import pandas.testing as pdt

from autogluon.common.utils.path_converter import PathConverter
from autogluon.common.utils.utils import get_default_base_path
from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION
from autogluon.core.metrics import METRICS
from autogluon.core.models import AbstractModel, AuxiliaryParams, BaggedEnsembleModel
from autogluon.core.stacked_overfitting.utils import check_stacked_overfitting_from_leaderboard
from autogluon.core.testing.global_context_snapshot import GlobalContextSnapshot
from autogluon.core.utils import download, generate_train_test_split_combined, infer_problem_type, unzip
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.testing.generate_datasets import (
    generate_toy_binary_10_dataset,
    generate_toy_binary_dataset,
    generate_toy_multiclass_10_dataset,
    generate_toy_multiclass_30_dataset,
    generate_toy_multiclass_dataset,
    generate_toy_quantile_10_dataset,
    generate_toy_quantile_dataset,
    generate_toy_quantile_single_level_dataset,
    generate_toy_regression_10_dataset,
    generate_toy_regression_dataset,
)


class DatasetLoaderHelper:
    dataset_info_dict = dict(
        # Binary dataset
        adult={
            "url": "https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip",
            "name": "AdultIncomeBinaryClassification",
            "problem_type": BINARY,
            "label": "class",
        },
        # Multiclass big dataset with 7 classes, all features are numeric. Runs SLOW.
        covertype={
            "url": "https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification.zip",
            "name": "CoverTypeMulticlassClassification",
            "problem_type": MULTICLASS,
            "label": "Cover_Type",
        },
        # Subset of covertype dataset with 3k train/test rows. Ratio of labels is preserved.
        covertype_small={
            "url": "https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassificationSmall.zip",
            "name": "CoverTypeMulticlassClassificationSmall",
            "problem_type": MULTICLASS,
            "label": "Cover_Type",
        },
        # Regression with mixed feature-types, skewed Y-values.
        ames={
            "url": "https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression.zip",
            "name": "AmesHousingPriceRegression",
            "problem_type": REGRESSION,
            "label": "SalePrice",
        },
        # Regression with multiple text field and categorical
        sts={
            "url": "https://autogluon-text.s3.amazonaws.com/glue_sts.zip",
            "name": "glue_sts",
            "problem_type": REGRESSION,
            "label": "score",
        },
    )

    toy_map = dict(
        toy_binary=generate_toy_binary_dataset,
        toy_multiclass=generate_toy_multiclass_dataset,
        toy_regression=generate_toy_regression_dataset,
        toy_quantile=generate_toy_quantile_dataset,
        toy_quantile_single_level=generate_toy_quantile_single_level_dataset,
        toy_binary_10=generate_toy_binary_10_dataset,
        toy_multiclass_10=generate_toy_multiclass_10_dataset,
        toy_regression_10=generate_toy_regression_10_dataset,
        toy_quantile_10=generate_toy_quantile_10_dataset,
        toy_multiclass_30=generate_toy_multiclass_30_dataset,
    )

    @staticmethod
    def load_dataset(name: str, directory_prefix: str = "./datasets/") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        if name in DatasetLoaderHelper.toy_map:
            return DatasetLoaderHelper.toy_map[name]()
        dataset_info = copy.deepcopy(DatasetLoaderHelper.dataset_info_dict[name])
        train_file = dataset_info.pop("train_file", "train_data.csv")
        test_file = dataset_info.pop("test_file", "test_data.csv")
        name_inner = dataset_info.pop("name")
        url = dataset_info.pop("url", None)
        train_data, test_data = DatasetLoaderHelper.load_data(
            directory_prefix=directory_prefix,
            train_file=train_file,
            test_file=test_file,
            name=name_inner,
            url=url,
        )

        return train_data, test_data, dataset_info

    # TODO: Refactor this eventually, this is old code from 2019 that can be improved (use consistent local path for datasets, don't assume zip files, etc.)
    @staticmethod
    def load_data(
        directory_prefix: str,
        train_file: str,
        test_file: str,
        name: str,
        url: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Will check if files exist locally for:
            directory_prefix/name/train_file
            directory_prefix/name/test_file
        If either don't exist, then download the files from the `url` location.
        Then, load both train and test data and return them.

        Parameters
        ----------
        directory_prefix
        train_file
        test_file
        name
        url

        Returns
        -------
        train_data: pd.DataFrame
        test_data: pd.DataFrame
        """
        if not os.path.exists(directory_prefix):
            os.mkdir(directory_prefix)
        directory = directory_prefix + name + "/"
        train_file_path = directory + train_file
        test_file_path = directory + test_file
        if (not os.path.exists(train_file_path)) or (not os.path.exists(test_file_path)):
            # fetch files from s3:
            print("%s data not found locally, so fetching from %s" % (name, url))
            zip_name = download(url, directory_prefix)
            unzip(zip_name, directory_prefix)
            os.remove(zip_name)

        train_data = TabularDataset(train_file_path)
        test_data = TabularDataset(test_file_path)
        return train_data, test_data


class FitHelper:
    """
    Helper functions to test and verify predictors and models when fit through TabularPredictor's API.
    """

    @staticmethod
    def fit_and_validate_dataset(
        dataset_name: str,
        fit_args: dict[str, Any],
        init_args: dict[str, Any] | None = None,
        sample_size: int | None = 1000,  # FIXME: default to None
        refit_full: bool = True,
        delete_directory: bool = True,
        extra_metrics: list[str] | None = None,
        extra_info: bool = False,
        predictor_info: bool = False,
        expected_model_count: int | None = 2,
        fit_weighted_ensemble: bool = True,
        min_cls_count_train: int = 1,
        path_as_absolute: bool = False,
        compile: bool = False,
        compiler_configs: dict | None = None,
        allowed_dataset_features: list[str] | None = None,
        expected_stacked_overfitting_at_test: bool | None = None,
        expected_stacked_overfitting_at_val: bool | None = None,
        scikit_api: bool = False,
        use_test_data: bool = False,
        use_test_for_val: bool = False,
        raise_on_model_failure: bool | None = None,
        deepcopy_fit_args: bool = True,
        verify_model_seed: bool = False,
        verify_load_wo_cuda: bool = False,
        verify_single_prediction_equivalent_to_multi: bool = True,
    ) -> TabularPredictor:
        if compiler_configs is None:
            compiler_configs = {}
        directory_prefix = "./datasets/"
        train_data, test_data, dataset_info = DatasetLoaderHelper.load_dataset(
            name=dataset_name, directory_prefix=directory_prefix
        )
        if sample_size is not None and sample_size < len(test_data):
            test_data = test_data.sample(n=sample_size, random_state=0)
        label = dataset_info["label"]
        problem_type = dataset_info["problem_type"]
        _init_args = dict(
            label=label,
            problem_type=problem_type,
        )
        if "init_kwargs" in dataset_info:
            _init_args.update(dataset_info["init_kwargs"])
        if allowed_dataset_features is not None:
            train_data = train_data[allowed_dataset_features + [label]]
            test_data = test_data[allowed_dataset_features + [label]]

        if init_args is None:
            init_args = _init_args
        else:
            init_args = copy.deepcopy(init_args)
            _init_args.update(init_args)
            init_args = _init_args
        if "path" not in init_args:
            init_args["path"] = os.path.join(get_default_base_path(), dataset_name, f"AutogluonOutput_{uuid.uuid4()}")
        if path_as_absolute:
            init_args["path"] = PathConverter.to_absolute(path=init_args["path"])
            assert PathConverter._is_absolute(path=init_args["path"])
        save_path = init_args["path"]

        if deepcopy_fit_args:
            fit_args = copy.deepcopy(fit_args)
        if use_test_data:
            fit_args["test_data"] = test_data
            if use_test_for_val:
                fit_args["tuning_data"] = test_data
        if raise_on_model_failure is not None and "raise_on_model_failure" not in fit_args:
            fit_args["raise_on_model_failure"] = raise_on_model_failure
        if "fit_weighted_ensemble" not in fit_args:
            if not fit_weighted_ensemble and expected_model_count is not None:
                expected_model_count -= 1
            fit_args["fit_weighted_ensemble"] = fit_weighted_ensemble

        try:
            ctx_before = GlobalContextSnapshot.capture()

            predictor: TabularPredictor = FitHelper.fit_dataset(
                train_data=train_data,
                init_args=init_args,
                fit_args=fit_args,
                sample_size=sample_size,
                scikit_api=scikit_api,
                min_cls_count_train=min_cls_count_train,
            )

            ctx_after_fit = GlobalContextSnapshot.capture()
            ctx_before.assert_unchanged(ctx_after_fit)

            if compile:
                predictor.compile(models="all", compiler_configs=compiler_configs)
                predictor.persist(models="all")

            model_names = predictor.model_names()
            model_name = model_names[0]
            if expected_model_count is not None:
                assert len(model_names) == expected_model_count

            pred = predictor.predict(test_data)
            _verify_pred_well_formed(predictor=predictor, pred=pred, index=test_data.index)

            ctx_after_predict = GlobalContextSnapshot.capture()
            ctx_after_fit.assert_unchanged(ctx_after_predict)

            evaluate_result = predictor.evaluate(test_data)

            test_data_transform = predictor.transform_features(data=test_data, model=model_name)
            test_data_transform_before = test_data_transform.copy(deep=True)
            model = predictor._trainer.load_model(model_name=model_name)
            model.predict(X=test_data_transform)
            pdt.assert_frame_equal(test_data_transform, test_data_transform_before, check_dtype=True)

            if predictor.can_predict_proba:
                pred_proba = predictor.predict_proba(test_data)
                evaluate_predictions_result = predictor.evaluate_predictions(
                    y_true=test_data[label], y_pred=pred_proba
                )

                _verify_pred_proba_well_formed(
                    predictor=predictor, pred_proba=pred_proba, index=test_data.index, pred=pred
                )
                _verify_evaluate_equivalence(
                    predictor=predictor,
                    evaluate_result=evaluate_result,
                    evaluate_predictions_result=evaluate_predictions_result,
                )

                if predictor.problem_type == BINARY:
                    # The collapsed binary form must hold the same values as the positive-class column.
                    # `check_dtype=False`: the binary -> multiclass conversion widens float32 to float64.
                    pred_proba_pos = predictor.predict_proba(test_data, as_multiclass=False)
                    pdt.assert_series_equal(
                        pred_proba_pos,
                        pred_proba[predictor.positive_class],
                        check_names=False,
                        check_dtype=False,
                        obj="predict_proba(as_multiclass=False) vs positive class column",
                    )

                pred_proba_repeat = predictor.predict_proba(test_data)
                are_close = np.allclose(pred_proba, pred_proba_repeat, atol=1e-5)
                if not are_close:
                    raise AssertionError(
                        "Predictions differ when predicting on the same data multiple times\n"
                        f"First Predict:\n{pred_proba}\n"
                        f"Second Predict:\n{pred_proba_repeat}\n"
                    )

                pred_proba_1 = predictor.predict_proba(
                    test_data.head(1)
                )  # Verify model can predict on a single sample
                if verify_single_prediction_equivalent_to_multi:
                    pred_proba_1_from_multi = pred_proba.head(1)
                    are_close = np.allclose(pred_proba_1, pred_proba_1_from_multi, atol=1e-5)
                    if not are_close:
                        raise AssertionError(
                            "Predictions differ when predicting a single sample vs predicting multiple samples\n"
                            f"Single Sample:\n{pred_proba_1}\n"
                            f"Multi Sample:\n{pred_proba_1_from_multi}\n"
                        )
            else:
                try:
                    predictor.predict_proba(test_data)
                except AssertionError:
                    pass  # expected
                else:
                    raise AssertionError("Expected `predict_proba` to raise AssertionError, but it didn't!")

                _verify_evaluate_equivalence(
                    predictor=predictor,
                    evaluate_result=evaluate_result,
                    evaluate_predictions_result=predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred),
                )

            if refit_full:
                refit_model_names = predictor.refit_full()
                if expected_model_count is not None:
                    assert len(refit_model_names) == expected_model_count
                refit_model_name = refit_model_names[model_name]
                assert "_FULL" in refit_model_name
                predictor.predict(test_data, model=refit_model_name)
                if predictor.can_predict_proba:
                    predictor.predict_proba(test_data, model=refit_model_name)

                # verify that val_in_fit is False if the model supports refit_full
                model = predictor._trainer.load_model(refit_model_name)
                if isinstance(model, BaggedEnsembleModel):
                    model = model.load_child(model.models[0])
                model_info = model.get_info()
                can_refit_full = model._get_tags()["can_refit_full"]
                if can_refit_full:
                    assert not model_info["val_in_fit"], (
                        f"val data must not be present in refit model if `can_refit_full=True`. Maybe an exception occurred?"
                    )
                else:
                    assert model_info["val_in_fit"], (
                        f"val data must be present in refit model if `can_refit_full=False`"
                    )
            if verify_model_seed:
                names = predictor.model_names()
                for name in names:
                    model = predictor._trainer.load_model(name)
                    _verify_model_seed(model=model)

            if predictor_info:
                predictor.info()
            lb_kwargs = {}
            if extra_info:
                lb_kwargs["extra_info"] = True
            lb = predictor.leaderboard(test_data, extra_metrics=extra_metrics, **lb_kwargs)
            _verify_leaderboard_well_formed(predictor=predictor, leaderboard=lb)
            stacked_overfitting_assert(
                lb, predictor, expected_stacked_overfitting_at_val, expected_stacked_overfitting_at_test
            )

            predictor_load = predictor.load(path=predictor.path)
            # Compare against a fresh predict from the in-memory predictor rather than the `pred`
            # computed earlier: `refit_full` above can change which model is used by default.
            pred_before_load = predictor.predict(test_data)
            pred_load = predictor_load.predict(test_data)
            pdt.assert_frame_equal(
                pd.DataFrame(pred_before_load),
                pd.DataFrame(pred_load),
                check_names=False,
                obj="predictions before vs after save+load",
            )

            # TODO: This is expensive, only do this sparingly.
            if verify_load_wo_cuda:
                import torch

                if torch.cuda.is_available():
                    # Checks if the model is able to predict w/o CUDA.
                    # This verifies that a model artifact works on a CPU machine.
                    predictor_path = predictor.path

                    code = textwrap.dedent(f"""
                            import os
                            os.environ["CUDA_VISIBLE_DEVICES"] = ""
                            from autogluon.tabular import TabularPredictor

                            import torch
                            assert torch.cuda.is_available() is False
                            predictor = TabularPredictor.load(r"{predictor_path}")
                            X, y = predictor.load_data_internal()
                            predictor.persist("all")
                            predictor.predict_multi(X, transform_features=False)
                        """)
                    subprocess.run([sys.executable, "-c", code], check=True)

            assert os.path.realpath(save_path) == os.path.realpath(predictor.path)
            return predictor
        finally:
            if delete_directory:
                # Always remove the AutoGluon output directory, including when an assertion above
                # fails -- otherwise failing tests are exactly the ones that leak artifacts.
                shutil.rmtree(save_path, ignore_errors=True)

    @staticmethod
    def load_dataset(name: str, directory_prefix: str = "./datasets/") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        return DatasetLoaderHelper.load_dataset(name=name, directory_prefix=directory_prefix)

    @staticmethod
    def fit_dataset(
        train_data: pd.DataFrame,
        init_args: dict[str, Any],
        fit_args: dict[str, Any],
        sample_size: int | None = None,
        min_cls_count_train: int = 1,
        scikit_api: bool = False,
    ) -> TabularPredictor:
        if "problem_type" in init_args:
            problem_type = init_args["problem_type"]
        else:
            problem_type = infer_problem_type(train_data[init_args["label"]])

        if sample_size is not None and sample_size < len(train_data):
            train_data, _ = generate_train_test_split_combined(
                data=train_data,
                label=init_args["label"],
                problem_type=problem_type,
                test_size=len(train_data) - sample_size,
                min_cls_count_train=min_cls_count_train,
            )

        if scikit_api:
            from autogluon.tabular.experimental import TabularClassifier, TabularRegressor

            X = train_data.drop(columns=[init_args["label"]])
            y = train_data[init_args["label"]]
            if problem_type in [REGRESSION]:
                regressor = TabularRegressor(init_args=init_args, fit_args=fit_args)
                regressor.fit(X, y)
                return regressor.predictor_
            else:
                classifier = TabularClassifier(init_args=init_args, fit_args=fit_args)
                classifier.fit(X, y)
                return classifier.predictor_
        else:
            return TabularPredictor(**init_args).fit(train_data, **fit_args)

    @staticmethod
    def _verify_auxiliary_param_keys(model_cls: type[AbstractModel]) -> None:
        """Fail if the model declares `_default_auxiliary_params_extra` keys that nothing consumes.

        Known keys are the `AuxiliaryParams` schema fields plus the registered ag-params
        (`_ag_params_common()` and the model's `_ag_params()`). An unknown key is either a typo
        or a model-private param the wrapper reads without registering it — register such keys
        in the model's `_ag_params()`.
        """
        known_keys = AuxiliaryParams.known_keys()
        known_keys |= model_cls._ag_params_common()
        # `_ag_params` implementations are constant per class, so calling on an
        # uninitialized instance is safe.
        known_keys |= model_cls._ag_params(object.__new__(model_cls))
        declared_keys = set()
        for klass in model_cls.__mro__:
            declared_keys |= set(klass.__dict__.get("_default_auxiliary_params_extra") or {})
        unknown_keys = declared_keys - known_keys
        if unknown_keys:
            raise AssertionError(
                f"Model {model_cls.__name__} declares unknown auxiliary param key(s) in "
                f"`_default_auxiliary_params_extra`: {sorted(unknown_keys)}"
                f"\nEither fix the typo (known keys: {sorted(known_keys)}),"
                f"\nor, if the model's wrapper code consumes the key, register it in the model's `_ag_params()`."
            )

    @staticmethod
    def verify_model(
        model_cls: Type[AbstractModel],
        model_hyperparameters: dict[str, Any],
        bag: bool | str = "first",
        refit_full: bool | str = "first",
        extra_metrics: bool = False,
        require_known_problem_types: bool = True,
        raise_on_model_failure: bool = True,
        problem_types: list[str] | None = None,
        verify_model_seed: bool = True,
        verify_single_prediction_equivalent_to_multi: bool = True,
        use_larger_toy_datasets: bool = False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        model_cls
        model_hyperparameters
        bag
        refit_full
        extra_metrics
        require_known_problem_types
        raise_on_model_failure
        problem_types: list[str], optional
            If specified, checks the given problem_types.
            If None, checks `model_cls.supported_problem_types()`
        verify_model_seed: bool = True
        verify_single_prediction_equivalent_to_multi: bool = True
        **kwargs

        Returns
        -------

        """
        if verify_model_seed and model_cls.seed_name is not None:
            # verify that the seed logic works
            model_hyperparameters = model_hyperparameters.copy()
            model_hyperparameters[model_cls.seed_name] = 42

        fit_args = dict(
            hyperparameters={model_cls: model_hyperparameters},
        )
        if model_cls.supported_problem_types.__func__ is not AbstractModel.supported_problem_types.__func__:
            raise AssertionError(
                f"Model {model_cls.__name__} overrides `supported_problem_types()`. "
                f"Declare the `_supported_problem_types` class attribute instead of overriding the classmethod."
                f"""\nExample code:
    _supported_problem_types = ["binary", "multiclass", "regression", "quantile"]
        """
            )
        FitHelper._verify_auxiliary_param_keys(model_cls=model_cls)
        supported_problem_types = model_cls.supported_problem_types()
        if supported_problem_types is None:
            raise AssertionError(
                f"Model must specify `cls._supported_problem_types`"
                f"""\nExample code:
    _supported_problem_types = ["binary", "multiclass", "regression", "quantile"]
        """
            )
        assert isinstance(supported_problem_types, list)
        assert len(supported_problem_types) > 0

        known_problem_types = [
            "binary",
            "multiclass",
            "regression",
            "quantile",
            "softclass",
        ]

        if require_known_problem_types:
            for problem_type in supported_problem_types:
                if problem_type not in known_problem_types:
                    raise AssertionError(
                        f"Model {model_cls.__name__} supports an unknown problem_type: {problem_type}"
                        f"\nKnown problem types: {known_problem_types}"
                        f"\nEither remove the unknown problem_type from `model_cls.supported_problem_types` or set `require_known_problem_types=False`"
                    )

        if use_larger_toy_datasets:
            problem_type_dataset_map = {
                "binary": ["toy_binary_10"],
                "multiclass": ["toy_multiclass_10"],
                "regression": ["toy_regression_10"],
                "quantile": ["toy_quantile_10"],
            }
        else:
            problem_type_dataset_map = {
                "binary": ["toy_binary"],
                "multiclass": ["toy_multiclass"],
                "regression": ["toy_regression"],
                "quantile": ["toy_quantile", "toy_quantile_single_level"],
            }

        problem_types_refit_full = []
        if refit_full:
            if isinstance(refit_full, bool):
                problem_types_refit_full = supported_problem_types
            elif refit_full == "first":
                problem_types_refit_full = supported_problem_types[:1]

        if problem_types is None:
            problem_types_to_check = supported_problem_types
        else:
            problem_types_to_check = problem_types

        for problem_type in problem_types_to_check:
            if problem_type not in problem_type_dataset_map:
                print(f"WARNING: Skipping check on problem_type='{problem_type}': No dataset available")
                continue
            _extra_metrics = None
            if extra_metrics:
                _extra_metrics = METRICS.get(problem_type, None)
            refit_full = problem_type in problem_types_refit_full
            for dataset_name in problem_type_dataset_map[problem_type]:
                FitHelper.fit_and_validate_dataset(
                    dataset_name=dataset_name,
                    fit_args=fit_args,
                    fit_weighted_ensemble=False,
                    refit_full=refit_full,
                    extra_metrics=_extra_metrics,
                    raise_on_model_failure=raise_on_model_failure,
                    verify_model_seed=verify_model_seed,
                    verify_single_prediction_equivalent_to_multi=verify_single_prediction_equivalent_to_multi,
                    **kwargs,
                )

        if bag:
            model_params_bag = copy.deepcopy(model_hyperparameters)
            model_params_bag["ag.ens.fold_fitting_strategy"] = "sequential_local"
            fit_args_bag = dict(
                hyperparameters={model_cls: model_params_bag},
                num_bag_folds=2,
                num_bag_sets=1,
            )
            if isinstance(bag, bool):
                problem_types_bag = problem_types_to_check
            elif bag == "first":
                problem_types_bag = problem_types_to_check[:1]
            else:
                raise ValueError(f"Unknown 'bag' value: {bag}")

            for problem_type in problem_types_bag:
                _extra_metrics = None
                if extra_metrics:
                    _extra_metrics = METRICS.get(problem_type, None)
                refit_full = problem_type in problem_types_refit_full
                for dataset_name in problem_type_dataset_map[problem_type]:
                    FitHelper.fit_and_validate_dataset(
                        dataset_name=dataset_name,
                        fit_args=fit_args_bag,
                        fit_weighted_ensemble=False,
                        refit_full=refit_full,
                        extra_metrics=_extra_metrics,
                        raise_on_model_failure=raise_on_model_failure,
                        verify_model_seed=verify_model_seed,
                        verify_single_prediction_equivalent_to_multi=verify_single_prediction_equivalent_to_multi,
                        **kwargs,
                    )


def stacked_overfitting_assert(
    lb: pd.DataFrame,
    predictor: TabularPredictor,
    expected_stacked_overfitting_at_val: bool | None,
    expected_stacked_overfitting_at_test: bool | None,
):
    if expected_stacked_overfitting_at_val is not None:
        assert predictor._stacked_overfitting_occurred == expected_stacked_overfitting_at_val, (
            "Expected stacked overfitting at val mismatch!"
        )

    if expected_stacked_overfitting_at_test is not None:
        stacked_overfitting = check_stacked_overfitting_from_leaderboard(lb)
        assert stacked_overfitting == expected_stacked_overfitting_at_test, (
            "Expected stacked overfitting at test mismatch!"
        )


def _verify_pred_well_formed(predictor: TabularPredictor, pred, index) -> None:
    """Cheap structural checks on `predict` output, reusing the already-computed predictions."""
    assert pred.index.equals(index), "predict must preserve the index of the input data"
    assert not np.asarray(pd.isna(pred)).any(), f"predict returned NaN values:\n{pred}"

    if predictor.problem_type == QUANTILE:
        assert list(pred.columns) == list(predictor.quantile_levels), (
            f"predict columns must equal predictor.quantile_levels\n"
            f"columns: {list(pred.columns)}\nquantile_levels: {list(predictor.quantile_levels)}"
        )


def _verify_pred_proba_well_formed(predictor: TabularPredictor, pred_proba, index, pred=None) -> None:
    """Cheap structural checks on `predict_proba` output, reusing the already-computed predictions.

    Catches class-count/ordering mistakes (a model inferring `num_classes` from the training split
    rather than from `predictor.num_classes`), un-normalized probabilities, and index loss.
    """
    assert pred_proba.index.equals(index), "predict_proba must preserve the index of the input data"
    assert not np.isnan(np.asarray(pred_proba, dtype=float)).any(), f"predict_proba returned NaN values:\n{pred_proba}"

    class_labels = predictor.class_labels
    assert list(pred_proba.columns) == list(class_labels), (
        f"predict_proba columns must equal predictor.class_labels\n"
        f"columns: {list(pred_proba.columns)}\nclass_labels: {list(class_labels)}"
    )

    proba_values = np.asarray(pred_proba, dtype=float)
    # Small tolerance: compiled backends (e.g. ONNX) can emit values a few ULP outside [0, 1].
    assert (proba_values >= -1e-6).all() and (proba_values <= 1 + 1e-6).all(), (
        f"predict_proba values must lie in [0, 1]: min={proba_values.min()}, max={proba_values.max()}\n{pred_proba}"
    )
    assert np.allclose(proba_values.sum(axis=1), 1.0, atol=1e-5), f"predict_proba rows must sum to 1:\n{pred_proba}"

    if pred is not None:
        # `predict` must be derivable from `predict_proba` -- catches label/column misalignment.
        pred_from_proba = predictor.predict_from_proba(y_pred_proba=pred_proba)
        pdt.assert_series_equal(
            pred,
            pred_from_proba,
            check_names=False,
            obj="predict vs predict_from_proba(predict_proba)",
        )


def _verify_evaluate_equivalence(
    predictor: TabularPredictor, evaluate_result: dict, evaluate_predictions_result: dict
) -> None:
    """`predictor.evaluate(data)` is documented as a shortcut for `evaluate_predictions(y, predict(_proba)(data))`."""
    if predictor.sample_weight is not None:
        return  # `evaluate` forwards sample weights that the direct `evaluate_predictions` call does not
    assert set(evaluate_result.keys()) == set(evaluate_predictions_result.keys()), (
        f"evaluate and evaluate_predictions returned different metrics\n"
        f"evaluate: {sorted(evaluate_result.keys())}\n"
        f"evaluate_predictions: {sorted(evaluate_predictions_result.keys())}"
    )
    for metric, value in evaluate_result.items():
        other = evaluate_predictions_result[metric]
        if isinstance(value, (int, float, np.integer, np.floating)) and isinstance(
            other, (int, float, np.integer, np.floating)
        ):
            assert np.isclose(value, other, atol=1e-8, equal_nan=True), (
                f"evaluate and evaluate_predictions disagree on '{metric}': {value} vs {other}"
            )


def _verify_leaderboard_well_formed(predictor: TabularPredictor, leaderboard: pd.DataFrame) -> None:
    """The leaderboard must describe exactly the fitted models, with usable timings and val scores."""
    model_names = predictor.model_names()
    assert len(model_names) == len(set(model_names)), f"predictor.model_names() has duplicates: {model_names}"

    graph_nodes = set(predictor._trainer.model_graph.nodes)
    missing_from_graph = [m for m in model_names if m not in graph_nodes]
    assert not missing_from_graph, f"models absent from trainer.model_graph: {missing_from_graph}"

    lb_models = list(leaderboard["model"])
    assert len(lb_models) == len(set(lb_models)), f"leaderboard has duplicate rows: {lb_models}"
    assert set(lb_models) == set(model_names), (
        f"leaderboard must have exactly one row per fitted model\n"
        f"leaderboard: {sorted(lb_models)}\nmodel_names: {sorted(model_names)}"
    )

    fit_time = np.asarray(leaderboard["fit_time"], dtype=float)
    assert np.isfinite(fit_time).all() and (fit_time >= 0).all(), (
        f"leaderboard fit_time must be finite and non-negative:\n{leaderboard[['model', 'fit_time']]}"
    )

    # `_FULL` refit models are fit without validation data, so they legitimately have no val score.
    val_rows = leaderboard[~leaderboard["model"].str.endswith("_FULL")]
    score_val = np.asarray(val_rows["score_val"], dtype=float)
    assert np.isfinite(score_val).all(), (
        f"leaderboard score_val must be finite for non-refit models:\n{val_rows[['model', 'score_val']]}"
    )
    pred_time_val = np.asarray(val_rows["pred_time_val"], dtype=float)
    assert np.isfinite(pred_time_val).all() and (pred_time_val >= 0).all(), (
        f"leaderboard pred_time_val must be finite and non-negative:\n{val_rows[['model', 'pred_time_val']]}"
    )


def _verify_model_seed(model: AbstractModel):
    assert model.random_seed is None or isinstance(model.random_seed, int)
    if model.seed_name is not None:
        if model.seed_name in model._user_params:
            assert model.random_seed == model._user_params[model.seed_name]
        assert model.seed_name in model.params
        assert model.random_seed == model.params[model.seed_name]
    if isinstance(model, BaggedEnsembleModel):
        for child in model.models:
            child = model.load_child(child)
            _verify_model_seed(child)
