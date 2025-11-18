from __future__ import annotations

import copy
import os
import pandas as pd
import shutil
import sys
import subprocess
import textwrap
import uuid
from typing import Any, Type

from autogluon.common.utils.path_converter import PathConverter
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.metrics import METRICS
from autogluon.core.models import AbstractModel, BaggedEnsembleModel
from autogluon.core.stacked_overfitting.utils import check_stacked_overfitting_from_leaderboard
from autogluon.core.utils import download, generate_train_test_split_combined, infer_problem_type, unzip

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.testing.generate_datasets import (
    generate_toy_binary_dataset,
    generate_toy_binary_10_dataset,
    generate_toy_multiclass_dataset,
    generate_toy_regression_dataset,
    generate_toy_quantile_dataset,
    generate_toy_quantile_single_level_dataset,
    generate_toy_multiclass_10_dataset,
    generate_toy_regression_10_dataset,
    generate_toy_quantile_10_dataset,
    generate_toy_multiclass_30_dataset,
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
    ) -> TabularPredictor:
        if compiler_configs is None:
            compiler_configs = {}
        directory_prefix = "./datasets/"
        train_data, test_data, dataset_info = DatasetLoaderHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
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
            init_args["path"] = os.path.join(directory_prefix, dataset_name, f"AutogluonOutput_{uuid.uuid4()}")
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

        predictor: TabularPredictor = FitHelper.fit_dataset(
            train_data=train_data,
            init_args=init_args,
            fit_args=fit_args,
            sample_size=sample_size,
            scikit_api=scikit_api,
            min_cls_count_train=min_cls_count_train,
        )
        if compile:
            predictor.compile(models="all", compiler_configs=compiler_configs)
            predictor.persist(models="all")
        if sample_size is not None and sample_size < len(test_data):
            test_data = test_data.sample(n=sample_size, random_state=0)
        predictor.predict(test_data)
        predictor.evaluate(test_data)

        if predictor.can_predict_proba:
            pred_proba = predictor.predict_proba(test_data)
            predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba)
        else:
            try:
                predictor.predict_proba(test_data)
            except AssertionError:
                pass  # expected
            else:
                raise AssertionError("Expected `predict_proba` to raise AssertionError, but it didn't!")

        model_names = predictor.model_names()
        model_name = model_names[0]
        if expected_model_count is not None:
            assert len(model_names) == expected_model_count
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
                assert not model_info["val_in_fit"], f"val data must not be present in refit model if `can_refit_full=True`. Maybe an exception occurred?"
            else:
                assert model_info["val_in_fit"], f"val data must be present in refit model if `can_refit_full=False`"
        if verify_model_seed:
            model_names = predictor.model_names()
            for model_name in model_names:
                model = predictor._trainer.load_model(model_name)
                _verify_model_seed(model=model)

        if predictor_info:
            predictor.info()
        lb_kwargs = {}
        if extra_info:
            lb_kwargs["extra_info"] = True
        lb = predictor.leaderboard(test_data, extra_metrics=extra_metrics, **lb_kwargs)
        stacked_overfitting_assert(lb, predictor, expected_stacked_overfitting_at_val, expected_stacked_overfitting_at_test)

        predictor_load = predictor.load(path=predictor.path)
        predictor_load.predict(test_data)

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
        if delete_directory:
            shutil.rmtree(save_path, ignore_errors=True)  # Delete AutoGluon output directory to ensure runs' information has been removed.
        return predictor

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
        supported_problem_types = model_cls.supported_problem_types()
        if supported_problem_types is None:
            raise AssertionError(
                f"Model must specify `cls.supported_problem_types`"
                f"""\nExample code:
    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression", "quantile"]
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
                        **kwargs,
                    )


def stacked_overfitting_assert(
    lb: pd.DataFrame,
    predictor: TabularPredictor,
    expected_stacked_overfitting_at_val: bool | None,
    expected_stacked_overfitting_at_test: bool | None,
):
    if expected_stacked_overfitting_at_val is not None:
        assert predictor._stacked_overfitting_occurred == expected_stacked_overfitting_at_val, "Expected stacked overfitting at val mismatch!"

    if expected_stacked_overfitting_at_test is not None:
        stacked_overfitting = check_stacked_overfitting_from_leaderboard(lb)
        assert stacked_overfitting == expected_stacked_overfitting_at_test, "Expected stacked overfitting at test mismatch!"


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
