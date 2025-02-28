from __future__ import annotations

import copy
import os
import shutil
import uuid
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from autogluon.common.utils.path_converter import PathConverter
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE
from autogluon.core.data.label_cleaner import LabelCleaner
from autogluon.core.metrics import METRICS
from autogluon.core.models import AbstractModel, BaggedEnsembleModel
from autogluon.core.stacked_overfitting.utils import check_stacked_overfitting_from_leaderboard
from autogluon.core.utils import download, generate_train_test_split, generate_train_test_split_combined, infer_problem_type, unzip
from autogluon.features.generators import AbstractFeatureGenerator, AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--runregression", action="store_true", default=False, help="run regression tests")
    parser.addoption("--runpyodide", action="store_true", default=False, help="run Pyodide tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "regression: mark test as regression test")
    config.addinivalue_line("markers", "pyodide: mark test as pyodide test")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_regression = pytest.mark.skip(reason="need --runregression option to run")
    skip_pyodide = pytest.mark.skip(reason="need --runpyodide option to run")
    custom_markers = dict(slow=skip_slow, regression=skip_regression, pyodide=skip_pyodide)
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        custom_markers.pop("slow", None)
    if config.getoption("--runregression"):
        # --runregression given in cli: do not skip slow tests
        custom_markers.pop("regression", None)
    if config.getoption("--runpyodide"):
        # --runpyodide given in cli: do not skip pyodide tests
        custom_markers.pop("pyodide", None)

    for item in items:
        for marker in custom_markers:
            if marker in item.keywords:
                item.add_marker(custom_markers[marker])

    # Normalize the file paths and use a consistent comparison method
    normalized_path = lambda p: os.path.normpath(str(p))
    resource_allocation_path = normalized_path("tests/unittests/resource_allocation")

    # Reordering logic to ensure tests under ./unittests/resource_allocation run last
    # TODO: Fix this once resource_allocation tests are robost enough to run with other tests without ordering issues
    resource_allocation_tests = [item for item in items if resource_allocation_path in normalized_path(item.fspath)]
    other_tests = [item for item in items if resource_allocation_path not in normalized_path(item.fspath)]

    items.clear()
    items.extend(other_tests)
    items.extend(resource_allocation_tests)


def generate_toy_binary_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3],
        label: [0, 0, 1, 1],
    }

    dataset_info = {
        "problem_type": BINARY,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_multiclass_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3, 4, 5],
        label: [0, 0, 1, 1, 2, 2],
    }

    dataset_info = {
        "problem_type": MULTICLASS,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_regression_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3],
        label: [0.1, 0.9, 1.1, 1.9],
    }

    dataset_info = {
        "problem_type": REGRESSION,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_quantile_dataset():
    train_data, test_data, dataset_info = generate_toy_regression_dataset()
    dataset_info["problem_type"] = QUANTILE
    dataset_info["init_kwargs"] = {"quantile_levels": [0.25, 0.5, 0.75]}
    return train_data, test_data, dataset_info


def generate_toy_binary_10_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        label: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    }

    dataset_info = {
        "problem_type": BINARY,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_multiclass_10_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        label: [0, 0, 1, 1, 2, 2, 0, 0, 1, 1],
    }

    dataset_info = {
        "problem_type": MULTICLASS,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_regression_10_dataset():
    label = "label"
    dummy_dataset = {
        "int": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        label: [0.1, 0.9, 1.1, 1.9, 0.2, 0.8, 1.2, 1.8, -0.1, 0.7],
    }

    dataset_info = {
        "problem_type": REGRESSION,
        "label": label,
    }

    train_data = pd.DataFrame(dummy_dataset)
    test_data = train_data
    return train_data, test_data, dataset_info


def generate_toy_quantile_10_dataset():
    train_data, test_data, dataset_info = generate_toy_regression_10_dataset()
    dataset_info["problem_type"] = QUANTILE
    dataset_info["init_kwargs"] = {"quantile_levels": [0.25, 0.5, 0.75]}
    return train_data, test_data, dataset_info


def generate_toy_multiclass_30_dataset():
    label = "label"
    train_data = generate_toy_multiclass_n_dataset(n_samples=30, n_features=2, n_classes=3)
    test_data = train_data

    dataset_info = {
        "problem_type": MULTICLASS,
        "label": label,
    }
    return train_data, test_data, dataset_info


def generate_toy_multiclass_n_dataset(n_samples, n_features, n_classes) -> pd.DataFrame:
    from sklearn.datasets import make_blobs
    X, y = make_blobs(centers=n_classes, n_samples=n_samples, n_features=n_features, cluster_std=0.5, random_state=0)
    data = pd.DataFrame(X)
    data["label"] = y
    return data


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
        toy_binary_10=generate_toy_binary_10_dataset,
        toy_multiclass_10=generate_toy_multiclass_10_dataset,
        toy_regression_10=generate_toy_regression_10_dataset,
        toy_quantile_10=generate_toy_quantile_10_dataset,
        toy_multiclass_30=generate_toy_multiclass_30_dataset,
    )

    @staticmethod
    def load_dataset_toy(name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        return DatasetLoaderHelper.toy_map[name]()

    @staticmethod
    def load_dataset(name: str, directory_prefix: str = "./datasets/") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        if name in DatasetLoaderHelper.toy_map:
            return DatasetLoaderHelper.load_dataset_toy(name=name)

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

    @staticmethod
    def load_data(directory_prefix, train_file, test_file, name, url=None):
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
    @staticmethod
    def fit_and_validate_dataset(
        dataset_name,
        fit_args,
        init_args=None,
        sample_size=1000,
        refit_full=True,
        delete_directory=True,
        extra_metrics=None,
        extra_info=False,
        predictor_info=False,
        expected_model_count: int | None = 2,
        fit_weighted_ensemble: bool = True,
        min_cls_count_train=1,
        path_as_absolute=False,
        compile=False,
        compiler_configs=None,
        allowed_dataset_features=None,
        expected_stacked_overfitting_at_test=None,
        expected_stacked_overfitting_at_val=None,
        scikit_api=False,
        use_test_data=False,
        use_test_for_val=False,
        raise_on_model_failure: bool | None = None,
        deepcopy_fit_args: bool = True,
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
            with pytest.raises(AssertionError):
                predictor.predict_proba(test_data)

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

        if predictor_info:
            predictor.info()
        lb_kwargs = {}
        if extra_info:
            lb_kwargs["extra_info"] = True
        lb = predictor.leaderboard(test_data, extra_metrics=extra_metrics, **lb_kwargs)
        stacked_overfitting_assert(lb, predictor, expected_stacked_overfitting_at_val, expected_stacked_overfitting_at_test)

        predictor_load = predictor.load(path=predictor.path)
        predictor_load.predict(test_data)

        assert os.path.realpath(save_path) == os.path.realpath(predictor.path)
        if delete_directory:
            shutil.rmtree(save_path, ignore_errors=True)  # Delete AutoGluon output directory to ensure runs' information has been removed.
        return predictor

    @staticmethod
    def load_dataset(name: str, directory_prefix: str = "./datasets/") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        return DatasetLoaderHelper.load_dataset(name=name, directory_prefix=directory_prefix)

    @staticmethod
    def fit_dataset(train_data, init_args, fit_args, sample_size=None, min_cls_count_train=1, scikit_api=False) -> TabularPredictor:
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
        model_cls,
        model_hyperparameters,
        bag: bool | str = "first",
        refit_full: bool | str = "first",
        extra_metrics: bool = False,
        require_known_problem_types: bool = True,
        raise_on_model_failure: bool = True,
        **kwargs,
    ):
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
            "binary": "toy_binary",
            "multiclass": "toy_multiclass",
            "regression": "toy_regression",
            "quantile": "toy_quantile",
        }

        problem_types_refit_full = []
        if refit_full:
            if isinstance(refit_full, bool):
                problem_types_refit_full = supported_problem_types
            elif refit_full == "first":
                problem_types_refit_full = supported_problem_types[:1]

        for problem_type in supported_problem_types:
            if problem_type not in problem_type_dataset_map:
                print(f"WARNING: Skipping check on problem_type='{problem_type}': No dataset available")
                continue
            _extra_metrics = None
            if extra_metrics:
                _extra_metrics = METRICS.get(problem_type, None)
            refit_full = problem_type in problem_types_refit_full
            dataset_name = problem_type_dataset_map[problem_type]
            FitHelper.fit_and_validate_dataset(
                dataset_name=dataset_name,
                fit_args=fit_args,
                fit_weighted_ensemble=False,
                refit_full=refit_full,
                extra_metrics=_extra_metrics,
                raise_on_model_failure=raise_on_model_failure,
                **kwargs,
            )

        if bag:
            model_params_bag = copy.deepcopy(model_hyperparameters)
            model_params_bag["ag_args_ensemble"] = {"fold_fitting_strategy": "sequential_local"}
            fit_args_bag = dict(
                hyperparameters={model_cls: model_params_bag},
                num_bag_folds=2,
                num_bag_sets=1,
            )
            if isinstance(bag, bool):
                problem_types_bag = supported_problem_types
            elif bag == "first":
                problem_types_bag = supported_problem_types[:1]
            else:
                raise ValueError(f"Unknown 'bag' value: {bag}")

            for problem_type in problem_types_bag:
                _extra_metrics = None
                if extra_metrics:
                    _extra_metrics = METRICS.get(problem_type, None)
                refit_full = problem_type in problem_types_refit_full
                dataset_name = problem_type_dataset_map[problem_type]
                FitHelper.fit_and_validate_dataset(
                    dataset_name=dataset_name,
                    fit_args=fit_args_bag,
                    fit_weighted_ensemble=False,
                    refit_full=refit_full,
                    extra_metrics=_extra_metrics,
                    raise_on_model_failure=raise_on_model_failure,
                    **kwargs,
                )



# Helper functions for training models outside of predictors
class ModelFitHelper:
    @staticmethod
    def fit_and_validate_dataset(
        dataset_name: str, model: AbstractModel, fit_args: dict, sample_size: int = 1000, check_predict_children: bool = False
    ) -> AbstractModel:
        directory_prefix = "./datasets/"
        train_data, test_data, dataset_info = DatasetLoaderHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
        label = dataset_info["label"]
        model, label_cleaner, feature_generator = ModelFitHelper.fit_dataset(
            train_data=train_data, model=model, label=label, fit_args=fit_args, sample_size=sample_size
        )
        if sample_size is not None and sample_size < len(test_data):
            test_data = test_data.sample(n=sample_size, random_state=0)

        X_test = test_data.drop(columns=[label])
        X_test = feature_generator.transform(X_test)

        y_pred = model.predict(X_test)
        assert isinstance(y_pred, np.ndarray), f"Expected np.ndarray as model.predict(X_test) output. Got: {y_pred.__class__}"

        y_pred_proba = model.predict_proba(X_test)
        assert isinstance(y_pred_proba, np.ndarray), f"Expected np.ndarray as model.predict_proba(X_test) output. Got: {y_pred.__class__}"
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


@contextmanager
def mock_system_resourcses(num_cpus=None, num_gpus=None):
    original_get_cpu_count = ResourceManager.get_cpu_count
    original_get_gpu_count = ResourceManager.get_gpu_count
    if num_cpus is not None:
        ResourceManager.get_cpu_count = lambda: num_cpus
    if num_gpus is not None:
        ResourceManager.get_gpu_count = lambda: num_gpus
    yield
    ResourceManager.get_cpu_count = original_get_cpu_count
    ResourceManager.get_gpu_count = original_get_gpu_count


@pytest.fixture
def dataset_loader_helper():
    return DatasetLoaderHelper


@pytest.fixture
def fit_helper():
    return FitHelper


@pytest.fixture
def model_fit_helper():
    return ModelFitHelper


@pytest.fixture
def mock_system_resources_ctx_mgr():
    return mock_system_resourcses


@pytest.fixture
def mock_num_cpus():
    return 16


@pytest.fixture
def mock_num_gpus():
    return 2


@pytest.fixture
def k_fold():
    return 2


@pytest.fixture
def stacked_overfitting_assert_func():
    return stacked_overfitting_assert


def stacked_overfitting_assert(lb, predictor, expected_stacked_overfitting_at_val, expected_stacked_overfitting_at_test):
    if expected_stacked_overfitting_at_val is not None:
        assert predictor._stacked_overfitting_occurred == expected_stacked_overfitting_at_val, "Expected stacked overfitting at val mismatch!"

    if expected_stacked_overfitting_at_test is not None:
        stacked_overfitting = check_stacked_overfitting_from_leaderboard(lb)
        assert stacked_overfitting == expected_stacked_overfitting_at_test, "Expected stacked overfitting at test mismatch!"
