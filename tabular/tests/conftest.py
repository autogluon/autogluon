import copy
import os
import shutil
import uuid
import pytest
import tempfile

import autogluon.core as ag
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.tabular import TabularPrediction as task
from autogluon.tabular.task.tabular_prediction.predictor_v2 import TabularPredictorV2


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


class DatasetLoaderHelper:
    dataset_info_dict = dict(
        # Binary dataset
        adult={
            'url': 'https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip',
            'name': 'AdultIncomeBinaryClassification',
            'problem_type': BINARY,
            'label_column': 'class',
        },
        # Multiclass big dataset with 7 classes, all features are numeric. Runs SLOW.
        covertype={
            'url': 'https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification.zip',
            'name': 'CoverTypeMulticlassClassification',
            'problem_type': MULTICLASS,
            'label_column': 'Cover_Type',
        },
        # Regression with mixed feature-types, skewed Y-values.
        ames={
            'url': 'https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression.zip',
            'name': 'AmesHousingPriceRegression',
            'problem_type': REGRESSION,
            'label_column': 'SalePrice',
        },
        # Regression with multiple text field and categorical
        sts={
            'url': 'https://autogluon-text.s3-us-west-2.amazonaws.com/glue_sts.zip',
            'name': 'SemanticTextualSimilarity',
            'problem_type': REGRESSION,
            'label_column': 'score',
        }
    )

    @staticmethod
    def load_dataset(name: str, directory_prefix: str = './datasets/'):
        dataset_info = copy.deepcopy(DatasetLoaderHelper.dataset_info_dict[name])
        train_file = dataset_info.pop('train_file', 'train_data.csv')
        test_file = dataset_info.pop('test_file', 'test_data.csv')
        name_inner = dataset_info.pop('name')
        url = dataset_info.pop('url', None)
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
            zip_name = ag.download(url, directory_prefix)
            ag.unzip(zip_name, directory_prefix)
            os.remove(zip_name)

        train_data = task.Dataset(file_path=train_file_path)
        test_data = task.Dataset(file_path=test_file_path)
        return train_data, test_data


class FitHelper:
    @staticmethod
    def fit_and_validate_dataset(dataset_name, fit_args, sample_size=1000, refit_full=True, delete_directory=True):
        directory_prefix = './datasets/'
        train_data, test_data, dataset_info = DatasetLoaderHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
        label_column = dataset_info['label_column']
        savedir = os.path.join(directory_prefix, dataset_name, f'AutogluonOutput_{uuid.uuid4()}')
        init_args = dict(
            label=label_column,
            path=savedir,
        )
        predictor = FitHelper.fit_dataset(train_data=train_data, init_args=init_args, fit_args=fit_args, sample_size=sample_size)
        if sample_size is not None and sample_size < len(test_data):
            test_data = test_data.sample(n=sample_size, random_state=0)
        predictor.predict(test_data)
        predictor.predict_proba(test_data)
        model_names = predictor.get_model_names()
        model_name = model_names[0]
        assert len(model_names) == 2
        if refit_full:
            refit_model_names = predictor.refit_full()
            assert len(refit_model_names) == 2
            refit_model_name = refit_model_names[model_name]
            assert '_FULL' in refit_model_name
            predictor.predict(test_data, model=refit_model_name)
            predictor.predict_proba(test_data, model=refit_model_name)
        predictor.info()
        predictor.leaderboard(test_data, extra_info=True)
        assert os.path.realpath(savedir) == os.path.realpath(predictor.output_directory)
        if delete_directory:
            shutil.rmtree(savedir, ignore_errors=True)  # Delete AutoGluon output directory to ensure runs' information has been removed.
        return predictor

    @staticmethod
    def fit_dataset(train_data, init_args, fit_args, sample_size=None):
        if sample_size is not None and sample_size < len(train_data):
            train_data = train_data.sample(n=sample_size, random_state=0)
        return TabularPredictorV2(**init_args).fit(train_data, **fit_args)


@pytest.fixture
def dataset_loader_helper():
    return DatasetLoaderHelper


@pytest.fixture
def fit_helper():
    return FitHelper