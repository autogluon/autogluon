import os
import shutil
import tempfile
from unittest import mock

import numpy.testing as npt
import pandas as pd
import pytest
import torch

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor

from ..utils.utils import get_home_dir

DOC_PATH_COL = "doc_path"


def path_expander(path, base_folder):
    path_l = path.split(";")
    return ";".join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])


def get_rvl_cdip_sample_data(
    zip_file="https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample.zip",
    dataset_folder="rvl_cdip_sample",
    dataset_file_name="rvl_cdip_train_data.csv",
):

    download_dir = "./ag_automm_rvl_cdip"
    load_zip.unzip(zip_file, unzip_dir=download_dir)

    dataset_path = os.path.join(download_dir, dataset_folder)
    rvl_cdip_data = pd.read_csv(f"{dataset_path}/{dataset_file_name}")
    train_data = rvl_cdip_data.sample(frac=0.8, random_state=200)
    test_data = rvl_cdip_data.drop(train_data.index)

    train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
    test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
    return train_data, test_data


def test_doc_classifier_standalone():
    requests_gag = mock.patch(
        "requests.Session.request",
        mock.Mock(side_effect=RuntimeError("Please use the `responses` library to mock HTTP in your tests.")),
    )
    predictor = MultiModalPredictor(label="label", verbosity=5)
    train_data, test_data = get_rvl_cdip_sample_data()

    save_path = os.path.join(get_home_dir(), "standalone_doc", "false")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.document_transformer.checkpoint_name": "microsoft/layoutlm-base-uncased",
            "env.num_workers": 0,
        },
        time_limit=150,
        save_path=save_path,
    )

    save_path_standalone = os.path.join(get_home_dir(), "standalone_doc", "true")

    predictor.save(
        path=save_path_standalone,
        standalone=True,
    )

    del predictor
    torch.cuda.empty_cache()

    loaded_online_predictor = MultiModalPredictor.load(path=save_path)
    online_predictions = loaded_online_predictor.predict(test_data, as_pandas=False)
    del loaded_online_predictor

    with requests_gag:
        # No internet connection here. If any command require internet connection, a RuntimeError will be raised!
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.hub.set_dir(tmpdirname)  # block reading files in `.cache`.
            loaded_offline_predictor = MultiModalPredictor.load(path=save_path_standalone)

    offline_predictions = loaded_offline_predictor.predict(test_data, as_pandas=False)
    del loaded_offline_predictor
    npt.assert_equal(online_predictions[0], offline_predictions[0])
