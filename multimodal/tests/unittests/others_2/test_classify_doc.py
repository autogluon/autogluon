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
from autogluon.multimodal.utils import path_expander

from ..utils import get_home_dir

DOC_PATH_COL = "doc_path"


def get_rvl_cdip_sample_data(
    zip_file="https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample.zip",
    dataset_folder="rvl_cdip_sample",
    dataset_file_name="rvl_cdip_train_data.csv",
    download_dir="./ag_automm_doc_image",
):
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    load_zip.unzip(zip_file, unzip_dir=download_dir)

    dataset_path = os.path.join(download_dir, dataset_folder)
    rvl_cdip_data = pd.read_csv(f"{dataset_path}/{dataset_file_name}")
    train_data = rvl_cdip_data.sample(frac=0.8, random_state=200)
    test_data = rvl_cdip_data.drop(train_data.index)

    train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
    test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=download_dir))
    return train_data, test_data


@pytest.mark.parametrize(
    "checkpoint_name",
    [("google/electra-small-discriminator"), ("microsoft/layoutlmv3-base")],
)
def test_doc_classification(checkpoint_name):
    predictor = MultiModalPredictor(label="label")
    train_data, test_data = get_rvl_cdip_sample_data()
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.document_transformer.checkpoint_name": checkpoint_name,
            "env.num_workers": 0,
        },
        time_limit=30,
    )

    predictor.evaluate(test_data)

    doc_path = test_data.iloc[0][DOC_PATH_COL]
    # make predictions
    predictions = predictor.predict({DOC_PATH_COL: [doc_path]})
    # output probability
    proba = predictor.predict_proba({DOC_PATH_COL: [doc_path]})
    # extract embeddings
    feature = predictor.extract_embedding({DOC_PATH_COL: [doc_path]})


@pytest.mark.parametrize(
    "checkpoint_name",
    [("google/electra-small-discriminator"), ("microsoft/layoutlmv3-base")],
)
def test_pdf_doc_classification(checkpoint_name):
    predictor = MultiModalPredictor(label="label")
    train_data, test_data = get_rvl_cdip_sample_data(
        zip_file="https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample_pdf.zip",
        dataset_folder="rvl_cdip_sample_pdf",
        dataset_file_name="rvl_cdip_pdf.csv",
        download_dir="./ag_automm_doc_pdf",
    )
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.document_transformer.checkpoint_name": checkpoint_name,
            "env.per_gpu_batch_size": 2,
            "env.num_workers": 0,
        },
        time_limit=30,
    )

    predictor.evaluate(test_data)

    doc_path = test_data.iloc[0][DOC_PATH_COL]
    # make predictions
    predictions = predictor.predict({DOC_PATH_COL: [doc_path]})
    # output probability
    proba = predictor.predict_proba({DOC_PATH_COL: [doc_path]})
    # extract embeddings
    feature = predictor.extract_embedding({DOC_PATH_COL: [doc_path]})


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
