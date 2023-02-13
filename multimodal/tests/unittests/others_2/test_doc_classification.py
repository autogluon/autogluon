import os

import pandas as pd
import pytest

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor

DOC_PATH_COL = "doc_path"


def path_expander(path, base_folder):
    path_l = path.split(";")
    return ";".join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])


def get_rvl_cdip_sample_data():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/doc_classification/rvl_cdip_sample.zip"

    download_dir = "./ag_automm_rvl_cdip"
    load_zip.unzip(zip_file, unzip_dir=download_dir)

    dataset_path = os.path.join(download_dir, "rvl_cdip_sample")
    rvl_cdip_data = pd.read_csv(f"{dataset_path}/rvl_cdip_train_data.csv")
    train_data = rvl_cdip_data.sample(frac=0.8, random_state=200)
    test_data = rvl_cdip_data.drop(train_data.index)

    train_data[DOC_PATH_COL] = train_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[DOC_PATH_COL] = test_data[DOC_PATH_COL].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

    return train_data, test_data


# Both text only backbones and document foundation models are supported.
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
