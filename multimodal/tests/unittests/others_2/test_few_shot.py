import uuid

import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import BINARY, FEW_SHOT_CLASSIFICATION, MULTICLASS
from autogluon.multimodal.utils.misc import shopee_dataset

from ..predictor.test_predictor import verify_predictor_save_load, verify_realtime_inference


def verify_predict_predict_proba(test_data, predictor):
    preds = predictor.predict(test_data)
    proba = predictor.predict_proba(test_data, as_pandas=False)
    assert len(proba) == len(test_data)
    assert (proba.argmax(axis=1) == preds).all()


def verify_predict_as_pandas_multiclass(test_data, predictor):
    pandas_pred = predictor.predict(test_data, as_pandas=True)
    pandas_proba = predictor.predict_proba(test_data, as_pandas=True)
    pandas_proba_as_multiclass = predictor.predict_proba(test_data, as_pandas=True, as_multiclass=True)
    pandas_proba_no_multiclass = predictor.predict_proba(test_data, as_pandas=True, as_multiclass=False)

    proba_as_multiclass = predictor.predict_proba(test_data, as_pandas=False, as_multiclass=True)
    proba_as_multiclass = predictor.predict_proba(test_data, as_pandas=False, as_multiclass=True)


def verify_predict_single_column(test_data, predictor):
    test_column = test_data.drop(columns=["label"], axis=1)
    preds = predictor.predict(test_column)
    assert len(preds) == len(test_data)

    acc = (preds == test_data["label"].to_numpy()).sum() / len(test_data)

    preds2 = predictor.predict(test_data)
    assert len(preds2) == len(test_data)
    assert (preds == preds2).all()
    return preds


@pytest.mark.single_gpu
def test_fewshot_svm_fit_predict():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)
    save_path = f"./tmp/{uuid.uuid4().hex}-automm_stanfordcars-8shot-en"
    predictor = MultiModalPredictor(
        label="label",
        problem_type=FEW_SHOT_CLASSIFICATION,
        eval_metric="acc",
        path=save_path,
    )
    predictor.fit(train_data)
    verify_predictor_save_load(predictor, test_data, verify_embedding=True)
    verify_realtime_inference(predictor, test_data, verify_embedding=True)
    verify_predict_single_column(test_data, predictor)
    verify_predict_predict_proba(test_data, predictor)
    verify_predict_as_pandas_multiclass(test_data, predictor)


def test_fewshot_svm_save_load():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)
    save_path = f"./tmp/{uuid.uuid4().hex}-automm_stanfordcars-8shot-en"
    predictor = MultiModalPredictor(
        label="label",
        problem_type=FEW_SHOT_CLASSIFICATION,
        eval_metric="acc",
        path=save_path,
    )

    predictor.fit(train_data)
    results = predictor.evaluate(test_data)
    preds = predictor.predict(test_data.drop(columns=["label"], axis=1))
    predictor2 = MultiModalPredictor.load(save_path)
    results2 = predictor2.evaluate(test_data)
    preds2 = predictor.predict(test_data.drop(columns=["label"], axis=1))
    assert results == results2
    assert (preds == preds2).all()
    predictor2.fit(train_data)
