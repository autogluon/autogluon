import numpy as np

from autogluon.multimodal.utils.few_shot_learning import FewShotSVMPredictor
from autogluon.multimodal.utils.misc import shopee_dataset


def test_fewshot_fit_predict():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    hyperparameters = {
        "model.names": ["timm_image"],
        "env.num_workers": 2,
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.eval_batch_size_ratio": 1,
    }

    import uuid

    model_path = f"./tmp/{uuid.uuid4().hex}-automm_stanfordcars-8shot-en"
    predictor = FewShotSVMPredictor(
        label="label",  # column name of the label
        hyperparameters=hyperparameters,
        eval_metric="acc",
        path=model_path,  # path to save model and artifacts
    )
    predictor.fit(train_data)
    test_column = test_data.drop(columns=["label"], axis=1)
    preds = predictor.predict(test_column)
    assert len(preds) == len(test_data)

    acc = (preds == test_data["label"].to_numpy()).sum() / len(test_data)

    preds2 = predictor.predict(test_data)
    assert len(preds2) == len(test_data)
    assert (preds == preds2).all()

    results = predictor.evaluate(test_data)

    proba = predictor.predict_proba(test_data)
    assert len(proba) == len(test_data)
    assert (proba.argmax(axis=1) == preds).all()

    pandas_pred = predictor.predict(test_data, as_pandas=True)
    pandas_proba = predictor.predict_proba(test_data, as_pandas=True)
    pandas_proba_as_multiclass = predictor.predict_proba(test_data, as_pandas=True, as_multiclass=True)
    pandas_proba_no_multiclass = predictor.predict_proba(test_data, as_pandas=True, as_multiclass=False)

    proba_as_multiclass = predictor.predict_proba(test_data, as_pandas=False, as_multiclass=True)
    proba_as_multiclass = predictor.predict_proba(test_data, as_pandas=False, as_multiclass=True)


def test_fewshot_save_load():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    hyperparameters = {
        "model.names": ["timm_image"],
        "env.num_workers": 2,
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.eval_batch_size_ratio": 1,
    }

    import uuid

    model_path = f"./tmp/{uuid.uuid4().hex}-automm_stanfordcars-8shot-en"
    predictor = FewShotSVMPredictor(
        label="label",  # column name of the label
        hyperparameters=hyperparameters,
        eval_metric="acc",
        path=model_path,  # path to save model and artifacts
    )

    predictor.fit(train_data)
    results = predictor.evaluate(test_data)
    preds = predictor.predict(test_data.drop(columns=["label"], axis=1))

    predictor2 = FewShotSVMPredictor.load(model_path)
    results2 = predictor2.evaluate(test_data)
    preds2 = predictor.predict(test_data.drop(columns=["label"], axis=1))
    assert results == results2
    assert (preds == preds2).all()


if __name__ == "__main__":
    test_fewshot_fit_predict()
    # test_fewshot_save_load()
