from autogluon.multimodal.utils.few_shot_learning import FewShotSVMPredictor
from autogluon.multimodal.utils.misc import shopee_dataset


def test_fewshot_fit_predict():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    hyperparameters = {
        "model.names": ["clip"],
        "model.clip.max_text_len": 0,
        "env.num_workers": 2,
        "model.clip.checkpoint_name": "swin_tiny_patch4_window7_224",
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
    print(train_data)
    predictor.fit(train_data)
    test_column = test_data.drop(columns=["label"], axis=1)
    preds = predictor.predict(test_column)

    print(preds)
    assert len(preds) == len(test_data)

    acc = (preds == test_data["label"].to_numpy()).sum() / len(test_data)
    print(acc)


def test_fewshot_fit_eval():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    hyperparameters = {
        "model.names": ["clip"],
        "model.clip.max_text_len": 0,
        "env.num_workers": 2,
        "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
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
    print(train_data)
    print(test_data)
    predictor.fit(train_data)
    results = predictor.evaluate(test_data)

    print(results)


def test_fewshot_save_load():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    hyperparameters = {
        "model.names": ["clip"],
        "model.clip.max_text_len": 0,
        "env.num_workers": 2,
        "model.clip.checkpoint_name": "openai/clip-vit-large-patch14-336",
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

    predictor2 = FewShotSVMPredictor.load(model_path)
    results2 = predictor2.evaluate(test_data)

    assert results == results2


if __name__ == "__main__":
    test_fewshot_fit_predict()
