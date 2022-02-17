import os
import json
import pytest

from autogluon.text.automm import AutoMMPredictor
from autogluon.text.automm.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
    BINARY,
    MULTICLASS,
)
from datasets import (
    PetFinderDataset,
    HatefulMeMesDataset,
    AEDataset,
)
from utils import get_home_dir

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
}


@pytest.mark.parametrize(
    "dataset_name,"
    "model_config,"
    "text_backbone,"
    "image_backbone",
    # "score",
    [
        (
            "petfinder",
            "fusion_mlp_image_text_tabular",
            "prajjwal1/bert-tiny",
            "swin_tiny_patch4_window7_224",
            # 0.15,  # 0.1847
        ),

        (
            "hateful_memes",
            "fusion_mlp_image_text",
            "prajjwal1/bert-tiny",
            "swin_tiny_patch4_window7_224",
            # 0.65,  # 0.6815
        ),

        (
            "hateful_memes",
            "hf_text",
            "prajjwal1/bert-tiny",
            None,
            # 0.6,  # 0.63687
        ),

        (
            "hateful_memes",
            "timm_image",
            None,
            "swin_tiny_patch4_window7_224",
            # 0.6,  # 0.6312
        ),

        (
            "hateful_memes",
            "clip",
            None,
            None,
            # 0.7,  # 0.7318
        ),

        (
            "ae",
            "hf_text",
            "prajjwal1/bert-tiny",
            None,
            # 0.3,  # 0.3367
        ),

    ]
)
def test_predictor(
        dataset_name,
        model_config,
        text_backbone,
        image_backbone,
        # score,
):
    dataset = ALL_DATASETS[dataset_name]()
    metric_name = dataset.metric
    test_metric_name = dataset.test_metric if hasattr(dataset, "test_metric") else metric_name

    if metric_name.lower() == "r2":
        # For regression, we use rmse as the evaluation metric, but use r2 for the test metric
        metric_name = "rmse"

    predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_config_path = os.path.join(cur_path, f"configs/model/{model_config}.yaml")
    data_config_path = os.path.join(cur_path, "configs/data/default.yaml")
    optimization_config_path = os.path.join(cur_path, "configs/optimization/adamw.yaml")
    environemnt_config_path = os.path.join(cur_path, "configs/environment/default.yaml")
    config = {
        MODEL: model_config_path,
        DATA: data_config_path,
        OPTIMIZATION: optimization_config_path,
        ENVIRONMENT: environemnt_config_path,
    }
    hyperparameters = {"optimization.max_epochs": 1}
    if text_backbone is not None:
        hyperparameters.update({
            "model.hf_text.checkpoint_name": text_backbone,
        })
    if image_backbone is not None:
        hyperparameters.update({
            "model.timm_image.checkpoint_name": image_backbone,
        })
    save_path = os.path.join(get_home_dir(), "outputs", dataset_name, model_config)
    if text_backbone is not None:
        save_path = os.path.join(save_path, text_backbone)
    if image_backbone is not None:
        save_path = os.path.join(save_path, image_backbone)
    print(f"save_path: {save_path}")
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=save_path,
    )
    scores, y_pred = predictor.evaluate(
        data=dataset.test_df,
        metrics=[test_metric_name],
        return_pred=True,
    )
    if predictor.problem_type in [BINARY, MULTICLASS]:
        y_pred_prob = predictor.predict_proba(
            data=dataset.test_df,
        )
    with open(os.path.join(predictor.path, 'test_metrics.json'), 'w') as fp:
        json.dump(scores, fp)

    # assert scores[test_metric_name] >= score

    # test saving and loading
    predictor = AutoMMPredictor.load(
        path=predictor.path,
    )
    scores_2, y_pred_2 = predictor.evaluate(
        data=dataset.test_df,
        metrics=[test_metric_name],
        return_pred=True,
    )
    assert scores == scores_2
    assert y_pred.equals(y_pred_2)

    if predictor.problem_type in [BINARY, MULTICLASS]:
        y_pred_prob_2 = predictor.predict_proba(
            data=dataset.test_df,
        )
        assert y_pred_prob.equals(y_pred_prob_2)
