import os
import shutil
import tempfile

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import BEST

from ..utils import PetFinderDataset, verify_predictor_save_load


def test_mixup():
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "optim.top_k_average_method": BEST,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "data.mixup.turn_on": True,
        "model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
    }

    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor.fit(
            train_data=dataset.train_df,
            time_limit=10,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)


def test_trivialaugment():
    dataset = PetFinderDataset()
    metric_name = dataset.metric

    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    hyperparameters = {
        "optim.max_epochs": 1,
        "optim.top_k_average_method": BEST,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "data.categorical.convert_to_text": False,
        "data.numerical.convert_to_text": False,
        "data.mixup.turn_on": True,
        "model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model.hf_text.text_trivial_aug_maxscale": 0.1,
        "model.hf_text.text_aug_detect_length": 10,
        "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
        "model.timm_image.train_transforms": ["resize_shorter_side", "center_crop", "trivial_augment"],
    }

    with tempfile.TemporaryDirectory() as save_path:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        predictor.fit(
            train_data=dataset.train_df,
            time_limit=30,
            save_path=save_path,
            hyperparameters=hyperparameters,
        )

        score = predictor.evaluate(dataset.test_df)
        verify_predictor_save_load(predictor, dataset.test_df)
