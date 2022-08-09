import os
import shutil

from test_predictor import verify_predictor_save_load
from unittest_datasets import AEDataset, HatefulMeMesDataset, PetFinderDataset
from utils import get_home_dir

from autogluon.multimodal import MultiModalPredictor

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
}


def test_distillation():
    dataset = PetFinderDataset()

    teacher_hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["hf_text", "timm_image", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    # test for distillation with different model structures
    student_hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["hf_text", "timm_image", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "distiller.temperature": 5.0,
        "distiller.hard_label_weight": 0.1,
        "distiller.soft_label_weight": 1.0,
        "distiller.output_feature_loss_weight": 0.01,
        "distiller.rkd_distance_loss_weight": 0.01,
        "distiller.rkd_angle_loss_weight": 0.02,
        "distiller.output_feature_loss_type": "mse",
    }

    teacher_predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    teacher_save_path = os.path.join(get_home_dir(), "petfinder", "teacher")
    if os.path.exists(teacher_save_path):
        shutil.rmtree(teacher_save_path)

    teacher_predictor = teacher_predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=teacher_hyperparameters,
        time_limit=30,
        save_path=teacher_save_path,
    )

    # test for distillation
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    student_save_path = os.path.join(get_home_dir(), "petfinder", "student")
    if os.path.exists(student_save_path):
        shutil.rmtree(student_save_path)

    predictor = predictor.fit(
        train_data=dataset.train_df,
        teacher_predictor=teacher_predictor,
        hyperparameters=student_hyperparameters,
        time_limit=30,
        save_path=student_save_path,
    )
    verify_predictor_save_load(predictor, dataset.test_df)

    # test for distillation with teacher predictor path
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    student_save_path = os.path.join(get_home_dir(), "petfinder", "student")
    if os.path.exists(student_save_path):
        shutil.rmtree(student_save_path)

    predictor = predictor.fit(
        train_data=dataset.train_df,
        teacher_predictor=teacher_predictor.path,
        hyperparameters=student_hyperparameters,
        time_limit=30,
        save_path=student_save_path,
    )

    verify_predictor_save_load(predictor, dataset.test_df)
